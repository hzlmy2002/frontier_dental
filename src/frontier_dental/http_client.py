"""Async HTTP client with retry/backoff and a token-bucket rate limiter.

Used by the deterministic Tier 1 of the Extractor (and anywhere else we want
HTTP without spinning up Chromium).
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import Settings

log = structlog.get_logger(__name__)


class TokenBucket:
    """Simple async token bucket. ``rps`` tokens replenish per second."""

    def __init__(self, rps: float, capacity: float | None = None) -> None:
        self.rps = max(rps, 0.001)
        self.capacity = capacity if capacity is not None else max(rps, 1.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rps)
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self.rps
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class RateLimitedClient:
    """``httpx.AsyncClient`` wrapper that enforces rate limiting + retries."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bucket = TokenBucket(settings.rate_limit_rps)
        self._client = httpx.AsyncClient(
            timeout=settings.request_timeout_s,
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )

    async def __aenter__(self) -> RateLimitedClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(self, url: str) -> httpx.Response:
        await self._bucket.acquire()
        retryer: AsyncRetrying = AsyncRetrying(
            stop=stop_after_attempt(self._settings.max_retries),
            wait=wait_exponential_jitter(initial=1, max=10),
            retry=retry_if_exception_type(
                (httpx.TransportError, httpx.HTTPStatusError, httpx.TimeoutException)
            ),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                resp = await self._client.get(url)
                if resp.status_code >= 500:
                    log.warning("upstream_5xx", url=url, status=resp.status_code)
                    raise httpx.HTTPStatusError(
                        f"{resp.status_code}", request=resp.request, response=resp
                    )
                resp.raise_for_status()
                return resp
        raise RuntimeError("unreachable")

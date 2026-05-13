"""Playwright-based page fetcher (Tier 2 of the extractor pipeline).

Lazily launches a single Chromium browser for the lifetime of a pipeline run.
Used when ``httpx`` returned HTML insufficient for extraction (typically
Vue/Algolia listings or PDPs whose structured data only appears after
hydration).

Uses the **async Playwright API**. Tests do not exercise this class directly —
they replace the fetcher with a stub. Real usage is covered by the
end-to-end ``--max-products-per-category`` smoke run in Phase 5.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

import structlog

from .config import Settings, get_settings

log = structlog.get_logger(__name__)


class PlaywrightFetcher:
    """Render a URL with Chromium and return the post-hydration HTML."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._playwright: Any = None
        self._browser: Any = None

    async def __aenter__(self) -> PlaywrightFetcher:
        await self._start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def _start(self) -> None:
        if self._playwright is not None:
            return
        # Lazy import — keeps the module importable for tests that mock the
        # fetcher and don't have Chromium installed.
        from playwright.async_api import async_playwright  # type: ignore[import-not-found]

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=True)
        log.info("playwright_started")

    async def fetch(self, url: str) -> str:
        await self._start()
        assert self._browser is not None
        ctx = await self._browser.new_context(user_agent=self._settings.user_agent)
        try:
            page = await ctx.new_page()
            await page.goto(url, wait_until="networkidle", timeout=self._settings.request_timeout_s * 1000)
            html = await page.content()
            return html  # type: ignore[no-any-return]
        finally:
            await ctx.close()

    async def aclose(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            finally:
                self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            finally:
                self._playwright = None
        log.info("playwright_stopped")

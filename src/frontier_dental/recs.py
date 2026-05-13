"""Adobe Commerce recommendations API client.

Safco's storefront is powered by Adobe Commerce, which exposes a public
``/recs/v1/precs/preconfigured`` endpoint that returns recommended /
alternative products for a given SKU. We use it to populate the
``alternative_products`` field on every ``ProductRecord`` — the field
that JSON-LD and DOM parsing leave empty.

The endpoint:
- is unauthenticated (uses a public ``recs_open`` API key);
- is stateless when called with empty ``userViewHistory`` /
  ``cartSkus`` / ``userPurchaseHistory`` arrays;
- returns ``results[]`` (one entry per recommendation unit, e.g.
  "Trending Products", "More Like This"), each with a ``products[]``
  list carrying ``sku``, ``name``, ``url``, ``categories``, prices, and
  rich attributes (``manufacturer_name``, ``manufacturer_part_number``).

We dedupe across units by SKU, drop the current SKU itself if it
echoes back, normalize protocol-relative URLs (``//host/path`` →
``https://host/path``), and cap output at
``Settings.recs_max_alternatives``.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from .config import Settings, get_settings

log = structlog.get_logger(__name__)


class RecsClient:
    """Fetch alternative-product URLs by SKU from Adobe's recs API.

    Owns its own ``httpx.AsyncClient`` by default so callers don't have to
    plumb one through. Pass ``client=`` to share an external client (e.g.
    in tests with ``httpx.MockTransport``).
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    async def fetch_alternatives(self, sku: str) -> tuple[str, ...]:
        """Return up to ``recs_max_alternatives`` product URLs related to
        ``sku``. Empty tuple on any error — recs are non-critical, the
        rest of the record stays valid."""

        if not sku:
            return ()
        client = self._client or self._build_client()
        try:
            payload = await self._request(client, sku)
        except Exception as e:  # noqa: BLE001 — recs is non-critical
            log.warning("recs_fetch_failed", sku=sku, error=str(e))
            return ()
        finally:
            if self._owns_client and self._client is None:
                # Per-call client created on the fly; close it.
                await client.aclose()

        urls = _parse_alternative_urls(
            payload,
            current_sku=sku,
            max_count=self._settings.recs_max_alternatives,
        )
        log.info("recs_fetched", sku=sku, count=len(urls))
        return urls

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=4.0),
        reraise=True,
    )
    async def _request(self, client: httpx.AsyncClient, sku: str) -> dict[str, Any]:
        s = self._settings
        body: dict[str, Any] = {
            "environmentId": s.recs_environment_id,
            "alternateEnvironmentId": "",
            "storeCode": s.recs_store_code,
            "storeViewCode": s.recs_store_view_code,
            "websiteCode": s.recs_website_code,
            "pageType": "Product",
            "category": "",
            "currentSku": sku,
            "cartSkus": [],
            "userViewHistorySkus": [],
            "userViewHistory": [],
            "userPurchaseHistory": [],
            "defaultStoreViewCode": "",
            "customerGroupCode": "",
        }
        origin = s.base_url.rstrip("/")
        resp = await client.post(
            s.recs_endpoint,
            headers={
                "Accept": "*/*",
                "Content-Type": "application/json",
                "X-Api-Key": s.recs_api_key,
                "Origin": origin,
                "Referer": origin + "/",
            },
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise httpx.HTTPError(f"recs API returned non-object: {type(data).__name__}")
        return data

    def _build_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=self._settings.request_timeout_s,
            headers={"User-Agent": self._settings.user_agent},
        )

    async def aclose(self) -> None:
        if self._client is not None and not self._owns_client:
            return
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_alternative_urls(
    payload: dict[str, Any], *, current_sku: str, max_count: int
) -> tuple[str, ...]:
    """Extract product URLs from a recs API response: deduped by URL,
    current SKU filtered out, capped at ``max_count``, rank-order
    preserved across units."""

    seen: set[str] = set()
    out: list[str] = []
    results = payload.get("results") or []
    if not isinstance(results, list):
        return ()
    for unit in results:
        if not isinstance(unit, dict):
            continue
        products = unit.get("products") or []
        if not isinstance(products, list):
            continue
        for product in products:
            if not isinstance(product, dict):
                continue
            sku = product.get("sku")
            if not sku or sku == current_sku:
                continue
            url = _normalize_safco_url(product.get("url") or "")
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(url)
            if len(out) >= max_count:
                return tuple(out)
    return tuple(out)


def _normalize_safco_url(url: str) -> str:
    """Adobe returns protocol-relative URLs (``//host/path``); upgrade to
    ``https://host/path`` and pass through already-absolute URLs."""

    if not isinstance(url, str):
        return ""
    candidate = url.strip()
    if not candidate:
        return ""
    if candidate.startswith("//"):
        return "https:" + candidate
    if candidate.startswith("/product/"):
        return "https://www.safcodental.com" + candidate
    if candidate.startswith(("http://", "https://")):
        return candidate
    return ""

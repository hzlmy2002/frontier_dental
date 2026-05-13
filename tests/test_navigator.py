"""Tests for the Navigator (category enumeration) and ListingCrawler
(per-category product URL extraction). The production runner (Playwright +
LangChain) is bypassed via an injected fake so tests never touch the
network or spin up Chromium.
"""

from __future__ import annotations

import json

import httpx
import pytest

from frontier_dental.agents.navigator import (
    DiscoveredCategories,
    ListingCrawler,
    Navigator,
    StaticCategoryDiscoverer,
    _parse_algolia_config,
    filter_categories,
    filter_product_urls,
)
from frontier_dental.models import CategoryRef


def _cat(name: str, url: str) -> CategoryRef:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return CategoryRef(name=name, url=url, slug=slug)


# --- filter_categories -------------------------------------------------------


def test_filter_categories_dedupes_and_drops_non_catalog_urls() -> None:
    raw = [
        _cat("Gloves", "https://www.safcodental.com/catalog/gloves"),
        _cat("Gloves dup", "https://www.safcodental.com/catalog/gloves/"),
        _cat("Sutures", "https://www.safcodental.com/catalog/sutures-surgical-products"),
        _cat("Off-domain", "https://other.example.com/catalog/x"),
        _cat("Catalog root", "https://www.safcodental.com/catalog"),
        _cat("Product (not category)", "https://www.safcodental.com/product/abc"),
    ]
    out = filter_categories(raw, base_url="https://www.safcodental.com")
    assert [c.url for c in out] == [
        "https://www.safcodental.com/catalog/gloves",
        "https://www.safcodental.com/catalog/sutures-surgical-products",
    ]
    assert [c.slug for c in out] == ["gloves", "sutures-surgical-products"]


def test_filter_categories_drops_subcategory_urls() -> None:
    """Subcategory URLs (multi-segment paths) are top-level-only territory:
    the per-category ListingCrawler reaches subcategory products from the
    parent listing, so we never want a subcategory URL fed back into
    discovery as a fresh category."""

    raw = [
        _cat("Gloves", "https://www.safcodental.com/catalog/gloves"),
        _cat("Nitrile", "https://www.safcodental.com/catalog/gloves/nitrile-gloves"),
        _cat("Surgical", "https://www.safcodental.com/catalog/gloves/surgical-gloves"),
    ]
    out = filter_categories(raw, base_url="https://www.safcodental.com")
    assert [c.slug for c in out] == ["gloves"]


def test_filter_categories_promotes_relative_urls() -> None:
    raw = [_cat("Gloves", "/catalog/gloves")]
    out = filter_categories(raw, base_url="https://www.safcodental.com")
    assert len(out) == 1
    assert out[0].url == "https://www.safcodental.com/catalog/gloves"


def test_filter_categories_drops_magento_route_prefix() -> None:
    """``_CATEGORY_PATH_RE`` truncates legacy URLs like
    ``/catalog/category/view/s/rubber-dam/id/913/`` to the prefix
    ``/catalog/category`` (because '/' isn't in the slug character class).
    Without explicit filtering that bogus 'category' slug would surface as a
    discovered category and its listing crawl would 404."""

    raw = [
        _cat("Gloves", "https://www.safcodental.com/catalog/gloves"),
        _cat("Magento route prefix", "https://www.safcodental.com/catalog/category"),
    ]
    out = filter_categories(raw, base_url="https://www.safcodental.com")
    assert [c.slug for c in out] == ["gloves"]


# --- Navigator (category discovery) -----------------------------------------


@pytest.mark.asyncio
async def test_navigator_returns_all_categories_by_default() -> None:
    captured: dict[str, str | None] = {}

    async def runner(intent: str | None) -> DiscoveredCategories:
        captured["intent"] = intent
        return DiscoveredCategories(
            categories=[
                _cat("Gloves", "https://www.safcodental.com/catalog/gloves"),
                _cat("Sutures", "https://www.safcodental.com/catalog/sutures-surgical-products"),
                _cat("Anesthetics", "https://www.safcodental.com/catalog/anesthetics"),
            ]
        )

    nav = Navigator(runner=runner)
    out = await nav.discover_categories()
    assert captured["intent"] is None
    assert [c.slug for c in out] == ["gloves", "sutures-surgical-products", "anesthetics"]


@pytest.mark.asyncio
async def test_navigator_forwards_intent_to_runner() -> None:
    captured: dict[str, str | None] = {}

    async def runner(intent: str | None) -> DiscoveredCategories:
        captured["intent"] = intent
        return DiscoveredCategories(
            categories=[_cat("Gloves", "https://www.safcodental.com/catalog/gloves")]
        )

    nav = Navigator(runner=runner)
    out = await nav.discover_categories(intent="I want gloves and surgical supplies")
    assert captured["intent"] == "I want gloves and surgical supplies"
    assert [c.slug for c in out] == ["gloves"]


@pytest.mark.asyncio
async def test_navigator_returns_empty_list_on_runner_exception() -> None:
    async def boom(intent: str | None) -> DiscoveredCategories:
        del intent  # signature dictated by NavigatorRunner protocol
        raise RuntimeError("LLM down")

    nav = Navigator(runner=boom)
    out = await nav.discover_categories()
    assert out == []


# --- ListingCrawler ---------------------------------------------------------


def test_filter_product_urls_keeps_only_safco_product_urls() -> None:
    urls = [
        "https://www.safcodental.com/product/alasta-pro",
        "https://www.safcodental.com/product/alasta-pro/",
        "https://www.safcodental.com/catalog/gloves",
        "https://example.com/product/xyz",
        "/product/relative-form",
        "",
        "https://www.safcodental.com/product/Has_Caps",
    ]
    out = filter_product_urls(urls, base_url="https://www.safcodental.com")
    assert out == [
        "https://www.safcodental.com/product/alasta-pro",
        "https://www.safcodental.com/product/relative-form",
    ]


@pytest.mark.asyncio
async def test_listing_crawler_paginates_via_algolia_api() -> None:
    """End-to-end happy path: scrape Algolia config from the catalog HTML,
    then walk pages until ``nbPages`` is exhausted. Verifies dedup across
    pages and that the request body actually carries the scraped
    ``categoryIds`` filter so the wrong category can't accidentally be
    returned without the test failing."""

    catalog_html = """
    <html><body>
      <script>
      window.algoliaConfig = {
        "applicationId": "TESTAPP123",
        "indexName": "testidx",
        "apiKey": "ZmFrZVNlYXJjaEtleQ=="
      };
      window.dataLayer.push({"event":"page_data","category_id":"385","category_name":"Gloves"});
      </script>
      <a href="/product/placeholder">irrelevant placeholder anchor</a>
    </body></html>
    """

    pages = {
        0: {
            "results": [{
                "hits": [
                    {"url": "https://www.safcodental.com/product/p1"},
                    {"url": "https://www.safcodental.com/product/p2"},
                ],
                "nbPages": 2,
                "nbHits": 3,
            }]
        },
        1: {
            "results": [{
                "hits": [
                    {"url": "https://www.safcodental.com/product/p1"},  # cross-page dup
                    {"url": "https://www.safcodental.com/product/p3"},
                ],
                "nbPages": 2,
                "nbHits": 3,
            }]
        },
    }

    posted_pages: list[int] = []
    posted_filters: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=catalog_html, request=request)
        # Algolia POST
        body = json.loads(request.content.decode())
        params = body["requests"][0]["params"]
        page = int(params.split("page=")[1].split("&")[0])
        posted_pages.append(page)
        posted_filters.append(params)
        return httpx.Response(200, json=pages[page], request=request)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = ListingCrawler(client=client)
        urls = await crawler.list_products("https://www.safcodental.com/catalog/gloves")

    assert posted_pages == [0, 1]
    assert all("categoryIds%3A385" in p for p in posted_filters)
    assert urls == [
        "https://www.safcodental.com/product/p1",
        "https://www.safcodental.com/product/p2",
        "https://www.safcodental.com/product/p3",
    ]


@pytest.mark.asyncio
async def test_listing_crawler_returns_empty_on_missing_algolia_config() -> None:
    """If the storefront HTML doesn't contain the inline Algolia config (e.g.
    template change, blocked, error page), bail out cleanly rather than
    raising. The pipeline keeps moving on to the next category."""

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text="<html>no algolia here</html>", request=request)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = ListingCrawler(client=client)
        urls = await crawler.list_products("https://www.safcodental.com/catalog/gloves")

    assert urls == []


@pytest.mark.asyncio
async def test_listing_crawler_returns_empty_on_catalog_fetch_failure() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, request=request)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        crawler = ListingCrawler(client=client)
        urls = await crawler.list_products("https://www.safcodental.com/catalog/gloves")

    assert urls == []


def test_parse_algolia_config_extracts_all_fields() -> None:
    html = """
      "applicationId":"A5ULKNTM8N","indexName":"safco_prod_default","apiKey":"abc=="
      ...later... "category_id":"385","category_name":"Gloves"
    """
    cfg = _parse_algolia_config(html)
    assert cfg is not None
    assert cfg.application_id == "A5ULKNTM8N"
    assert cfg.products_index == "safco_prod_default_products"
    assert cfg.api_key == "abc=="
    assert cfg.category_id == "385"


def test_parse_algolia_config_returns_none_when_any_field_missing() -> None:
    # Missing category_id
    html = '"applicationId":"X","indexName":"i","apiKey":"k"'
    assert _parse_algolia_config(html) is None


# --- StaticCategoryDiscoverer ----------------------------------------------


@pytest.mark.asyncio
async def test_static_discoverer_extracts_top_level_categories_from_html() -> None:
    payload = """
    <html><body>
      <nav>
        <a href="https://www.safcodental.com/catalog/gloves">Gloves</a>
        <a href="https://www.safcodental.com/catalog/anesthetics">Anesthetics</a>
        <a href="https://www.safcodental.com/catalog/gloves">dup</a>
        <a href="https://www.safcodental.com/catalog/gloves/nitrile-gloves">sub</a>
        <a href="https://www.safcodental.com/product/alasta-pro">prod</a>
        <a href="https://other.example.com/catalog/x">off-domain</a>
        <a href="https://www.safcodental.com/customer/account">acct</a>
      </nav>
    </body></html>
    """

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=payload, request=request)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        d = StaticCategoryDiscoverer(client=client)
        cats = await d.discover_categories()

    # Top-level only, deduped, in source order.
    assert [c.slug for c in cats] == ["gloves", "anesthetics"]
    assert all(c.url.startswith("https://www.safcodental.com/catalog/") for c in cats)


@pytest.mark.asyncio
async def test_static_discoverer_recovers_legacy_url_categories() -> None:
    """Some categories (e.g. rubber-dam, id 913) appear on the home page
    ONLY via the Magento legacy URL ``/catalog/category/view/s/<slug>/id/<id>/``.
    The plain ``/catalog/<slug>`` regex truncates that to the bogus prefix
    ``/catalog/category`` and ``filter_categories`` then drops it. This test
    pins the recovery path: we additionally regex-match the legacy URL,
    extract the slug, and synthesize a modern URL for it."""

    payload = """
    <html><body>
      <a href="https://www.safcodental.com/catalog/gloves">Gloves</a>
      <a href="https://www.safcodental.com/catalog/category/view/s/rubber-dam/id/913/">Rubber Dam</a>
    </body></html>
    """

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, text=payload, request=request)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        d = StaticCategoryDiscoverer(client=client)
        cats = await d.discover_categories()

    slugs = [c.slug for c in cats]
    assert "rubber-dam" in slugs
    assert "category" not in slugs  # the legacy-route prefix must NOT leak through
    rubber = next(c for c in cats if c.slug == "rubber-dam")
    assert rubber.url == "https://www.safcodental.com/catalog/rubber-dam"


@pytest.mark.asyncio
async def test_static_discoverer_returns_empty_on_http_error() -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    transport = httpx.MockTransport(boom)
    async with httpx.AsyncClient(transport=transport) as client:
        d = StaticCategoryDiscoverer(client=client)
        cats = await d.discover_categories()

    assert cats == []

"""Pipeline orchestrator tests.

Mocks Navigator, ListingCrawler, the HTTP client, and the Playwright fetcher
so the test suite never touches the network or spawns Chromium.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from frontier_dental.agents.extractor import Extractor
from frontier_dental.agents.navigator import (
    DiscoveredCategories,
    ListingCrawler,
    Navigator,
)
from frontier_dental.config import get_settings
from frontier_dental.http_client import RateLimitedClient
from frontier_dental.models import CategoryRef, CrawlStatus
from frontier_dental.pipeline import Pipeline
from frontier_dental.storage import Storage


CATEGORY_GLOVES = CategoryRef(
    name="Gloves", url="https://www.safcodental.com/catalog/gloves", slug="gloves"
)
URL_GOOD = "https://www.safcodental.com/product/good"
URL_NEEDS_RENDER = "https://www.safcodental.com/product/needs-render"
URL_BROKEN = "https://www.safcodental.com/product/broken"


def _good_jsonld_html() -> str:
    return """
    <html><head>
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"Product",
     "sku":"SKU-GOOD","name":"Good Gloves",
     "description":"a desc",
     "brand":{"@type":"Brand","name":"Acme"},
     "image":["https://x.com/i.jpg"],
     "url":"https://www.safcodental.com/product/good",
     "offers":{"@type":"Offer","price":"12.34","priceCurrency":"USD",
               "availability":"https://schema.org/InStock"}}
    </script>
    <script type="application/ld+json">
    {"@context":"https://schema.org","@type":"BreadcrumbList",
     "itemListElement":[
       {"@type":"ListItem","position":1,"name":"Home"},
       {"@type":"ListItem","position":2,"name":"Catalog"},
       {"@type":"ListItem","position":3,"name":"Gloves"},
       {"@type":"ListItem","position":4,"name":"Good Gloves"}]}
    </script>
    </head><body></body></html>
    """


def _empty_html() -> str:
    return "<html><body>nothing structured</body></html>"


def _category_listing_html(*product_paths: str) -> str:
    """Catalog HTML that satisfies the Algolia-driven ListingCrawler's
    config scraper. Product URLs are returned via the Algolia mock in
    :func:`_algolia_listing_transport`, not from anchors here."""

    return """
    <html><body>
      "applicationId":"TESTAPP","indexName":"testidx","apiKey":"ZmFrZQ=="
      "category_id":"385","category_name":"Test"
    </body></html>
    """


def _algolia_listing_transport(*product_paths: str) -> httpx.MockTransport:
    """A MockTransport that responds to BOTH:

    - GET <category_url> with the catalog HTML containing inline Algolia
      config (so the crawler can scrape applicationId / apiKey / category_id), and
    - POST <algolia>/1/indexes/*/queries with a single-page result whose
      hits carry ``url`` set to each ``product_paths`` entry.
    """

    catalog_html = _category_listing_html()
    algolia_payload = {
        "results": [{
            "hits": [{"url": p} for p in product_paths],
            "nbPages": 1,
            "nbHits": len(product_paths),
        }]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, text=catalog_html, request=request)
        return httpx.Response(200, json=algolia_payload, request=request)

    return httpx.MockTransport(handler)


@pytest.fixture
def storage(tmp_path: Path) -> Storage:
    return Storage(tmp_path / "p.sqlite")


@pytest.fixture
def http_client_mock() -> RateLimitedClient:
    """Real RateLimitedClient with the underlying httpx client patched."""

    settings = get_settings()
    client = RateLimitedClient(settings)

    async def fake_get(url: str) -> httpx.Response:
        if url == CATEGORY_GLOVES.url:
            return httpx.Response(
                200,
                text=_category_listing_html(URL_GOOD, URL_NEEDS_RENDER, URL_BROKEN),
                request=httpx.Request("GET", url),
            )
        if url == URL_GOOD:
            return httpx.Response(200, text=_good_jsonld_html(), request=httpx.Request("GET", url))
        if url in (URL_NEEDS_RENDER, URL_BROKEN):
            return httpx.Response(200, text=_empty_html(), request=httpx.Request("GET", url))
        raise AssertionError(f"unexpected URL {url}")

    client.get = fake_get  # type: ignore[method-assign]
    return client


def _make_navigator(slugs_to_return: list[CategoryRef]) -> Navigator:
    async def runner(intent: str | None) -> DiscoveredCategories:
        del intent
        return DiscoveredCategories(categories=slugs_to_return)

    return Navigator(runner=runner)


def _make_listing_crawler(http_client_mock: RateLimitedClient) -> ListingCrawler:
    """Wire a ListingCrawler against an in-memory transport that handles
    both the catalog config-scrape GET and the Algolia POST. The
    ``http_client_mock`` parameter is kept in the signature for symmetry
    with the test fixture wiring even though the crawler owns its own
    transport here."""

    del http_client_mock
    inner = httpx.AsyncClient(
        transport=_algolia_listing_transport(URL_GOOD, URL_NEEDS_RENDER, URL_BROKEN)
    )
    return ListingCrawler(client=inner)


class FakePlaywrightFetcher:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def fetch(self, url: str) -> str:
        self.calls.append(url)
        if url == URL_NEEDS_RENDER:
            return (
                _good_jsonld_html()
                .replace("SKU-GOOD", "SKU-RENDERED")
                .replace("Good Gloves", "Rendered Gloves")
                .replace("/product/good", "/product/needs-render")
            )
        if url == URL_BROKEN:
            return _empty_html()
        raise AssertionError(f"unexpected URL {url}")

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_pipeline_navigator_discovers_categories_then_crawls(
    storage: Storage, http_client_mock: RateLimitedClient
) -> None:
    """Default flow: Navigator runs, returns categories, listing crawler picks
    products, extractor produces records via JSON-LD."""

    fake_playwright = FakePlaywrightFetcher()
    fake_llm = AsyncMock(side_effect=AssertionError("LLM must not be called"))

    pipeline = Pipeline(
        storage=storage,
        navigator=_make_navigator([CATEGORY_GLOVES]),
        listing_crawler=_make_listing_crawler(http_client_mock),
        extractor=Extractor(llm=fake_llm),
        http_client=http_client_mock,
        playwright_fetcher=fake_playwright,
    )
    stats = await pipeline.run()  # no intent => all categories

    assert stats.discovered_categories == 1
    assert stats.categories == ["gloves"]
    assert stats.extracted_jsonld == 1  # only URL_GOOD has JSON-LD
    assert stats.extracted_playwright == 1  # URL_NEEDS_RENDER recovers via Playwright
    assert stats.failed == 1  # URL_BROKEN can't be saved
    products = storage.all_products()
    assert {p.sku for p in products} == {"SKU-GOOD", "SKU-RENDERED"}


@pytest.mark.asyncio
async def test_pipeline_explicit_categories_skip_navigator(
    storage: Storage, http_client_mock: RateLimitedClient
) -> None:
    """When explicit categories are passed, Navigator is never invoked."""

    async def must_not_be_called(intent: str | None) -> DiscoveredCategories:
        del intent
        raise AssertionError("Navigator must not run")

    fake_playwright = FakePlaywrightFetcher()
    fake_llm = AsyncMock(side_effect=AssertionError("LLM must not be called"))

    pipeline = Pipeline(
        storage=storage,
        navigator=Navigator(runner=must_not_be_called),
        listing_crawler=_make_listing_crawler(http_client_mock),
        extractor=Extractor(llm=fake_llm),
        http_client=http_client_mock,
        playwright_fetcher=fake_playwright,
    )
    stats = await pipeline.run(categories=[CATEGORY_GLOVES])

    assert stats.discovered_categories == 1
    assert stats.extracted_jsonld == 1


@pytest.mark.asyncio
async def test_pipeline_intent_is_forwarded_to_navigator(
    storage: Storage, http_client_mock: RateLimitedClient
) -> None:
    captured: dict[str, str | None] = {}

    async def runner(intent: str | None) -> DiscoveredCategories:
        captured["intent"] = intent
        return DiscoveredCategories(categories=[CATEGORY_GLOVES])

    pipeline = Pipeline(
        storage=storage,
        navigator=Navigator(runner=runner),
        listing_crawler=_make_listing_crawler(http_client_mock),
        extractor=Extractor(),
        http_client=http_client_mock,
        playwright_fetcher=None,
    )
    await pipeline.run(intent="I want gloves")
    assert captured["intent"] == "I want gloves"


@pytest.mark.asyncio
async def test_pipeline_skips_already_done_urls(
    storage: Storage, http_client_mock: RateLimitedClient
) -> None:
    """Resumability: pre-marking a URL DONE means Pipeline never fetches it."""

    from frontier_dental.storage import make_state as ms

    storage.upsert_state(ms(URL_GOOD, "gloves", CrawlStatus.DONE, sku="OLD-SKU"))
    storage.upsert_state(ms(URL_NEEDS_RENDER, "gloves", CrawlStatus.DONE, sku="OLD2"))
    storage.upsert_state(ms(URL_BROKEN, "gloves", CrawlStatus.DONE, sku="OLD3"))

    fake_llm = AsyncMock(side_effect=AssertionError("must not be called"))

    pipeline = Pipeline(
        storage=storage,
        navigator=_make_navigator([CATEGORY_GLOVES]),
        listing_crawler=_make_listing_crawler(http_client_mock),
        extractor=Extractor(llm=fake_llm),
        http_client=http_client_mock,
        playwright_fetcher=None,
    )
    stats = await pipeline.run()
    assert stats.skipped_done == 3
    assert stats.extracted_jsonld == 0


@pytest.mark.asyncio
async def test_pipeline_max_categories_caps_navigator_output(
    storage: Storage, http_client_mock: RateLimitedClient
) -> None:
    cats = [
        CATEGORY_GLOVES,
        CategoryRef(
            name="Sutures",
            url="https://www.safcodental.com/catalog/sutures-surgical-products",
            slug="sutures-surgical-products",
        ),
        CategoryRef(
            name="Anesthetics",
            url="https://www.safcodental.com/catalog/anesthetics",
            slug="anesthetics",
        ),
    ]
    pipeline = Pipeline(
        storage=storage,
        navigator=_make_navigator(cats),
        listing_crawler=_make_listing_crawler(http_client_mock),
        extractor=Extractor(),
        http_client=http_client_mock,
        playwright_fetcher=None,
    )
    stats = await pipeline.run(max_categories=1)
    assert stats.discovered_categories == 1
    assert stats.categories == ["gloves"]


@pytest.mark.asyncio
async def test_extractor_marks_failed_when_all_tiers_exhausted(
    storage: Storage, http_client_mock: RateLimitedClient
) -> None:
    """No JSON-LD, Playwright doesn't help, LLM returns nothing => FAILED."""

    fake_llm = AsyncMock(return_value={})

    # Listing returns only the broken URL.
    inner = httpx.AsyncClient(transport=_algolia_listing_transport(URL_BROKEN))
    listing = ListingCrawler(client=inner)

    pipeline = Pipeline(
        storage=storage,
        navigator=_make_navigator([CATEGORY_GLOVES]),
        listing_crawler=listing,
        extractor=Extractor(llm=fake_llm),
        http_client=http_client_mock,
        playwright_fetcher=FakePlaywrightFetcher(),
    )
    stats = await pipeline.run()
    assert stats.failed == 1
    assert stats.extracted_jsonld == 0
    state = storage.get_state(URL_BROKEN)
    assert state is not None
    assert state.status == CrawlStatus.FAILED

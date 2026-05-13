"""End-to-end pipeline orchestrator.

Two-stage discovery + tiered extraction:

1. ``Navigator`` — a LangGraph ReAct agent driving Playwright via LangChain
   Community's browser toolkit. Starts at the storefront landing page,
   discovers the catalog index itself, and enumerates every product
   category. With an optional ``intent`` argument, the LLM filters the
   enumerated list against the user's natural-language description; with
   no intent (the default), every discovered category is returned.

2. For each category, ``ListingCrawler`` (deterministic httpx + parse)
   extracts ``/product/<slug>`` URLs from the server-rendered listing HTML.

3. For each product URL, the multi-tier ``Extractor`` produces a
   ``ProductRecord``:

      Tier 1 — httpx + JSON-LD (+ DOM enrichments + heuristic pack-size).
      Tier 2 — re-fetch with Playwright when Tier 1 leaves required fields blank.
      Tier 3 — LLM structured-output fallback as a last resort.

4. Records are upserted (SKU-keyed, idempotent) and the per-URL state is
   tracked in ``crawl_state`` for resumability.

Resumability contract
---------------------
For every product URL discovered, ``crawl_state`` records:

* ``DISCOVERED`` — listing crawler emitted it; not yet fetched.
* ``DONE``       — Extractor produced a record, Storage upserted it.
* ``FAILED``     — All tiers exhausted; ``error`` column carries the reason.

Re-running the pipeline only revisits ``DISCOVERED`` and ``FAILED`` rows.
``DONE`` rows are skipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import httpx
import structlog

from .agents.extractor import Extractor
from .agents.navigator import ListingCrawler, Navigator
from .agents.validator import dedupe
from .config import Settings, get_settings
from .http_client import RateLimitedClient
from .models import CategoryRef, CrawlStatus, ExtractionMethod
from .recs import RecsClient
from .storage import Storage, make_state

log = structlog.get_logger(__name__)


class PageFetcher(Protocol):
    """Protocol matching ``PlaywrightFetcher`` — keeps tests free of Chromium."""

    async def fetch(self, url: str) -> str: ...

    async def aclose(self) -> None: ...


@dataclass
class PipelineStats:
    discovered_categories: int = 0
    discovered: int = 0
    extracted_jsonld: int = 0
    extracted_playwright: int = 0
    extracted_llm: int = 0
    failed: int = 0
    skipped_done: int = 0
    categories: list[str] = field(default_factory=list)
    intent: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent,
            "categories": list(self.categories),
            "discovered_categories": self.discovered_categories,
            "discovered_products": self.discovered,
            "extracted_jsonld": self.extracted_jsonld,
            "extracted_playwright": self.extracted_playwright,
            "extracted_llm": self.extracted_llm,
            "failed": self.failed,
            "skipped_done": self.skipped_done,
        }


class Pipeline:
    """Run the full discovery → extraction → storage pipeline."""

    def __init__(
        self,
        *,
        storage: Storage,
        navigator: Navigator,
        listing_crawler: ListingCrawler,
        extractor: Extractor,
        http_client: RateLimitedClient,
        playwright_fetcher: PageFetcher | None = None,
        recs: RecsClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._storage = storage
        self._navigator = navigator
        self._listings = listing_crawler
        self._extractor = extractor
        self._http = http_client
        self._playwright = playwright_fetcher
        self._recs = recs
        self._settings = settings or get_settings()

    async def run(
        self,
        *,
        intent: str | None = None,
        categories: Sequence[CategoryRef] | None = None,
        max_categories: int | None = None,
    ) -> PipelineStats:
        """Execute the full pipeline.

        * If ``categories`` is provided, the LLM Navigator is skipped and those
          categories are crawled directly. Useful for CI / explicit control.
        * Otherwise the Navigator is invoked. ``intent`` is forwarded to it as
          an optional natural-language filter.
        * ``max_categories`` caps the number of categories crawled (useful for
          dev-time runs against the full catalog).
        """

        if categories is None:
            categories = await self._navigator.discover_categories(intent=intent)

        if max_categories is not None:
            categories = list(categories)[:max_categories]

        stats = PipelineStats(
            intent=intent,
            categories=[c.slug for c in categories],
            discovered_categories=len(categories),
        )
        for cat in categories:
            await self._run_category(cat, stats)
        return stats

    async def _run_category(self, category: CategoryRef, stats: PipelineStats) -> None:
        urls = await self._listings.list_products(category.url)

        cap = self._settings.max_products_per_category
        if cap is not None:
            urls = urls[:cap]

        for url in urls:
            existing = self._storage.get_state(url)
            if existing is not None and existing.status == CrawlStatus.DONE:
                stats.skipped_done += 1
                continue
            if existing is None:
                self._storage.upsert_state(
                    make_state(url, category.slug, CrawlStatus.DISCOVERED)
                )
                stats.discovered += 1

            await self._extract_and_persist(url, category.slug, stats)

    async def _extract_and_persist(
        self, url: str, category_slug: str, stats: PipelineStats
    ) -> None:
        # --- Tier 1 --- httpx fetch + JSON-LD/DOM extraction (no LLM yet)
        try:
            static_html = await self._fetch_static(url)
        except Exception as e:  # noqa: BLE001
            log.warning("static_fetch_failed", url=url, error=str(e))
            self._storage.upsert_state(
                make_state(url, category_slug, CrawlStatus.FAILED, error=f"static fetch: {e}")
            )
            stats.failed += 1
            return

        outcome = await self._extractor.extract(
            url=url,
            html=static_html,
            source_category=category_slug,
            enable_llm_fallback=False,
        )

        # --- Tier 2 --- Playwright re-render if Tier 1 didn't produce a record
        if outcome.record is None and self._playwright is not None:
            try:
                rendered_html = await self._playwright.fetch(url)
            except Exception as e:  # noqa: BLE001
                log.warning("playwright_fetch_failed", url=url, error=str(e))
                rendered_html = None
            if rendered_html:
                outcome = await self._extractor.extract(
                    url=url,
                    html=rendered_html,
                    source_category=category_slug,
                    enable_llm_fallback=True,
                    method_override=ExtractionMethod.PLAYWRIGHT_DOM,
                )

        if outcome.record is None and self._playwright is None:
            outcome = await self._extractor.extract(
                url=url,
                html=static_html,
                source_category=category_slug,
                enable_llm_fallback=True,
            )

        if outcome.record is not None:
            record = outcome.record
            if self._recs is not None:
                alternatives = await self._recs.fetch_alternatives(record.sku)
                if alternatives:
                    record = record.model_copy(update={"alternative_products": alternatives})
            self._storage.upsert_product(record)
            self._storage.upsert_state(
                make_state(
                    url,
                    category_slug,
                    CrawlStatus.DONE,
                    sku=record.sku,
                )
            )
            _bump_method_stat(stats, outcome.method)
        else:
            self._storage.upsert_state(
                make_state(
                    url,
                    category_slug,
                    CrawlStatus.FAILED,
                    error=outcome.error or "unknown",
                )
            )
            stats.failed += 1

    async def _fetch_static(self, url: str) -> str:
        resp: httpx.Response = await self._http.get(url)
        return resp.text

    async def aclose(self) -> None:
        await self._http.aclose()
        if self._playwright is not None:
            await self._playwright.aclose()
        if self._recs is not None:
            await self._recs.aclose()


# --- helpers -----------------------------------------------------------------


def category_ref_from_slug(slug: str, base_url: str) -> CategoryRef:
    """Build a CategoryRef from a CLI-style slug like ``gloves`` or
    ``gloves/nitrile-gloves`` (or a full URL)."""

    if slug.startswith("http://") or slug.startswith("https://"):
        url = slug.rstrip("/")
    else:
        url = f"{base_url.rstrip('/')}/catalog/{slug.strip('/')}"
    deepest = url.rsplit("/", 1)[-1]
    return CategoryRef(name=deepest, url=url, slug=deepest)


def _bump_method_stat(stats: PipelineStats, method: ExtractionMethod) -> None:
    if method in (ExtractionMethod.JSON_LD, ExtractionMethod.DOM):
        stats.extracted_jsonld += 1
    elif method == ExtractionMethod.PLAYWRIGHT_DOM:
        stats.extracted_playwright += 1
    elif method == ExtractionMethod.LLM_FALLBACK:
        stats.extracted_llm += 1


def dedupe_storage(storage: Storage) -> int:
    """Run the validator/deduper across persisted rows. Returns the count
    of rows kept after dedupe. The persistence layer is already SKU-keyed so
    this is mostly a sanity check; it returns the deduped record count."""

    return len(dedupe(storage.all_products()))

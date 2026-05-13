"""Two-stage discovery layer.

Stage 1 — **Navigator (LLM-driven browser agent: LangChain + Playwright)**.
    A LangGraph ReAct agent drives a real Chromium instance via LangChain
    Community's Playwright toolkit. The agent starts at the storefront
    *landing page* — it does NOT assume any specific catalog URL — and
    must discover the catalog index itself by reading the navigation,
    clicking through, and reading hyperlinks. Once it has enumerated the
    categories (optionally filtered against a user-supplied ``intent``),
    it calls a custom ``submit_categories`` tool to return a structured
    ``DiscoveredCategories`` payload and stop.

    The tool surface exposed to the LLM:

    * ``navigate_browser`` — open a URL
    * ``click_element`` — click a CSS selector
    * ``extract_hyperlinks`` — list every ``<a>`` on the current page
    * ``extract_text`` — read visible text
    * ``get_elements`` — query elements with selected attributes
    * ``current_webpage`` — current URL
    * ``previous_webpage`` — go back
    * ``submit_categories`` — finalize and return the result

Stage 2 — **ListingCrawler (deterministic, Algolia API)**.
    Safco's storefront is Magento 2 + the ``magento2-algoliasearch``
    extension: the catalog HTML server-renders a static set of placeholder
    anchors, and the real product grid hydrates client-side from Algolia
    InstantSearch. Plain HTTP pagination on ``?page=N`` therefore returns
    the same anchors for every page.

    Instead, the crawler fetches the catalog page once over httpx, scrapes
    the inline Algolia config (``applicationId``, ``indexName``, ``apiKey``)
    and the Magento ``category_id`` from the GTM dataLayer, then paginates
    via direct POST to ``{appId}-dsn.algolia.net``. The search-only
    ``apiKey`` rotates (~24h ``validUntil``); we re-scrape it per run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from urllib.parse import quote

import httpx
import structlog
from pydantic import BaseModel, Field

from ..config import Settings, get_settings
from ..models import CategoryRef

log = structlog.get_logger(__name__)


_PRODUCT_URL_RE = re.compile(r"^https?://[^/]*safcodental\.com/product/[a-z0-9\-]+/?$", re.I)
_CATEGORY_URL_RE = re.compile(
    r"^https?://[^/]*safcodental\.com/catalog/[a-z0-9][a-z0-9\-]*/?$", re.I
)
# Top-level categories only (no subpaths); subcategories are reachable via
# their parent's listing page. Matches /catalog/<slug> embedded in HTML.
_CATEGORY_PATH_RE = re.compile(
    r"https?://[^/\s\"']*safcodental\.com/catalog/[a-z0-9][a-z0-9\-]*", re.I
)
# Legacy Magento category URL: /catalog/category/view/s/<slug>/id/<id>/.
# The home page links to a handful of categories ONLY via this legacy form
# (e.g. rubber-dam, id 913). The modern /catalog/<slug> URL still works for
# them — we just can't discover the slug from the modern regex above.
_LEGACY_CATEGORY_PATH_RE = re.compile(
    r"/catalog/category/view/s/(?P<slug>[a-z0-9][a-z0-9\-]*)/id/\d+", re.I
)
# ``category`` is the Magento route prefix in /catalog/category/view/s/...,
# which ``_CATEGORY_PATH_RE`` truncates to the bogus slug ``category``.
_RESERVED_CATEGORY_SLUGS: frozenset[str] = frozenset({"category"})


# --- Stage 1: Navigator ------------------------------------------------------


class DiscoveredCategories(BaseModel):
    """Pydantic schema the Navigator's LLM must emit at end-of-run."""

    categories: list[CategoryRef] = Field(
        default_factory=list,
        description=(
            "Every product category visible on the catalog page. Each entry "
            "carries the human-readable name, the absolute URL, and the "
            "trailing-path slug. Filter by user intent if provided."
        ),
    )


NavigatorRunner = Callable[[str | None], Awaitable[DiscoveredCategories]]
"""Signature: ``(intent_or_None) -> DiscoveredCategories``. Tests inject a
fake; production uses ``_default_navigator_runner``."""


def filter_categories(
    categories: list[CategoryRef], base_url: str
) -> list[CategoryRef]:
    """Sanity-pass the LLM's output: keep only safcodental ``/catalog/...``
    URLs, dedupe by URL, drop the catalog root itself."""

    seen: set[str] = set()
    out: list[CategoryRef] = []
    base = base_url.rstrip("/")
    for c in categories:
        url = (c.url or "").strip()
        if url.startswith("/"):
            url = base + url
        if not _CATEGORY_URL_RE.match(url):
            continue
        url = url.rstrip("/")
        if url == f"{base}/catalog":
            continue
        slug = url.rsplit("/", 1)[-1]
        if slug in _RESERVED_CATEGORY_SLUGS:
            # ``/catalog/category/...`` is Magento's legacy route prefix,
            # not a real category. The route prefix is captured by
            # ``_CATEGORY_PATH_RE`` when scanning HTML for ``/catalog/<slug>``,
            # so we filter it out here at the canonicalization step.
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append(CategoryRef(name=(c.name or slug).strip(), url=url, slug=slug))
    return out


class Navigator:
    """Discover categories on the storefront. Returns every category by
    default; with an ``intent`` string, returns only those that the LLM
    judges relevant to the user's natural-language description."""

    def __init__(
        self,
        *,
        runner: NavigatorRunner | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._runner = runner or _default_navigator_runner

    async def discover_categories(self, intent: str | None = None) -> list[CategoryRef]:
        log.info("navigator_starting", intent=intent)
        try:
            result = await self._runner(intent)
        except Exception as e:  # noqa: BLE001 — top-level boundary
            log.error("navigator_failed", error=str(e))
            return []
        cats = filter_categories(result.categories, self._settings.base_url)
        log.info(
            "navigator_finished",
            raw_count=len(result.categories),
            kept=len(cats),
            slugs=[c.slug for c in cats],
        )
        return cats


# --- Stage 1 (alternate): StaticCategoryDiscoverer --------------------------


class StaticCategoryDiscoverer:
    """Deterministic, LLM-free counterpart to ``Navigator``.

    Fetches the storefront landing page over plain HTTP and regex-extracts
    every ``/catalog/<slug>`` URL embedded in its server-rendered navigation.
    No browser, no LLM, no cost. The result is run through
    :func:`filter_categories` for the same dedup / sanity pass the agent's
    output goes through.

    Suitable when no natural-language intent filtering is needed (i.e. the
    user wants every category). When an intent is provided the LLM
    ``Navigator`` is the better choice — only it can interpret the intent.
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

    async def discover_categories(self) -> list[CategoryRef]:
        landing = self._settings.base_url
        client = self._client or httpx.AsyncClient(
            timeout=self._settings.request_timeout_s,
            headers={"User-Agent": self._settings.user_agent},
            follow_redirects=True,
        )
        try:
            log.info("static_category_discovery_starting", url=landing)
            resp = await client.get(landing)
            resp.raise_for_status()
            modern_urls = _CATEGORY_PATH_RE.findall(resp.text)
            legacy_slugs = _LEGACY_CATEGORY_PATH_RE.findall(resp.text)
        except Exception as e:  # noqa: BLE001 — top-level boundary
            log.warning("static_category_discovery_failed", url=landing, error=str(e))
            return []
        finally:
            if self._owns_client:
                await client.aclose()

        base = self._settings.base_url.rstrip("/")
        raws: list[CategoryRef] = [
            CategoryRef(name=u.rsplit("/", 1)[-1], url=u, slug=u.rsplit("/", 1)[-1])
            for u in modern_urls
        ]
        # The modern slug URL works for legacy-only categories too — Safco
        # 200s on /catalog/<slug> and serves the same Algolia config — so we
        # synthesize a modern URL from the legacy match and let
        # filter_categories canonicalize it alongside the rest.
        for slug in legacy_slugs:
            raws.append(
                CategoryRef(
                    name=slug, url=f"{base}/catalog/{slug}", slug=slug
                )
            )
        cats = filter_categories(raws, self._settings.base_url)
        log.info(
            "static_category_discovery_finished",
            kept=len(cats),
            raw_count=len(modern_urls) + len(legacy_slugs),
            legacy_count=len(legacy_slugs),
            slugs=[c.slug for c in cats],
        )
        return cats


# --- Stage 2: ListingCrawler -------------------------------------------------


def filter_product_urls(urls: list[str], base_url: str) -> list[str]:
    """Keep only canonical product detail URLs under the configured site,
    deduped, preserving order."""

    seen: set[str] = set()
    out: list[str] = []
    for raw in urls:
        if not raw:
            continue
        candidate = raw.strip()
        if candidate.startswith("/product/"):
            candidate = base_url.rstrip("/") + candidate
        if not _PRODUCT_URL_RE.match(candidate):
            continue
        candidate = candidate.rstrip("/")
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


_ALGOLIA_APP_RE = re.compile(r'"applicationId"\s*:\s*"(?P<v>[A-Za-z0-9]+)"')
_ALGOLIA_INDEX_RE = re.compile(r'"indexName"\s*:\s*"(?P<v>[A-Za-z0-9_]+)"')
_ALGOLIA_KEY_RE = re.compile(r'"apiKey"\s*:\s*"(?P<v>[A-Za-z0-9+/=]+)"')
_MAGENTO_CATEGORY_ID_RE = re.compile(r'"category_id"\s*:\s*"(?P<v>\d+)"')


class _AlgoliaConfig(BaseModel):
    """Per-category Algolia config scraped from the storefront HTML."""

    application_id: str
    api_key: str
    products_index: str
    category_id: str


def _parse_algolia_config(html: str) -> _AlgoliaConfig | None:
    """Extract Algolia search params from the inline ``magento2-algoliasearch``
    config and the GTM dataLayer category_id. Returns ``None`` if any field
    is missing — caller logs and bails out.

    The Magento extension inlines its config object near the bottom of the
    page; the GTM dataLayer push appears separately. We grep each field
    independently rather than locking in their relative order, which has
    drifted across extension versions in the wild.
    """

    app = _ALGOLIA_APP_RE.search(html)
    idx = _ALGOLIA_INDEX_RE.search(html)
    key = _ALGOLIA_KEY_RE.search(html)
    cat = _MAGENTO_CATEGORY_ID_RE.search(html)
    if not (app and idx and key and cat):
        return None
    return _AlgoliaConfig(
        application_id=app.group("v"),
        api_key=key.group("v"),
        products_index=f"{idx.group('v')}_products",
        category_id=cat.group("v"),
    )


class ListingCrawler:
    """Per-category product URL extraction via Safco's Algolia backend.

    Step 1: httpx GET the category page once and scrape the inline Magento
    Algolia config (``applicationId``, ``indexName``, ``apiKey``) plus the
    GTM dataLayer ``category_id``.

    Step 2: POST page=0..nbPages-1 to ``{appId}-dsn.algolia.net`` filtered by
    ``categoryIds:{cat_id}`` + ``visibility_catalog=1``. Each hit's ``url``
    field is the canonical PDP URL.

    The Algolia search-only ``apiKey`` rotates ~24h (``validUntil`` is
    embedded in it), which is why we re-scrape per run rather than caching.
    It is the same public key the browser sends — not a credential.
    """

    _HITS_PER_PAGE = 40

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self._owns_client = client is None

    async def list_products(self, category_url: str) -> list[str]:
        client = self._client or httpx.AsyncClient(
            timeout=self._settings.request_timeout_s,
            headers={"User-Agent": self._settings.user_agent},
            follow_redirects=True,
        )
        seen: set[str] = set()
        all_urls: list[str] = []
        pages_fetched = 0
        max_pages = self._settings.listing_max_pages
        try:
            log.info("listing_crawler_starting", category_url=category_url)
            try:
                resp = await client.get(category_url)
                resp.raise_for_status()
            except Exception as e:  # noqa: BLE001 — config fetch boundary
                log.warning(
                    "listing_crawler_config_fetch_failed",
                    category_url=category_url,
                    error=str(e),
                )
                return []

            cfg = _parse_algolia_config(resp.text)
            if cfg is None:
                log.warning(
                    "listing_crawler_config_missing",
                    category_url=category_url,
                )
                return []

            algolia_url = (
                f"https://{cfg.application_id.lower()}-dsn.algolia.net"
                "/1/indexes/*/queries"
                f"?x-algolia-application-id={cfg.application_id}"
                f"&x-algolia-api-key={cfg.api_key}"
            )
            facet = quote(json.dumps([[f"categoryIds:{cfg.category_id}"]]))
            visibility = quote(json.dumps(["visibility_catalog=1"]))

            for page in range(max_pages):
                params = (
                    f"hitsPerPage={self._HITS_PER_PAGE}"
                    f"&page={page}"
                    "&query="
                    f"&facetFilters={facet}"
                    f"&numericFilters={visibility}"
                )
                body = json.dumps(
                    {"requests": [{"indexName": cfg.products_index, "params": params}]}
                )
                try:
                    r = await client.post(
                        algolia_url,
                        headers={
                            "content-type": "application/x-www-form-urlencoded",
                            "Origin": self._settings.base_url,
                            "Referer": self._settings.base_url + "/",
                        },
                        content=body,
                    )
                    r.raise_for_status()
                    payload = r.json()
                except Exception as e:  # noqa: BLE001 — per-page boundary
                    log.warning(
                        "listing_crawler_algolia_failed",
                        category_url=category_url,
                        page=page,
                        error=str(e),
                    )
                    break

                pages_fetched += 1
                results = payload.get("results") or []
                if not results:
                    break
                res = results[0]
                hit_urls = [h.get("url") for h in res.get("hits", []) if h.get("url")]
                page_urls = filter_product_urls(hit_urls, self._settings.base_url)
                new_urls = [u for u in page_urls if u not in seen]
                for u in new_urls:
                    seen.add(u)
                    all_urls.append(u)

                nb_pages = res.get("nbPages", page + 1)
                if page + 1 >= nb_pages:
                    log.info(
                        "listing_crawler_pagination_exhausted",
                        category_url=category_url,
                        last_page=page,
                        nb_pages=nb_pages,
                    )
                    break
            else:
                log.warning(
                    "listing_crawler_hit_page_cap",
                    category_url=category_url,
                    cap=max_pages,
                )
        finally:
            if self._owns_client:
                await client.aclose()

        log.info(
            "listing_crawler_finished",
            category_url=category_url,
            kept=len(all_urls),
            pages_fetched=pages_fetched,
        )
        return all_urls


# --- Default LangChain + Playwright browser-agent runner ------------------


class _NavigatorCategoryArg(BaseModel):
    """Shape the LLM emits per category when calling submit_categories."""

    name: str = Field(description="Human-readable category name")
    url: str = Field(description="Absolute URL of the category landing page")
    slug: str = Field(description="Trailing path segment, e.g. 'gloves'")


class _NavigatorSubmitArgs(BaseModel):
    """args_schema for the submit_categories tool."""

    categories: list[_NavigatorCategoryArg] = Field(
        description=(
            "Every category you discovered (after filtering, if an intent "
            "was given). Each entry must include name, url, and slug."
        )
    )


_NAVIGATOR_SYSTEM_PROMPT = """\
You are a browser agent driving a real Chromium instance to enumerate every
product category of a dental supply storefront. You start at the storefront
landing page only — you do NOT know the catalog URL in advance and must
discover it by exploring the site (e.g. reading the main navigation,
clicking the "catalog"/"shop"/"products" entry).

You have these tools (names exactly as registered, all operate on the same
shared browser):

- navigate_browser(url): open a URL
- click_element(selector): click the first match of a CSS selector
- extract_hyperlinks(absolute_urls=true): list every <a> on the current page
- extract_text(): read all visible text on the current page
- get_elements(selector, attributes): query elements + chosen attributes
- current_webpage(): report the current URL
- previous_webpage(): go back one history step
- submit_categories(categories): finalize and return your result

Operating rules:
- ALWAYS finish by calling submit_categories exactly once, even if you only
  managed a partial list. Never stop without calling it.
- Each category you submit must include name (display text), url (absolute),
  and slug (the trailing path segment, e.g. "gloves" for "/catalog/gloves").
- Submit ONLY top-level categories — URLs of the form
  /catalog/<single-slug> (e.g. /catalog/gloves). Do NOT submit subcategory
  URLs like /catalog/gloves/nitrile-gloves; the per-category listing crawler
  reaches subcategory products from the parent listing.
- Do NOT browse individual product detail pages (URLs containing /product/).
- Do NOT log in, add to cart, or open the checkout flow.
- If a popup/banner blocks the page, dismiss it via click_element and move
  on — do not loop on it.
- Prefer extract_hyperlinks over multiple click_element calls when one page
  read can yield the whole category list.
- Be efficient: aim for the minimum number of tool calls.
"""


def _build_navigator_user_prompt(base_url: str, intent: str | None) -> str:
    landing = base_url.rstrip("/")
    if intent:
        intent_clause = (
            f'Filter the enumerated categories to those matching this user '
            f'intent: "{intent.strip()}". When in doubt, include rather than '
            "exclude — multiple plausible matches are fine."
        )
    else:
        intent_clause = (
            "Return EVERY product category you can enumerate. Do not filter."
        )
    return (
        f"Open the storefront landing page: {landing}\n"
        "From there, locate the catalog/category index (it is linked from "
        "the main navigation; the exact URL is not known to you in advance) "
        "and enumerate every product category listed.\n\n"
        f"{intent_clause}\n\n"
        "When you have your final list, call submit_categories with the "
        "categories and stop."
    )


async def _default_navigator_runner(intent: str | None) -> DiscoveredCategories:
    """Production runner — a LangGraph ReAct agent driving Playwright via
    LangChain Community's browser toolkit.

    All third-party imports are local so tests that inject a fake runner
    don't pay the import cost or require Playwright/LangChain installed.
    """

    from langchain_community.agent_toolkits import (  # type: ignore[import-not-found]
        PlayWrightBrowserToolkit,
    )
    from langchain_core.tools import StructuredTool  # type: ignore[import-not-found]
    from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    from langgraph.errors import GraphRecursionError  # type: ignore[import-not-found]
    from langgraph.prebuilt import create_react_agent  # type: ignore[import-not-found]
    from playwright.async_api import async_playwright  # type: ignore[import-not-found]

    s = get_settings()
    submitted: dict[str, DiscoveredCategories] = {}

    async def _submit(categories: list[_NavigatorCategoryArg]) -> str:
        items = [CategoryRef(name=c.name, url=c.url, slug=c.slug) for c in categories]
        submitted["result"] = DiscoveredCategories(categories=items)
        return f"Submitted {len(items)} categories. Stop now."

    submit_tool = StructuredTool.from_function(
        coroutine=_submit,
        name="submit_categories",
        description=(
            "Finalize and return the discovered categories. Call this exactly "
            "once at the end of your run with the complete list."
        ),
        args_schema=_NavigatorSubmitArgs,
    )

    # Start Playwright directly — langchain_community's
    # create_async_playwright_browser is a SYNC helper that internally calls
    # loop.run_until_complete, which collides with our running asyncio loop.
    pw = await async_playwright().start()
    async_browser = await pw.chromium.launch(headless=True)
    try:
        # Pre-create a context with our user agent so the toolkit's tools
        # (which fall back to the most recent context's last page) inherit it.
        context = await async_browser.new_context(user_agent=s.user_agent)
        await context.new_page()

        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
        tools = [*toolkit.get_tools(), submit_tool]

        llm = ChatOpenAI(
            model=s.vllm_model,
            base_url=s.vllm_base_url,
            api_key=s.vllm_api_key,
            temperature=s.vllm_temperature,
            max_tokens=s.vllm_max_tokens,
        )
        agent = create_react_agent(llm, tools)

        messages = [
            {"role": "system", "content": _NAVIGATOR_SYSTEM_PROMPT},
            {"role": "user", "content": _build_navigator_user_prompt(s.base_url, intent)},
        ]
        # langgraph counts each node visit; a ReAct round-trip is ~2 nodes.
        recursion_limit = max(8, 2 * s.navigator_max_steps + 4)

        log.info("navigator_agent_starting", intent=intent, max_steps=s.navigator_max_steps)
        try:
            await agent.ainvoke(
                {"messages": messages},
                config={"recursion_limit": recursion_limit},
            )
        except GraphRecursionError as e:
            log.warning("navigator_agent_hit_step_cap", error=str(e))
        except Exception as e:  # noqa: BLE001 — top-level boundary for the agent loop
            log.warning("navigator_agent_failed", error=str(e))
    finally:
        try:
            await async_browser.close()
        except Exception as e:  # noqa: BLE001
            log.warning("navigator_browser_close_failed", error=str(e))
        try:
            await pw.stop()
        except Exception as e:  # noqa: BLE001
            log.warning("navigator_playwright_stop_failed", error=str(e))

    return submitted.get("result") or DiscoveredCategories(categories=[])

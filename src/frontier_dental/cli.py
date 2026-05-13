"""Command-line entry point.

Three subcommands:

* ``run``    — discover categories (LLM-driven by default), then crawl product
               listings and extract product records.
* ``status`` — print crawl_state counts and product counts.
* ``export`` — write ``output/products.csv`` and ``output/products.json``.

Detailed, man-page-style help is exposed via ``-h`` / ``--help`` on each
command.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import click
import structlog

from .agents.extractor import Extractor
from .agents.navigator import (
    ListingCrawler,
    Navigator,
    StaticCategoryDiscoverer,
)
from .config import get_settings
from .http_client import RateLimitedClient
from .pipeline import Pipeline, PipelineStats, category_ref_from_slug
from .playwright_fetcher import PlaywrightFetcher
from .recs import RecsClient
from .storage import Storage


# --- Help formatting ------------------------------------------------------


CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
    "max_content_width": 100,
}


# Click preserves indentation only inside a paragraph that starts with `\b\n`.
# Each preformatted block below therefore begins with a `\b` line.

_GROUP_EPILOG = """\
\b
PIPELINE
    1. Discovery     — either:
                       a) StaticCategoryDiscoverer (default when no --intent
                          and no --category): deterministic httpx + regex
                          extraction of /catalog/<slug> links from the
                          storefront landing page. No LLM, no browser.
                       b) Navigator (when --intent is set): LangGraph ReAct
                          agent driving Chromium via LangChain Community's
                          Playwright toolkit. Required for natural-language
                          intent filtering.
    2. ListingCrawler — deterministic httpx + regex extraction of /product/<slug>.
    3. Extractor     — tiered: JSON-LD → Playwright re-render → LLM fallback.
    4. Validator     — pydantic schema + SKU-keyed dedup.
    5. Storage       — SQLite (canonical) with CSV / JSON export.

\b
ENVIRONMENT (read from .env via pydantic-settings; see .env.example)
    BASE_URL                   Storefront root.                Default: https://www.safcodental.com
    USER_AGENT                 HTTP / browser user agent string.
    RATE_LIMIT_RPS             Outbound HTTP rate limit (rps). Default: 1.0
    REQUEST_TIMEOUT_S          HTTP request timeout (sec).     Default: 30.0
    MAX_RETRIES                HTTP retry count.               Default: 3
    MAX_PRODUCTS_PER_CATEGORY  Per-category product cap.       Default: unset (no cap)
    NAVIGATOR_MAX_STEPS        Cap on Navigator agent calls.   Default: 60
    VLLM_BASE_URL              OpenAI-compatible LLM endpoint. Default: http://localhost:8000/v1
    VLLM_API_KEY               API key for the LLM endpoint.   Default: EMPTY
    VLLM_MODEL                 Model name.                     Default: Qwen/Qwen3.5-35B-A3B
    VLLM_TEMPERATURE           LLM sampling temperature.       Default: 0.0
    VLLM_MAX_TOKENS            Max output tokens per LLM call. Default: 2048
    SQLITE_PATH                Products SQLite path.           Default: output/products.sqlite
    OUTPUT_DIR                 CSV/JSON export directory.      Default: output/
    LOG_LEVEL                  INFO | DEBUG | WARNING.         Default: INFO
    LOG_JSON                   true=JSON logs, false=human.    Default: true

\b
EXAMPLES

\b
    # Default no-intent run — deterministic static discovery (no LLM, no browser)
    python main.py run

\b
    # Filter categories by a natural-language intent (LLM Navigator)
    python main.py run --intent "gloves and surgical supplies"

\b
    # Skip discovery entirely, crawl explicit categories
    python main.py run --category gloves --category sutures-surgical-products

\b
    # Show progress / counts after / during a run
    python main.py status

\b
    # Export the SQLite store to CSV + JSON in output/
    python main.py export --format both

\b
EXIT STATUS
    0   Success (per-URL failures are logged but do not cause non-zero exit).
    1   Configuration / runtime error.
    2   Click usage error (bad flags, unknown subcommand, etc.).

\b
FILES
    .env                       Local config (see .env.example for template).
    output/products.sqlite     Canonical SKU-keyed product store + crawl_state.
    output/products.csv        CSV export (run `export` to refresh).
    output/products.json       JSON export (run `export` to refresh).

\b
SEE ALSO
    README.md         Architecture, design notes, and limitations.
    .env.example      Full settings template.
"""


_RUN_EPILOG = """\
\b
DISCOVERY MODES (mutually exclusive)

\b
    Default (no --intent, no --category)
        StaticCategoryDiscoverer fetches the storefront landing page over
        plain HTTP and regex-extracts every /catalog/<slug> link in the
        server-rendered navigation. No LLM, no Chromium, no cost.
        The full catalog tree is then crawled.

\b
    --intent "<text>"
        The Navigator agent (LangGraph ReAct + Playwright) enumerates
        categories AND filters them against the natural-language intent.
        The LLM judges relevance from category names; when in doubt, it
        includes rather than excludes.

\b
    --category <slug-or-url>  (repeatable)
        Skip ALL discovery — no Navigator, no static fetch, no LLM, no
        browser. The given slugs or full URLs are crawled directly.

\b
RESUMABILITY
    URLs already in crawl_state.DONE are skipped on re-run. FAILED rows
    are retried. Killing the process mid-run and re-invoking does not
    duplicate work and does not drop URLs.

\b
ENVIRONMENT
    Notable env vars consulted by `run`:
      NAVIGATOR_MAX_STEPS         Cap the Navigator agent's tool-call budget.
      MAX_PRODUCTS_PER_CATEGORY   Same as --max-products-per-category.
      VLLM_BASE_URL / VLLM_API_KEY / VLLM_MODEL   LLM endpoint config.
      RATE_LIMIT_RPS              Outbound HTTP rate limit.
      REQUEST_TIMEOUT_S           HTTP request timeout.
      MAX_RETRIES                 Retry count for transient HTTP errors.
    See `python main.py --help` for the full ENVIRONMENT block.

\b
OUTPUT
    On completion, prints a JSON PipelineStats object to stdout summarizing:
      intent, categories, discovered_categories, discovered_products,
      extracted_jsonld, extracted_playwright, extracted_llm, failed,
      skipped_done.

\b
EXAMPLES

\b
    # Default — every category (deterministic static discovery, no LLM)
    python main.py run

\b
    # Intent-filtered (LLM Navigator)
    python main.py run --intent "gloves"

\b
    # Explicit categories
    python main.py run \\
        --category gloves \\
        --category sutures-surgical-products

\b
    # CI-friendly: skip Tier 2 Playwright re-render in the Extractor
    python main.py run --no-playwright

\b
SEE ALSO
    `python main.py --help`     Top-level overview, full ENVIRONMENT block.
    `python main.py status`     Counts after a run.
    `python main.py export`     Persist SQLite to CSV / JSON.
"""


_STATUS_EPILOG = """\
\b
OUTPUT (JSON to stdout)
    sqlite_path     Path to the SQLite store (echo of SQLITE_PATH).
    products        Total rows in the products table.
    crawl_state     Counts grouped by state: DISCOVERED | DONE | FAILED.

\b
EXAMPLES

\b
    python main.py status

\b
SEE ALSO
    `python main.py run`        Populate the store.
    `python main.py export`     Dump the store to CSV / JSON.
"""


_EXPORT_EPILOG = """\
\b
OUTPUT FILES
    products.csv    Flattened representation suitable for spreadsheets:
                      category_path           '>'-joined string
                      specifications          JSON-encoded string
                      image_urls              '|'-joined string
                      alternative_products    '|'-joined string
    products.json   Native nested ProductRecord shape (canonical for tooling).

\b
EXAMPLES

\b
    # Both formats into the configured OUTPUT_DIR (default: output/)
    python main.py export

\b
    # CSV only, into a custom directory
    python main.py export --format csv --output-dir ./snapshots

\b
SEE ALSO
    `python main.py run`        Populate the store first.
    `python main.py status`     Inspect counts before exporting.
"""


# --- Logging --------------------------------------------------------------


def _setup_logging() -> None:
    s = get_settings()
    logging.basicConfig(level=s.log_level, format="%(message)s", stream=sys.stderr)
    processors: list[structlog.types.Processor] = [
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.add_log_level,
    ]
    if s.log_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, s.log_level, logging.INFO)
        ),
    )


# --- Commands -------------------------------------------------------------


@click.group(context_settings=CONTEXT_SETTINGS, epilog=_GROUP_EPILOG)
def main() -> None:
    """Frontier Dental — agent-based scraper for safcodental.com.

    A LangGraph ReAct agent driving a real Chromium instance via LangChain
    Community's Playwright toolkit discovers the storefront's product
    catalog (starting from the bare landing page, with no hard-coded URL
    hints). A deterministic crawler turns each category into product URLs.
    A tiered extractor (JSON-LD → Playwright re-render → LLM structured
    output) produces ProductRecord rows. Records are validated, deduped by
    SKU, and persisted to SQLite, with optional CSV / JSON export.
    """

    _setup_logging()


@main.command("run", context_settings=CONTEXT_SETTINGS, epilog=_RUN_EPILOG)
@click.option(
    "--intent",
    type=str,
    default=None,
    metavar="TEXT",
    help=(
        "Natural-language description of categories to crawl, passed to the "
        "LLM Navigator. Mutually exclusive with --category. Omit (or pass an "
        "empty string) to crawl every category in the catalog via the "
        "deterministic, LLM-free static discoverer. "
        'Example: "I want gloves and surgical supplies".'
    ),
)
@click.option(
    "--category",
    "categories",
    multiple=True,
    metavar="SLUG-OR-URL",
    help=(
        "Explicit category slug (e.g. 'gloves') or absolute URL "
        "(e.g. 'https://www.safcodental.com/catalog/gloves'). Repeatable. "
        "When provided, all discovery is skipped (no browser, no LLM, no "
        "static fetch) and these categories are crawled directly. Mutually "
        "exclusive with --intent."
    ),
)
@click.option(
    "--no-playwright",
    is_flag=True,
    help=(
        "Skip the Tier 2 Playwright re-render fallback inside the Extractor. "
        "Tier 1 (JSON-LD over static HTML) and Tier 3 (LLM) still run. "
        "Useful in CI environments where Chromium isn't installed for "
        "extraction. Note that the Navigator itself still requires Chromium "
        "unless you also pass --category to skip discovery."
    ),
)
@click.option(
    "--max-products-per-category",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Cap the number of product URLs scraped per category. Overrides the "
        "MAX_PRODUCTS_PER_CATEGORY env var. Unset (no cap) by default."
    ),
)
@click.option(
    "--max-categories",
    type=int,
    default=None,
    metavar="N",
    help=(
        "Cap the total number of categories crawled. Useful when the "
        "Navigator discovers the full catalog tree but you only want to "
        "sample a few branches. Unset (all discovered categories) by default."
    ),
)
def cmd_run(
    intent: str | None,
    categories: Sequence[str],
    no_playwright: bool,
    max_products_per_category: int | None,
    max_categories: int | None,
) -> None:
    """Discover categories, crawl listings, and extract products.

    Three discovery modes are available — see DISCOVERY MODES below for
    details and EXAMPLES for typical invocations. Output is persisted to
    the SQLite store at SQLITE_PATH and can subsequently be exported with
    `python main.py export`.
    """

    s = get_settings()
    if max_products_per_category is not None:
        s.max_products_per_category = max_products_per_category

    # Treat --intent "" the same as no intent.
    if intent is not None:
        intent = intent.strip() or None

    storage = Storage(s.sqlite_path)
    explicit = (
        [category_ref_from_slug(c, s.base_url) for c in categories] if categories else None
    )

    asyncio.run(
        _run_pipeline(
            intent=intent,
            explicit_categories=explicit,
            max_categories=max_categories,
            use_playwright=not no_playwright,
            storage=storage,
        )
    )


async def _run_pipeline(
    *,
    intent: str | None,
    explicit_categories: list | None,
    max_categories: int | None,
    use_playwright: bool,
    storage: Storage,
) -> None:
    s = get_settings()
    http = RateLimitedClient(s)
    extractor = Extractor()
    navigator = Navigator()
    listing_crawler = ListingCrawler()
    playwright = PlaywrightFetcher() if use_playwright else None
    recs = RecsClient()

    # Discovery routing:
    #   - explicit --category: skip all discovery
    #   - --intent: LLM Navigator (only mode that can filter by NL)
    #   - else: deterministic StaticCategoryDiscoverer (no LLM, no browser)
    if explicit_categories is not None:
        resolved_categories: list | None = explicit_categories
    elif intent:
        resolved_categories = None  # let Pipeline.run invoke the Navigator
    else:
        resolved_categories = await StaticCategoryDiscoverer().discover_categories()

    pipeline = Pipeline(
        storage=storage,
        navigator=navigator,
        listing_crawler=listing_crawler,
        extractor=extractor,
        http_client=http,
        playwright_fetcher=playwright,
        recs=recs,
    )
    try:
        stats: PipelineStats = await pipeline.run(
            intent=intent,
            categories=resolved_categories,
            max_categories=max_categories,
        )
    finally:
        await http.aclose()
        if playwright is not None:
            await playwright.aclose()
        await recs.aclose()

    click.echo(json.dumps(stats.as_dict(), indent=2))


@main.command("status", context_settings=CONTEXT_SETTINGS, epilog=_STATUS_EPILOG)
def cmd_status() -> None:
    """Print crawl-state counts and total product count.

    Reads the SQLite store at SQLITE_PATH (no network, no LLM) and emits a
    JSON summary to stdout: total products, plus a histogram of crawl_state
    rows grouped by status (DISCOVERED / DONE / FAILED). Useful for
    checking progress mid-run or sanity-checking persisted state.
    """

    s = get_settings()
    storage = Storage(s.sqlite_path)
    counts = storage.status_counts()
    payload = {
        "sqlite_path": str(s.sqlite_path),
        "products": storage.product_count(),
        "crawl_state": counts,
    }
    click.echo(json.dumps(payload, indent=2))


@main.command("export", context_settings=CONTEXT_SETTINGS, epilog=_EXPORT_EPILOG)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "json", "both"]),
    default="both",
    show_default=True,
    help=(
        "Output format(s). 'csv' flattens nested fields (category_path "
        "'>'-joined; specifications JSON-encoded; image_urls and "
        "alternative_products '|'-joined). 'json' keeps the native nested "
        "ProductRecord shape. 'both' writes both files."
    ),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    metavar="PATH",
    help=(
        "Override the output directory for export files. Defaults to the "
        "OUTPUT_DIR env var (output/ if unset). The directory is created "
        "if it does not already exist."
    ),
)
def cmd_export(fmt: str, output_dir: Path | None) -> None:
    """Write products.csv and/or products.json from the SQLite store.

    The CSV format is suitable for spreadsheet inspection (with nested
    fields flattened); the JSON format preserves the native ProductRecord
    nesting for programmatic consumers. Existing files at the target paths
    are overwritten in place.
    """

    s = get_settings()
    out_dir = output_dir or s.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = Storage(s.sqlite_path)

    written: dict[str, str] = {}
    if fmt in ("csv", "both"):
        path = out_dir / "products.csv"
        n = storage.export_csv(path)
        written[str(path)] = f"{n} rows"
    if fmt in ("json", "both"):
        path = out_dir / "products.json"
        n = storage.export_json(path)
        written[str(path)] = f"{n} rows"

    click.echo(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()

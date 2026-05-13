"""Tiered PDP extractor.

Order of attempts (cheap → expensive):

1. **Tier 1 — JSON-LD on static HTML.** Safco PDPs ship a ``Product`` and
   ``BreadcrumbList`` JSON-LD block server-side, which already carries
   ``sku, name, description, image, brand, offers.price/currency/availability``
   plus the full category hierarchy. Fully deterministic, no JS needed.

2. **Tier 2 — Playwright DOM.** Some fields (pack size,
   specifications, alternate products, additional images) only render after
   client-side hydration. The pipeline re-fetches the page with Playwright and
   re-runs this same extractor against the rendered HTML. Augments — does not
   replace — Tier 1 output.

3. **Tier 3 — LLM fallback.** When required fields are still missing, the
   LLM is asked to extract structured JSON over the raw HTML. Tier 3 records
   set ``extraction_method = LLM_FALLBACK``.

The extractor is a *pure HTML → record* function. It does not fetch HTML; the
pipeline owns I/O. This keeps unit-testing trivial.
"""

from __future__ import annotations

import html as html_mod
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import structlog
from selectolax.parser import HTMLParser

from ..llm import structured_extract
from ..models import (
    ExtractionMethod,
    ProductRecord,
    normalize_availability,
)

log = structlog.get_logger(__name__)


REQUIRED_FIELDS: frozenset[str] = frozenset({"sku", "name", "price"})
"""Fields that MUST be present for an extraction to be considered successful.

If Tier 1 + Tier 2 leave any of these blank, Tier 3 (LLM) is invoked. If
Tier 3 still can't supply them, the extraction is marked as failed.
"""


LLMExtractor = Callable[[str, str], Awaitable[dict[str, Any]]]
"""Type for the LLM fallback callable. Tests inject an AsyncMock."""


# --- JSON-LD helpers ---------------------------------------------------------

_JSONLD_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)


_DESCRIPTION_HTML_HINT_RE = re.compile(r"&[a-z]+;|<[a-z/]")
_WS_RUN_RE = re.compile(r"[ \t]+")
_BLANKLINE_RUN_RE = re.compile(r"\n{3,}")


def _clean_description(raw: str | None) -> str:
    """Normalize a JSON-LD ``description`` value to clean plaintext.

    Safco's CMS lets editors paste rich HTML (sometimes with Microsoft Teams
    wrapper spans) into the description field. JSON-LD then string-encodes
    those tags, producing doubly-escaped strings like
    ``"&lt;p&gt;&lt;span ...&gt;Body &amp;nbsp; text&lt;/p&gt;"``.

    For most products the field is plain text with ``\\r\\n`` line breaks; for
    the messy minority it carries entities and tags. This helper handles both:

    1. Always strip leading/trailing whitespace.
    2. If the string contains HTML entities or tags, ``html.unescape`` it,
       strip remaining tags, decode any second-layer entities, normalize
       non-breaking spaces, and collapse whitespace runs.
    """

    if not raw:
        return ""
    text = raw.strip()
    if not _DESCRIPTION_HTML_HINT_RE.search(text):
        return text

    decoded = html_mod.unescape(text)
    if "<" in decoded:
        try:
            tree = HTMLParser(decoded)
            body = tree.body or tree
            decoded = body.text(separator=" ", strip=False) or ""
        except Exception:  # noqa: BLE001 — parser pathologies fall back to raw decoded
            pass
    decoded = html_mod.unescape(decoded)  # second pass for inner entities
    decoded = decoded.replace(" ", " ")
    decoded = _WS_RUN_RE.sub(" ", decoded)
    decoded = _BLANKLINE_RUN_RE.sub("\n\n", decoded)
    return decoded.strip()


def _iter_jsonld_blocks(html: str) -> list[Any]:
    """Return every parseable JSON-LD payload (some sites ship a list per block)."""

    blocks: list[Any] = []
    for match in _JSONLD_RE.finditer(html):
        raw = match.group(1).strip()
        try:
            blocks.append(json.loads(raw))
        except json.JSONDecodeError:
            log.debug("jsonld_block_unparseable", preview=raw[:120])
    return blocks


def _flatten_jsonld(blocks: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, list):
            out.extend(x for x in b if isinstance(x, dict))
        elif isinstance(b, dict):
            if "@graph" in b and isinstance(b["@graph"], list):
                out.extend(x for x in b["@graph"] if isinstance(x, dict))
            else:
                out.append(b)
    return out


def parse_product_jsonld(html: str) -> dict[str, Any]:
    """Extract a partial record dict from any ``Product`` JSON-LD blocks."""

    for entity in _flatten_jsonld(_iter_jsonld_blocks(html)):
        if entity.get("@type") != "Product":
            continue

        offers = entity.get("offers") or {}
        if isinstance(offers, list):
            offers = offers[0] if offers else {}

        images = entity.get("image") or []
        if isinstance(images, str):
            images = [images]

        brand = entity.get("brand") or {}
        if isinstance(brand, dict):
            brand_name = brand.get("name")
        else:
            brand_name = brand if isinstance(brand, str) else None

        price_raw = offers.get("price") if isinstance(offers, dict) else None
        price: Decimal | None = None
        if price_raw not in (None, ""):
            try:
                price = Decimal(str(price_raw))
            except InvalidOperation:
                price = None

        return {
            "sku": (entity.get("sku") or "").strip() or None,
            "name": (entity.get("name") or "").strip() or None,
            "description": _clean_description(entity.get("description")),
            "brand": brand_name,
            "price": price,
            "currency": offers.get("priceCurrency") if isinstance(offers, dict) else None,
            "availability": normalize_availability(
                offers.get("availability") if isinstance(offers, dict) else None
            ),
            "image_urls": tuple(str(u) for u in images if u),
            "product_url": entity.get("url"),
        }
    return {}


def parse_breadcrumb_jsonld(html: str) -> tuple[str, ...]:
    """Return the category hierarchy from any ``BreadcrumbList`` JSON-LD block.

    Drops the first crumb (\"Home\") and the last crumb (the product itself) —
    callers only care about the *category* path leading to the product.
    """

    for entity in _flatten_jsonld(_iter_jsonld_blocks(html)):
        if entity.get("@type") != "BreadcrumbList":
            continue
        items = entity.get("itemListElement") or []
        names: list[str] = []
        for item in sorted(items, key=lambda i: i.get("position", 0)):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        if len(names) <= 2:
            return ()  # only Home + Product, no category path
        return tuple(names[1:-1])
    return ()


_PACK_SIZE_PATTERNS = (
    re.compile(r"\b(\d{2,4})\s*(?:gloves?|count|ct|pcs?|pieces?)\s*(?:per|/)\s*(box|case|bag|pack)\b", re.I),
    re.compile(r"\b(\d{2,4})\s*(?:per|/)\s*(box|case|bag|pack)\b", re.I),
    re.compile(r"\b(box|bag|pack|case)\s*of\s*(\d{2,4})\b", re.I),
    re.compile(r"\b(box|bag|pack|case)\s+contains?\s+(\d{2,4})\b", re.I),
    re.compile(r"\b(\d{2,4})\s*(?:gloves?|count|ct|pcs?|pieces?)\s+per\s+(box|case|bag|pack)\b", re.I),
)


def _guess_pack_size(*texts: str) -> str | None:
    """Pull a pack-size string out of free-form text (e.g. a product description).

    Recognizes patterns like:
      "200 gloves per box"   → "200/box"
      "100 per case"         → "100/case"
      "Box of 50"            → "50/box"

    Returns ``None`` when nothing matches.
    """

    for text in texts:
        if not text:
            continue
        for pat in _PACK_SIZE_PATTERNS:
            m = pat.search(text)
            if not m:
                continue
            groups = m.groups()
            # Last-form puts the unit first; normalise to "<count>/<unit>".
            if groups[0].isdigit():
                count, unit = groups[0], groups[1]
            else:
                count, unit = groups[1], groups[0]
            return f"{count}/{unit.lower()}"
    return None


# --- DOM enrichments (Tier 2 reuses these against the Playwright-rendered HTML) -


_MASTER_DATA_RE = re.compile(
    r'window\.masterData\s*=\s*"((?:\\.|[^"\\])*)"',
)


def parse_master_data(html: str) -> dict[str, Any]:
    """Decode the ``window.masterData`` JS variable that Safco PDPs emit
    server-side and pull out per-variant Item # / Mfr # / brand.

    The blob is a percent-and-unicode-escaped JSON object keyed by Safco
    catalog number. Each entry carries:

      * ``sku``                       — the variant's catalog number (Item #)
      * ``manufacturer_part_number``  — the upstream Mfr #
      * ``manufacturer_name``         — the upstream brand name (used as
                                        ``brand`` since JSON-LD reports the
                                        seller "Safco Dental" instead).
      * ``stock_availability_label``  — variant-level stock label

    Returns a partial record dict (empty when the page has no masterData).
    Order of returned tuples matches the variant ordering in the source.
    """

    m = _MASTER_DATA_RE.search(html)
    if not m:
        return {}
    raw = m.group(1)
    try:
        decoded = bytes(raw, "utf-8").decode("unicode_escape")
        data = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        log.debug("master_data_unparseable", error=str(e))
        return {}
    if not isinstance(data, dict):
        return {}

    # Preserve source ordering by 'position' when available, fall back to dict order.
    variants = [v for v in data.values() if isinstance(v, dict)]
    variants.sort(key=lambda v: v.get("position") if isinstance(v.get("position"), int) else 0)

    item_numbers: list[str] = []
    mfr_numbers: list[str] = []
    brands: list[str] = []
    for v in variants:
        sku = v.get("sku")
        if isinstance(sku, str) and sku.strip():
            item_numbers.append(sku.strip())
        mfr = v.get("manufacturer_part_number")
        if isinstance(mfr, str) and mfr.strip():
            mfr_numbers.append(mfr.strip())
        mname = v.get("manufacturer_name")
        if isinstance(mname, str) and mname.strip():
            brands.append(mname.strip())

    out: dict[str, Any] = {}
    if item_numbers:
        out["item_numbers"] = tuple(item_numbers)
    if mfr_numbers:
        out["mfr_numbers"] = tuple(mfr_numbers)
    # On Safco the JSON-LD ``brand`` field always reports the seller ("Safco
    # Dental") rather than the actual product brand, so when masterData
    # exposes a consistent ``manufacturer_name`` we surface it as the real
    # ``brand`` and let it override JSON-LD downstream.
    distinct = {b for b in brands if b}
    chosen: str | None = None
    if len(distinct) == 1:
        chosen = next(iter(distinct))
    elif len(distinct) > 1:
        chosen = max(distinct, key=brands.count)
    if chosen:
        out["brand"] = chosen
    return out


_PACK_SIZE_SELECTORS = (
    ".product-pack-size",
    "[data-pack-size]",
)


def parse_dom_enrichments(html: str) -> dict[str, Any]:
    """Pull richer fields that JSON-LD doesn't carry. Safe on un-hydrated HTML
    (returns mostly empty values) and on hydrated HTML (returns populated values).
    """

    tree = HTMLParser(html)
    out: dict[str, Any] = {}

    for sel in _PACK_SIZE_SELECTORS:
        node = tree.css_first(sel)
        if node and node.text(strip=True):
            out["pack_size"] = node.text(strip=True)
            break

    # Specifications: any <table> with rows of <th>/<td> or <dt>/<dd> pairs.
    specs: dict[str, str] = {}
    for table in tree.css(".additional-attributes-table tr, table.data tr"):
        th = table.css_first("th")
        td = table.css_first("td")
        if th and td:
            specs[th.text(strip=True)] = td.text(strip=True)
    if specs:
        out["specifications"] = specs

    # Alternative / related product URLs.
    related: list[str] = []
    for link in tree.css(
        ".related-products a, .upsell-products a, .product-items a.product-item-link"
    ):
        href = link.attributes.get("href")
        if href and "/product/" in href:
            related.append(href)
    if related:
        # dedupe preserving order
        seen: set[str] = set()
        out["alternative_products"] = tuple(x for x in related if not (x in seen or seen.add(x)))

    return out


# --- Outcome + Extractor -----------------------------------------------------


@dataclass(frozen=True)
class ExtractorOutcome:
    """What ``Extractor.extract`` returns. Either ``record`` is non-None
    (success) or ``error`` is non-None (failure)."""

    record: ProductRecord | None
    method: ExtractionMethod
    error: str | None = None


class Extractor:
    """Tiered extractor. Inject an ``llm`` callable to override the LLM client
    (tests pass a mock; production passes ``llm.structured_extract``)."""

    def __init__(self, llm: LLMExtractor | None = None) -> None:
        self._llm = llm if llm is not None else _default_llm

    async def extract(
        self,
        *,
        url: str,
        html: str,
        source_category: str | None = None,
        enable_llm_fallback: bool = True,
        method_override: ExtractionMethod | None = None,
    ) -> ExtractorOutcome:
        # --- Tier 1 + DOM enrichment ---
        merged: dict[str, Any] = {}
        merged.update(parse_product_jsonld(html))
        merged.update(parse_dom_enrichments(html))
        # Safco-specific: pick up Item # / Mfr # / real brand from the
        # `window.masterData` JSON blob the storefront emits server-side.
        # masterData is more authoritative than JSON-LD for ``brand`` —
        # JSON-LD reports the seller (Safco Dental) instead of the upstream
        # brand, so let masterData overwrite that field.
        _MASTER_DATA_AUTHORITATIVE = frozenset({"brand"})
        for k, v in parse_master_data(html).items():
            if v in (None, "", (), [], {}):
                continue
            if k in _MASTER_DATA_AUTHORITATIVE or merged.get(k) in (None, "", (), [], {}):
                merged[k] = v
        category_path = parse_breadcrumb_jsonld(html)

        # Best-effort pack-size from description text — JSON-LD doesn't carry it
        # but most Safco PDPs describe the unit count in plain English.
        if not merged.get("pack_size"):
            guessed = _guess_pack_size(merged.get("description") or "", merged.get("name") or "")
            if guessed:
                merged["pack_size"] = guessed

        method = method_override or (
            ExtractionMethod.JSON_LD if merged else ExtractionMethod.DOM
        )

        if not _has_required(merged):
            if not enable_llm_fallback:
                # Caller (e.g. the pipeline) wants to try a richer fetch first.
                return ExtractorOutcome(
                    record=None,
                    method=method,
                    error=f"required fields missing without LLM fallback: {sorted(_missing(merged))}",
                )
            # --- Tier 3: LLM fallback ---
            log.info("extractor_invoking_llm_fallback", url=url, missing=_missing(merged))
            try:
                llm_data = await self._llm(html, url)
            except Exception as e:  # network / vLLM failures should not crash the run
                log.warning("llm_fallback_errored", url=url, error=str(e))
                llm_data = {}
            merged = _merge_llm(merged, llm_data)
            if llm_data and not category_path:
                cp = llm_data.get("category_path")
                if isinstance(cp, list):
                    category_path = tuple(str(x) for x in cp if x)
            method = ExtractionMethod.LLM_FALLBACK

            if not _has_required(merged):
                return ExtractorOutcome(
                    record=None,
                    method=ExtractionMethod.LLM_FALLBACK,
                    error=f"required fields missing after LLM fallback: {sorted(_missing(merged))}",
                )

        record = _build_record(
            merged,
            url=url,
            method=method,
            source_category=source_category,
            category_path=category_path,
        )
        return ExtractorOutcome(record=record, method=method)


# --- Implementation helpers --------------------------------------------------


def _has_required(d: dict[str, Any]) -> bool:
    return not _missing(d)


def _missing(d: dict[str, Any]) -> set[str]:
    return {f for f in REQUIRED_FIELDS if not d.get(f)}


def _merge_llm(existing: dict[str, Any], llm_data: dict[str, Any]) -> dict[str, Any]:
    """LLM only fills fields the deterministic tiers missed."""

    out = dict(existing)
    for k, v in llm_data.items():
        if v in (None, "", [], {}):
            continue
        if out.get(k) in (None, "", [], {}):
            out[k] = v
    if "price" in out and isinstance(out["price"], str):
        try:
            out["price"] = Decimal(out["price"])
        except InvalidOperation:
            out["price"] = None
    if "image_urls" in out and isinstance(out["image_urls"], list):
        out["image_urls"] = tuple(out["image_urls"])
    if "alternative_products" in out and isinstance(out["alternative_products"], list):
        out["alternative_products"] = tuple(out["alternative_products"])
    if "item_numbers" in out and isinstance(out["item_numbers"], list):
        out["item_numbers"] = tuple(str(x) for x in out["item_numbers"] if x)
    if "mfr_numbers" in out and isinstance(out["mfr_numbers"], list):
        out["mfr_numbers"] = tuple(str(x) for x in out["mfr_numbers"] if x)
    return out


def _build_record(
    data: dict[str, Any],
    *,
    url: str,
    method: ExtractionMethod,
    source_category: str | None,
    category_path: tuple[str, ...],
) -> ProductRecord:
    image_urls = data.get("image_urls") or ()
    if isinstance(image_urls, list):
        image_urls = tuple(image_urls)
    alts = data.get("alternative_products") or ()
    if isinstance(alts, list):
        alts = tuple(alts)
    item_numbers = data.get("item_numbers") or ()
    if isinstance(item_numbers, list):
        item_numbers = tuple(str(x) for x in item_numbers if x)
    mfr_numbers = data.get("mfr_numbers") or ()
    if isinstance(mfr_numbers, list):
        mfr_numbers = tuple(str(x) for x in mfr_numbers if x)
    avail = data.get("availability")
    if avail not in (None, "InStock", "OutOfStock", "PreOrder", "Discontinued", "Unknown"):
        avail = "Unknown"
    return ProductRecord(
        sku=str(data["sku"]),
        item_numbers=item_numbers,
        mfr_numbers=mfr_numbers,
        name=str(data["name"]),
        product_url=str(data.get("product_url") or url),
        category_path=category_path,
        brand=data.get("brand"),
        price=data.get("price"),
        currency=data.get("currency"),
        availability=avail,
        pack_size=data.get("pack_size"),
        description=str(data.get("description") or ""),
        specifications=dict(data.get("specifications") or {}),
        image_urls=image_urls,
        alternative_products=alts,
        extracted_at=datetime.now(UTC),
        extraction_method=method,
        source_category=source_category,
    )


# --- Default LLM caller ------------------------------------------------------


_LLM_SYSTEM_PROMPT = """You extract a single product record from raw HTML for a dental supply catalog.

Return ONLY a JSON object with these keys (use null/empty for unknown values):
  sku, item_numbers (array), mfr_numbers (array),
  name, brand, price, currency, availability,
  pack_size, description, specifications (object), image_urls (array),
  alternative_products (array of URLs), category_path (array)

Field guidance:
  - sku           — the parent product code (often shown in the URL slug).
  - item_numbers  — Safco's per-variant catalog numbers, labeled "Item #" on
                    the page. ONE entry per size/variant row.
  - mfr_numbers   — manufacturer part numbers, labeled "Mfr #" on the page.
                    ONE entry per size/variant row, aligned with item_numbers.
  - brand         — the upstream product brand (NOT the seller "Safco Dental",
                    which is what JSON-LD reports). Prefer the manufacturer
                    name surfaced in the variant tables when present.

availability must be one of: InStock, OutOfStock, PreOrder, Discontinued, Unknown.
price must be a numeric string with no currency symbol (e.g. "23.49").
"""


_LLM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sku": {"type": ["string", "null"]},
        "item_numbers": {"type": ["array", "null"], "items": {"type": "string"}},
        "mfr_numbers": {"type": ["array", "null"], "items": {"type": "string"}},
        "name": {"type": ["string", "null"]},
        "brand": {"type": ["string", "null"]},
        "price": {"type": ["string", "null"]},
        "currency": {"type": ["string", "null"]},
        "availability": {"type": ["string", "null"]},
        "pack_size": {"type": ["string", "null"]},
        "description": {"type": ["string", "null"]},
        "specifications": {"type": ["object", "null"]},
        "image_urls": {"type": ["array", "null"], "items": {"type": "string"}},
        "alternative_products": {"type": ["array", "null"], "items": {"type": "string"}},
        "category_path": {"type": ["array", "null"], "items": {"type": "string"}},
    },
    "required": ["sku", "name"],
    "additionalProperties": False,
}


async def _default_llm(html: str, url: str) -> dict[str, Any]:
    # Truncate to keep prompts cheap; real-world rendered HTML can be huge.
    snippet = html[:60_000]
    user = (
        f"Source URL: {url}\n\n"
        f"HTML (truncated to 60k chars):\n```\n{snippet}\n```"
    )
    return await structured_extract(
        system=_LLM_SYSTEM_PROMPT,
        user=user,
        json_schema=_LLM_SCHEMA,
        schema_name="product_record",
    )

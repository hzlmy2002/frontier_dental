"""Domain models.

``ProductRecord`` is the canonical, persistence-ready shape that every other
component speaks. ``CrawlState`` tracks per-URL pipeline progress so a
crashed run can resume where it left off.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ExtractionMethod(StrEnum):
    JSON_LD = "json_ld"
    DOM = "dom"
    PLAYWRIGHT_DOM = "playwright_dom"
    LLM_FALLBACK = "llm_fallback"


class CrawlStatus(StrEnum):
    DISCOVERED = "discovered"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    DONE = "done"
    FAILED = "failed"


_ImmutableModel = ConfigDict(frozen=True, str_strip_whitespace=True, populate_by_name=True)


class ProductRecord(BaseModel):
    """Canonical product record. Immutable; constructed by the Extractor and
    persisted by the Storage layer.

    Identifier fields:
      - ``sku``  — Safco's parent product code (e.g. ``DRCDK``). Stable, used
                   as primary key. Also the value carried by the JSON-LD
                   ``Product.sku`` field.
      - ``item_numbers`` — Safco's per-variant catalog numbers (e.g.
                   ``("4681214","4681216",...)``). What customers paste into
                   order forms; what the PDP labels "Item #".
      - ``mfr_numbers``  — Manufacturer part numbers, one per variant (e.g.
                   ``("ALGA200XS","ALGA200S",...)``). What the PDP labels
                   "Mfr #".

    Both ``item_numbers`` and ``mfr_numbers`` are tuples because most Safco
    PDPs are *grouped products* with several size/variant rows, each
    carrying its own pair. Single-variant products yield one-element tuples.
    """

    model_config = _ImmutableModel

    sku: str
    item_numbers: tuple[str, ...] = ()
    mfr_numbers: tuple[str, ...] = ()
    name: str
    product_url: str
    category_path: tuple[str, ...] = ()
    brand: str | None = None
    price: Decimal | None = None
    currency: str | None = None
    availability: Literal["InStock", "OutOfStock", "PreOrder", "Discontinued", "Unknown"] | None = (
        None
    )
    pack_size: str | None = None
    description: str = ""
    specifications: dict[str, str] = Field(default_factory=dict)
    image_urls: tuple[str, ...] = ()
    alternative_products: tuple[str, ...] = ()
    extracted_at: datetime
    extraction_method: ExtractionMethod
    source_category: str | None = None  # which top-level category drove discovery


class CrawlState(BaseModel):
    """Resumability checkpoint row."""

    model_config = ConfigDict(populate_by_name=True)

    url: str
    category: str
    status: CrawlStatus
    sku: str | None = None
    error: str | None = None
    updated_at: datetime


class DiscoveredUrl(BaseModel):
    """Output of the listing crawler."""

    model_config = _ImmutableModel

    url: HttpUrl
    category: str


class CategoryRef(BaseModel):
    """A category surfaced by the Navigator. ``slug`` is the trailing path
    segment, used as a stable identifier in storage and CLI."""

    model_config = _ImmutableModel

    name: str
    url: str
    slug: str


def normalize_availability(schema_org_value: str | None) -> str | None:
    """Map a schema.org availability URI (or bare token) to our enum string."""

    if not schema_org_value:
        return None
    token = schema_org_value.rsplit("/", 1)[-1].strip()
    return token if token in {"InStock", "OutOfStock", "PreOrder", "Discontinued"} else "Unknown"

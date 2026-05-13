"""Validator + deduper.

The Extractor already validates fields via Pydantic, so this layer's
responsibility is **deduplication**: collapse multiple records pointing at
the same SKU (or same product URL when SKU is missing) into a single
canonical record. The merge prefers more-informative records over
less-informative ones — concretely:

* deterministic methods beat LLM fallback,
* records with ``brand``/``pack_size``/``specifications`` filled in
  beat ones without.
"""

from __future__ import annotations

from collections.abc import Iterable

import structlog

from ..models import ExtractionMethod, ProductRecord

log = structlog.get_logger(__name__)


_METHOD_RANK: dict[ExtractionMethod, int] = {
    ExtractionMethod.JSON_LD: 0,
    ExtractionMethod.DOM: 1,
    ExtractionMethod.PLAYWRIGHT_DOM: 2,
    ExtractionMethod.LLM_FALLBACK: 3,
}


def _completeness(p: ProductRecord) -> int:
    """Higher = more complete. Used to break ties when two records share a SKU."""

    score = 0
    if p.brand:
        score += 1
    if p.pack_size:
        score += 1
    if p.specifications:
        score += 1
    if p.alternative_products:
        score += 1
    if len(p.image_urls) > 1:
        score += 1
    return score


def _better(a: ProductRecord, b: ProductRecord) -> ProductRecord:
    """Pick the more informative of two records that share a key."""

    a_rank = _METHOD_RANK.get(a.extraction_method, 99)
    b_rank = _METHOD_RANK.get(b.extraction_method, 99)
    if a_rank != b_rank:
        return a if a_rank < b_rank else b
    a_score = _completeness(a)
    b_score = _completeness(b)
    if a_score != b_score:
        return a if a_score > b_score else b
    return a if a.extracted_at >= b.extracted_at else b


def dedupe(records: Iterable[ProductRecord]) -> list[ProductRecord]:
    """Return a deduped list keyed by ``sku`` (falls back to ``product_url``)."""

    by_key: dict[str, ProductRecord] = {}
    for r in records:
        key = r.sku.strip() or r.product_url
        if key in by_key:
            by_key[key] = _better(by_key[key], r)
        else:
            by_key[key] = r
    return list(by_key.values())

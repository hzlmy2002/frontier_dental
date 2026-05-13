"""Tests for the validator/deduper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from frontier_dental.agents.validator import dedupe
from frontier_dental.models import ExtractionMethod, ProductRecord


def _r(
    sku: str,
    *,
    method: ExtractionMethod = ExtractionMethod.JSON_LD,
    age_minutes: int = 0,
    pack_size: str | None = None,
    brand: str | None = None,
    specifications: dict[str, str] | None = None,
    image_urls: tuple[str, ...] = (),
    url: str | None = None,
) -> ProductRecord:
    return ProductRecord(
        sku=sku,
        name=f"Product {sku}",
        product_url=url or f"https://example.com/p/{sku.lower()}",
        category_path=("Cat",),
        price=Decimal("1.00"),
        currency="USD",
        description="x",
        pack_size=pack_size,
        brand=brand,
        specifications=specifications or {},
        image_urls=image_urls,
        extracted_at=datetime.now(UTC) - timedelta(minutes=age_minutes),
        extraction_method=method,
    )


def test_dedupe_keeps_unique_records() -> None:
    out = dedupe([_r("A"), _r("B")])
    assert sorted(p.sku for p in out) == ["A", "B"]


def test_dedupe_prefers_deterministic_method() -> None:
    llm = _r("A", method=ExtractionMethod.LLM_FALLBACK, brand="Foo")
    json_ld = _r("A", method=ExtractionMethod.JSON_LD)
    out = dedupe([llm, json_ld])
    assert len(out) == 1
    assert out[0].extraction_method == ExtractionMethod.JSON_LD


def test_dedupe_prefers_more_complete_record_when_methods_tie() -> None:
    bare = _r("A")
    rich = _r("A", brand="Foo", pack_size="100/box", specifications={"k": "v"})
    out = dedupe([bare, rich])
    assert len(out) == 1
    assert out[0].brand == "Foo"
    assert out[0].pack_size == "100/box"


def test_dedupe_falls_back_to_url_when_sku_blank() -> None:
    a = _r("", url="https://example.com/p/x").model_copy(update={"sku": ""})
    b = _r("", url="https://example.com/p/x").model_copy(
        update={"sku": "", "brand": "Foo"}
    )
    out = dedupe([a, b])
    assert len(out) == 1
    assert out[0].brand == "Foo"

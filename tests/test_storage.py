"""Storage round-trip + export tests."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from frontier_dental.models import CrawlStatus, ExtractionMethod, ProductRecord
from frontier_dental.storage import Storage, make_state


def _record(sku: str = "TEST1") -> ProductRecord:
    return ProductRecord(
        sku=sku,
        item_numbers=("4681214", "4681216"),
        mfr_numbers=("ALGA200XS", "ALGA200S"),
        name="Test Product",
        product_url=f"https://example.com/p/{sku.lower()}",
        category_path=("A", "B"),
        brand="Brand",
        price=Decimal("9.99"),
        currency="USD",
        availability="InStock",
        pack_size="100/box",
        description="hello",
        specifications={"Material": "Nitrile"},
        image_urls=("https://img.example.com/1.jpg",),
        alternative_products=("https://example.com/p/other",),
        extracted_at=datetime.now(UTC),
        extraction_method=ExtractionMethod.JSON_LD,
        source_category="gloves",
    )


def test_upsert_and_read_back(tmp_path: Path) -> None:
    s = Storage(tmp_path / "db.sqlite")
    s.upsert_product(_record("SKU1"))
    products = s.all_products()
    assert len(products) == 1
    assert products[0].sku == "SKU1"
    assert products[0].item_numbers == ("4681214", "4681216")
    assert products[0].mfr_numbers == ("ALGA200XS", "ALGA200S")
    assert products[0].price == Decimal("9.99")
    assert products[0].specifications == {"Material": "Nitrile"}
    assert products[0].image_urls == ("https://img.example.com/1.jpg",)


def test_upsert_is_idempotent_on_sku(tmp_path: Path) -> None:
    s = Storage(tmp_path / "db.sqlite")
    s.upsert_product(_record("SKU1"))
    updated = _record("SKU1").model_copy(update={"name": "Renamed"})
    s.upsert_product(updated)
    products = s.all_products()
    assert len(products) == 1
    assert products[0].name == "Renamed"


def test_crawl_state_round_trip(tmp_path: Path) -> None:
    s = Storage(tmp_path / "db.sqlite")
    state = make_state("https://x.com/a", "gloves", CrawlStatus.DISCOVERED)
    s.upsert_state(state)
    got = s.get_state("https://x.com/a")
    assert got is not None
    assert got.status == CrawlStatus.DISCOVERED
    s.upsert_state(make_state("https://x.com/a", "gloves", CrawlStatus.DONE, sku="SKU1"))
    counts = s.status_counts()
    assert counts == {"done": 1}


def test_export_json_and_csv(tmp_path: Path) -> None:
    s = Storage(tmp_path / "db.sqlite")
    s.upsert_product(_record("SKU1"))
    s.upsert_product(_record("SKU2"))

    json_path = tmp_path / "out.json"
    s.export_json(json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0]["specifications"] == {"Material": "Nitrile"}

    csv_path = tmp_path / "out.csv"
    s.export_csv(csv_path)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["category_path"] == "A > B"
    assert rows[0]["image_urls"] == "https://img.example.com/1.jpg"


def test_urls_by_status_filters_correctly(tmp_path: Path) -> None:
    s = Storage(tmp_path / "db.sqlite")
    s.upsert_state(make_state("https://x.com/a", "g", CrawlStatus.DISCOVERED))
    s.upsert_state(make_state("https://x.com/b", "g", CrawlStatus.DONE))
    s.upsert_state(make_state("https://x.com/c", "g", CrawlStatus.FAILED, error="boom"))

    discovered = s.urls_by_status(CrawlStatus.DISCOVERED)
    assert [d.url for d in discovered] == ["https://x.com/a"]

    multi = s.urls_by_status(CrawlStatus.DISCOVERED, CrawlStatus.FAILED)
    assert sorted(d.url for d in multi) == ["https://x.com/a", "https://x.com/c"]

"""SQLite-backed persistence + CSV/JSON export.

Two tables:
- ``products`` — canonical ``ProductRecord`` rows, keyed by SKU (idempotent upsert).
- ``crawl_state`` — per-URL pipeline status, used for resumability.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .models import CrawlState, CrawlStatus, ExtractionMethod, ProductRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    item_numbers TEXT NOT NULL DEFAULT '[]',
    mfr_numbers TEXT NOT NULL DEFAULT '[]',
    name TEXT NOT NULL,
    product_url TEXT NOT NULL UNIQUE,
    category_path TEXT NOT NULL,
    brand TEXT,
    price TEXT,
    currency TEXT,
    availability TEXT,
    pack_size TEXT,
    description TEXT NOT NULL DEFAULT '',
    specifications TEXT NOT NULL DEFAULT '{}',
    image_urls TEXT NOT NULL DEFAULT '[]',
    alternative_products TEXT NOT NULL DEFAULT '[]',
    extracted_at TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    source_category TEXT
);

CREATE TABLE IF NOT EXISTS crawl_state (
    url TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    status TEXT NOT NULL,
    sku TEXT,
    error TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_crawl_status ON crawl_state(status);
CREATE INDEX IF NOT EXISTS idx_crawl_category ON crawl_state(category);
"""


class Storage:
    """Thin wrapper around an on-disk SQLite database.

    Each method opens its own short-lived connection so the class is safe to
    share across asyncio tasks (writes serialize at the SQLite level).
    """

    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.sqlite_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            conn.close()

    # --- Product persistence ---

    def upsert_product(self, product: ProductRecord) -> None:
        row = _product_to_row(product)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO products (
                    sku, item_numbers, mfr_numbers, name, product_url,
                    category_path, brand,
                    price, currency, availability, pack_size, description,
                    specifications, image_urls, alternative_products,
                    extracted_at, extraction_method, source_category
                ) VALUES (
                    :sku, :item_numbers, :mfr_numbers, :name, :product_url,
                    :category_path, :brand,
                    :price, :currency, :availability, :pack_size, :description,
                    :specifications, :image_urls, :alternative_products,
                    :extracted_at, :extraction_method, :source_category
                )
                ON CONFLICT(sku) DO UPDATE SET
                    item_numbers=excluded.item_numbers,
                    mfr_numbers=excluded.mfr_numbers,
                    name=excluded.name,
                    product_url=excluded.product_url,
                    category_path=excluded.category_path,
                    brand=excluded.brand,
                    price=excluded.price,
                    currency=excluded.currency,
                    availability=excluded.availability,
                    pack_size=excluded.pack_size,
                    description=excluded.description,
                    specifications=excluded.specifications,
                    image_urls=excluded.image_urls,
                    alternative_products=excluded.alternative_products,
                    extracted_at=excluded.extracted_at,
                    extraction_method=excluded.extraction_method,
                    source_category=excluded.source_category
                """,
                row,
            )

    def all_products(self) -> list[ProductRecord]:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM products ORDER BY sku")
            cur.row_factory = sqlite3.Row
            return [_row_to_product(r) for r in cur.fetchall()]

    def product_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM products").fetchone()[0])

    # --- Crawl state ---

    def upsert_state(self, state: CrawlState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crawl_state (url, category, status, sku, error, updated_at)
                VALUES (:url, :category, :status, :sku, :error, :updated_at)
                ON CONFLICT(url) DO UPDATE SET
                    category=excluded.category,
                    status=excluded.status,
                    sku=excluded.sku,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                {
                    "url": state.url,
                    "category": state.category,
                    "status": state.status.value,
                    "sku": state.sku,
                    "error": state.error,
                    "updated_at": state.updated_at.isoformat(),
                },
            )

    def get_state(self, url: str) -> CrawlState | None:
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM crawl_state WHERE url = ?", (url,))
            cur.row_factory = sqlite3.Row
            row = cur.fetchone()
            if row is None:
                return None
            return CrawlState(
                url=row["url"],
                category=row["category"],
                status=CrawlStatus(row["status"]),
                sku=row["sku"],
                error=row["error"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def status_counts(self) -> dict[str, int]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT status, COUNT(*) FROM crawl_state GROUP BY status ORDER BY status"
            )
            return {row[0]: int(row[1]) for row in cur.fetchall()}

    def urls_by_status(self, *statuses: CrawlStatus) -> list[CrawlState]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._connect() as conn:
            cur = conn.execute(
                f"SELECT * FROM crawl_state WHERE status IN ({placeholders}) ORDER BY url",
                tuple(s.value for s in statuses),
            )
            cur.row_factory = sqlite3.Row
            return [
                CrawlState(
                    url=r["url"],
                    category=r["category"],
                    status=CrawlStatus(r["status"]),
                    sku=r["sku"],
                    error=r["error"],
                    updated_at=datetime.fromisoformat(r["updated_at"]),
                )
                for r in cur.fetchall()
            ]

    # --- Exports ---

    def export_json(self, path: Path) -> int:
        products = self.all_products()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([_product_to_jsonable(p) for p in products], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return len(products)

    def export_csv(self, path: Path) -> int:
        products = self.all_products()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_HEADERS)
            for p in products:
                writer.writerow(_product_to_csv_row(p))
        return len(products)


# --- Row mappers -------------------------------------------------------------

_CSV_HEADERS = (
    "sku",
    "item_numbers",
    "mfr_numbers",
    "name",
    "product_url",
    "category_path",
    "brand",
    "price",
    "currency",
    "availability",
    "pack_size",
    "description",
    "specifications_json",
    "image_urls",
    "alternative_products",
    "extracted_at",
    "extraction_method",
    "source_category",
)


def _product_to_row(p: ProductRecord) -> dict[str, str | None]:
    return {
        "sku": p.sku,
        "item_numbers": json.dumps(list(p.item_numbers)),
        "mfr_numbers": json.dumps(list(p.mfr_numbers)),
        "name": p.name,
        "product_url": p.product_url,
        "category_path": json.dumps(list(p.category_path)),
        "brand": p.brand,
        "price": str(p.price) if p.price is not None else None,
        "currency": p.currency,
        "availability": p.availability,
        "pack_size": p.pack_size,
        "description": p.description,
        "specifications": json.dumps(p.specifications),
        "image_urls": json.dumps(list(p.image_urls)),
        "alternative_products": json.dumps(list(p.alternative_products)),
        "extracted_at": p.extracted_at.isoformat(),
        "extraction_method": p.extraction_method.value,
        "source_category": p.source_category,
    }


def _row_to_product(row: sqlite3.Row) -> ProductRecord:
    return ProductRecord(
        sku=row["sku"],
        item_numbers=tuple(json.loads(row["item_numbers"] or "[]")),
        mfr_numbers=tuple(json.loads(row["mfr_numbers"] or "[]")),
        name=row["name"],
        product_url=row["product_url"],
        category_path=tuple(json.loads(row["category_path"])),
        brand=row["brand"],
        price=Decimal(row["price"]) if row["price"] is not None else None,
        currency=row["currency"],
        availability=row["availability"],
        pack_size=row["pack_size"],
        description=row["description"] or "",
        specifications=json.loads(row["specifications"] or "{}"),
        image_urls=tuple(json.loads(row["image_urls"] or "[]")),
        alternative_products=tuple(json.loads(row["alternative_products"] or "[]")),
        extracted_at=datetime.fromisoformat(row["extracted_at"]),
        extraction_method=ExtractionMethod(row["extraction_method"]),
        source_category=row["source_category"],
    )


def _product_to_jsonable(p: ProductRecord) -> dict[str, object]:
    return {
        "sku": p.sku,
        "item_numbers": list(p.item_numbers),
        "mfr_numbers": list(p.mfr_numbers),
        "name": p.name,
        "product_url": p.product_url,
        "category_path": list(p.category_path),
        "brand": p.brand,
        "price": str(p.price) if p.price is not None else None,
        "currency": p.currency,
        "availability": p.availability,
        "pack_size": p.pack_size,
        "description": p.description,
        "specifications": p.specifications,
        "image_urls": list(p.image_urls),
        "alternative_products": list(p.alternative_products),
        "extracted_at": p.extracted_at.isoformat(),
        "extraction_method": p.extraction_method.value,
        "source_category": p.source_category,
    }


def _product_to_csv_row(p: ProductRecord) -> tuple[str, ...]:
    return (
        p.sku,
        "|".join(p.item_numbers),
        "|".join(p.mfr_numbers),
        p.name,
        p.product_url,
        " > ".join(p.category_path),
        p.brand or "",
        str(p.price) if p.price is not None else "",
        p.currency or "",
        p.availability or "",
        p.pack_size or "",
        p.description,
        json.dumps(p.specifications, ensure_ascii=False),
        "|".join(p.image_urls),
        "|".join(p.alternative_products),
        p.extracted_at.isoformat(),
        p.extraction_method.value,
        p.source_category or "",
    )


def _now() -> datetime:
    return datetime.now(UTC)


def make_state(
    url: str, category: str, status: CrawlStatus, *, sku: str | None = None, error: str | None = None
) -> CrawlState:
    """Helper for callers that don't want to import datetime everywhere."""

    return CrawlState(url=url, category=category, status=status, sku=sku, error=error, updated_at=_now())


def bulk_upsert_products(storage: Storage, products: Iterable[ProductRecord]) -> int:
    n = 0
    for p in products:
        storage.upsert_product(p)
        n += 1
    return n

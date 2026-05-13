"""Tests for the tiered Extractor.

Tier 1 — httpx + JSON-LD on static HTML (always tried first).
Tier 2 — Playwright re-renders when static fetch fields are insufficient.
        (Tier 2 is exercised by integration tests in Phase 4 — unit tests here
         pass already-rendered HTML to the same parsing function.)
Tier 3 — LLM fallback when the above leave required fields blank.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from frontier_dental.agents.extractor import (
    REQUIRED_FIELDS,
    Extractor,
    parse_breadcrumb_jsonld,
    parse_master_data,
    parse_product_jsonld,
)
from frontier_dental.models import ExtractionMethod


# --- Tier 1: JSON-LD parsing -------------------------------------------------


def test_parse_product_jsonld_extracts_core_fields(alasta_pro_html: str) -> None:
    parsed = parse_product_jsonld(alasta_pro_html)
    assert parsed["sku"] == "DRCDK"
    assert parsed["name"] == "Alasta Pro"
    assert parsed["brand"] == "Safco Dental"
    assert parsed["price"] == Decimal("23.49")
    assert parsed["currency"] == "USD"
    assert parsed["availability"] == "InStock"
    assert "Alasta" in parsed["description"]
    assert any("drcdk.jpg" in url for url in parsed["image_urls"])


def test_parse_breadcrumb_jsonld_yields_full_path(alasta_pro_html: str) -> None:
    path = parse_breadcrumb_jsonld(alasta_pro_html)
    # First crumb ("Home") is dropped; final crumb ("Alasta Pro") is the product itself, also dropped.
    assert path == ("Dental Supplies", "Dental Exam Gloves", "Nitrile gloves")


def test_parse_product_jsonld_handles_missing_blocks() -> None:
    html = "<html><body>no structured data here</body></html>"
    parsed = parse_product_jsonld(html)
    assert parsed == {}


@pytest.mark.asyncio
async def test_extractor_returns_jsonld_when_sufficient(alasta_pro_html: str) -> None:
    extractor = Extractor(llm=_llm_should_not_be_called())
    outcome = await extractor.extract(
        url="https://www.safcodental.com/product/alasta-pro",
        html=alasta_pro_html,
        source_category="gloves",
    )
    assert outcome.method == ExtractionMethod.JSON_LD
    assert outcome.record is not None
    p = outcome.record
    assert p.sku == "DRCDK"
    assert p.price == Decimal("23.49")
    assert p.category_path == ("Dental Supplies", "Dental Exam Gloves", "Nitrile gloves")
    assert p.source_category == "gloves"
    assert p.extraction_method == ExtractionMethod.JSON_LD


# --- Tier 3: LLM fallback ----------------------------------------------------


@pytest.mark.asyncio
async def test_extractor_falls_back_to_llm_for_irregular_pdp(irregular_pdp_html: str) -> None:
    fake_llm = AsyncMock(
        return_value={
            "sku": "GP-2000-MYST",
            "name": "Glove-Pro 2000 Mystery Pack",
            "brand": "Glove-Pro",
            "price": "42.50",
            "currency": "USD",
            "availability": "InStock",
            "pack_size": "150/box",
            "description": "Premium nitrile gloves with extra grip. Latex-free. Powder-free.",
            "specifications": {"Material": "Nitrile", "Color": "Blue", "Size range": "XS-XL"},
            "image_urls": [
                "https://example.com/img/glovepro1.jpg",
                "https://example.com/img/glovepro2.jpg",
            ],
            "alternative_products": [
                "https://example.com/product/glovepro-junior",
                "https://example.com/product/grip-master-3000",
            ],
            "category_path": ["Gloves"],
        }
    )
    extractor = Extractor(llm=fake_llm)
    outcome = await extractor.extract(
        url="https://example.com/product/mystery",
        html=irregular_pdp_html,
        source_category="gloves",
    )
    assert outcome.method == ExtractionMethod.LLM_FALLBACK
    assert fake_llm.await_count == 1
    p = outcome.record
    assert p is not None
    assert p.sku == "GP-2000-MYST"
    assert p.pack_size == "150/box"
    assert p.specifications["Material"] == "Nitrile"
    assert p.extraction_method == ExtractionMethod.LLM_FALLBACK


@pytest.mark.asyncio
async def test_extractor_returns_failure_when_llm_returns_empty(irregular_pdp_html: str) -> None:
    """LLM fallback that returns insufficient fields => extraction fails cleanly."""

    fake_llm = AsyncMock(return_value={})
    extractor = Extractor(llm=fake_llm)
    outcome = await extractor.extract(
        url="https://example.com/product/mystery",
        html=irregular_pdp_html,
        source_category="gloves",
    )
    assert outcome.record is None
    assert outcome.method == ExtractionMethod.LLM_FALLBACK
    assert outcome.error is not None
    assert "missing" in outcome.error.lower() or "failed" in outcome.error.lower()


@pytest.mark.asyncio
async def test_extractor_skips_llm_when_fallback_disabled(irregular_pdp_html: str) -> None:
    """The pipeline uses ``enable_llm_fallback=False`` for its first pass so it
    can try a Playwright re-render before paying for an LLM call."""

    fake_llm = AsyncMock(side_effect=AssertionError("LLM must not be called"))
    extractor = Extractor(llm=fake_llm)
    outcome = await extractor.extract(
        url="https://example.com/product/mystery",
        html=irregular_pdp_html,
        source_category="gloves",
        enable_llm_fallback=False,
    )
    assert outcome.record is None
    assert fake_llm.await_count == 0
    assert outcome.error is not None


# --- Helpers -----------------------------------------------------------------


def _llm_should_not_be_called() -> AsyncMock:
    m = AsyncMock(side_effect=AssertionError("LLM must not be invoked when JSON-LD is sufficient"))
    return m


def test_required_fields_match_record_schema() -> None:
    """Sanity: REQUIRED_FIELDS exists so the fallback trigger is documented."""

    assert "sku" in REQUIRED_FIELDS
    assert "name" in REQUIRED_FIELDS


@pytest.mark.parametrize(
    "text,expected",
    [
        ("200 gloves per box", "200/box"),
        ("100 per case", "100/case"),
        ("Box of 50", "50/box"),
        ("Bag of 24", "24/bag"),
        ("100 ct/box", "100/box"),
        ("Quantity: a few", None),
        ("", None),
    ],
)
def test_guess_pack_size(text: str, expected: str | None) -> None:
    from frontier_dental.agents.extractor import _guess_pack_size

    assert _guess_pack_size(text) == expected


@pytest.mark.asyncio
async def test_extractor_recovers_pack_size_from_description(alasta_pro_html: str) -> None:
    """The Alasta Pro description says 'Each box contains 200 ... gloves' — the
    extractor should surface that as a structured pack_size."""

    extractor = Extractor()
    outcome = await extractor.extract(
        url="https://www.safcodental.com/product/alasta-pro",
        html=alasta_pro_html,
        source_category="gloves",
    )
    assert outcome.record is not None
    # Pattern 'box of 200' or '200 ... per box' — either acceptable.
    assert outcome.record.pack_size is not None
    assert "200" in outcome.record.pack_size
    assert "box" in outcome.record.pack_size.lower()


# --- Tier 1: Safco-specific window.masterData (Item # / Mfr #) -------------


def test_parse_master_data_extracts_per_variant_item_and_mfr_numbers(
    alasta_pro_html: str,
) -> None:
    """Each Safco PDP ships a `window.masterData` JS blob with one entry per
    size/variant. We pull Item # (= variant sku) and Mfr # (= manufacturer
    part number) into ordered tuples — Alasta Pro has 5 sizes (XS-XL+)."""

    parsed = parse_master_data(alasta_pro_html)
    assert isinstance(parsed.get("item_numbers"), tuple)
    assert isinstance(parsed.get("mfr_numbers"), tuple)

    item_numbers = parsed["item_numbers"]
    mfr_numbers = parsed["mfr_numbers"]
    # Five variants on the Alasta Pro PDP at fixture-capture time.
    assert len(item_numbers) >= 1
    assert len(mfr_numbers) >= 1
    # Item numbers are Safco's 7-digit numeric catalog codes.
    assert all(s.isdigit() for s in item_numbers)
    # Mfr numbers carry the ALGA200* shape on this product.
    assert all("ALGA200" in s for s in mfr_numbers)


def test_parse_master_data_returns_empty_when_blob_missing() -> None:
    assert parse_master_data("<html><body>no masterData here</body></html>") == {}


# --- Description cleaning --------------------------------------------------


def test_clean_description_passes_plain_text_through_unchanged() -> None:
    from frontier_dental.agents.extractor import _clean_description

    raw = "Make glove changes faster and easier with Alasta® PRO"
    assert _clean_description(raw) == raw


def test_clean_description_strips_doubly_escaped_html_and_teams_wrappers() -> None:
    """The Safco CMS lets editors paste Teams-flavoured HTML into the
    description; JSON-LD then string-encodes it, producing values like
    ``"&lt;p&gt;&lt;span ...&gt;Body&amp;nbsp;text&lt;/p&gt;"``. We want clean
    plaintext out the other side."""

    from frontier_dental.agents.extractor import _clean_description

    raw = (
        "&lt;p&gt;&lt;span data-teams=\"true\"&gt;&lt;span class=\"ui-provider a b\" "
        "dir=\"ltr\"&gt;Grab incredible deals on open-box clearance items.&lt;br /&gt;"
        "&lt;br /&gt;Perfect for quality and savings.&amp;nbsp;These like-new items "
        "at unbeatable prices!&amp;ndash; please contact your account manager."
        "&lt;/span&gt;&lt;/span&gt;&lt;/p&gt;"
    )
    out = _clean_description(raw)
    assert "<" not in out and ">" not in out
    assert "&lt;" not in out and "&amp;" not in out and "&nbsp;" not in out
    assert "data-teams" not in out and "ui-provider" not in out
    assert "Grab incredible deals" in out
    assert "Perfect for quality" in out
    # – = en-dash, decoded from &ndash;
    assert "–" in out or "-" in out
    # No multi-space runs
    assert "  " not in out


@pytest.mark.asyncio
async def test_extractor_populates_item_and_mfr_numbers_from_masterdata(
    alasta_pro_html: str,
) -> None:
    extractor = Extractor()
    outcome = await extractor.extract(
        url="https://www.safcodental.com/product/alasta-pro",
        html=alasta_pro_html,
        source_category="gloves",
    )
    assert outcome.record is not None
    assert len(outcome.record.item_numbers) >= 1
    assert len(outcome.record.mfr_numbers) >= 1
    # Brand should now reflect the upstream manufacturer surfaced by
    # masterData, overriding the JSON-LD seller name.
    assert outcome.record.brand is not None and outcome.record.brand.strip()
    assert outcome.record.brand != "Safco Dental"


@pytest.mark.asyncio
async def test_extractor_lignospan_overrides_jsonld_brand_with_real_manufacturer(
    lignospan_html: str,
) -> None:
    """Regression: JSON-LD reports ``brand="Safco Dental"`` (the seller) for
    every Safco PDP. ``window.masterData`` carries the actual brand
    (``Septodont`` for Lignospan); the extractor must prefer it."""

    extractor = Extractor()
    outcome = await extractor.extract(
        url="https://www.safcodental.com/product/lignospan-reg",
        html=lignospan_html,
        source_category="anesthetics",
    )
    assert outcome.record is not None
    p = outcome.record
    assert p.item_numbers == ("3540755",)
    assert p.mfr_numbers == ("01A1100",)
    assert p.brand == "Septodont"

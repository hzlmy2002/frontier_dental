"""Tests for the Adobe Commerce recommendations API client.

The HTTP layer is mocked via ``httpx.MockTransport`` so tests never hit
the network. We exercise:

* the response parser (``_parse_alternative_urls``) against the real
  shape we observed from a live API call;
* end-to-end ``RecsClient.fetch_alternatives`` with a mocked transport,
  including dedupe, current-SKU filtering, and cap enforcement;
* graceful degradation: empty tuple when the API errors.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from frontier_dental.recs import RecsClient, _normalize_safco_url, _parse_alternative_urls


# --- _normalize_safco_url --------------------------------------------------


@pytest.mark.unit
def test_normalize_safco_url_protocol_relative() -> None:
    assert (
        _normalize_safco_url("//www.safcodental.com/product/foo")
        == "https://www.safcodental.com/product/foo"
    )


@pytest.mark.unit
def test_normalize_safco_url_path_only() -> None:
    assert (
        _normalize_safco_url("/product/foo")
        == "https://www.safcodental.com/product/foo"
    )


@pytest.mark.unit
def test_normalize_safco_url_passthrough_absolute() -> None:
    assert (
        _normalize_safco_url("https://www.safcodental.com/product/foo")
        == "https://www.safcodental.com/product/foo"
    )


@pytest.mark.unit
def test_normalize_safco_url_rejects_garbage() -> None:
    assert _normalize_safco_url("") == ""
    assert _normalize_safco_url("not a url") == ""


# --- _parse_alternative_urls ------------------------------------------------


def _payload(units: list[dict[str, Any]]) -> dict[str, Any]:
    return {"results": units, "totalResults": len(units)}


def _unit(products: list[dict[str, Any]]) -> dict[str, Any]:
    return {"unitName": "Trending", "products": products}


def _product(sku: str, slug: str) -> dict[str, Any]:
    return {"sku": sku, "name": sku, "url": f"//www.safcodental.com/product/{slug}"}


@pytest.mark.unit
def test_parse_dedupes_across_units_and_filters_current_sku() -> None:
    payload = _payload(
        [
            _unit([_product("AAA", "alpha"), _product("DRCDD", "self"), _product("BBB", "bravo")]),
            _unit([_product("BBB", "bravo"), _product("CCC", "charlie")]),
        ]
    )
    out = _parse_alternative_urls(payload, current_sku="DRCDD", max_count=10)
    assert out == (
        "https://www.safcodental.com/product/alpha",
        "https://www.safcodental.com/product/bravo",
        "https://www.safcodental.com/product/charlie",
    )


@pytest.mark.unit
def test_parse_respects_max_count_across_units() -> None:
    payload = _payload(
        [
            _unit([_product("A1", "a1"), _product("A2", "a2"), _product("A3", "a3")]),
            _unit([_product("A4", "a4"), _product("A5", "a5")]),
        ]
    )
    out = _parse_alternative_urls(payload, current_sku="X", max_count=2)
    assert out == (
        "https://www.safcodental.com/product/a1",
        "https://www.safcodental.com/product/a2",
    )


@pytest.mark.unit
def test_parse_handles_missing_or_malformed_fields() -> None:
    payload = {
        "results": [
            None,
            {"products": "not-a-list"},
            _unit([{"sku": "OK", "url": "//www.safcodental.com/product/ok"}]),
            _unit([{"sku": "NO-URL"}]),
            _unit([{"url": "//www.safcodental.com/product/no-sku"}]),  # missing sku
        ]
    }
    out = _parse_alternative_urls(payload, current_sku="X", max_count=10)
    assert out == ("https://www.safcodental.com/product/ok",)


@pytest.mark.unit
def test_parse_empty_results() -> None:
    assert _parse_alternative_urls({"results": []}, current_sku="X", max_count=10) == ()
    assert _parse_alternative_urls({}, current_sku="X", max_count=10) == ()


# --- RecsClient.fetch_alternatives ------------------------------------------


@pytest.mark.asyncio
async def test_fetch_alternatives_happy_path() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = httpx._content.json_dumps  # noqa: SLF001 — placeholder
        body = request.read().decode()
        captured["body"] = body
        return httpx.Response(
            200,
            json=_payload([_unit([_product("AAA", "alpha"), _product("BBB", "bravo")])]),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        recs = RecsClient(client=client)
        out = await recs.fetch_alternatives("DRCDD")

    assert out == (
        "https://www.safcodental.com/product/alpha",
        "https://www.safcodental.com/product/bravo",
    )
    # Verify the request went where we expect with the right auth header.
    assert "commerce.adobe.io/recs/v1/precs/preconfigured" in captured["url"]
    assert captured["headers"]["x-api-key"] == "recs_open"
    assert '"currentSku":"DRCDD"' in captured["body"]


@pytest.mark.asyncio
async def test_fetch_alternatives_returns_empty_on_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request  # signature dictated by httpx.MockTransport
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        recs = RecsClient(client=client)
        out = await recs.fetch_alternatives("ANY")

    assert out == ()


@pytest.mark.asyncio
async def test_fetch_alternatives_returns_empty_for_blank_sku() -> None:
    recs = RecsClient()  # no transport set up — proves no request is made
    assert await recs.fetch_alternatives("") == ()

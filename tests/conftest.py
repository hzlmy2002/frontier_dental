"""Pytest fixtures shared across the test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from frontier_dental import config as config_mod

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test gets its own ``output/`` directory + a fresh Settings singleton."""

    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "products.sqlite"))
    monkeypatch.setenv("VLLM_BASE_URL", "http://test-llm/v1")
    monkeypatch.setenv("VLLM_API_KEY", "test")
    monkeypatch.setenv("VLLM_MODEL", "test-model")
    monkeypatch.setenv("RATE_LIMIT_RPS", "1000")  # don't slow tests down
    config_mod.reset_settings_for_tests()
    yield
    config_mod.reset_settings_for_tests()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def alasta_pro_html() -> str:
    return (FIXTURES_DIR / "pdp_alasta_pro.html").read_text(encoding="utf-8")


@pytest.fixture
def lignospan_html() -> str:
    return (FIXTURES_DIR / "pdp_lignospan.html").read_text(encoding="utf-8")


@pytest.fixture
def irregular_pdp_html() -> str:
    return (FIXTURES_DIR / "pdp_irregular.html").read_text(encoding="utf-8")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:  # noqa: ARG001
    if os.environ.get("RUN_NETWORK_TESTS"):
        return
    skip_network = pytest.mark.skip(reason="set RUN_NETWORK_TESTS=1 to run network tests")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)

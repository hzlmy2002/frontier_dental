"""CLI routing tests. Patches ``_run_pipeline`` so we can assert the args
flowing into it without actually constructing the pipeline / hitting any
network or browser."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from frontier_dental import cli


@pytest.fixture
def captured_args() -> dict[str, Any]:
    return {}


@pytest.fixture
def patched_run_pipeline(captured_args: dict[str, Any]):
    async def _capture(**kwargs: Any) -> None:
        captured_args.update(kwargs)

    with patch.object(cli, "_run_pipeline", side_effect=_capture) as p:
        yield p


def test_empty_intent_string_normalizes_to_none(
    patched_run_pipeline, captured_args: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["run", "--intent", ""])
    assert result.exit_code == 0, result.output
    assert captured_args["intent"] is None


def test_whitespace_only_intent_normalizes_to_none(
    patched_run_pipeline, captured_args: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["run", "--intent", "   "])
    assert result.exit_code == 0, result.output
    assert captured_args["intent"] is None


def test_real_intent_passes_through(
    patched_run_pipeline, captured_args: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["run", "--intent", "I want gloves"])
    assert result.exit_code == 0, result.output
    assert captured_args["intent"] == "I want gloves"


def test_no_flags_default_routing(
    patched_run_pipeline, captured_args: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["run"])
    assert result.exit_code == 0, result.output
    assert captured_args["intent"] is None
    assert captured_args["explicit_categories"] is None


def test_explicit_categories_are_resolved(
    patched_run_pipeline, captured_args: dict[str, Any]
) -> None:
    runner = CliRunner()
    result = runner.invoke(cli.main, ["run", "--category", "gloves"])
    assert result.exit_code == 0, result.output
    cats = captured_args["explicit_categories"]
    assert cats is not None
    assert len(cats) == 1
    assert cats[0].slug == "gloves"

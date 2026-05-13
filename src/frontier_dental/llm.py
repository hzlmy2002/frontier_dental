"""OpenAI-compatible LLM client (vLLM endpoint).

Used by:
- ``extractor.py`` Tier 3 fallback (structured-output extraction)

The Navigator uses LangChain's ``ChatOpenAI`` directly; that path does not
go through this module.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import structlog
from openai import AsyncOpenAI

from .config import Settings, get_settings

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def get_async_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(base_url=s.vllm_base_url, api_key=s.vllm_api_key)


def get_settings_for_llm() -> Settings:
    return get_settings()


async def structured_extract(
    *,
    system: str,
    user: str,
    json_schema: dict[str, Any],
    schema_name: str = "extraction",
) -> dict[str, Any]:
    """Ask the LLM for a JSON object matching ``json_schema``.

    Falls back from OpenAI ``response_format=json_schema`` (strict) to plain
    ``json_object`` mode when the model rejects the strict format — this keeps
    the prototype compatible with vLLM-served models that don't yet support the
    full strict schema spec.
    """

    client = get_async_client()
    s = get_settings_for_llm()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        resp = await client.chat.completions.create(
            model=s.vllm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=s.vllm_temperature,
            max_tokens=s.vllm_max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema, "strict": True},
            },
        )
    except Exception as e:  # pragma: no cover — server-side compatibility shim
        log.info("structured_output_strict_unavailable_falling_back", error=str(e))
        resp = await client.chat.completions.create(
            model=s.vllm_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=s.vllm_temperature,
            max_tokens=s.vllm_max_tokens,
            response_format={"type": "json_object"},
        )

    content = resp.choices[0].message.content or "{}"
    return _safe_load_json(content)


def _safe_load_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # Strip ``` fences if a model decides to wrap its output.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        log.warning("llm_returned_non_json", preview=text[:300])
        return {}

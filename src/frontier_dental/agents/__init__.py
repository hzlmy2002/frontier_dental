"""Agent layer.

- ``navigator``: LangGraph ReAct agent driving Playwright (LangChain Community
  browser toolkit) — discovers the catalog and enumerates categories.
- ``extractor``: tiered (httpx + JSON-LD → Playwright DOM → LLM) PDP extractor.
- ``validator``: pydantic schema validation + SKU-keyed dedup.
"""

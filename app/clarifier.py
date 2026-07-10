"""Clarification classifier: decides whether to ask back or query GraphRAG.

``classify()`` asks an OpenAI model whether an incoming user message is
specific enough to answer or too vague/ambiguous, returning a
``ClarifierResult`` (whether clarification is needed, plus an optional
follow-up question and answer options). The system prompt is assembled
from ``prompts/clarifier.txt`` with the domain entity types injected from
``settings.yaml`` (``extract_graph.entity_types``), so clarifications stay
scoped to the indexed domain.
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from openai import AsyncOpenAI

_SETTINGS_PATH = Path("settings.yaml")
_PROMPT_PATH = Path("prompts/clarifier.txt")


@dataclass(frozen=True, slots=True)
class ClarifierResult:
    needs_clarification: bool
    question: str | None
    options: list[str] | None


@functools.lru_cache(maxsize=1)
def _entity_types() -> str:
    data = yaml.safe_load(_SETTINGS_PATH.read_text(encoding="utf-8"))
    types = data.get("extract_graph", {}).get("entity_types", [])
    if not types:
        raise ValueError("settings.yaml is missing extract_graph.entity_types")
    return ", ".join(types)


@functools.lru_cache(maxsize=1)
def _system_prompt() -> str:
    template = _PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{entity_types}", _entity_types())


async def classify(message: str) -> ClarifierResult:
    """Ask the LLM whether the message is specific enough to query GraphRAG."""
    api_key = os.environ.get("GRAPHRAG_API_KEY")
    if not api_key:
        raise ValueError("GRAPHRAG_API_KEY is not set")

    model = os.environ.get("CLARIFIER_MODEL", "gpt-4.1-mini")
    client = AsyncOpenAI(api_key=api_key)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": message},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)

    return ClarifierResult(
        needs_clarification=bool(data.get("needs_clarification", False)),
        question=data.get("question"),
        options=data.get("options"),
    )

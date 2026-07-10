"""In-process GraphRAG query execution via ``graphrag.api``.

Loads the GraphRAG config and the index parquet tables needed for the
configured ``SearchMethod`` (local/global/drift/basic), dispatches the
matching search, enforces a per-query timeout, and strips ``[Data: ...]``
citation markers from answers. ``run_query_with_context``
returns the retrieved context object so measurement code can
score grounding without re-querying.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

import graphrag.api as graphrag_api
from graphrag.callbacks.noop_query_callbacks import NoopQueryCallbacks
from graphrag.cli import query as graphrag_query
from graphrag.config.enums import SearchMethod
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig

from app.config import QuerySettings

_resolve_output_files = getattr(
    graphrag_query,
    "resolve_output_files",
    graphrag_query._resolve_output_files,
)

# Parquet tables required (and optionally loaded) per search method.
_SEARCH_OUTPUTS: dict[SearchMethod, tuple[list[str], list[str]]] = {
    SearchMethod.GLOBAL: (
        ["entities", "communities", "community_reports"],
        [],
    ),
    SearchMethod.LOCAL: (
        [
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
        ["covariates"],
    ),
    SearchMethod.DRIFT: (
        [
            "communities",
            "community_reports",
            "text_units",
            "relationships",
            "entities",
        ],
        [],
    ),
    SearchMethod.BASIC: (["text_units"], []),
}

_DATA_REFERENCE_RE = re.compile(r"\s*\[Data:[^\]]+\]")
_INCOMPLETE_DATA_REF_SUFFIX = re.compile(r"\[Data:[^\]]*$")


def _strip_data_references(answer: str) -> str:
    """Remove GraphRAG record citations from user-facing answers."""
    return _DATA_REFERENCE_RE.sub("", answer)


class CitationStreamStripper:
    """Strip ``[Data: ...]`` markers from a streaming token sequence."""

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending = ""

    def feed(self, chunk: str) -> str:
        self._pending += chunk
        cleaned = _DATA_REFERENCE_RE.sub("", self._pending)
        # Hold back text that might
        # be the start of a citation split across chunks.
        match = _INCOMPLETE_DATA_REF_SUFFIX.search(cleaned)
        if match:
            self._pending = cleaned[match.start() :]
            return cleaned[: match.start()].rstrip()
        self._pending = ""
        return cleaned

    def flush(self) -> str:
        remainder = _DATA_REFERENCE_RE.sub("", self._pending)
        self._pending = ""
        return remainder


def _load_config(settings: QuerySettings) -> GraphRagConfig:
    # Load GraphRAG config,
    # pointing output storage at ``settings.data_dir``.
    return load_config(
        root_dir=settings.root,
        cli_overrides={
            "output_storage": {
                "base_dir": str(settings.data_dir.resolve()),
            },
        },
    )


async def _load_dataframes(
    config: GraphRagConfig,
    method: SearchMethod,
) -> dict:
    """Load the parquet index tables needed for the given search method."""
    required, optional = _SEARCH_OUTPUTS[method]
    return await asyncio.to_thread(
        _resolve_output_files,
        config=config,
        output_list=required,
        optional_list=optional,
    )


async def _dispatch(
    config: GraphRagConfig,
    settings: QuerySettings,
    query: str,
    dfs: dict,
) -> tuple[str, Any]:
    shared = {
        "config": config,
        "query": query,
        "response_type": settings.response_type,
        "verbose": settings.verbose,
    }

    match settings.method:
        case SearchMethod.GLOBAL:
            text, context = await graphrag_api.global_search(
                **shared,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                community_level=settings.community_level,
                dynamic_community_selection=(
                    settings.dynamic_community_selection
                ),
            )
        case SearchMethod.LOCAL:
            text, context = await graphrag_api.local_search(
                **shared,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                text_units=dfs["text_units"],
                relationships=dfs["relationships"],
                covariates=dfs["covariates"],
                community_level=settings.community_level,
            )
        case SearchMethod.DRIFT:
            text, context = await graphrag_api.drift_search(
                **shared,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                text_units=dfs["text_units"],
                relationships=dfs["relationships"],
                community_level=settings.community_level,
            )
        case SearchMethod.BASIC:
            text, context = await graphrag_api.basic_search(
                **shared,
                text_units=dfs["text_units"],
            )

    answer = text if isinstance(text, str) else str(text)
    return answer, context


async def _run(settings: QuerySettings, query: str) -> tuple[str, Any]:
    """Load config and tables, run search, apply ``timeout_s``."""
    config = _load_config(settings)
    dfs = await _load_dataframes(config, settings.method)
    text, context = await asyncio.wait_for(
        _dispatch(config, settings, query, dfs),
        timeout=settings.timeout_s,
    )
    return _strip_data_references(text), context


async def run_query_with_context(
    settings: QuerySettings, query: str
) -> tuple[str, Any]:
    """Run search; return the answer and the GraphRAG retrieval context."""
    return await _run(settings, query)


def _stream(
    config: GraphRagConfig,
    settings: QuerySettings,
    query: str,
    dfs: dict,
    *,
    callbacks: list | None = None,
) -> AsyncIterator[Any]:
    """Return the GraphRAG streaming generator for the configured method.

    Yields answer text chunks only; retrieval context is delivered via
    ``QueryCallbacks.on_context`` (see :func:`run_query_stream`).
    """
    shared = {
        "config": config,
        "query": query,
        "response_type": settings.response_type,
        "verbose": settings.verbose,
        "callbacks": callbacks or [],
    }

    match settings.method:
        case SearchMethod.GLOBAL:
            return graphrag_api.global_search_streaming(
                **shared,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                community_level=settings.community_level,
                dynamic_community_selection=(
                    settings.dynamic_community_selection
                ),
            )
        case SearchMethod.LOCAL:
            return graphrag_api.local_search_streaming(
                **shared,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                text_units=dfs["text_units"],
                relationships=dfs["relationships"],
                covariates=dfs["covariates"],
                community_level=settings.community_level,
            )
        case SearchMethod.DRIFT:
            return graphrag_api.drift_search_streaming(
                **shared,
                entities=dfs["entities"],
                communities=dfs["communities"],
                community_reports=dfs["community_reports"],
                text_units=dfs["text_units"],
                relationships=dfs["relationships"],
                community_level=settings.community_level,
            )
        case SearchMethod.BASIC:
            return graphrag_api.basic_search_streaming(
                **shared,
                text_units=dfs["text_units"],
            )


class _ContextCapture(NoopQueryCallbacks):
    """Collect retrieval context from GraphRAG's streaming search callbacks."""

    context: Any = None

    def on_context(self, context: Any) -> None:
        self.context = context


async def run_query_stream(
    settings: QuerySettings, query: str
) -> AsyncIterator[tuple[str, Any]]:
    """Yield ``(kind, payload)`` events for one streaming GraphRAG query.

    Events:
      - ``("context", context_dict)`` — the retrieval context, once.
      - ``("chunk", text)``           — answer fragment, citations stripped.
      - ``("done", None)``            — terminal sentinel.
    """
    config = _load_config(settings)
    dfs = await _load_dataframes(config, settings.method)
    capture = _ContextCapture()
    generator = _stream(config, settings, query, dfs, callbacks=[capture])

    context_emitted = False
    stripper = CitationStreamStripper()
    async for item in generator:
        if not context_emitted and capture.context is not None:
            yield ("context", capture.context)
            context_emitted = True
        text = item if isinstance(item, str) else str(item)
        cleaned = stripper.feed(text)
        if cleaned:
            yield ("chunk", cleaned)

    remainder = stripper.flush()
    if remainder:
        yield ("chunk", remainder)

    if not context_emitted and capture.context is not None:
        yield ("context", capture.context)

    yield ("done", None)

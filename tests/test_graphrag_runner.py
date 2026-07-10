"""Tests for GraphRAG query response helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from app.graphrag_runner import (
    CitationStreamStripper,
    _strip_data_references,
    run_query_stream,
)


def test_strip_data_references_removes_graphrag_citations() -> None:
    answer = (
        "A data source node manages data origins "
        "[Data: Entities (72, 266, 1572); Relationships (1466, +more)]. "
        "See the [Kubernetes documentation](https://kubernetes.io/docs/concepts/) "
        "for details [Data: Sources (1)]."
    )

    assert _strip_data_references(answer) == (
        "A data source node manages data origins. "
        "See the [Kubernetes documentation](https://kubernetes.io/docs/concepts/) "
        "for details."
    )


def test_citation_stream_stripper_handles_split_markers() -> None:
    stripper = CitationStreamStripper()
    parts = [
        "Answer text",
        " [Data: Entities (1958); Relationships (3142, 3143, 3144, 3141, "
        "3140, 8766)]",
    ]

    emitted = "".join(stripper.feed(part) for part in parts)
    emitted += stripper.flush()

    assert emitted == "Answer text"


def test_citation_stream_stripper_handles_marker_split_across_chunks() -> None:
    stripper = CitationStreamStripper()
    parts = ["Answer text [Data: Enti", "ties (1)]"]

    emitted = "".join(stripper.feed(part) for part in parts)
    emitted += stripper.flush()

    assert emitted == "Answer text"


def test_run_query_stream_emits_first_text_chunk(monkeypatch) -> None:
    """Streaming must not treat the first generator item as context."""

    async def fake_chunks():
        yield "Sym"
        yield "pathy for Data"

    def fake_stream(_config, _settings, _query, _dfs, *, callbacks=None):
        if callbacks:
            callbacks[0].on_context({"sources": []})
        return fake_chunks()

    async def fake_load_dataframes(_config, _method):
        return {}

    monkeypatch.setattr("app.graphrag_runner._load_config", MagicMock())
    monkeypatch.setattr(
        "app.graphrag_runner._load_dataframes",
        fake_load_dataframes,
    )
    monkeypatch.setattr("app.graphrag_runner._stream", fake_stream)

    async def collect():
        return [event async for event in run_query_stream(MagicMock(), "q")]

    events = asyncio.run(collect())

    assert events[0] == ("context", {"sources": []})
    assert events[1] == ("chunk", "Sym")
    assert events[2] == ("chunk", "pathy for Data")
    assert events[-1] == ("done", None)

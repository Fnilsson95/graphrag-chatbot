"""GraphRAG query options environment variables.

Single source of truth for how one GraphRAG query is parameterised —
index root/data dir, search method, community level, response type,
verbosity, and timeout. ``QuerySettings.from_env()`` builds an instance
from ``GRAPHRAG_*`` environment variables and validates the search method,
raising a clear error for unknown values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from graphrag.config.enums import SearchMethod

_DEFAULT_RESPONSE_TYPE = (
    "Answer concretely from the retrieved Kubernetes documentation. "
    "Lead with the direct answer in 2-4 sentences and include the specific "
    "names, resource types, or API fields when they appear in the context. "
    "If the context only partially supports the question, give what is "
    "supported and clearly mark which part is uncertain rather than refusing. "
    "Do not invent facts not in the context. Do not include GraphRAG record "
    'citations or bracketed data references such as "[Data: ...]". When '
    "relevant documentation URLs appear in the retrieved context, end with a "
    "clickable Markdown link to the source page. No preamble."
)


@dataclass(frozen=True, slots=True)
class QuerySettings:
    """Paths, search mode, and tuning for one query."""

    root: Path = field(default_factory=lambda: Path("."))
    data_dir: Path = field(default_factory=lambda: Path("output"))
    method: SearchMethod = SearchMethod.LOCAL
    community_level: int = 2
    dynamic_community_selection: bool = False
    response_type: str = _DEFAULT_RESPONSE_TYPE
    verbose: bool = False
    timeout_s: float = 180.0

    @classmethod
    def from_env(cls) -> QuerySettings:
        root = Path(os.environ.get("GRAPHRAG_ROOT", "."))

        method_raw = (
            os.environ.get("GRAPHRAG_QUERY_METHOD", "local").strip().lower()
        )
        try:
            method = SearchMethod(method_raw)
        except ValueError:
            valid = ", ".join(m.value for m in SearchMethod)
            raise ValueError(
                f"Invalid GRAPHRAG_QUERY_METHOD: {method_raw!r}. "
                f"Expected one of: {valid}"
            ) from None

        return cls(
            root=root,
            data_dir=Path(
                os.environ.get("GRAPHRAG_DATA_DIR", str(root / "output")),
            ),
            method=method,
            community_level=int(
                os.environ.get("GRAPHRAG_COMMUNITY_LEVEL", "2"),
            ),
            dynamic_community_selection=os.environ.get(
                "GRAPHRAG_DYNAMIC_COMMUNITY_SELECTION", ""
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            response_type=os.environ.get(
                "GRAPHRAG_RESPONSE_TYPE", _DEFAULT_RESPONSE_TYPE
            ),
            verbose=os.environ.get("GRAPHRAG_QUERY_VERBOSE", "")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            timeout_s=float(
                os.environ.get("GRAPHRAG_QUERY_TIMEOUT_S", "180"),
            ),
        )

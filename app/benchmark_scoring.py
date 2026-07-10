"""Token-free proxy scoring for benchmark answers.

NOT a substitute for an LLM judge. Two signals, different trust levels:
- groundedness: fraction of the answer's content words that also appear
  in the retrieved GraphRAG context. Free (item 2 surfaces the context)
  and a meaningful anti-hallucination / "is it grounded" proxy.
- assertion_overlap: lexical overlap of the answer with each assertion's
  content words. Weak for abstract or negative assertions; a smoke
  signal only, never treat as ground truth.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
# Common English stop words excluded from overlap scoring.
_STOP = {
    "the",
    "and",
    "for",
    "are",
    "was",
    "with",
    "that",
    "this",
    "from",
    "should",
    "response",
    "documentation",
    "data",
    "not",
    "any",
    "have",
    "use",
    "used",
    "its",
    "their",
    "into",
    "than",
    "then",
    "also",
}


def _content_words(text: str) -> set[str]:
    return {
        w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOP
    }


def _context_text(context: Any) -> str:
    """Flatten GraphRAG's per-method context object to plain text."""
    try:
        import pandas as pd

        if isinstance(context, dict):
            parts: list[str] = []
            for value in context.values():
                if isinstance(value, pd.DataFrame):
                    parts.append(value.to_csv(index=False))
                else:
                    parts.append(str(value))
            return " ".join(parts)
    except Exception:  # noqa: BLE001 - shape varies; fall back to str()
        pass
    return str(context)


def score_groundedness(answer: str, context: Any) -> float:
    """Fraction of answer content words found in the retrieved context."""
    words = _content_words(answer)
    if not words:
        return 0.0
    ctx = _content_words(_context_text(context))
    return round(len(words & ctx) / len(words), 4)


def _assertion_overlap(answer_words: set[str], assertion: str) -> float:
    target = _content_words(assertion)
    if not target:
        return 0.0
    return round(len(target & answer_words) / len(target), 4)


def score_record(
    answer: str,
    assertions: Sequence[str],
    context: Any,
) -> dict:
    """Return groundedness + per/mean assertion-overlap for one answer."""
    answer_words = _content_words(answer)
    overlaps = [_assertion_overlap(answer_words, a) for a in assertions]
    return {
        "groundedness": score_groundedness(answer, context),
        "assertion_overlap": overlaps,
        "assertion_overlap_mean": (
            round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0
        ),
    }

"""Generate BenchmarkQED assertion question sets from GraphRAG output.

CLI that reads entity/community titles from the GraphRAG output parquet
files and emits question objects with deterministic (uuid5) ids and
generic assertions, producing the auto-generated local and global
activity assertion files that ``benchmark_qed.py`` then answers. The
curated ``input/evaluation_set.json`` is hand-written and not produced
here.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _AssertionQuestion:
    question_id: str
    question_text: str
    assertions: list[str]


def _stable_id(text: str) -> str:
    # Deterministic question id so re-runs produce the same BenchmarkQED ids.
    namespace = f"graphrag-chatbot-benchmark-qed:{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, namespace))


def _read_parquet_titles(
    path: Path,
    *,
    title_col: str = "title",
    sort_col: str | None = None,
    limit: int = 10,
) -> list[str]:
    # Loaded lazily to keep import surface small for non-users.
    import pandas as pd  # type: ignore[import-not-found]

    df = pd.read_parquet(path)
    if title_col not in df.columns:
        raise ValueError(f"Expected column {title_col!r} in {path}")

    titles = df[title_col].dropna()
    if sort_col and sort_col in df.columns:
        df2 = df[[title_col, sort_col]].dropna()
        df2 = df2.sort_values(sort_col, ascending=False)
        titles = df2[title_col]

    seen: set[str] = set()
    result: list[str] = []
    for raw in titles.astype(str).tolist():
        title = " ".join(raw.split())
        if not title or title in seen:
            continue
        seen.add(title)
        result.append(title)
        if len(result) >= limit:
            break
    return result


def _generate_global_questions(
    entities: list[str], communities: list[str]
) -> list[_AssertionQuestion]:
    entity_list = ", ".join(entities[:8]) if entities else "the dataset"
    community_list = (
        ", ".join(communities[:6]) if communities else "the dataset"
    )

    prompts: list[tuple[str, list[str]]] = [
        (
            (
                "Across the dataset, what are the main themes and how do they "
                f"connect to key entities like {entity_list}?"
            ),
            [
                (
                    "The response should summarize 2-5 high-level themes "
                    "grounded in the dataset."
                ),
                (
                    "The response should reference multiple specific entities "
                    "and explain their role in those themes."
                ),
                (
                    "The response should connect themes to concrete evidence "
                    "from the dataset (examples, facts, or summaries)."
                ),
            ],
        ),
        (
            (
                "Across the dataset, summarize what the major communities "
                f"(e.g., {community_list}) are about and how they relate to "
                "each other."
            ),
            [
                (
                    "The response should describe several communities with "
                    "brief, accurate summaries."
                ),
                (
                    "The response should compare or relate communities "
                    "(overlaps, contrasts, or hierarchy)."
                ),
                (
                    "The response should stay consistent with the dataset and "
                    "avoid invented entities or communities."
                ),
            ],
        ),
    ]

    questions: list[_AssertionQuestion] = []
    for question_text, assertions in prompts:
        questions.append(
            _AssertionQuestion(
                question_id=_stable_id(question_text),
                question_text=question_text,
                assertions=assertions,
            )
        )
    return questions


def _generate_local_questions(
    entities: list[str], communities: list[str], *, limit: int
) -> list[_AssertionQuestion]:
    targets: list[str] = []
    targets.extend(entities[: max(0, limit // 2)])
    targets.extend(communities[: max(0, limit - len(targets))])
    if not targets:
        targets = ["the most important entity in the dataset"]

    questions: list[_AssertionQuestion] = []
    for target in targets[:limit]:
        question_text = (
            f"What does the dataset say about {target}, and what evidence "
            "supports that?"
        )
        questions.append(
            _AssertionQuestion(
                question_id=_stable_id(question_text),
                question_text=question_text,
                assertions=[
                    (
                        "The response should describe the target accurately "
                        "and consistently with the dataset."
                    ),
                    (
                        "The response should cite supporting evidence from "
                        "the dataset (examples, details, or quotes)."
                    ),
                    (
                        "The response should avoid hallucinated details not "
                        "supported by the dataset."
                    ),
                ],
            )
        )
    return questions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BenchmarkQED assertions JSON from GraphRAG output."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="GraphRAG output dir (default: $GRAPHRAG_DATA_DIR or ./output).",
    )
    parser.add_argument(
        "--global-out",
        type=Path,
        default=Path("input/activity_global_assertions.json"),
        help="Output path for global assertion questions JSON.",
    )
    parser.add_argument(
        "--local-out",
        type=Path,
        default=Path("input/activity_local_assertions.json"),
        help="Output path for local assertion questions JSON.",
    )
    parser.add_argument(
        "--top-entities",
        type=int,
        default=12,
        help="Number of entities to consider (default: 12).",
    )
    parser.add_argument(
        "--top-communities",
        type=int,
        default=12,
        help="Number of communities to consider (default: 12).",
    )
    parser.add_argument(
        "--local-questions",
        type=int,
        default=16,
        help="Number of local questions to generate (default: 16).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = (
        args.data_dir
        if args.data_dir is not None
        else Path(os.environ.get("GRAPHRAG_DATA_DIR", "output"))
    )

    entities_path = data_dir / "entities.parquet"
    community_reports_path = data_dir / "community_reports.parquet"
    if not entities_path.exists():
        raise FileNotFoundError(f"Missing {entities_path}")
    if not community_reports_path.exists():
        raise FileNotFoundError(f"Missing {community_reports_path}")

    entities = _read_parquet_titles(
        entities_path,
        title_col="title",
        sort_col="frequency",
        limit=args.top_entities,
    )
    communities = _read_parquet_titles(
        community_reports_path,
        title_col="title",
        sort_col="rank",
        limit=args.top_communities,
    )

    global_questions = _generate_global_questions(entities, communities)
    local_questions = _generate_local_questions(
        entities, communities, limit=args.local_questions
    )

    args.global_out.parent.mkdir(parents=True, exist_ok=True)
    args.local_out.parent.mkdir(parents=True, exist_ok=True)

    args.global_out.write_text(
        json.dumps(
            [asdict(q) for q in global_questions],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.local_out.write_text(
        json.dumps(
            [asdict(q) for q in local_questions],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

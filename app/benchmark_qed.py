"""Run a BenchmarkQED question set through GraphRAG and measure it.

CLI that answers every question in a BenchmarkQED-style input file with
bounded concurrency, writing two outputs: the BenchmarkQED-schema answers
file (consumed by ``benchmark-qed autoe``) and a token-free
``*.summary.json`` sidecar with per-question timing and an aggregate
report — latency min/mean/p50/p95/max, error rate, and the
groundedness / assertion-overlap proxies. A failing question is recorded
with its error rather than aborting the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.benchmark_scoring import score_record
from app.config import QuerySettings
from app.graphrag_runner import run_query_with_context


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    """Question payload required by BenchmarkQED."""

    question_id: str
    question_text: str
    assertions: tuple[str, ...] = ()


@dataclass(slots=True)
class AnswerRecord:
    """One answered question plus timing/reliability metadata."""

    question_id: str
    question_text: str
    answer: str
    elapsed_s: float
    error: str | None = None
    groundedness: float = 0.0
    assertion_overlap_mean: float = 0.0


def _read_questions(path: Path) -> list[BenchmarkQuestion]:
    """Load questions from assertions or question JSON files."""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of question objects.")

    questions: list[BenchmarkQuestion] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        qid = item.get("question_id")
        qtext = item.get("question_text")
        if isinstance(qid, str) and isinstance(qtext, str):
            raw = item.get("assertions")
            assertions = (
                tuple(a for a in raw if isinstance(a, str))
                if isinstance(raw, list)
                else ()
            )
            questions.append(
                BenchmarkQuestion(
                    question_id=qid,
                    question_text=qtext,
                    assertions=assertions,
                )
            )

    if not questions:
        raise ValueError("No valid questions found in input file.")
    return questions


async def _answer_questions(
    questions: Sequence[BenchmarkQuestion],
    settings: QuerySettings,
    max_concurrency: int,
) -> list[AnswerRecord]:
    """Run GraphRAG over all questions with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[AnswerRecord | None] = [None] * len(questions)

    async def _run_one(index: int, question: BenchmarkQuestion) -> None:
        async with semaphore:
            started = time.perf_counter()
            error: str | None = None
            answer = ""
            scores = {"groundedness": 0.0, "assertion_overlap_mean": 0.0}
            try:
                answer, context = await run_query_with_context(
                    settings, question.question_text
                )
                scores = score_record(answer, question.assertions, context)
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                error = repr(exc)
            elapsed = time.perf_counter() - started
            results[index] = AnswerRecord(
                question_id=question.question_id,
                question_text=question.question_text,
                answer=answer.strip(),
                elapsed_s=round(elapsed, 3),
                error=error,
                groundedness=scores["groundedness"],
                assertion_overlap_mean=scores["assertion_overlap_mean"],
            )

    await asyncio.gather(*(_run_one(i, q) for i, q in enumerate(questions)))
    return [row for row in results if row is not None]


def _pct(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0, 1]); empty -> 0.0."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    fr = pos - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fr


def _summarize(
    records: Sequence[AnswerRecord],
    wall_clock_s: float,
) -> dict:
    """Build an aggregate latency + reliability summary."""
    n = len(records)
    errors = [r for r in records if r.error is not None]
    ok = [r for r in records if r.error is None]
    lat = sorted(r.elapsed_s for r in ok)
    mean = sum(lat) / len(lat) if lat else 0.0
    return {
        "questions": n,
        "ok": len(ok),
        "errors": len(errors),
        "error_rate": round(len(errors) / n, 4) if n else 0.0,
        "wall_clock_s": round(wall_clock_s, 1),
        "latency_s": {
            "min": round(lat[0], 3) if lat else 0.0,
            "mean": round(mean, 3),
            "p50": round(_pct(lat, 0.50), 3),
            "p95": round(_pct(lat, 0.95), 3),
            "max": round(lat[-1], 3) if lat else 0.0,
        },
        "accuracy_proxy": {
            "groundedness_mean": round(
                sum(r.groundedness for r in ok) / len(ok), 4
            )
            if ok
            else 0.0,
            "assertion_overlap_mean": round(
                sum(r.assertion_overlap_mean for r in ok) / len(ok), 4
            )
            if ok
            else 0.0,
            "_note": "proxy signals, not LLM-judged accuracy",
        },
        "failed_question_ids": [r.question_id for r in errors],
        "failures": [
            {
                "question_id": r.question_id,
                "question_text": r.question_text,
                "error": r.error,
            }
            for r in errors
        ],
        "per_question": [
            {
                "question_id": r.question_id,
                "elapsed_s": r.elapsed_s,
                "ok": r.error is None,
            }
            for r in records
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BenchmarkQED answer JSON from local GraphRAG.",
    )
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="Path to a BenchmarkQED-style question/assertions JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON path to write answers.",
    )
    parser.add_argument(
        "--max-concurrency",
        default=4,
        type=int,
        help="Concurrent GraphRAG queries (default: 4).",
    )
    return parser.parse_args()


async def _async_main() -> None:
    args = _parse_args()
    if args.max_concurrency < 1:
        raise ValueError("--max-concurrency must be >= 1")

    questions = _read_questions(args.questions)
    settings = QuerySettings.from_env()

    started = time.perf_counter()
    records = await _answer_questions(
        questions=questions,
        settings=settings,
        max_concurrency=args.max_concurrency,
    )
    wall_clock_s = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # BenchmarkQED-compatible answers file: schema unchanged.
    answers = [
        {
            "question_id": r.question_id,
            "question_text": r.question_text,
            "answer": r.answer,
        }
        for r in records
    ]
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(answers, file, ensure_ascii=False, indent=2)

    # Latency + reliability sidecar, next to the answers file.
    summary = _summarize(records, wall_clock_s)
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    lat = summary["latency_s"]
    print(
        f"{summary['ok']}/{summary['questions']} ok, "
        f"{summary['errors']} errors | "
        f"mean {lat['mean']}s  p50 {lat['p50']}s  "
        f"p95 {lat['p95']}s  max {lat['max']}s | "
        f"wall {summary['wall_clock_s']}s\n"
        f"answers: {args.output}\nsummary: {summary_path}"
    )


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()

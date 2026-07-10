"""Tests for BenchmarkQED answer generation helpers."""

from __future__ import annotations

import json

import pytest

from app.benchmark_qed import _read_questions


def test_read_questions_accepts_assertions_shape(tmp_path) -> None:
    input_path = tmp_path / "assertions.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q-1",
                    "question_text": "What is X?",
                    "assertions": ["a1", "a2"],
                },
                {
                    "question_id": "q-2",
                    "question_text": "What is Y?",
                    "assertions": ["b1"],
                },
            ]
        ),
        encoding="utf-8",
    )

    questions = _read_questions(input_path)
    assert [q.question_id for q in questions] == ["q-1", "q-2"]
    assert [q.question_text for q in questions] == ["What is X?", "What is Y?"]


def test_read_questions_requires_valid_entries(tmp_path) -> None:
    input_path = tmp_path / "empty.json"
    input_path.write_text(json.dumps([{"question_id": 1}]), encoding="utf-8")

    with pytest.raises(ValueError, match="No valid questions"):
        _read_questions(input_path)

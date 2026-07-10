"""Unit tests for clarification option resolution."""

from __future__ import annotations

from app.redis_cache.conversation_history import (
    ConversationMessage,
    build_contextual_query,
    format_clarification,
    match_option,
    parse_options,
    resolve_option_pick,
    strip_trailing_options,
)


def test_format_clarification_includes_bullets() -> None:
    text = format_clarification(
        "What would you like explained?",
        ["Overview of the platform", "Node-based workflow system"],
    )
    assert "- Overview of the platform" in text
    assert "- Node-based workflow system" in text


def test_parse_options_reads_trailing_bullets() -> None:
    content = (
        "What aspect should I explain?\n\n"
        "- Overview of the platform\n"
        "- Node-based workflow system\n"
        "- Supported data types"
    )
    assert parse_options(content) == [
        "Overview of the platform",
        "Node-based workflow system",
        "Supported data types",
    ]


def test_parse_options_ignores_single_bullet() -> None:
    assert parse_options("Here is a list:\n- only one") == []


def test_match_option_prefix() -> None:
    options = [
        "Overview of the platform",
        "Node-based workflow system",
    ]
    assert match_option("Overview", options) == "Overview of the platform"


def test_match_option_exact() -> None:
    options = ["Overview of the platform", "Node-based workflow system"]
    assert match_option("Node-based workflow system", options) == (
        "Node-based workflow system"
    )


def test_resolve_option_pick_expands_short_reply() -> None:
    history = [
        ConversationMessage(role="user", content="Explain pods"),
        ConversationMessage(
            role="assistant",
            content=format_clarification(
                "What aspect of Kubernetes should I explain?",
                [
                    "Overview of the platform",
                    "Node-based workflow system",
                    "Supported data types",
                    "Machine learning features",
                ],
            ),
        ),
    ]
    resolved = resolve_option_pick("Overview", history)
    assert resolved == (
        "Explain pods — specifically: Overview of the platform"
    )


def test_resolve_option_pick_returns_none_without_options() -> None:
    history = [
        ConversationMessage(role="user", content="hello"),
        ConversationMessage(role="assistant", content="Hi there."),
    ]
    assert resolve_option_pick("Overview", history) is None


def test_strip_trailing_options_keeps_question_only() -> None:
    content = format_clarification(
        "What aspect should I explain?",
        ["Overview of the platform", "Node-based workflow system"],
    )
    assert strip_trailing_options(content) == "What aspect should I explain?"


def test_build_contextual_query_omits_option_bullets() -> None:
    history = [
        ConversationMessage(role="user", content="Explain pods"),
        ConversationMessage(
            role="assistant",
            content=format_clarification(
                "What aspect should I explain?",
                ["Overview of the platform", "Node-based workflow system"],
            ),
        ),
    ]
    query = build_contextual_query("Overview", history)
    assert "Conversation history:" in query
    assert "- Overview of the platform" not in query
    assert "What aspect should I explain?" in query

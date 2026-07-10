"""Transport-agnostic prompt pipeline shared by both API endpoints.

``run_pipeline`` is the single source of truth for one prompt turn:
rate-limit, load history, build the contextual query, resolve settings,
check the cache on the opening question only (keyed by that message, not
context), run the clarifier, and (on a clear, uncached question) query
GraphRAG. Follow-up turns always bypass the cache.

The router layer is responsible only for turning these events into the
appropriate transport.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.clarifier import classify
from app.config import QuerySettings
from app.graphrag_runner import (
    _strip_data_references,
    run_query_stream,
    run_query_with_context,
)
from app.redis_cache.client import get_redis
from app.redis_cache.conversation_history import (
    append_turn,
    build_contextual_query,
    format_clarification,
    load_history,
    new_conversation_id,
    resolve_option_pick,
)
from app.redis_cache.prompt_cache import CacheLookup, get_cached, set_cached
from app.redis_cache.rate_limit import RateLimitExceeded, check_rate_limit
from app.sources import Source, extract_sources, format_documentation_links

logger = logging.getLogger(__name__)

_DEFAULT_CLARIFICATION = "Could you clarify your question?"


def _finalize_answer(text: str, sources: list[Source]) -> tuple[str, str]:
    """Strip graph citations and append documentation links when available."""
    answer = _strip_data_references(text.strip())
    footer = format_documentation_links(sources)
    if not footer or footer.strip() in answer:
        return answer, ""
    return f"{answer.rstrip()}{footer}", footer


# EVENTS
# Each event is a step the transport layer renders. They are emitted in a
# fixed order: Meta, then either Clarification, or (Cache, [Sources],
# Chunk+, [Answer]); a terminal Error replaces whatever would follow.


@dataclass(frozen=True, slots=True)
class Meta:
    """Conversation id assigned for this turn."""

    conversation_id: str


@dataclass(frozen=True, slots=True)
class Cache:
    """Which cache tier served (or missed) this turn."""

    tier: str


@dataclass(frozen=True, slots=True)
class Clarification:
    """Clarifier wants a follow-up; the turn ends here."""

    message: str
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Sources:
    """Citations derived from the retrieval context."""

    sources: list[Source]


@dataclass(frozen=True, slots=True)
class Chunk:
    """A fragment of answer text (the whole answer when not streaming)."""

    text: str


@dataclass(frozen=True, slots=True)
class Answer:
    """Terminal success marker carrying the fully assembled answer."""

    text: str
    sources: list[Source] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Error:
    """Terminal failure with an HTTP-style status and a user-facing detail."""

    status: int
    detail: str


# PIPELINE
PipelineEvent = Meta | Cache | Clarification | Sources | Chunk | Answer | Error


async def run_pipeline(
    message: str,
    conversation_id: str | None,
    client_ip: str,
    *,
    stream: bool = False,
) -> AsyncIterator[PipelineEvent]:
    """Drive one prompt turn, yielding events in order."""
    redis = get_redis()
    conversation_id = conversation_id or new_conversation_id()
    yield Meta(conversation_id)

    try:
        await check_rate_limit(redis, client_ip)
    except RateLimitExceeded:
        yield Error(429, "Too many requests")
        return

    history = await load_history(redis, conversation_id)
    resolved_query = resolve_option_pick(message, history)
    if resolved_query:
        query_message = resolved_query
    else:
        query_message = build_contextual_query(message, history)

    try:
        settings = QuerySettings.from_env()
    except ValueError as exc:
        logger.info("invalid query settings: %s", exc)
        yield Error(400, str(exc))
        return

    is_first_turn = not history
    if is_first_turn:
        lookup = await get_cached(redis, settings, message.strip())
    else:
        lookup = CacheLookup(None, "miss")
    yield Cache(lookup.tier)

    if lookup.answer is not None:
        answer, _ = _finalize_answer(lookup.answer, [])
        await append_turn(redis, conversation_id, message, answer)
        yield Chunk(answer)
        yield Answer(answer)
        return

    if not resolved_query:
        try:
            decision = await classify(query_message)
        except Exception as exc:
            logger.exception("clarifier failed")
            yield Error(500, f"Clarifier failed: {exc}")
            return

        if decision.needs_clarification:
            text = decision.question or _DEFAULT_CLARIFICATION
            assistant_text = format_clarification(text, decision.options)
            await append_turn(redis, conversation_id, message, assistant_text)
            yield Clarification(text, decision.options or [])
            return

    async for event in _query(
        redis,
        settings,
        conversation_id,
        message,
        query_message,
        message.strip() if is_first_turn else None,
        stream,
    ):
        yield event


async def _query(
    redis,
    settings: QuerySettings,
    conversation_id: str,
    message: str,
    contextual_message: str,
    cache_key: str | None,
    stream: bool,
) -> AsyncIterator[PipelineEvent]:
    """Run a live GraphRAG query, emit chunks/sources, cache the answer."""
    collected: list[str] = []
    sources: list[Source] = []
    try:
        async for kind, payload in _answer_events(
            settings, contextual_message, stream
        ):
            if kind == "sources":
                sources = payload
                if sources:
                    yield Sources(sources)
            else:  # "chunk"
                collected.append(payload)
                yield Chunk(payload)
    except TimeoutError:
        logger.info("graphrag timeout method=%s", settings.method.value)
        yield Error(504, "GraphRAG query timed out")
        return
    except FileNotFoundError:
        logger.exception(
            "graphrag index not found data_dir=%s", settings.data_dir
        )
        yield Error(503, "GraphRAG index not found")
        return
    except Exception:
        logger.exception(
            "graphrag query failed method=%s", settings.method.value
        )
        yield Error(500, "GraphRAG query failed")
        return

    answer, footer = _finalize_answer("".join(collected), sources)
    if footer:
        yield Chunk(footer)

    if not answer:
        yield Error(500, "Empty GraphRAG response")
        return

    if cache_key is not None:
        await set_cached(redis, settings, cache_key, answer)
    await append_turn(redis, conversation_id, message, answer)
    yield Answer(answer, sources)


async def _answer_events(
    settings: QuerySettings, contextual_message: str, stream: bool
) -> AsyncIterator[tuple[str, object]]:
    """Normalize streaming and buffered GraphRAG runners to a common shape.

    Both paths yield ``("sources", list[Source])`` once, then one or more
    ``("chunk", str)`` events so ``_query`` can treat them identically.
    """
    if stream:
        async for kind, payload in run_query_stream(
            settings, contextual_message
        ):
            if kind == "context":
                yield ("sources", extract_sources(payload))
            elif kind == "chunk":
                yield ("chunk", payload)
    else:
        answer, context = await run_query_with_context(
            settings, contextual_message
        )
        yield ("sources", extract_sources(context))
        yield ("chunk", answer)

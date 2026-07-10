"""Conversation history backed by Redis with an in-process fallback.

Stores user/assistant turns in a Redis list keyed by conversation id,
trimmed to the most recent N messages with a sliding TTL; if Redis is
unavailable it falls back to a TTL-pruned in-process dict.
``build_contextual_query`` prepends recent turns to a follow-up question
so the model can resolve references, while leaving first-turn questions
untouched.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Literal

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant"]
_DEFAULT_TTL_SECONDS = 90 * 24 * 60 * 60  # 90 days
_DEFAULT_MAX_MESSAGES = 12
_LOCAL_HISTORY: dict[str, tuple[float, list[str]]] = {}


@dataclass(frozen=True, slots=True)
class ConversationMessage:
    role: Role
    content: str


def new_conversation_id() -> str:
    """Return a new public conversation identifier."""
    return str(uuid.uuid4())


def _ttl_seconds() -> int:
    return int(
        os.environ.get(
            "CONVERSATION_HISTORY_TTL_SECONDS",
            _DEFAULT_TTL_SECONDS,
        )
    )


def _max_messages() -> int:
    return int(
        os.environ.get(
            "CONVERSATION_HISTORY_MAX_MESSAGES",
            _DEFAULT_MAX_MESSAGES,
        )
    )


def _key(conversation_id: str) -> str:
    return f"conversation:{conversation_id}:messages"


def _encode(message: ConversationMessage) -> str:
    return json.dumps(asdict(message), separators=(",", ":"))


def _decode(raw: str) -> ConversationMessage | None:
    try:
        data = json.loads(raw)
        role = data["role"]
        content = data["content"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None

    if role not in {"user", "assistant"} or not isinstance(content, str):
        return None
    return ConversationMessage(role=role, content=content)


def _prune_local(now: float) -> None:
    for key, (expires_at, _) in list(_LOCAL_HISTORY.items()):
        if expires_at <= now:
            del _LOCAL_HISTORY[key]


def _local_load(conversation_id: str) -> list[ConversationMessage]:
    now = time.time()
    _prune_local(now)
    raw = _LOCAL_HISTORY.get(conversation_id)
    if raw is None:
        return []

    _, items = raw
    messages = [_decode(item) for item in items[-_max_messages() :]]
    return [message for message in messages if message is not None]


def _local_append(conversation_id: str, message: ConversationMessage) -> None:
    now = time.time()
    _prune_local(now)
    _, items = _LOCAL_HISTORY.get(conversation_id, (0, []))
    items = [*items, _encode(message)][-_max_messages() :]
    _LOCAL_HISTORY[conversation_id] = (now + _ttl_seconds(), items)


async def load_history(
    redis: Redis | None,
    conversation_id: str,
) -> list[ConversationMessage]:
    """Load recent messages for a conversation."""
    if redis is None:
        return _local_load(conversation_id)

    try:
        raw = await redis.lrange(_key(conversation_id), -_max_messages(), -1)
    except RedisError as exc:
        logger.warning("Conversation history falling back to memory: %s", exc)
        return _local_load(conversation_id)

    messages = [_decode(item) for item in raw]
    return [message for message in messages if message is not None]


async def append_message(
    redis: Redis | None,
    conversation_id: str,
    message: ConversationMessage,
) -> None:
    """Append one message and keep only the recent conversation window."""
    if redis is None:
        _local_append(conversation_id, message)
        return

    try:
        key = _key(conversation_id)
        await redis.rpush(key, _encode(message))
        await redis.ltrim(key, -_max_messages(), -1)
        await redis.expire(key, _ttl_seconds())
    except RedisError as exc:
        logger.warning("Conversation history falling back to memory: %s", exc)
        _local_append(conversation_id, message)


async def append_turn(
    redis: Redis | None,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Append a user/assistant turn to the conversation."""
    await append_message(
        redis,
        conversation_id,
        ConversationMessage(role="user", content=user_message),
    )
    await append_message(
        redis,
        conversation_id,
        ConversationMessage(role="assistant", content=assistant_message),
    )


def format_clarification(message: str, options: list[str] | None) -> str:
    """Format a clarification question with bullet options for history."""
    if not options:
        return message
    bullets = "\n".join(f"- {option}" for option in options)
    return f"{message.rstrip()}\n\n{bullets}"


def parse_options(content: str) -> list[str]:
    """Extract trailing bullet options from an assistant clarification."""
    lines = content.rstrip().splitlines()
    options: list[str] = []
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("- "):
            options.insert(0, stripped[2:].strip())
        elif not stripped and options:
            continue
        elif options:
            break
    return options if len(options) >= 2 else []


def match_option(pick: str, options: list[str]) -> str | None:
    """Match a short user reply to one of the offered options."""
    normalized_pick = pick.strip().casefold()
    if not normalized_pick:
        return None

    for option in options:
        if option.casefold() == normalized_pick:
            return option

    prefix_matches = [
        option
        for option in options
        if option.casefold().startswith(normalized_pick)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        return min(prefix_matches, key=len)

    word_matches = [
        option
        for option in options
        if normalized_pick in {word.casefold() for word in option.split()}
    ]
    if len(word_matches) == 1:
        return word_matches[0]

    return None


def strip_trailing_options(content: str) -> str:
    """Drop bullet options when replaying clarifications in history."""
    if not parse_options(content):
        return content.strip()

    lines = content.rstrip().splitlines()
    body: list[str] = []
    for line in lines:
        if line.strip().startswith("- "):
            break
        body.append(line)
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body).strip()


def resolve_option_pick(
    message: str,
    history: list[ConversationMessage],
) -> str | None:
    """Expand a short option pick into a full query using prior context."""
    if not history or history[-1].role != "assistant":
        return None

    options = parse_options(history[-1].content)
    matched = match_option(message, options)
    if matched is None:
        return None

    prior_user = ""
    for item in reversed(history[:-1]):
        if item.role == "user" and item.content.strip():
            prior_user = item.content.strip()
            break

    if prior_user:
        return f"{prior_user} — specifically: {matched}"
    return matched


def build_contextual_query(
    message: str,
    history: list[ConversationMessage],
) -> str:
    """Add prior turns to follow-up questions while preserving first turns."""
    clean_history = [
        item for item in history[-_max_messages() :] if item.content.strip()
    ]
    if not clean_history:
        return message

    lines = [
        "Answer the current question. Use the conversation history only to "
        "resolve follow-up references.",
        "",
        "Conversation history:",
    ]
    for item in clean_history:
        speaker = "User" if item.role == "user" else "Assistant"
        text = item.content.strip()
        if item.role == "assistant":
            text = strip_trailing_options(text)
        lines.append(f"{speaker}: {text}")

    lines.extend(["", "Current user question:", message])
    return "\n".join(lines)

"""Redis-backed prompt cache with two lookup tiers.

Exact: SHA-256 of (search method, community level, response type, opening
user question), namespaced by an index-version counter so re-indexing
invalidates entries. Only the first turn of a conversation is cached;
follow-ups always query GraphRAG live.
Semantic (optional, gated by ``SEMANTIC_CACHE_ENABLED``): KNN paraphrase
match via ``semantic_cache.py``. Entries are TTL-bounded; lookups and
writes swallow Redis errors and fall back to a live query.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Literal

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import QuerySettings
from app.redis_cache.semantic_cache import (
    is_enabled as semantic_enabled,
)
from app.redis_cache.semantic_cache import (
    semantic_get,
    semantic_set,
)

CacheTier = Literal["exact", "semantic", "miss"]


@dataclass(slots=True)
class CacheLookup:
    answer: str | None
    tier: CacheTier


logger = logging.getLogger(__name__)

INDEX_VERSION_KEY = "prompt_index_version"
_DEFAULT_TTL_SECONDS = 90 * 24 * 60 * 60  # 90 days


def _ttl_seconds() -> int:
    # Read each call so tests can override via env var.
    return int(
        os.environ.get("PROMPT_CACHE_TTL_SECONDS", _DEFAULT_TTL_SECONDS),
    )


async def _index_version(redis: Redis) -> int:
    raw = await redis.get(INDEX_VERSION_KEY)
    if raw is None:
        # First run: initialise the version counter
        # used to namespace cache keys.
        await redis.set(INDEX_VERSION_KEY, 1, nx=True)
        raw = await redis.get(INDEX_VERSION_KEY)
    return int(raw)


def _build_key(version: int, settings: QuerySettings, message: str) -> str:
    # Include search settings in the hash
    # so different methods/levels don't collide.
    payload = "|".join(
        [
            settings.method.value,
            str(settings.community_level),
            settings.response_type,
            message,
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"prompt:v{version}:{digest}"


async def get_cached(
    redis: Redis | None, settings: QuerySettings, message: str
) -> CacheLookup:
    """Return a ``CacheLookup``. Tries exact first, then semantic."""

    if redis is None:
        return CacheLookup(None, "miss")
    try:
        version = await _index_version(redis)
        exact = await redis.get(_build_key(version, settings, message))
    except RedisError as exc:
        logger.warning("Cache lookup failed, falling back to live: %s", exc)
        return CacheLookup(None, "miss")

    if exact is not None:
        logger.info("cache_tier=exact")
        return CacheLookup(exact, "exact")

    if semantic_enabled():
        hit = await semantic_get(redis, settings, message, version)
        if hit is not None:
            logger.info(
                "cache_tier=semantic similarity=%.3f matched=%r",
                hit.similarity,
                hit.matched_question[:80],
            )
            return CacheLookup(hit.answer, "semantic")

    logger.info("cache_tier=miss")
    return CacheLookup(None, "miss")


async def set_cached(
    redis: Redis | None, settings: QuerySettings, message: str, answer: str
) -> None:
    """Store an answer with TTL in both tiers (semantic tier if enabled)."""
    if redis is None:
        return
    try:
        version = await _index_version(redis)
        ttl = _ttl_seconds()
        await redis.set(
            _build_key(version, settings, message),
            answer,
            ex=ttl,
        )
    except RedisError as exc:
        logger.warning("Cache write failed: %s", exc)
        return

    if semantic_enabled():
        await semantic_set(redis, settings, message, answer, version, ttl)

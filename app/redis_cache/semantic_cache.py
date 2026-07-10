"""Semantic prompt-answer cache backed by RediSearch (Redis Stack).

Complements the exact-match cache in ``prompt_cache.py``. When the exact
key misses, we embed the message with a local sentence-transformer,
KNN-search a per-fingerprint vector index, and serve the cached answer
if cosine similarity clears ``SEMANTIC_CACHE_THRESHOLD``.

Entries are written alongside the exact cache (see ``prompt_cache.py``).
All Redis errors degrade to a live query; the feature is fully gated by
``SEMANTIC_CACHE_ENABLED`` so it can ship dark.
"""

from __future__ import annotations

import hashlib
import logging
import os
import struct
from dataclasses import dataclass
from functools import lru_cache
from uuid import uuid4

from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

from app.config import QuerySettings

logger = logging.getLogger(__name__)

_INDEX_NAME = "prompt_vec_idx"
_KEY_PREFIX = "prompt:vec:"
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_DIM = 384


def is_enabled() -> bool:
    return os.environ.get("SEMANTIC_CACHE_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _threshold() -> float:
    # Cosine similarity cutoff. RediSearch returns cosine *distance*
    # (1 - similarity), so we convert at the call site.
    return float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", "0.93"))


def _top_k() -> int:
    return int(os.environ.get("SEMANTIC_CACHE_TOP_K", "1"))


@lru_cache(maxsize=1)
def _model():
    # Imported lazily so test runs that never touch the semantic cache
    # don't pay the ~2s model load.
    from sentence_transformers import SentenceTransformer

    logger.info("loading sentence-transformer model=%s", _MODEL_NAME)
    return SentenceTransformer(_MODEL_NAME)


def _embed(text: str) -> bytes:
    vec = _model().encode(
        text, normalize_embeddings=True, convert_to_numpy=True
    )
    # RediSearch expects raw float32 little-endian bytes.
    return struct.pack(f"<{_EMBED_DIM}f", *vec.tolist())


def _fingerprint(settings: QuerySettings) -> str:
    payload = "|".join(
        [
            settings.method.value,
            str(settings.community_level),
            settings.response_type,
        ]
    )
    # Short tag; RediSearch TAG values must avoid separators.
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class SemanticHit:
    answer: str
    similarity: float
    matched_question: str


async def ensure_index(redis: Redis | None) -> None:
    """Create the RediSearch vector index if it doesn't exist."""
    if redis is None or not is_enabled():
        return
    try:
        await redis.execute_command(
            "FT.CREATE",
            _INDEX_NAME,
            "ON",
            "HASH",
            "PREFIX",
            "1",
            _KEY_PREFIX,
            "SCHEMA",
            "fingerprint",
            "TAG",
            "version",
            "NUMERIC",
            "question",
            "TEXT",
            "answer",
            "TEXT",
            "NOINDEX",
            "embedding",
            "VECTOR",
            "HNSW",
            "6",
            "TYPE",
            "FLOAT32",
            "DIM",
            str(_EMBED_DIM),
            "DISTANCE_METRIC",
            "COSINE",
        )
        logger.info("created RediSearch index name=%s", _INDEX_NAME)
    except ResponseError as exc:
        if "Index already exists" in str(exc):
            return
        logger.warning("FT.CREATE failed (semantic cache disabled): %s", exc)
    except RedisError as exc:
        logger.warning(
            "redis unavailable during FT.CREATE (semantic cache disabled): %s",
            exc,
        )


async def semantic_get(
    redis: Redis | None,
    settings: QuerySettings,
    message: str,
    version: int,
) -> SemanticHit | None:
    """KNN-lookup the message; return a hit only if above threshold."""
    if redis is None or not is_enabled():
        return None
    try:
        vec = _embed(message)
        fp = _fingerprint(settings)
        top_k = _top_k()
        # Filter by settings fingerprint + index version,
        # then rank by cosine distance.
        query = (
            f"(@fingerprint:{{{fp}}} @version:[{version} {version}])"
            f"=>[KNN {top_k} @embedding $vec AS score]"
        )
        raw = await redis.execute_command(
            "FT.SEARCH",
            _INDEX_NAME,
            query,
            "PARAMS",
            "2",
            "vec",
            vec,
            "RETURN",
            "3",
            "answer",
            "question",
            "score",
            "SORTBY",
            "score",
            "DIALECT",
            "2",
        )
    except RedisError as exc:
        logger.warning("semantic lookup failed: %s", exc)
        return None

    # FT.SEARCH reply: [total, key1, [field, val, field, val, ...], key2, ...]
    if not raw or raw[0] == 0:
        return None
    fields = raw[2]
    record = {fields[i]: fields[i + 1] for i in range(0, len(fields), 2)}
    distance = float(record.get("score", "1.0"))
    similarity = 1.0 - distance
    if similarity < _threshold():
        logger.info(
            "semantic miss similarity=%.3f threshold=%.3f",
            similarity,
            _threshold(),
        )
        return None
    return SemanticHit(
        answer=record.get("answer", ""),
        similarity=similarity,
        matched_question=record.get("question", ""),
    )


async def semantic_set(
    redis: Redis | None,
    settings: QuerySettings,
    message: str,
    answer: str,
    version: int,
    ttl_seconds: int,
) -> None:
    """Persist a question/answer pair as a vector record."""
    if redis is None or not is_enabled():
        return
    try:
        vec = _embed(message)
        fp = _fingerprint(settings)
        key = f"{_KEY_PREFIX}v{version}:{uuid4().hex}"
        await redis.hset(
            key,
            mapping={
                "fingerprint": fp,
                "version": version,
                "question": message,
                "answer": answer,
                "embedding": vec,
            },
        )
        await redis.expire(key, ttl_seconds)
    except RedisError as exc:
        logger.warning("semantic write failed: %s", exc)

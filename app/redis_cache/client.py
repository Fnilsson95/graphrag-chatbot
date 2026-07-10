"""Shared async Redis client

Holds a single module-level ``redis.asyncio`` client created from
``REDIS_URL`` (defaulting to localhost, ``decode_responses=True``).
``init_redis``/``close_redis`` are driven by the FastAPI lifespan;
``get_redis`` returns the client or ``None`` so callers can no-op cleanly
when Redis was never initialised.
"""

from __future__ import (
    annotations,
)

import os

from redis.asyncio import Redis

_client: Redis | None = None


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


async def init_redis() -> None:
    """Create the shared client. Called from FastAPI on startup."""
    global _client
    # decode_responses=True returns str instead of bytes for all string values.
    _client = Redis.from_url(_redis_url(), decode_responses=True)


async def close_redis() -> None:
    """Close the shared client. Called from FastAPI on shutdown"""

    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def get_redis() -> Redis | None:
    """Return the shared client (or None if not initialized)"""
    return _client

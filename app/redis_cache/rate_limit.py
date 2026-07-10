"""Per-IP fixed-window rate limiter (Redis and in-memory fallback).

Limits each client IP to a configurable number of requests per time
window (default 30 / 60s, tunable via ``RATE_LIMIT_*`` env vars) using a
Redis counter on a time-bucketed key with expiry. Falls back to an
in-process per-bucket counter when Redis is down, and raises
``RateLimitExceeded`` when the cap is hit so the endpoint can return
HTTP 429.
"""

from __future__ import annotations

import logging
import os
import time

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

# default 30 requests per window
_DEFAULT_REQUESTS = 30
# default 60 seconds per window (30 requests/min)
_DEFAULT_WINDOW_SECONDS = 60

# Fallback counter
_local_counters: dict[tuple[str, int], int] = {}


def _limit() -> int:
    return int(os.environ.get("RATE_LIMIT_REQUESTS", _DEFAULT_REQUESTS))


def _window() -> int:
    return int(
        os.environ.get("RATE_LIMIT_WINDOW_SECONDS", _DEFAULT_WINDOW_SECONDS)
    )


class RateLimitExceeded(Exception):
    """Raised when the caller has exceeded the allowed request rate."""


def _local_increment(client_ip: str, bucket: int) -> int:
    for stale_key in [k for k in _local_counters if k[1] < bucket]:
        del _local_counters[stale_key]
    key = (client_ip, bucket)
    _local_counters[key] = _local_counters.get(key, 0) + 1
    return _local_counters[key]


async def check_rate_limit(redis: Redis | None, client_ip: str) -> None:
    """Raise RateLimitExceeded if over the limit."""
    window = _window()
    # Fixed-window bucket: all requests in the same window share one counter.
    bucket = int(time.time()) // window

    if redis is None:
        count = _local_increment(client_ip, bucket)
    else:
        try:
            key = f"ratelimit: {client_ip}:{bucket}"
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, window)
        except RedisError as exc:
            logger.warning("Rate limiter falling back to in-memory: %s", exc)
            count = _local_increment(client_ip, bucket)

    if count > _limit():
        raise RateLimitExceeded(
            f"Rate limit exceeded: {count}/{_limit()} per {window}s"
        )

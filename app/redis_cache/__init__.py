"""Redis-backed support for the ``/prompt`` path.

Houses the shared async Redis client plus the three features layered on
top of it: the answer cache, the per-IP rate limiter, and conversation
history. Each feature degrades gracefully to an in-process fallback when
Redis is unavailable, so the API keeps working (without cross-process
sharing) during a Redis outage.
"""

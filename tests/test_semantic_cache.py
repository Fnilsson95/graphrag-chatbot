"""Unit tests for the two-tier prompt cache.

The semantic tier needs RediSearch (Redis Stack), which fakeredis does
not implement. Rather than spinning up a real Stack in CI, these tests
stub ``semantic_get``/``semantic_set`` and verify the *tiering logic* in
``prompt_cache.get_cached`` — i.e. exact wins over semantic, semantic
fires only when exact misses and the feature flag is on, and the tier
label is reported correctly.

The live RediSearch round-trip (embedding + KNN) is covered separately by
the semantic cache module against a running Redis Stack.
"""

from __future__ import annotations

import asyncio

import fakeredis

from app.config import QuerySettings
from app.redis_cache import prompt_cache
from app.redis_cache.prompt_cache import CacheLookup, get_cached, set_cached
from app.redis_cache.semantic_cache import SemanticHit


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _fresh_redis():
    return fakeredis.FakeAsyncRedis(decode_responses=True)


def test_exact_hit_short_circuits_semantic(monkeypatch):
    redis = _fresh_redis()
    settings = QuerySettings()
    _run(set_cached(redis, settings, "what is X", "answer-X"))

    called = {"n": 0}

    async def fake_semantic(*_a, **_kw):
        called["n"] += 1
        return SemanticHit(
            answer="WRONG", similarity=0.99, matched_question=""
        )

    monkeypatch.setattr(prompt_cache, "semantic_enabled", lambda: True)
    monkeypatch.setattr(prompt_cache, "semantic_get", fake_semantic)

    lookup = _run(get_cached(redis, settings, "what is X"))
    assert lookup == CacheLookup(answer="answer-X", tier="exact")
    assert called["n"] == 0


def test_semantic_hit_when_exact_misses(monkeypatch):
    redis = _fresh_redis()
    settings = QuerySettings()

    async def fake_semantic(*_a, **_kw):
        return SemanticHit(
            answer="paraphrase-answer",
            similarity=0.97,
            matched_question="original Q",
        )

    monkeypatch.setattr(prompt_cache, "semantic_enabled", lambda: True)
    monkeypatch.setattr(prompt_cache, "semantic_get", fake_semantic)

    lookup = _run(get_cached(redis, settings, "paraphrased question"))
    assert lookup.tier == "semantic"
    assert lookup.answer == "paraphrase-answer"


def test_semantic_skipped_when_flag_off(monkeypatch):
    redis = _fresh_redis()
    settings = QuerySettings()

    async def boom(*_a, **_kw):
        raise AssertionError("semantic_get must not be called when disabled")

    monkeypatch.setattr(prompt_cache, "semantic_enabled", lambda: False)
    monkeypatch.setattr(prompt_cache, "semantic_get", boom)

    lookup = _run(get_cached(redis, settings, "anything"))
    assert lookup == CacheLookup(answer=None, tier="miss")


def test_miss_when_semantic_below_threshold(monkeypatch):
    redis = _fresh_redis()
    settings = QuerySettings()

    async def fake_semantic(*_a, **_kw):
        return None  # semantic_cache filters by threshold before returning

    monkeypatch.setattr(prompt_cache, "semantic_enabled", lambda: True)
    monkeypatch.setattr(prompt_cache, "semantic_get", fake_semantic)

    lookup = _run(get_cached(redis, settings, "low-similarity question"))
    assert lookup.tier == "miss"
    assert lookup.answer is None

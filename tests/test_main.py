"""API tests: health, cache hit, cache invalidation, fallback, rate limit"""

from __future__ import annotations

import asyncio

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.redis_cache import client as redis_client
from app.redis_cache.prompt_cache import INDEX_VERSION_KEY


@pytest.fixture(autouse=True)
def reset_local_counters(monkeypatch):
    """Reset in-memory state and disable the semantic cache between tests.

    The semantic tier needs RediSearch, which fakeredis lacks; .env may set
    SEMANTIC_CACHE_ENABLED=1, so force it off for the API tests.
    """
    from app.redis_cache import conversation_history, rate_limit

    monkeypatch.setenv("SEMANTIC_CACHE_ENABLED", "0")
    rate_limit._local_counters.clear()
    conversation_history._LOCAL_HISTORY.clear()
    yield
    rate_limit._local_counters.clear()
    conversation_history._LOCAL_HISTORY.clear()


@pytest.fixture
def fake_redis(monkeypatch):
    """Replace the Redis client with an in-memory fake"""
    fake = fakeredis.FakeAsyncRedis(decode_responses=True)
    monkeypatch.setattr(redis_client, "_client", fake)
    return fake


@pytest.fixture
def graphrag_calls(monkeypatch):
    """Capture and stub the GraphRAG runner so tests don't hit it."""

    calls: list[str] = []

    async def fake_run(_settings, message):
        calls.append(message)
        return f"answer for {message}", {}

    monkeypatch.setattr("app.pipeline.run_query_with_context", fake_run)
    return calls


@pytest.fixture
def graphrag_stream(monkeypatch):
    """Stub run_query_stream so the SSE endpoint emits context + chunks."""

    async def fake_stream(_settings, message):
        yield ("context", {})
        for token in ("answer ", "for ", message):
            yield ("chunk", token)
        yield ("done", None)

    monkeypatch.setattr("app.pipeline.run_query_stream", fake_stream)


@pytest.fixture
def stub_clarifier(monkeypatch):
    """Stub the clarifier so tests don't hit the LLM"""
    from app.clarifier import ClarifierResult

    async def fake_classify(_message: str) -> ClarifierResult:
        return ClarifierResult(
            needs_clarification=False,
            question=None,
            options=None,
        )

    monkeypatch.setattr("app.pipeline.classify", fake_classify)
    return fake_classify


@pytest.fixture
def client(fake_redis, graphrag_calls, stub_clarifier) -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_identical_prompt_is_served_from_cache(
    client: TestClient,
    graphrag_calls: list[str],
) -> None:
    requestOne = client.post("/prompt", json={"message": "hello"})
    requestTwo = client.post("/prompt", json={"message": "hello"})

    # First call --> cache is empty --> goes through to (fake) GraphRAG
    # Second call --> cache is stored
    assert requestOne.status_code == 200
    assert requestTwo.status_code == 200
    assert requestOne.headers["X-Cache-Tier"] == "miss"
    assert requestTwo.headers["X-Cache-Tier"] == "exact"
    assert requestOne.json()["message"] == requestTwo.json()["message"]
    assert len(graphrag_calls) == 1


def test_cache_resets_when_index_is_rebuilt(
    client: TestClient, fake_redis, graphrag_calls: list[str]
) -> None:
    # First call --> cached under V1
    # Second call --> cached under V2
    client.post("/prompt", json={"message": "hello"})
    asyncio.get_event_loop().run_until_complete(
        fake_redis.incr(INDEX_VERSION_KEY)
    )
    client.post("/prompt", json={"message": "hello"})
    assert len(graphrag_calls) == 2


def test_cache_miss_falls_back_to_live_path(
    client: TestClient, graphrag_calls: list[str]
) -> None:
    response = client.post(
        "/prompt",
        json={"message": "fresh query", "conversationID": "fresh-query"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "kind": "answer",
        "message": "answer for fresh query",
        "options": None,
        "sources": None,
        "conversationID": "fresh-query",
    }
    assert graphrag_calls == ["fresh query"]


def test_response_includes_generated_conversation_id(
    client: TestClient,
) -> None:
    response = client.post("/prompt", json={"message": "hello"})

    assert response.status_code == 200
    assert response.json()["conversationID"]


def test_follow_up_query_includes_conversation_history(
    client: TestClient,
    graphrag_calls: list[str],
) -> None:
    first = client.post(
        "/prompt",
        json={
            "message": "Tell me about Alice",
            "conversationID": "conversation-1",
        },
    )
    second = client.post(
        "/prompt",
        json={
            "message": "What about her role?",
            "conversationID": "conversation-1",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(graphrag_calls) == 2
    assert "Conversation history:" in graphrag_calls[1]
    assert "User: Tell me about Alice" in graphrag_calls[1]
    assert "Assistant: answer for Tell me about Alice" in graphrag_calls[1]
    assert "Current user question:\nWhat about her role?" in graphrag_calls[1]


def test_follow_up_bypasses_cache(
    client: TestClient,
    graphrag_calls: list[str],
) -> None:
    conversation_id = "follow-up-live"
    first = client.post(
        "/prompt",
        json={
            "message": "Tell me about Alice",
            "conversationID": conversation_id,
        },
    )
    second = client.post(
        "/prompt",
        json={
            "message": "What about her role?",
            "conversationID": conversation_id,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["X-Cache-Tier"] == "miss"
    assert second.headers["X-Cache-Tier"] == "miss"
    assert len(graphrag_calls) == 2
    assert "Conversation history:" in graphrag_calls[1]


def test_same_follow_up_is_isolated_by_conversation(
    client: TestClient,
    graphrag_calls: list[str],
) -> None:
    client.post(
        "/prompt",
        json={"message": "Tell me about Alice", "conversationID": "alice"},
    )
    client.post(
        "/prompt",
        json={"message": "Tell me about Bob", "conversationID": "bob"},
    )
    client.post(
        "/prompt",
        json={"message": "What about their role?", "conversationID": "alice"},
    )
    client.post(
        "/prompt",
        json={"message": "What about their role?", "conversationID": "bob"},
    )

    assert len(graphrag_calls) == 4
    assert "Tell me about Alice" in graphrag_calls[2]
    assert "Tell me about Bob" in graphrag_calls[3]


def test_stream_emits_meta_chunks_and_done(
    fake_redis, graphrag_stream, stub_clarifier
) -> None:
    client = TestClient(app)
    response = client.post(
        "/prompt/stream",
        json={"message": "fresh", "conversationID": "stream-1"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: meta" in body
    assert '"conversationID": "stream-1"' in body
    assert "event: cache" in body
    assert "event: chunk" in body
    assert "answer " in body  # streamed token
    assert body.rstrip().endswith("event: done\ndata: {}")


def test_stream_reports_clarification(fake_redis, monkeypatch) -> None:
    from app.clarifier import ClarifierResult

    async def needs_clarification(_message: str) -> ClarifierResult:
        return ClarifierResult(
            needs_clarification=True,
            question="Which dataset?",
            options=["A", "B"],
        )

    monkeypatch.setattr("app.pipeline.classify", needs_clarification)
    client = TestClient(app)
    response = client.post(
        "/prompt/stream", json={"message": "vague", "conversationID": "c-1"}
    )

    assert response.status_code == 200
    assert "event: clarification" in response.text
    assert "Which dataset?" in response.text


def test_clarification_option_pick_runs_graphrag(
    client: TestClient,
    graphrag_calls: list[str],
    monkeypatch,
) -> None:
    from app.clarifier import ClarifierResult

    calls = 0

    async def classify(message: str) -> ClarifierResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ClarifierResult(
                needs_clarification=True,
                question="What aspect should I explain?",
                options=[
                    "Overview of the platform",
                    "Node-based workflow system",
                ],
            )
        return ClarifierResult(
            needs_clarification=False,
            question=None,
            options=None,
        )

    monkeypatch.setattr("app.pipeline.classify", classify)

    conversation_id = "clarify-pick-1"
    first = client.post(
        "/prompt",
        json={
            "message": "Explain pods",
            "conversationID": conversation_id,
        },
    )
    second = client.post(
        "/prompt",
        json={"message": "Overview", "conversationID": conversation_id},
    )

    assert first.status_code == 200
    assert first.json()["kind"] == "clarification"
    assert second.status_code == 200
    assert second.json()["kind"] == "answer"
    assert len(graphrag_calls) == 1
    assert graphrag_calls[0] == (
        "Explain pods — specifically: Overview of the platform"
    )
    assert calls == 1


def test_cache_hit_skips_clarifier(
    client: TestClient,
    graphrag_calls: list[str],
    fake_redis,
    monkeypatch,
) -> None:
    from app.clarifier import ClarifierResult

    calls = 0

    async def classify_once(_message: str) -> ClarifierResult:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("clarifier should not run on cache hit")
        return ClarifierResult(
            needs_clarification=False,
            question=None,
            options=None,
        )

    monkeypatch.setattr("app.pipeline.classify", classify_once)

    first = client.post("/prompt", json={"message": "cached question"})
    assert first.status_code == 200

    second = client.post("/prompt", json={"message": "cached question"})
    assert second.status_code == 200
    assert len(graphrag_calls) == 1
    assert calls == 1


def test_rate_limit_return_429(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")

    for i in range(3):
        ok = client.post("/prompt", json={"message": f"m{i}"})
        assert ok.status_code == 200

    blocked = client.post("/prompt", json={"message": "m4"})
    assert blocked.status_code == 429


def test_rate_limit_works_without_redis(
    graphrag_calls: list[str],
    stub_clarifier,
    monkeypatch,
) -> None:
    monkeypatch.setattr(redis_client, "_client", None)
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "60")
    client = TestClient(app)

    for i in range(3):
        ok = client.post("/prompt", json={"message": f"m{i}"})
        assert ok.status_code == 200

    blocked = client.post("/prompt", json={"message": "m4"})
    assert blocked.status_code == 429

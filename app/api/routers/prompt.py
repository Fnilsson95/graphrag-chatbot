"""HTTP transport for the prompt pipeline.

``POST /prompt`` drains :func:`app.pipeline.run_pipeline` and returns the
final answer as JSON. ``POST /prompt/stream`` forwards each pipeline event
as a Server-Sent Event so the frontend can render tokens as they arrive.
"""

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.api import sse
from app.api.schemas import PromptRequest, PromptResponse
from app.api.schemas import Source as SourceSchema
from app.pipeline import (
    Answer,
    Cache,
    Chunk,
    Clarification,
    Error,
    Meta,
    Sources,
    run_pipeline,
)
from app.sources import Source

router = APIRouter(tags=["prompt"])


def _to_schema(sources: list[Source]) -> list[SourceSchema]:
    return [SourceSchema(title=s.title, url=s.url) for s in sources]


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post(
    "/prompt",
    response_model=PromptResponse,
    summary="Prompt endpoint",
)
async def prompt(
    body: PromptRequest,
    request: Request,
    response: Response,
) -> PromptResponse:
    """Run the pipeline and return the assembled answer or clarification."""
    conversation_id = body.conversation_id
    answer: Answer | None = None

    # Non-streaming path: consume Chunk events silently,
    # return the final Answer.
    async for event in run_pipeline(
        body.message, body.conversation_id, _client_ip(request)
    ):
        match event:
            case Meta():
                conversation_id = event.conversation_id
            case Cache():
                response.headers["X-Cache-Tier"] = event.tier
            case Clarification():
                return PromptResponse(
                    kind="clarification",
                    message=event.message,
                    options=event.options or None,
                    conversation_id=conversation_id,
                )
            case Error():
                raise HTTPException(
                    status_code=event.status, detail=event.detail
                )
            case Answer():
                answer = event

    if answer is None:  # pragma: no cover - run_pipeline always terminates
        raise HTTPException(status_code=500, detail="Empty GraphRAG response")

    return PromptResponse(
        kind="answer",
        message=answer.text,
        sources=_to_schema(answer.sources) or None,
        conversation_id=conversation_id,
    )


@router.post("/prompt/stream", summary="Streaming prompt endpoint (SSE)")
async def prompt_stream(
    body: PromptRequest, request: Request
) -> StreamingResponse:
    """Forward each pipeline event to the client as an SSE frame."""
    return StreamingResponse(
        _sse_frames(body, _client_ip(request)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse_frames(body: PromptRequest, client_ip: str):
    """Translate pipeline events into SSE byte frames."""
    async for event in run_pipeline(
        body.message, body.conversation_id, client_ip, stream=True
    ):
        match event:
            case Meta():
                yield sse.encode(
                    sse.EVENT_META, {"conversationID": event.conversation_id}
                )
            case Cache():
                yield sse.encode(sse.EVENT_CACHE, {"tier": event.tier})
            case Clarification():
                yield sse.encode(
                    sse.EVENT_CLARIFICATION,
                    {"message": event.message, "options": event.options},
                )
                yield sse.encode(sse.EVENT_DONE, {})
                return
            case Sources():
                yield sse.encode(
                    sse.EVENT_SOURCES,
                    {"sources": [s.to_dict() for s in event.sources]},
                )
            case Chunk():
                yield sse.encode(sse.EVENT_CHUNK, {"text": event.text})
            case Error():
                yield sse.encode(
                    sse.EVENT_ERROR,
                    {"status": event.status, "detail": event.detail},
                )
                return
            case Answer():
                yield sse.encode(sse.EVENT_DONE, {})
                return

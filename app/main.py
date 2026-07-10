"""FastAPI application entry point.

Builds the GraphRAG Chatbot API: configures CORS (restrictive by
default, localhost allowed in dev), mounts the ``/prompt`` router and a
``/health`` check, and manages an async lifespan that loads ``.env`` and
opens/closes the shared Redis client around request serving. ``main()``
runs it under uvicorn for local/Docker use.
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.routers import prompt as prompt_router
from app.redis_cache.client import close_redis, get_redis, init_redis
from app.redis_cache.semantic_cache import ensure_index

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


def _split_csv_env(value: str) -> list[str]:
    """Parse a comma-separated env var into a list of non-empty strings."""
    return [v.strip() for v in value.split(",") if v.strip()]


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load ``.env`` before serving."""
    load_dotenv()
    await init_redis()
    await ensure_index(get_redis())
    try:
        yield
    finally:
        await close_redis()


def create_app() -> FastAPI:
    """FastAPI app, CORS, routers, ``/health``."""
    application = FastAPI(
        title="GraphRAG Chatbot",
        description="Chatbot Generation API",
        lifespan=lifespan,
    )

    # CORS defaults are intentionally restrictive. For local dev, the default
    # regex allows localhost on any port. Configure explicit origins in prod.
    allow_origins_raw = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
    allow_origin_regex = os.environ.get(
        "CORS_ALLOW_ORIGIN_REGEX",
        r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    ).strip()
    if allow_origins_raw == "*":
        allow_origins = ["*"]
    else:
        allow_origins = _split_csv_env(allow_origins_raw)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=None if allow_origins else allow_origin_regex,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(prompt_router.router)

    @application.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        """Liveness check."""
        return {"status": "ok"}

    return application


app = create_app()


def main() -> None:
    """Run uvicorn for local development and Docker."""
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        loop="asyncio",
    )


if __name__ == "__main__":
    main()

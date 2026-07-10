"""Server-Sent Events encoding for the streaming prompt endpoint.

Each pipeline event maps to a named SSE frame (``event:`` + JSON ``data:``).
The frontend adapter in ``webb/src/lib/prompt-api-adapter.ts`` parses these.
"""

from __future__ import annotations

import json

EVENT_META = "meta"
EVENT_CACHE = "cache"
EVENT_SOURCES = "sources"
EVENT_CHUNK = "chunk"
EVENT_CLARIFICATION = "clarification"
EVENT_ERROR = "error"
EVENT_DONE = "done"


def encode(event: str, data: dict) -> bytes:
    """Encode one SSE frame as ``event:``/``data:`` lines."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

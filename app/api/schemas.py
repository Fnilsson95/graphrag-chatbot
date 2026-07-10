"""Pydantic models for the ``/prompt`` request/response contract.

``PromptRequest`` carries the user ``message`` and an optional stable
``conversation_id`` (length/charset constrained, accepted under both
``conversationID`` and ``conversation_id`` aliases). ``PromptResponse``
distinguishes an ``answer`` from a ``clarification`` via ``kind``, and
carries the message, optional clarification ``options``, and the
conversation id so the client can continue the thread.
"""

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic.types import StringConstraints

ConversationID = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        # Safe for Redis keys and URL paths; must start with alphanumeric.
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]


class PromptRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(
        ...,
        min_length=1,
        description="The user's question",
    )
    conversation_id: ConversationID | None = Field(
        default=None,
        validation_alias=AliasChoices("conversationID", "conversation_id"),
        serialization_alias="conversationID",
        description="Stable identifier for one conversation",
    )


class Source(BaseModel):
    title: str
    url: str | None = None


class PromptResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Literal["answer", "clarification"]
    message: str
    options: list[str] | None = None
    sources: list[Source] | None = None
    conversation_id: ConversationID = Field(
        validation_alias=AliasChoices("conversationID", "conversation_id"),
        serialization_alias="conversationID",
    )

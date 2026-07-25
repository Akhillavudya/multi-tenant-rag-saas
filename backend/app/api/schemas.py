"""Request and response models for the HTTP API.

Pydantic models, separate from the ORM tables, and the separation is the point:
these define what a CLIENT may send and what a client may see, which is a
smaller and differently-shaped set than what the database stores.

**The rule that matters: no request model has an `org_id` field.**

Not one. Tenant scope arrives from the verified session token via
`security.clerk.Principal` and is applied by the router; if a client could put
`org_id` in a JSON body, the strongest possible Qdrant filter would simply be
filtering on a value the attacker chose. Every request model here is
`extra="forbid"`, so a body containing `org_id` is rejected outright rather
than silently ignored — a rejected request is visible in the logs, an ignored
field is not, and "ignored" is one careless `**payload` away from "accepted".

Response models are equally deliberate about what they omit. `ChatbotOut`
returns `public_key` because the dashboard must display the embed snippet;
nothing here returns `clerk_org_id`, internal user ids, or filesystem paths.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.core.chat import MAX_QUESTION_CHARS
from app.db.tables import (
    Channel,
    ChatbotStatus,
    DocumentStatus,
    MessageRole,
    PlanTier,
    SourceType,
)

# Applied to every INBOUND model. `extra="forbid"` turns an unexpected field
# into a 422 instead of a shrug; see the module docstring on why silence is the
# dangerous option here.
_STRICT_IN = ConfigDict(extra="forbid")

# Applied to every OUTBOUND model. from_attributes lets FastAPI build these
# straight from ORM objects, so a handler returns the row and Pydantic performs
# the field selection — which means the whitelist below is what ships, not
# whatever the ORM happens to have loaded.
_FROM_ORM = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------
# Chatbots
# --------------------------------------------------------------------------


class ChatbotCreate(BaseModel):
    """Body for POST /api/bots.

    Note what is absent as much as what is present: no org_id (from the token),
    no embedding_model or dimension (from the configured provider — letting a
    client pick a dimension would let it create a bot whose vectors can never
    be written), no public_key (generated server-side by a CSPRNG), and no
    status (a new bot is always DRAFT until a document is ready).
    """

    model_config = _STRICT_IN

    name: str = Field(min_length=1, max_length=120)
    system_prompt: str | None = Field(default=None, max_length=8_000)
    model_overrides: dict = Field(default_factory=dict)


class ChatbotUpdate(BaseModel):
    """Body for PATCH /api/bots/{id}. Every field optional.

    All-optional is what makes this a PATCH rather than a PUT: the dashboard's
    settings form saves one field at a time, and a PUT would require it to send
    back every value it did not touch — which is how a concurrent edit gets
    silently reverted.

    `status` is settable so a bot can be disabled without deleting it (which
    would destroy its vectors); `embedding_model` and `dimension` are not,
    because changing them is `scripts/reembed.py`'s job, not a form field's.
    """

    model_config = _STRICT_IN

    name: str | None = Field(default=None, min_length=1, max_length=120)
    system_prompt: str | None = Field(default=None, max_length=8_000)
    model_overrides: dict | None = None
    status: ChatbotStatus | None = None


class ChatbotOut(BaseModel):
    """A chatbot as the dashboard sees it."""

    model_config = _FROM_ORM

    id: uuid.UUID
    name: str
    system_prompt: str | None
    model_overrides: dict
    embedding_model: str
    dimension: int
    status: ChatbotStatus
    # The widget credential. Returned deliberately — Phase 5's embed snippet
    # contains it, so the dashboard must be able to show it. It is a PUBLIC key
    # in the sense that it ships inside a <script> tag on the customer's own
    # site: it identifies a bot and authorizes chatting with it, and nothing
    # else. It can never read or modify documents, which is why the chat
    # endpoint in Phase 3 resolves it to org+bot and stops there.
    public_key: str
    created_at: datetime


# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


class DocumentOut(BaseModel):
    """A document and its ingestion state.

    This is what the sources tab polls. `status` plus `error` are the whole
    user-facing contract of async ingestion: a document sitting in PROCESSING
    means work is happening, FAILED with an error string means the user can
    usually fix it themselves (a scanned PDF, an encrypted file).

    `chunk_count` is included because it is the one number that tells a user
    their upload actually produced something searchable. A READY document with
    zero chunks is a bug, and surfacing the count makes it visible rather than
    leaving them to wonder why the bot ignores their file.
    """

    model_config = _FROM_ORM

    id: uuid.UUID
    chatbot_id: uuid.UUID
    filename: str
    source_type: SourceType
    source_url: str | None
    status: DocumentStatus
    error: str | None
    pages: int | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------
# Chat (Phase 3)
# --------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Body for both chat endpoints.

    Read the absent fields first, because they are the security design:

    - **no `chatbot_id`**. The widget route derives it from the bot key and the
      dashboard route from the URL path, both server-side. A body field would
      let anyone holding one public key chat with every bot in the system.
    - **no `org_id`**, for the reason stated at the top of this module.
    - **no `system_prompt`, `model`, `top_k` or `max_tokens`**. Every one of
      those is a knob that costs money or changes behaviour, and a caller who
      can set them from a `<script>` tag on someone else's website is a caller
      who can run our LLM budget at their chosen size. They live on the chatbot
      row, editable only by an authenticated owner.

    `conversation_id` IS accepted, and it is the one field that could be forged
    — so the handler verifies it belongs to the same (org, chatbot) before using
    it, and creates a new conversation if it does not. See `api/chat.py`.
    """

    model_config = _STRICT_IN

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    # Continue an existing thread. Absent means "start a new one" — which is the
    # normal first message from a widget that has just loaded.
    conversation_id: uuid.UUID | None = None

    # Idempotency key, supplied by the client (see `messages.client_message_id`).
    # A widget whose connection dropped mid-answer retries with the same value
    # and receives the stored answer instead of paying for a second generation.
    # Free-form because it comes from a browser: a UUID in practice, but nothing
    # depends on its shape, only on its stability.
    client_message_id: str | None = Field(default=None, max_length=80)


class CitationOut(BaseModel):
    """One source under an answer.

    Mirrors `core.chat.Citation` rather than reusing it: that is a domain type
    free to change, this is a wire format that clients depend on.
    """

    citation: str
    document_id: str
    source: str
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None


class MessageOut(BaseModel):
    """One stored turn, as the dashboard's conversation viewer sees it.

    Includes the observability columns because the viewer is where you go to ask
    "why was this answer bad?" — and the answer is usually visible in the
    citations (retrieval missed) or the latency (a provider was struggling).
    """

    model_config = _FROM_ORM

    id: uuid.UUID
    role: MessageRole
    content: str
    citations: list[CitationOut] | None
    tokens_in: int | None
    tokens_out: int | None
    latency_ms: int | None
    created_at: datetime


class ConversationOut(BaseModel):
    """A thread, without its messages.

    Deliberately not nested: the list view renders hundreds of these and needs
    none of the message bodies, and loading them would be the classic list-page
    N+1. One extra request fetches a transcript when a row is actually opened.
    """

    model_config = _FROM_ORM

    id: uuid.UUID
    chatbot_id: uuid.UUID
    channel: Channel
    started_at: datetime
    last_message_at: datetime


class ConversationDetailOut(ConversationOut):
    """One thread with its full transcript, in order."""

    messages: list[MessageOut]


# --------------------------------------------------------------------------
# Organization / usage
# --------------------------------------------------------------------------


class OrganizationOut(BaseModel):
    """The caller's own org. Never another's — there is no endpoint that takes
    an org id, precisely so there is nothing to forge."""

    model_config = _FROM_ORM

    id: uuid.UUID
    name: str
    plan: PlanTier
    created_at: datetime


class UsageOut(BaseModel):
    """Current consumption against plan caps.

    Returned on the 429 when a limit is hit as well as from the usage endpoint,
    so a client that gets refused has the numbers to explain why without a
    second round trip.

    Three different shapes of limit live here, and the difference matters when
    reading the dashboard: chatbots and documents are STOCKS (delete one, free a
    slot), while messages are a monthly FLOW that resets — which is why only the
    message counters carry a `period`.
    """

    plan: PlanTier
    chatbots_used: int
    chatbots_limit: int
    documents_used: int
    documents_limit: int

    # Phase 3. "YYYY-MM" in UTC; see `core/metering.current_period`.
    period: str
    messages_used: int
    messages_limit: int

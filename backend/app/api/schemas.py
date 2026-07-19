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

from app.db.tables import ChatbotStatus, DocumentStatus, PlanTier, SourceType

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
    second round trip. Message metering arrives in Phase 3; the counts that
    exist in Phase 2 are the ones this phase can honestly report.
    """

    plan: PlanTier
    chatbots_used: int
    chatbots_limit: int
    documents_used: int
    documents_limit: int

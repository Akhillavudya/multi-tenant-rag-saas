"""Model registry — the single import surface for Alembic.

Alembic's autogenerate only sees models that have actually been imported into
the process. This module imports them all, and `migrations/env.py` imports this
module. Adding a model therefore means adding one import line here, and nothing
else, forever.

Phase 2 adds (PLAN §4): Organization, User, Chatbot, Document, Conversation,
Message, UsageRecord, Subscription.
"""

from app.db.base import Base

# The noqa is required: these imports look "unused" to the linter, but the
# import itself is the point — it registers the model on Base.metadata.
from app.db.tables import (  # noqa: F401
    Chatbot,
    ChatbotStatus,
    Document,
    DocumentStatus,
    Organization,
    PlanTier,
    SourceType,
    User,
    UserRole,
)

__all__ = [
    "Base",
    "Chatbot",
    "ChatbotStatus",
    "Document",
    "DocumentStatus",
    "Organization",
    "PlanTier",
    "SourceType",
    "User",
    "UserRole",
]

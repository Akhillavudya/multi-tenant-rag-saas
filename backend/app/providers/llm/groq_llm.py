"""Groq LLM provider (groq SDK).

The fallback generator (PLAN §2, §3 rule 3): when Gemini raises a *retryable*
ProviderError — almost always a free-tier 429 — Phase 3 re-runs the same request
through this provider. It is also the whole point of `LLM_PROVIDER=groq` being a
one-line env change.

This is the ONLY file allowed to import `groq` — Hard Rule #2.

Two responsibilities, mirroring gemini.py so the two files stay diff-able:

1. **Shape translation.** Groq speaks the OpenAI-compatible chat format, so the
   translation is nearly free: our `Message(role, content)` is already the right
   shape, and the system prompt is simply the first message rather than an
   out-of-band config field the way Gemini wants it.
2. **Error translation.** Every groq exception becomes a ProviderError subclass,
   so the API layer maps failures to HTTP codes without knowing the vendor, and
   the fallback logic branches on `retryable` instead of on Groq-specific types.

API note (groq >= 0.13): the async `chat.completions.create(stream=True)` is a
*coroutine returning an AsyncStream*, not an async generator — the same trap as
Gemini's `generate_content_stream`. Await it first, then iterate.
"""

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from app.providers.base import (
    Message,
    ProviderAuthError,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

if TYPE_CHECKING:
    from groq import AsyncGroq

logger = logging.getLogger(__name__)

# Groq's strongest free-tier instruction-follower. Fast enough that it is a
# genuine fallback rather than a degraded mode.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Same reasoning as Gemini: the facts live in the retrieved context, so
# creativity here only buys drift away from them.
DEFAULT_TEMPERATURE = 0.2

# Unlike Gemini, no role remapping is needed — "system"/"user"/"assistant" are
# exactly the wire values. The set exists to reject anything else loudly.
_ALLOWED_ROLES = {"user", "assistant"}


class GroqLLM:
    """LLMProvider backed by Groq's OpenAI-compatible chat API.

    Satisfies the Protocol structurally; does not inherit it.
    """

    name = "groq"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            # Fail at construction so a missing key is a boot error, not a
            # user's chat request dying mid-sentence.
            raise ProviderAuthError("GROQ_API_KEY is not set", provider=self.name)
        self._api_key = api_key
        self.model = model
        self._client: AsyncGroq | None = None

    def _get_client(self) -> "AsyncGroq":
        """Lazily construct the SDK client.

        No lock: this is a cheap config object wrapping an httpx session, so a
        racing double-construction is harmless. The local models are the ones
        that need locking — there a double load means two copies of the weights.
        """
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self._api_key)
        return self._client

    async def stream(
        self,
        system: str,
        messages: list[Message],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield answer text chunk by chunk.

        This is an async GENERATOR: calling it runs none of this body. The
        validation below, and any auth failure, happen on the first `async for`.
        Callers must wrap the *iteration* in try/except, not the call.
        """
        if not messages:
            raise ProviderInvalidRequestError(
                "at least one message is required", provider=self.name
            )
        if max_tokens <= 0:
            raise ProviderInvalidRequestError(
                f"max_tokens must be positive, got {max_tokens}", provider=self.name
            )

        client = self._get_client()
        payload = self._to_chat_messages(system, messages)

        try:
            # Await first, THEN iterate — see the API note in the module
            # docstring. This is the version-sensitive line in this file.
            stream = await client.chat.completions.create(
                model=self.model,
                messages=payload,
                max_tokens=max_tokens,
                temperature=DEFAULT_TEMPERATURE,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    # Groq sends a final usage-only frame with no choices.
                    continue
                text = chunk.choices[0].delta.content
                # `delta.content` is None on the role-announcing first frame and
                # on the finish frame; yielding None would break concatenation
                # downstream.
                if text:
                    yield text
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    # -- internals ----------------------------------------------------------

    def _to_chat_messages(
        self, system: str, messages: list[Message]
    ) -> list[dict[str, str]]:
        """Build the OpenAI-format message list.

        The system prompt is prepended as a real message here (Groq has no
        out-of-band system field). A `system` role arriving inside `messages` is
        still a caller bug — it would produce two system turns, and the model's
        behaviour when they conflict is undefined.
        """
        payload: list[dict[str, str]] = []
        if system:
            payload.append({"role": "system", "content": system})
        for msg in messages:
            if msg.role not in _ALLOWED_ROLES:
                raise ProviderInvalidRequestError(
                    f"role {msg.role!r} cannot be sent as a message; "
                    "the system prompt is passed separately",
                    provider=self.name,
                )
            payload.append({"role": msg.role, "content": msg.content})
        return payload

    def _translate(self, exc: Exception) -> ProviderError:
        """Map an SDK exception onto our error hierarchy.

        Mapping is by HTTP status where one exists, because that is the stable
        contract; SDK message strings change between releases.
        """
        import groq

        if isinstance(exc, groq.APITimeoutError):
            # Subclass of APIConnectionError, so it must be checked first.
            return ProviderTimeoutError(str(exc), provider=self.name)
        if isinstance(exc, groq.APIConnectionError):
            # Never reached the server: DNS, refused connection, TLS.
            return ProviderUnavailableError(str(exc), provider=self.name)

        if isinstance(exc, groq.APIStatusError):
            code = exc.status_code
            message = str(exc)

            if code in (401, 403):
                return ProviderAuthError(message, provider=self.name)
            if code == 429:
                # The one that actually happens on a free tier.
                return ProviderRateLimitError(message, provider=self.name)
            if code in (408, 504):
                return ProviderTimeoutError(message, provider=self.name)
            if 400 <= code < 500:
                # Our fault: context too long, bad model name, bad parameter.
                # Deliberately NOT retryable — the same bad request would fail
                # identically against any other provider.
                return ProviderInvalidRequestError(message, provider=self.name)
            return ProviderUnavailableError(message, provider=self.name)

        if isinstance(exc, TimeoutError):
            return ProviderTimeoutError(str(exc), provider=self.name)

        # Unknown failure: treat as retryable. A transient network fault is far
        # likelier here than a logic bug, and failing over is cheap.
        return ProviderUnavailableError(
            f"unexpected Groq failure: {exc}", provider=self.name
        )

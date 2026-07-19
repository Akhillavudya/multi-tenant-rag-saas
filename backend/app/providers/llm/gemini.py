"""Gemini LLM provider (google-genai SDK).

The default generator (PLAN §2): free tier, fast, good instruction-following.
This is the ONLY file in the project allowed to import google.genai — Hard
Rule #2. Everything above it sees `LLMProvider.stream()` and our own exceptions.

Two responsibilities, and they are equally important:

1. **Shape translation.** Our vendor-neutral `Message(role, content)` becomes
   Gemini's `Content(role, parts)`. Gemini calls the assistant role "model", and
   takes the system prompt out-of-band in the config rather than as a message.
2. **Error translation.** Every google.genai exception becomes a ProviderError
   subclass, so the API layer maps failures to HTTP codes without knowing which
   vendor produced them — and so Phase 3's Gemini→Groq fallback can act on
   `retryable` instead of on Google-specific types.

API note (verified against google-genai 2.12.1): the async
`generate_content_stream` is a *coroutine returning an AsyncIterator*, not an
async generator. It must be awaited first, then iterated. Writing
`async for c in client.aio.models.generate_content_stream(...)` fails.
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
    from google.genai import Client

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"

# Grounded RAG answers should be near-deterministic: the facts are in the
# context, so creative variation is only an opportunity to drift from them.
DEFAULT_TEMPERATURE = 0.2

# Gemini's role vocabulary. "system" never appears here — it goes into
# GenerateContentConfig.system_instruction instead, which is why the mapping
# below only has to cover two cases.
_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiLLM:
    """LLMProvider backed by Google's Gemini API.

    Satisfies the Protocol structurally; does not inherit it.
    """

    name = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            # Fail at construction, not mid-stream. The factory builds this at
            # startup, so a missing key surfaces as a boot error rather than as
            # a user's chat request dying halfway through a sentence.
            raise ProviderAuthError("GEMINI_API_KEY is not set", provider=self.name)
        self._api_key = api_key
        self.model = model
        self._client: Client | None = None

    def _get_client(self) -> "Client":
        """Lazily construct the SDK client.

        No lock needed, unlike the local models: this is a cheap object holding
        config and an HTTP session, so an accidental double-construction costs
        almost nothing. Contrast with the embedder, where two concurrent loads
        would mean two copies of the weights in RAM.
        """
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def stream(
        self,
        system: str,
        messages: list[Message],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield answer text chunk by chunk.

        This is an async GENERATOR: calling it runs none of this body. Everything
        below — including the validation and any auth failure — happens on the
        first `async for`. Callers must wrap the iteration in try/except, not
        the call.
        """
        if not messages:
            raise ProviderInvalidRequestError(
                "at least one message is required", provider=self.name
            )
        if max_tokens <= 0:
            raise ProviderInvalidRequestError(
                f"max_tokens must be positive, got {max_tokens}", provider=self.name
            )

        from google.genai import types

        client = self._get_client()
        contents = self._to_gemini_contents(messages)
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            max_output_tokens=max_tokens,
            temperature=DEFAULT_TEMPERATURE,
        )

        try:
            # Await first, THEN iterate — see the API note in the module
            # docstring. This is the version-sensitive line in this file.
            stream = await client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                text = chunk.text
                # Chunks can legitimately carry no text — safety verdicts,
                # usage metadata, or an empty final chunk. Yielding None here
                # would break string concatenation downstream.
                if text:
                    yield text
        except ProviderError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    # -- internals ----------------------------------------------------------

    def _to_gemini_contents(self, messages: list[Message]) -> list[object]:
        """Convert our Messages into Gemini's Content objects.

        A `system` role reaching this point is a caller bug: the system prompt
        belongs in `system_instruction`, and silently folding it in as a user
        turn would change the model's behaviour in a way that is very hard to
        notice later.
        """
        from google.genai import types

        contents: list[object] = []
        for msg in messages:
            role = _ROLE_MAP.get(msg.role)
            if role is None:
                raise ProviderInvalidRequestError(
                    f"role {msg.role!r} cannot be sent as a message; "
                    "the system prompt goes in system_instruction",
                    provider=self.name,
                )
            contents.append(
                types.Content(role=role, parts=[types.Part(text=msg.content)])
            )
        return contents

    def _translate(self, exc: Exception) -> ProviderError:
        """Map an SDK exception onto our error hierarchy.

        The mapping is by HTTP status because that is the stable contract;
        message strings change without warning between SDK releases.
        """
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            code = getattr(exc, "code", None)
            message = getattr(exc, "message", None) or str(exc)

            if code in (401, 403):
                return ProviderAuthError(message, provider=self.name)
            if code == 429:
                # The one that actually happens on a free tier.
                return ProviderRateLimitError(message, provider=self.name)
            if code == 408 or code == 504:
                return ProviderTimeoutError(message, provider=self.name)
            if code is not None and 400 <= code < 500:
                # Our fault: context too long, bad model name, blocked prompt.
                # Deliberately NOT retryable — failing over to Groq would just
                # reproduce the same bad request against a second provider.
                return ProviderInvalidRequestError(message, provider=self.name)
            return ProviderUnavailableError(message, provider=self.name)

        if isinstance(exc, TimeoutError):
            return ProviderTimeoutError(str(exc), provider=self.name)

        # Unknown failure: treat as retryable. A transient network error is far
        # more likely here than a logic bug, and the fallback provider is a
        # cheap second opinion.
        return ProviderUnavailableError(
            f"unexpected Gemini failure: {exc}", provider=self.name
        )

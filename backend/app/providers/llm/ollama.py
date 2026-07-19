"""Ollama LLM provider (plain HTTP via httpx — no vendor SDK).

The local/offline generator (PLAN §2): an Ollama daemon running on the developer
machine, serving an open-weights model. Its value is that it costs nothing and
needs no network, so tests and experiments never touch a rate-limited free tier.

Deliberately no SDK. Ollama is an ordinary JSON-over-HTTP API and `httpx` is
already a dependency (FastAPI's TestClient uses it), so a sixth vendor package
would buy nothing but another thing to keep upgraded. Hard Rule #2 is about
*isolation*, not about SDKs specifically — the HTTP calls are confined here for
the same reason `google.genai` is confined to gemini.py.

Wire format notes:

- `POST /api/chat` takes the same `{role, content}` message list as Groq, so
  `_to_chat_messages` is near-identical. The system prompt goes in-band.
- With `"stream": true` the response is **NDJSON**: one complete JSON object per
  line, terminated by an object with `"done": true`. This is not SSE — there is
  no `data:` prefix and no blank-line separator, so lines are parsed directly.
- Generation limits live under `options`, and the output cap is called
  `num_predict`, not `max_tokens`.
"""

import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.providers.base import (
    Message,
    ProviderError,
    ProviderInvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)

# Small enough to run on a laptop without a GPU, good enough to exercise the
# grounded-answer prompt end to end.
DEFAULT_MODEL = "llama3.2"

# Same reasoning as the hosted providers: the facts are in the retrieved
# context, so creativity only buys drift away from them.
DEFAULT_TEMPERATURE = 0.2

# Generous by hosted-provider standards, and it needs to be: a cold local model
# is read off disk into RAM/VRAM on first request, which can take tens of
# seconds. Too tight a timeout here turns a normal first run into a failure.
DEFAULT_TIMEOUT_SECONDS = 300.0

# No renaming needed — Ollama uses the same role vocabulary we do. The set
# exists only to reject `system` arriving in the wrong place.
_ALLOWED_ROLES = {"user", "assistant"}


class OllamaLLM:
    """LLMProvider backed by a local Ollama daemon.

    Satisfies the Protocol structurally; does not inherit it.
    """

    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not base_url:
            # Not an auth error: Ollama has no credentials. This is a
            # misconfiguration, and it should stop the process at startup.
            raise ProviderInvalidRequestError(
                "OLLAMA_BASE_URL is not set", provider=self.name
            )
        # rstrip so both "http://host:11434" and ".../" produce one clean join.
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def stream(
        self,
        system: str,
        messages: list[Message],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Yield answer text chunk by chunk.

        This is an async GENERATOR: calling it runs none of this body. The
        validation below, and any connection failure, happen on the first
        `async for`. Callers must wrap the *iteration* in try/except.
        """
        if not messages:
            raise ProviderInvalidRequestError(
                "at least one message is required", provider=self.name
            )
        if max_tokens <= 0:
            raise ProviderInvalidRequestError(
                f"max_tokens must be positive, got {max_tokens}", provider=self.name
            )

        payload = {
            "model": self.model,
            "messages": self._to_chat_messages(system, messages),
            "stream": True,
            "options": {
                "temperature": DEFAULT_TEMPERATURE,
                "num_predict": max_tokens,
            },
        }

        try:
            # The client is built per call rather than cached on self. An
            # AsyncClient binds to the event loop that created it, and a
            # long-lived one on a provider constructed at import time would
            # eventually be used from a different loop (tests, Celery workers)
            # and fail in a confusing way. A local connection is cheap enough
            # that pooling is not worth that risk.
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/api/chat", json=payload
                ) as response:
                    if response.status_code >= 400:
                        # The body has not been read yet in streaming mode, so
                        # it must be pulled in explicitly before it can be
                        # included in the error message.
                        await response.aread()
                        raise self._translate_status(response)

                    async for line in response.aiter_lines():
                        text = self._extract_text(line)
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
        """Build the message list. System prompt is in-band, as the first entry.

        A `system` role inside `messages` is a caller bug: it would produce two
        system turns with undefined precedence.
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

    def _extract_text(self, line: str) -> str | None:
        """Parse one NDJSON line into its text delta, or None if it carries none.

        Blank lines, the terminating `"done": true` object, and any frame
        without message content all return None. A malformed line is logged and
        skipped rather than raised: one bad frame should not destroy an answer
        the user is already reading.
        """
        line = line.strip()
        if not line:
            return None
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("ollama: skipping unparseable stream line: %r", line[:200])
            return None

        # Errors can arrive mid-stream with HTTP 200 already sent — e.g. the
        # model is unloaded partway through. This must not be swallowed.
        if error := frame.get("error"):
            raise ProviderUnavailableError(str(error), provider=self.name)

        return frame.get("message", {}).get("content") or None

    def _translate_status(self, response: httpx.Response) -> ProviderError:
        """Map an HTTP error response onto our error hierarchy."""
        detail = response.text[:500] or response.reason_phrase
        code = response.status_code

        if code == 404:
            # Almost always "model not found" — the model was never pulled.
            # Not retryable: `ollama pull` is a human action, and failing over
            # to another provider would hide a setup mistake.
            return ProviderInvalidRequestError(
                f"model {self.model!r} not available on the Ollama host "
                f"(try `ollama pull {self.model}`): {detail}",
                provider=self.name,
            )
        if code == 408 or code == 504:
            return ProviderTimeoutError(detail, provider=self.name)
        if 400 <= code < 500:
            return ProviderInvalidRequestError(detail, provider=self.name)
        return ProviderUnavailableError(detail, provider=self.name)

    def _translate(self, exc: Exception) -> ProviderError:
        """Map a transport-level exception onto our error hierarchy."""
        if isinstance(exc, httpx.TimeoutException):
            return ProviderTimeoutError(str(exc), provider=self.name)
        if isinstance(exc, httpx.ConnectError):
            # By far the most common failure: the daemon is not running.
            # Retryable in the formal sense — a hosted fallback would succeed —
            # so the message has to carry the real cause for the logs.
            return ProviderUnavailableError(
                f"cannot reach the Ollama daemon at {self.base_url} "
                f"(is `ollama serve` running?): {exc}",
                provider=self.name,
            )
        if isinstance(exc, httpx.HTTPError):
            return ProviderUnavailableError(str(exc), provider=self.name)
        if isinstance(exc, TimeoutError):
            return ProviderTimeoutError(str(exc), provider=self.name)

        return ProviderUnavailableError(
            f"unexpected Ollama failure: {exc}", provider=self.name
        )

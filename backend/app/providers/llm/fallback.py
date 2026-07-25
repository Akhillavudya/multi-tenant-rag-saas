"""Automatic LLM failover: try the configured provider, then a backup.

PLAN §3 rule 3, made concrete: "providers raise a common `ProviderError`; the
API layer can fall back Gemini->Groq automatically." This is where that happens,
and the reason it can happen at all is Phase 1's error hierarchy — every vendor
already translates its own exceptions into ours, and every one of them sets
`retryable`. Without that groundwork this file would be a pile of
`except google.genai.errors.APIError` clauses that grow with each new vendor.

Composition, not inheritance: `FallbackLLM` holds two `LLMProvider`s and is
itself an `LLMProvider`. Nothing above it — `core/generation.py`, the chat
endpoint — can tell the difference between this and a single provider, which is
what keeps failover out of the business logic.

**The two rules that decide whether to fail over.**

1. *Only before the first token.* Once a fragment has reached the client, the
   answer has started; restarting on another provider would splice two different
   answers into one sentence, and the user would watch their reply contradict
   itself mid-flow. After the first token, the error propagates and the stream
   ends early — a truncated answer is recoverable by asking again, a spliced one
   is confusing in a way the user cannot diagnose.

2. *Only when `retryable` is true.* Rate limits, timeouts and 5xx are the free
   tier's normal weather and another provider genuinely helps. A bad API key or
   an over-length prompt is OUR bug: retrying it burns the backup's quota to
   reproduce the identical failure, and — worse — hides the misconfiguration
   behind an answer that silently comes from the wrong model.

**What this deliberately does not do.** No retry against the same provider, no
circuit breaker, no health tracking. Each would help; each is also a decision
that needs data we do not have yet, and a failover path complicated enough to
have its own bugs is worse than one simple enough to reason about at 3am. The
observable cost of the simple version is one extra failed request per outage
window, which the logs make obvious.
"""

import logging
from collections.abc import AsyncIterator

from app.providers.base import LLMProvider, Message, ProviderError

logger = logging.getLogger(__name__)


class FallbackLLM:
    """An `LLMProvider` that tries `primary`, then `secondary`.

    Satisfies the Protocol structurally, like every other provider here. Holds
    already-built providers rather than names, so the factory owns construction
    and this class owns only the decision.
    """

    def __init__(self, primary: LLMProvider, secondary: LLMProvider) -> None:
        if primary.name == secondary.name:
            # Falling back onto the provider that just failed reproduces the
            # outage and doubles the latency. Refused at construction, where it
            # is a config error, rather than at 3am, where it is a mystery.
            raise ValueError(
                f"fallback provider must differ from the primary "
                f"(both are {primary.name!r})"
            )
        self._primary = primary
        self._secondary = secondary

    @property
    def name(self) -> str:
        """Reported as `gemini->groq`, so a log line says which pair was in play
        rather than an opaque "fallback"."""
        return f"{self._primary.name}->{self._secondary.name}"

    @property
    def primary(self) -> LLMProvider:
        return self._primary

    @property
    def secondary(self) -> LLMProvider:
        return self._secondary

    async def stream(
        self,
        system: str,
        messages: list[Message],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stream from the primary, switching to the secondary if it fails
        cleanly before producing anything.

        An async generator, like every provider's `stream`: nothing below runs
        until the caller starts iterating, so a caller that wants to catch
        provider errors must wrap the `async for`, not the call.
        """
        delivered = False
        try:
            async for fragment in self._primary.stream(
                system=system, messages=messages, max_tokens=max_tokens
            ):
                # Set before the yield: if the consumer disconnects while we are
                # suspended here, the fragment has already left this generator.
                delivered = True
                yield fragment
            return
        except ProviderError as exc:
            if delivered:
                # Mid-stream. Rule 1: no splicing. The client keeps the partial
                # answer and the error ends the stream.
                logger.warning(
                    "provider failed mid-stream; not failing over",
                    extra={"provider": self._primary.name, "error": str(exc)},
                )
                raise
            if not exc.retryable:
                # Rule 2: our bug, not their outage.
                logger.error(
                    "provider failed unretryably; not failing over",
                    extra={"provider": self._primary.name, "error": str(exc)},
                )
                raise
            primary_error = exc
            logger.warning(
                "falling back to secondary LLM provider",
                extra={
                    "primary": self._primary.name,
                    "secondary": self._secondary.name,
                    "reason": type(exc).__name__,
                    "error": str(exc),
                },
            )

        # Outside the except block on purpose: yielding from inside one would
        # keep the primary's exception as the "currently handled" exception for
        # the whole second stream, and any unrelated error raised later would
        # arrive with a confusing chained traceback.
        try:
            async for fragment in self._secondary.stream(
                system=system, messages=messages, max_tokens=max_tokens
            ):
                yield fragment
        except ProviderError as exc:
            logger.error(
                "both LLM providers failed",
                extra={
                    "primary": self._primary.name,
                    "secondary": self._secondary.name,
                    "primary_error": str(primary_error),
                    "secondary_error": str(exc),
                },
            )
            # Chained deliberately: the secondary's failure is what the caller
            # sees, and `__cause__` preserves why we were on the secondary at
            # all. Without it, an operator reading the traceback would never
            # learn that the primary is the one that is down.
            raise exc from primary_error

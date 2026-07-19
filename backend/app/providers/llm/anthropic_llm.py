"""Anthropic LLM provider — PREMIUM STUB, not yet implemented.

Reserves the `anthropic` slot so PLAN §3 rule 4 (a paying client runs Claude
while everyone else runs Gemini, same codebase) stays a fill-in-the-blank rather
than a redesign under deadline. The env var, the factory branch and the config
field all exist from today; only the body below is missing.

Why a stub at all, instead of adding the file when it is needed? Because the
*shape* is the part worth committing early. Writing it now proves the
`LLMProvider` Protocol survives contact with a third vendor, and it does: nothing
about Claude's API needs a change to `base.py`. A contract that has only ever
been implemented by providers written in the same afternoon has not been tested.

**Not** listed in pyproject dependencies: `anthropic` is deliberately uninstalled
so nobody pays ~10MB and an import for a code path that raises. The SDK import
therefore lives inside the method, where it would fail loudly and locally.

To finish this file:
1. Add `anthropic>=0.40.0` to pyproject dependencies.
2. `client.messages.stream(...)` — Anthropic takes the system prompt
   out-of-band as a top-level `system=` parameter (like Gemini, unlike Groq),
   and `max_tokens` is REQUIRED, not optional.
3. Translate `anthropic.APIStatusError` by `.status_code` exactly as
   groq_llm.py does — the taxonomy is near-identical.
"""

from collections.abc import AsyncIterator

from app.providers.base import Message, ProviderAuthError

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicLLM:
    """LLMProvider backed by Anthropic's Messages API. Streaming not yet built.

    Constructs successfully on purpose: the factory tests instantiate every
    provider named in the registry, and raising here would force them to
    special-case the unfinished ones — which is exactly the kind of exception
    that quietly becomes permanent. The failure is deferred to `stream()`, where
    it is unmissable and cannot be mistaken for a configuration problem.
    """

    name = "anthropic"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY is not set", provider=self.name)
        self._api_key = api_key
        self.model = model

    async def stream(
        self,
        system: str,
        messages: list[Message],
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Not implemented. See the module docstring for the completion steps."""
        raise NotImplementedError(
            "the Anthropic provider is a stub; set LLM_PROVIDER=gemini, groq or "
            "ollama, or implement app/providers/llm/anthropic_llm.py"
        )
        # Unreachable, and that is the point: without a `yield` in the body this
        # is a plain coroutine, so the NotImplementedError would surface when the
        # caller *calls* stream() — while the real implementation, an async
        # generator, would raise only on first iteration. Keeping the yield here
        # means the stub fails at the same moment the finished version would.
        yield ""  # pragma: no cover

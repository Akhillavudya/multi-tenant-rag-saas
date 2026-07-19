"""OpenAI LLM provider — PREMIUM STUB, not yet implemented.

Reserves the `openai` slot. Mechanism and reasoning are identical to
anthropic_llm.py and are not repeated here — see that module's docstring for why
the stub constructs successfully, why the SDK is not a dependency, and why the
unreachable `yield` matters.

Kept as a separate stub rather than folded in with Anthropic because
`LLM_PROVIDER` selects exactly one, and because this one pairs with
`embeddings/openai_embed.py`: a client standardising on OpenAI usually wants both
the generator and the embedder, but the two are selected by *different* env vars
and must remain independently swappable. Nothing about running GPT for
generation implies re-embedding a corpus, and the config should not imply it
either.

To finish this file:
1. Add `openai>=1.60.0` to pyproject dependencies.
2. `AsyncOpenAI.chat.completions.create(..., stream=True)` — this is the wire
   format Groq copied, so `groq_llm.py` is the template for this file almost
   line for line, including the await-then-iterate trap.
3. Translate `openai.APIStatusError` by `.status_code`. Watch the same subclass
   ordering trap: `APITimeoutError` is a subclass of `APIConnectionError`.
"""

from collections.abc import AsyncIterator

from app.providers.base import Message, ProviderAuthError

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLM:
    """LLMProvider backed by OpenAI's chat completions API. Not yet built.

    Constructs successfully on purpose; the failure is deferred to `stream()`.
    See anthropic_llm.py for the full reasoning.
    """

    name = "openai"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        if not api_key:
            raise ProviderAuthError("OPENAI_API_KEY is not set", provider=self.name)
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
            "the OpenAI provider is a stub; set LLM_PROVIDER=gemini, groq or "
            "ollama, or implement app/providers/llm/openai_llm.py"
        )
        # Load-bearing despite being unreachable — see anthropic_llm.py.
        yield ""  # pragma: no cover

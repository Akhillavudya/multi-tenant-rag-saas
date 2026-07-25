"""Prompt templates, loaded from disk and rendered.

Prompts live as `.txt` files next to this module rather than as string literals
inside it (PLAN §3). They are edited on a different rhythm from code, by people
who may not read Python, and Phase 4 makes the system prompt editable per
chatbot — which only works if the prompt is already data. This module is the one
place that turns those files into strings.

The important thing here is `REFUSAL`. The prompt instructs the model to answer
with an exact sentence when the context does not contain the answer, and the
application needs to *detect* that sentence — to count refusals, to avoid
billing a message that answered nothing, and in Phase 3 to decide whether a
provider fallback is worth attempting. That makes the sentence a contract
between a `.txt` file and this code, and contracts between files drift. So the
loader verifies it, and fails at startup rather than silently losing the ability
to recognise a refusal.
"""

from functools import cache
from importlib.resources import files
from string import Template

GROUNDED_ANSWER = "grounded_answer.txt"

# Must match the prompt file exactly. `_load` asserts this on every load, so a
# reworded prompt fails loudly here instead of quietly breaking refusal
# detection everywhere downstream.
REFUSAL = "I don't have that information."


class PromptError(Exception):
    """A prompt file is missing, unreadable, or no longer matches its contract."""


@cache
def load_prompt(name: str = GROUNDED_ANSWER) -> str:
    """Read a prompt template from this package. Cached per process.

    Cached because a chat request must not touch the disk to fetch a file that
    never changes at runtime. The cost is that editing a prompt requires a
    restart — correct for production, and worth knowing during development.

    `importlib.resources` rather than `Path(__file__).parent`: it resolves
    package data correctly even when the application is packaged as a zip or a
    wheel, where `__file__` does not point at a real directory.
    """
    if "/" in name or "\\" in name or ".." in name:
        # These names come from code, not user input. The guard costs nothing
        # and means that stays true even if a future caller passes one through.
        raise PromptError(f"invalid prompt name: {name!r}")

    try:
        text = files(__package__).joinpath(name).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptError(f"prompt template {name!r} not found") from exc

    if not text.strip():
        raise PromptError(f"prompt template {name!r} is empty")

    if name == GROUNDED_ANSWER and REFUSAL not in text:
        raise PromptError(
            f"{name!r} no longer contains the exact refusal string "
            f"{REFUSAL!r}. Refusal detection depends on the prompt and the "
            "REFUSAL constant matching; update both together."
        )

    return text


def render_grounded_answer(context: str, persona: str | None = None) -> str:
    """Build the system prompt for one grounded answer.

    Takes the already-formatted context block (see `generation.build_context`),
    not the raw chunks: deciding how a chunk is labelled and cited is generation's
    job, and this module only fills a slot.

    `persona` is the chatbot's own `system_prompt` — the owner's description of
    the bot's voice and scope. It is inserted into a STYLE block ABOVE the rules,
    never appended below them, and the template's rule 11 states explicitly that
    style cannot relax grounding. Order is the cheap half of that defence:
    instructions nearer the end of a system prompt carry more weight, so the
    rules that must win are the ones that finish.

    That is mitigation, not a guarantee. A determined bot owner can still write a
    persona that degrades answer quality — which is acceptable, because it is
    THEIR bot and their documents. The threat this guards against is a persona
    that silently turns off citation or grounding and leaves the product looking
    like it hallucinates.

    `Template.substitute`, not `str.format`: prompt text accumulates literal
    braces over time — JSON examples, output formats, code snippets — and
    `format` raises on every one of them, turning a prose edit into a crash.
    Both are safe against hostile document content, since both scan the template
    and not the substituted values.
    """
    if not context.strip():
        # An empty context with a prompt that says "answer only from the
        # context" produces a confident refusal, which looks like a correct
        # answer and hides the real bug upstream in retrieval.
        raise PromptError("cannot render a grounded prompt with empty context")

    # The whole block is built here rather than in the template so that a bot
    # with no custom prompt gets no empty heading — a dangling "STYLE" with
    # nothing under it is the kind of prompt debris models react to oddly.
    persona_block = (
        f"STYLE\n\n{persona.strip()}\n" if persona and persona.strip() else ""
    )

    try:
        return Template(load_prompt(GROUNDED_ANSWER)).substitute(
            context=context, persona=persona_block
        )
    except KeyError as exc:
        # A placeholder in the file that this function does not supply.
        raise PromptError(
            f"prompt template expects an unknown placeholder: {exc}"
        ) from exc


def is_refusal(answer: str) -> bool:
    """Whether the model declined to answer from the context.

    Matches on a normalised prefix rather than exact equality. The model
    reliably *starts* with the refusal but may append a clarifying sentence, and
    treating "I don't have that information. The documents cover X instead."
    as an answer would inflate the answered-question rate with non-answers.

    The tradeoff is a genuine false positive: a reply that opens with the
    refusal and then answers anyway is counted as a refusal. That is the safer
    direction — over-counting refusals makes retrieval look worse than it is and
    prompts investigation, while under-counting hides the failure entirely.
    """
    return answer.strip().casefold().startswith(REFUSAL.casefold())

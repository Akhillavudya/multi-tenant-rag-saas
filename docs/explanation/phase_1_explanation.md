# Phase 1 — Provider layer & core RAG: file-by-file explanation

Build depth for every file created in Phase 1, in the order it was created.
Concept-level revision notes live in `LEARNING.md`; this file is about *these
specific files* and why they look the way they do.

**Phase 1 goal (PLAN.md §6):** build the provider abstraction of §3 exactly
(interfaces, factory, Gemini + Groq + Ollama LLMs, bge_local embedder,
bge_local + noop rerankers, premium stubs), then `app/core/`: chunking,
ingestion, retrieval (top-40 → rerank → top-6), generation (grounded prompt with
citations), plus a CLI to drive it end-to-end.
**Done when:** ingest a PDF and get a fast, cited, accurate streamed answer in
the terminal; `LLM_PROVIDER=groq` works with zero code edits;
`RERANK_PROVIDER=noop` visibly degrades answer quality.

---

## 1. `backend/app/providers/base.py`

**Purpose** — Declares the three contracts (`LLMProvider`, `EmbeddingProvider`,
`RerankProvider`) plus the shared `ProviderError` family that every vendor
adapter translates its SDK errors into. Without this file there is no shared
shape to program against: `core/retrieval.py` would import the Qdrant and
sentence-transformers SDKs directly, and Hard Rule #2 would be unenforceable —
"swap the provider" would mean editing business logic.

**Analogy** — A wall socket. The socket defines the shape (two pins, 230V); any
appliance with that plug works. The power station behind it can switch from coal
to solar and your lamp never learns about it. `base.py` is the socket; the files
under `llm/`, `embeddings/`, `rerank/` are power stations.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `ProviderError` | Base exception. Carries `provider` (which vendor failed) and the class-level `retryable` flag. | Raised inside adapters; caught in `core/` and the API layer. |
| `ProviderAuthError` | Bad/missing key. `retryable = False`. | On 401/403 from a vendor. |
| `ProviderRateLimitError` | Quota or RPM exceeded. `retryable = True`. | On 429 — the common one on free tiers. |
| `ProviderTimeoutError` | Deadline blown. `retryable = True`. | When the adapter's own timeout fires. |
| `ProviderUnavailableError` | 5xx, connection refused, Ollama still loading a model. `retryable = True`. | On transport failures. |
| `ProviderInvalidRequestError` | Context too long, bad model name, unsupported param. `retryable = False`. | On 400 — our bug, not theirs. |
| `Role` | `Literal["system","user","assistant"]`. | Type-check time only. |
| `Message` | Frozen dataclass `(role, content)`. Vendor-neutral conversation turn. | Built in `core/generation.py`, consumed by LLM adapters. |
| `ScoredIndex` | Alias for `tuple[int, float]` — `(index_into_docs, score)`. | Returned by rerankers. |
| `LLMProvider` | Protocol: `name`, `async stream(system, messages, max_tokens) -> AsyncIterator[str]`. | Implemented by `llm/*.py`. |
| `EmbeddingProvider` | Protocol: `name`, `model`, `dimension`, `async embed_documents`, `async embed_query`. | Implemented by `embeddings/*.py`. |
| `RerankProvider` | Protocol: `name`, `async rerank(query, docs, top_k) -> list[ScoredIndex]`. | Implemented by `rerank/*.py`. |

### Flow of the code

This file has **no runtime flow of its own** — nothing here executes during a
request. It runs exactly once, at import time, to create the exception classes,
the `Message` dataclass, and the Protocol objects. Its real "flow" is the
*type-check-time and design-time* one:

1. `factory.py` declares its return type as `LLMProvider`.
2. `core/generation.py` accepts an `LLMProvider` parameter and calls `.stream()`.
3. `llm/gemini.py` defines a class with a matching `stream` signature — **without
   importing this file for the Protocol** (it does import the exceptions).
4. mypy/your IDE checks structurally that Gemini's class fits the Protocol.
5. At runtime Python just calls the method; the Protocol was never involved.

The exception flow *is* real at runtime:
vendor SDK raises its own error → adapter catches it → re-raises the matching
`ProviderError` subclass → `core/` either falls back (if `retryable`) or lets it
bubble → API layer maps it to a clean JSON HTTP error.

### Math
None.

### Common gotchas / traps

- **`Protocol` is not `ABC`.** Inheriting from a Protocol is optional and gives
  you nothing at runtime. There is no "you forgot to implement `stream`" error at
  class-definition time — a missing method surfaces as `AttributeError` on first
  call. Static checking (mypy) is what catches it early; that is the price of
  keeping the dependency arrow one-directional.
- **`@runtime_checkable` only checks method *names*, not signatures.** An
  `isinstance(x, LLMProvider)` passes for a class with a `stream` method that
  takes completely wrong arguments. Treat it as a smoke test, never as validation.
- **An `async def` returning `AsyncIterator[str]` that uses `yield` is an async
  *generator*.** Calling it does not run any code — the body only starts on the
  first `async for`. So a provider that raises `ProviderAuthError` inside
  `stream()` raises it at first iteration, not at call time. Wrap the `async for`,
  not the call, in your try/except.
- **Mutable default trap avoided:** `Message` is frozen, so it can be safely put
  in sets/dicts and can't be mutated after being logged to the transcript.
- **Scores are not comparable across providers.** bge-reranker returns logits
  (roughly −11…+11), Cohere returns 0…1. Any code that thresholds on an absolute
  score value breaks the moment you swap rerankers — order only.
- **`dimension` must live in the DB, not just here.** Two embedding models with
  the same dimension (many are 768) will happily write into the same Qdrant
  collection and return silent garbage. The per-bot `embedding_model` column in
  Phase 2 is what actually prevents this; `dimension` here only prevents the
  crash-loud case.

### Interview questions

**Q: `Protocol` vs `ABC` — when do you pick which?**
`ABC` uses nominal typing: a class must explicitly inherit, and Python enforces
that abstract methods are implemented at instantiation time. That's the right
choice when you own all the implementations and want a shared base with real
code in it. `Protocol` uses structural typing ("duck typing with a type
checker"): conformance is by shape, no inheritance, no import. It's the right
choice when you want the *consumer* to define the interface and keep
implementations independent of it — including third-party classes you can't make
inherit from anything. Here it means `providers/llm/gemini.py` has zero imports
from the abstraction, so the vendor layer is a true leaf and cannot accidentally
become a place where shared logic accumulates.

**Q: Why does the error hierarchy carry a `retryable` flag instead of the caller
just catching specific types?**
Because the retry decision is a property of the failure, not of the call site. If
callers switch on concrete types, then every call site must be updated when a new
error type is added — and each one will get the classification subtly different.
One flag centralises the policy: `except ProviderError as e: if e.retryable:
try_fallback()`. It also makes the Gemini→Groq fallback in Phase 3 safe by
construction: an auth error or an over-long context will never be retried on the
backup provider, so a bug in our prompt-building can't silently burn the second
provider's free quota too.

---

## 2. `backend/app/providers/embeddings/bge_local.py`

**Purpose** — The default `EmbeddingProvider`: loads `BAAI/bge-small-en-v1.5`
locally and turns text into 384-dimensional unit vectors. Without it there is
nothing to store in Qdrant and no way to search by meaning — retrieval would
collapse back to keyword matching, which can't tell that "refund policy" and
"money back rules" are the same question.

**Analogy** — A translator who converts any sentence into GPS coordinates on a
map of ideas. Sentences about the same topic land in the same neighbourhood, so
"find relevant text" becomes "find nearby points" — a geometry problem a
computer is very good at.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `MODEL_NAME` / `DIMENSION` | Pin the model id and its 384-dim output. | Import time. |
| `QUERY_PREFIX` | The BGE instruction glued to queries only. | Every `embed_query`. |
| `DEFAULT_BATCH_SIZE` | 32 texts per forward pass on CPU. | Every `_encode_sync`. |
| `__init__` | Stores config, sets `_model = None`, creates the load lock. **Loads nothing.** | Factory construction. |
| `warmup()` | Public "load now" hook for FastAPI lifespan / Celery boot. | App startup. |
| `_get_model()` | Double-checked-locked lazy load; verifies reported dimension. | First embed call, then cached. |
| `_load_model_sync()` | The actual `SentenceTransformer(...)` construction; imports torch late. | Once per process. |
| `embed_documents()` | Validates non-empty, batch-embeds chunks, preserves order. | Ingestion, per batch. |
| `embed_query()` | Prefixes, embeds one string, unwraps the single vector. | Every user question. |
| `_encode()` | Offloads to a thread, translates unknown exceptions to `ProviderError`. | Both public methods. |
| `_encode_sync()` | `model.encode(..., normalize_embeddings=True)` → `.tolist()`. | Inside the worker thread. |

### Flow of the code

**Startup path (ideal):**
1. Factory constructs `BGELocalEmbedder()` — cheap, no model, no torch import.
2. Lifespan calls `await warmup()` → `_get_model()`.
3. Lock acquired; `asyncio.to_thread(_load_model_sync)` imports
   `sentence_transformers` (which imports torch) and constructs the model.
4. First ever run downloads ~130MB to the HuggingFace cache; later runs read it.
5. `get_sentence_embedding_dimension()` is checked against `DIMENSION` — mismatch
   raises rather than proceeding.
6. `self._model` set; lock released.

**Ingestion path (`embed_documents`):**
1. Empty list → return `[]` immediately (no thread, no model).
2. Whitespace-only check → `ProviderInvalidRequestError`.
3. `_encode(texts)` → `_get_model()` (already loaded → instant return).
4. `asyncio.to_thread(_encode_sync, ...)`: the event loop is free while this runs.
5. Inside the thread: tokenise → transformer forward pass → mean-pool → L2
   normalise → numpy array of shape `(len(texts), 384)`.
6. `.tolist()` → `list[list[float]]`, one vector per input, same order.

**Query path (`embed_query`):**
Same as above but with exactly one text, prefixed with `QUERY_PREFIX`, and the
single vector unwrapped with `vectors[0]`.

### Math

**L2 normalisation** — each vector is divided by its own length:

    v̂ = v / ||v||    where    ||v|| = √(Σᵢ vᵢ²)

After this, `||v̂|| = 1` for every vector. That matters because cosine similarity

    cos(θ) = (A·B) / (||A||·||B||)

collapses to just `A·B` when both lengths are 1. So we pay the normalisation cost
**once at write time** and every one of the millions of later comparisons becomes
a plain dot product — `Σᵢ aᵢbᵢ`, 384 multiply-adds, no square roots. It also
makes scores stable: without normalisation a long chunk gets a large-magnitude
vector and wins the dot product regardless of meaning.

**Dimension** — 384 floats × 4 bytes = 1.5KB per vector before Qdrant's index
overhead. 10,000 chunks ≈ 15MB. Cheap; the reason to care about `dimension` is
correctness, not storage.

**Complexity** — encoding is O(n · L²·d) in attention terms for n texts of L
tokens; practically, linear in the number of texts with a fixed per-text cost.
Batching changes the constant, never the order.

### Common gotchas / traps

- **Forgetting `QUERY_PREFIX`.** No crash, no error — just quietly worse
  retrieval. This is the single most common bge mistake and it's invisible
  without an eval set. It's also why the Protocol has `embed_query` and
  `embed_documents` as separate methods rather than one `embed()`.
- **Applying the prefix to documents too.** Equally wrong in the other
  direction, and equally silent.
- **Blocking the event loop.** `await model.encode(...)` isn't a thing;
  `model.encode(...)` called directly inside `async def` blocks *every* request
  in the process for the duration of the batch. `asyncio.to_thread` is the fix.
- **Loading the model at import time.** Would make `import app.main` take
  seconds, make tests slow, and load weights inside processes that never embed.
  Lazy + explicit `warmup()` gives fast imports *and* no first-request penalty.
- **Missing the double-check inside the lock.** Without the second
  `if self._model is not None`, every coroutine queued on the lock loads its own
  copy after the first releases it.
- **Mixing models in one collection.** Two different 768-dim models produce
  incompatible vectors that Qdrant will happily store and compare — returning
  plausible-looking nonsense. The dimension check here only catches the *loud*
  case; the per-bot `embedding_model` column in Phase 2 catches the quiet one.
- **`np.float32` is not JSON-serialisable.** Hence `.tolist()`, not `list()`.
- **Model download needs network on first run.** In CI or on a plane a cold
  cache raises `ProviderUnavailableError`. Pre-warm the cache in the Docker image
  in Phase 7.

### Interview questions

**Q: Why do queries and documents get embedded differently by the same model?**
Because BGE (like E5) is trained *asymmetrically*: during contrastive training,
queries carried an instruction prefix and passages did not, so the model learned
two slightly different input conventions that map into one shared space. A search
query is also a different kind of text from a passage — short, interrogative,
often keyword-like — and the prefix tells the model "this is a search query,
encode it for matching against passages." Using the wrong convention doesn't
error; it just shifts the vector enough to lose relevant hits, which is why this
class exposes two methods and validates the distinction in code rather than
leaving it to the caller's memory.

**Q: You have an `async def` that needs to run a 200ms CPU-bound library call.
What do you do, and why does it work in Python despite the GIL?**
Wrap it in `await asyncio.to_thread(fn, ...)`. The event loop is single-threaded
and cooperative — any blocking call inside a coroutine stalls all other tasks —
so the work must leave the loop's thread. The GIL would normally make threads
useless for CPU work, but native extensions like torch and numpy *release* the
GIL while executing their C/BLAS kernels, so the heavy math genuinely runs in
parallel and the loop keeps serving. For pure-Python CPU work the GIL would still
bite and you'd need `ProcessPoolExecutor` (or, in our architecture, a Celery
worker) instead.

---

## 3. `backend/app/providers/embeddings/__init__.py`

**Purpose** — Declares `app/providers/embeddings/` to be a Python *package* so
that `app.providers.embeddings.bge_local` is a valid import path and setuptools
ships the directory on `pip install -e .`. Without it the module written in §2 is
just a file on disk that nothing can reach.

**Analogy** — The nameplate on an office door. The room and everything in it
already exist; without the plate the building directory doesn't list the room, so
no visitor can be directed to it.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| The file's existence | Makes the directory a *regular package*, discoverable by `setuptools.packages.find`. | Install time and import time. |
| The docstring | Documents the one rule: no imports here. | Never executes meaningfully. |
| Absence of re-exports | Keeps `import app.providers.embeddings` free of torch. | Every import in the project. |

### Flow of the code

1. `pip install -e .` runs `setuptools.packages.find(include=["app*"])`, which
   walks the tree collecting directories that contain `__init__.py`.
2. `app.providers.embeddings` is added to the installed package list.
3. At runtime, `import app.providers.embeddings.bge_local` causes Python to first
   import each parent package in order: `app` → `app.providers` →
   `app.providers.embeddings` → then the module itself.
4. Step 3 is why this file's *contents* matter: whatever is written here executes
   before `bge_local` is even reached, on every single import of anything beneath
   it.
5. Because it is empty, that cost is ~zero, and the torch import stays confined
   to `bge_local` where it belongs.

### Math
None.

### Common gotchas / traps

- **Namespace packages make this failure confusing.** Since Python 3.3, a
  directory *without* `__init__.py` can still be imported as an implicit
  namespace package — so `import app.providers.embeddings.bge_local` may work
  fine when you run from `backend/`, and then fail after install, because
  `packages.find` never collected the directory. Symptom: "works locally, missing
  in Docker." The fix is this file; the general rule is to prefer regular
  packages unless you specifically need namespace behaviour.
- **The convenience re-export trap.** Writing
  `from app.providers.embeddings.bge_local import BGELocalEmbedder` here gives a
  nicer import path and costs you a torch import in every process that touches
  the package — including Alembic migrations and unit tests for pure functions.
  Convenience at the package root is paid for by everything downstream.
- **Side effects in `__init__.py` are invisible.** Code here runs implicitly, on
  an import the reader can't see in the file they're looking at. It is the single
  worst place in a Python project to put logic.
- **Circular imports usually start here.** A package `__init__` that imports its
  own submodules, which import the package, is the classic cycle.

### Interview questions

**Q: What's the difference between a regular package and a namespace package, and
when does it bite?**
A regular package has `__init__.py`; a namespace package (PEP 420) is just a
directory Python is willing to treat as importable, allowing one logical package
to be split across multiple locations on `sys.path`. The bite is packaging:
`setuptools.find_packages()` only finds regular packages, so a namespace
directory imports fine in development (where the source tree is on the path) and
vanishes from the installed wheel. You get an `ImportError` in CI or production
for code that ran perfectly on your machine. Use `find_namespace_packages()` if
you genuinely want namespace behaviour — most projects don't, and should just add
`__init__.py`.

**Q: Why keep `__init__.py` empty rather than re-exporting the package's classes?**
Because `__init__.py` runs on *any* import beneath the package, so every symbol
you re-export makes its whole import chain unavoidable. Here that chain is torch:
a re-export would mean a test for `core/chunking.py` — pure string manipulation
with no model involved — pays a multi-second import and ~1GB of memory. For a
lightweight library, re-exporting a clean public API is good practice; for a
package whose modules have heavy or optional dependencies, laziness at the root
is the correct default, and the factory does the concrete import at the moment a
provider is actually selected.

---

## 4. `backend/pyproject.toml` (Phase 1 dependency edit)

**Purpose** — Adds every library Phase 1 needs to the project's dependency list,
so `pip install -e ".[dev]"` builds an environment where the provider layer can
actually run. Without this edit, `bge_local.py` from §2 is unrunnable code:
`import sentence_transformers` raises `ModuleNotFoundError`.

**Analogy** — The ingredient list at the top of a recipe. The method steps are
your code; without the shopping list, the kitchen is missing flour and no amount
of correct instructions produces bread.

*(The Phase 0 sections of this file cover the rest of `pyproject.toml` — build
system, ruff, black, pytest config. This section is only about the Phase 1
dependency additions.)*

### Main pieces

| Added dependency | Why it's here | Used by |
|---|---|---|
| `sentence-transformers>=3.3.0` | The heavyweight. Pulls torch + transformers (~2GB). Gives the bge-small bi-encoder, the bge-reranker cross-encoder, and the model's own tokenizer. | `embeddings/bge_local.py`, `rerank/bge_local.py`, `core/chunking.py` |
| `qdrant-client>=1.12.0` | Talks to the vector DB running in Docker. | `core/retrieval.py`, `core/ingestion.py` |
| `google-genai>=0.3.0` | Gemini SDK — the default LLM. The current unified SDK, superseding `google-generativeai`. | `llm/gemini.py` only |
| `groq>=0.13.0` | Free-tier fallback LLM, very fast inference. | `llm/groq_llm.py` only |
| `pypdf>=5.1.0` | Pure-Python PDF text extraction. | `core/ingestion.py` |
| `python-docx>=1.1.2` | `.docx` parsing. Note the import name is `docx`, not `python_docx`. | `core/ingestion.py` |
| `httpx>=0.27.0` | Async HTTP client. Promoted from dev to runtime. | `llm/ollama.py` |
| *(no `ollama` SDK)* | Ollama is a plain REST API; httpx covers it. One less vendor dependency on the offline path. | — |
| *(no `tiktoken`)* | Deliberately absent — see Math below. | — |

### Flow of the code

A declaration, not a program, but it does drive an ordered process:

1. `pip install -e ".[dev]"` reads `[project].dependencies` plus the `dev` extra.
2. pip resolves the graph — `sentence-transformers` drags in `torch`,
   `transformers`, `tokenizers`, `huggingface-hub`, `numpy`, `scipy`.
3. Wheels download and install (torch is the slow one, several hundred MB).
4. `-e` (editable) installs *this* project as a link to the source tree, so your
   edits take effect without reinstalling.
5. `[tool.setuptools.packages.find] include = ["app*"]` decides which directories
   ship — which is exactly why §3's `__init__.py` had to exist first.
6. At runtime `import sentence_transformers` now resolves; the model *weights*
   are still not present and download on first `warmup()`.

Note steps 5 and 6 are the two separate failure modes people conflate: a missing
`__init__.py` breaks *your* code being found, a cold HuggingFace cache breaks the
*model* being found. Different errors, different fixes.

### Math

**Why no `tiktoken`.** Chunking needs to count tokens to hit "~600 tokens with
15% overlap" (PLAN §1). Token counts are **tokenizer-specific**: tiktoken
implements OpenAI's BPE, while bge-small uses a BERT WordPiece vocabulary of
30,522. The same string can differ by 10–20% between them. Since the constraint
we actually care about is bge-small's hard 512-token input limit — text beyond it
is silently truncated, and truncated text embeds to a vector that misrepresents
the chunk — we must count with *the model's own* tokenizer, which ships inside
`transformers`. Counting with the wrong tokenizer means chunks that look
compliant and are quietly over the limit.

Overlap arithmetic, for reference: at 600 tokens with 15% overlap, stride is
600 × (1 − 0.15) = 510 tokens, so a document of *T* tokens produces roughly
⌈(T − 600) / 510⌉ + 1 chunks — about 18% more chunks than non-overlapping
splitting, which is the storage price paid for not severing sentences that
straddle a boundary.

### Common gotchas / traps

- **Install name ≠ import name.** `python-docx` imports as `docx`;
  `sentence-transformers` imports as `sentence_transformers`; `google-genai`
  imports as `google.genai`. Guessing wrong produces a `ModuleNotFoundError` that
  looks like a failed install when the install was fine.
- **`>=` is a floor, not a pin.** A fresh `pip install` months from now may
  resolve to a newer major version with breaking changes. Acceptable while
  developing; Phase 7 needs a lockfile (`uv.lock` / `pip-compile`) so the Oracle
  box installs byte-identical versions.
- **torch is the whole install cost.** ~2GB on disk. On Windows, CPU-only torch
  is the default from PyPI, which is what we want — a CUDA build would be several
  GB more for no benefit on this machine.
- **Adding a dependency does not install it.** Editing this file changes nothing
  until you re-run `pip install -e ".[dev]"`. Symptom: "I added it, still
  `ModuleNotFoundError`."
- **Duplicating a dep in both lists.** `httpx` was in `dev`; leaving it in both
  after promoting it to runtime isn't an error but is a lie about intent — the
  dev extra should list only what's needed to *develop*, so that a production
  image installing without `[dev]` still works.
- **Vendor SDKs here are a licence to import them *only* in `providers/`.**
  Nothing in `pyproject.toml` enforces Hard Rule #2 — that is a discipline
  question, and a grep in code review is the real check.

### Interview questions

**Q: What does `pip install -e .` actually do differently from `pip install .`?**
A normal install copies your package into `site-packages`, so the running code is
a snapshot and every source edit requires a reinstall. Editable install instead
writes a link (historically an `.egg-link` + `.pth` file, now a PEP 660 hook)
pointing `site-packages` back at your source tree, so imports resolve to the
files you're editing and changes take effect on the next process start. It is a
development convenience and should never be how production installs — production
wants the immutable copy, from a pinned lockfile.

**Q: Why not just `pip freeze > requirements.txt`?**
Because it conflates two different lists. `pyproject.toml` records your *direct*
dependencies and the version ranges you actually support — the intent. `pip
freeze` dumps the fully-resolved transitive closure of one machine at one moment
— the outcome. If you only keep the freeze output, nobody can tell which of the
80 packages you actually chose versus which came along with torch, and you can
never safely relax a version. The correct pairing is a declarative manifest
(`pyproject.toml`) plus a generated lockfile, with the lockfile used for
deploys and CI reproducibility.

---

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
- **Scores are not comparable across providers.** bge-reranker under
  sentence-transformers 5.x returns 0…1, older versions returned a raw logit, and
  Cohere returns its own 0…1 on a different scale. Any code that thresholds on an
  absolute score value breaks the moment you swap rerankers *or upgrade a
  library* — order only.
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

## 5. `backend/app/providers/rerank/bge_local.py`

**Purpose** — The default `RerankProvider`: takes the ~40 candidate chunks that
vector search returned and rescores each one against the query with a
cross-encoder, so only the ~6 genuinely relevant ones reach the LLM. Without it
the model receives padded, loosely-related context — and an LLM given weak
context produces confidently wrong answers rather than admitting ignorance.

**Analogy** — Vector search is skimming the spines on a library shelf and pulling
40 plausible books. The reranker is opening each one, reading the relevant page,
and keeping the 6 that actually address your question. Skimming scales to the
whole library; reading doesn't. So you skim wide, then read narrow.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `MODEL_NAME` | Pins `BAAI/bge-reranker-v2-m3` (~568M params, multilingual). | Import time. |
| `MAX_LENGTH` | 512-token truncation for the joined (query, doc) pair. | Model construction. |
| `DEFAULT_BATCH_SIZE` | 16 pairs per forward pass. The main latency knob. | Every `_score_sync`. |
| `__init__` | Config + `_model = None` + load lock. Loads nothing. | Factory construction. |
| `warmup()` | Preload hook for lifespan / Celery boot. | App startup. |
| `_get_model()` | Double-checked lazy load. | First rerank, then cached. |
| `_load_model_sync()` | Constructs `CrossEncoder(...)`; late torch import. | Once per process. |
| `rerank()` | Validates, scores, sorts by score desc, truncates to `top_k`. | Every chat request. |
| `_score()` | Thread offload + error translation. | Inside `rerank`. |
| `_score_sync()` | Builds `(query, doc)` pairs, `model.predict(...)`. | Inside the worker thread. |

### Flow of the code

1. `retrieval.search()` has 40 candidate chunk objects from Qdrant, each with
   text plus `document_id`, `page`, `source`.
2. It passes **just the texts** as `docs`, in a known order, with `top_k=6`.
3. Guard clauses: empty `docs` returns `[]` (no model touched); `top_k <= 0` or a
   blank query raises `ProviderInvalidRequestError`.
4. `_score()` calls `_get_model()` (already warm), then `asyncio.to_thread(...)`.
5. In the thread: `pairs = [(query, doc), ...]` — 40 pairs.
6. `model.predict(pairs)` runs 40 forward passes in batches of 16. Each pass
   tokenises `[CLS] query [SEP] document [SEP]`, runs the transformer, and the
   classification head emits **one relevance score** (a raw logit, which
   sentence-transformers 5.x then squashes through the model's default sigmoid).
7. Back on the loop: `enumerate(scores)` pairs each score with its **original
   index**, then `sorted(..., reverse=True)` orders by score.
8. `ranked[:top_k]` produces `[(17, 8.42), (3, 6.10), ...]`.
9. The caller uses those indexes to pick its own chunk objects — metadata intact,
   ready for citations.

Step 7 is the load-bearing line. Sorting the scores without carrying the indexes
along would destroy the mapping back to the source chunks, and the failure is
silent: you'd still get 6 chunks and a fluent answer, just citing the wrong pages.

### Math

**Bi-encoder (the embedder):** encodes each text independently into a vector,
then scores with a dot product on normalized vectors:

    score(q, d) = e_q . e_d      where each e = v / ||v||   (unit length)

Document vectors are computed **once at ingestion** and indexed. Query time is
one encode plus an approximate nearest-neighbour lookup, so cost barely grows
with corpus size.

**Cross-encoder (this file):** no separate vectors exist. The pair is encoded
jointly and a head produces a scalar:

    score(q, d) = W . h_CLS( [CLS] q [SEP] d [SEP] )

Nothing is precomputable — the representation depends on *both* texts, so it must
be recomputed for every (query, document) pair.

**Why that ordering wins.** Let *N* = corpus chunks, *K* = candidates reranked
(40), *L* = sequence length, *d* = hidden size.

- Bi-encoder search: about O(log N . d) with HNSW. Cheap, approximate.
- Cross-encoder over the whole corpus: **O(N . L^2 . d)** — for N = 100,000 that
  is 100,000 transformer passes per question. Completely infeasible.
- Cross-encoder over K candidates: **O(K . L^2 . d)**, constant in N.

So pipeline cost is `O(log N . d) + O(K . L^2 . d)`. The first term finds
*probably* relevant text at corpus scale; the second decides *precisely* which of
those is best, at fixed cost. That recall-then-precision split is the whole
design.

**Attention is why it is more accurate.** Self-attention over the joined sequence
compares all tokens pairwise, so the document phrase "refund within 30 days" is
directly compared against the query words "how long". The bi-encoder had to
compress the entire chunk into 384 numbers *before ever seeing the question* —
any nuance it discarded is unrecoverable.

**Latency, honestly.** 40 pairs of 512 tokens through a 568M-parameter model on
CPU is the slowest step in the chat path — expect seconds, not milliseconds.
Levers: lower `MAX_LENGTH`, raise `DEFAULT_BATCH_SIZE`, cut candidates from 40 to
20, or switch to `BAAI/bge-reranker-base` (278M, English-only, roughly 2x
faster). Measure before choosing; PLAN section 7's p50/p95 work exists for
exactly this.

### Common gotchas / traps

- **Losing the index mapping.** `sorted(scores)` instead of
  `sorted(enumerate(scores), key=...)` silently decouples results from source
  chunks. Citations point at the wrong pages; nothing raises.
- **Forgetting `reverse=True`.** Higher score means more relevant. An ascending
  sort returns the six *least* relevant chunks — and the LLM will still answer
  fluently, which is what makes it hard to spot.
- **Assuming a score range instead of measuring it.** This was gotten wrong once
  already while writing this project: the comment in `bge_local.py` originally
  claimed raw logits in roughly -11..+11, and the first real run returned 0.977
  and 0.000 — sentence-transformers 5.x applies the model's default sigmoid
  inside `predict()`, older versions did not. A `score > 0.5` threshold would
  therefore behave differently across library versions, and differently again
  under Cohere Rerank. The range is a property of the version and config, never
  of the contract. Sort by scores; never threshold on them.
- **Reranking before filtering by tenant.** In Phase 2 the `org_id` filter must
  be applied at the *Qdrant* level. Reranking a cross-tenant candidate set and
  filtering afterwards would still have pulled another org's text into this
  request's processing — a leak even if it never reaches the response.
- **Passing the whole corpus.** The interface accepts any list; nothing stops a
  caller passing 10,000 docs and hanging the worker. Two-stage discipline lives
  in `retrieval.py`, not here.
- **Blocking the event loop** — same trap as the embedder, worse consequences:
  this model is about 4x larger, so the stall is longer.
- **Stable sort as a free tiebreak.** Python's sort is stable, so equal scores
  preserve the vector-search order. That is a sensible fallback, and it is
  accidental — worth knowing it is there.

### Interview questions

**Q: Explain the difference between a bi-encoder and a cross-encoder, and why a
RAG pipeline uses both.**
A bi-encoder maps query and document into a shared vector space *independently*.
Because documents never depend on the query, their vectors are computed once at
ingestion and stored in an ANN index, making retrieval sub-linear in corpus size
— but the model must compress each document before knowing what will be asked, so
matching is approximate. A cross-encoder concatenates query and document into one
sequence and runs full self-attention across both, so every query token attends
to every document token; it is substantially more accurate but produces nothing
reusable, requiring a forward pass per pair at query time. Using both gives you
the bi-encoder's scalability for *recall* (find 40 probably-relevant chunks out
of millions) and the cross-encoder's accuracy for *precision* (pick the best 6),
with total cost `O(log N) + O(K)` rather than `O(N)` — the classic
retrieve-then-rerank cascade.

**Q: Your reranker adds 2 seconds of latency. How do you decide whether to keep
it?**
Measure both quality and latency, and refuse to trade on vibes. Build a fixed
eval set of about 20 real questions with known correct source chunks, then
compare `RERANK_PROVIDER=bge_local` against `noop` on retrieval metrics —
Recall@6, MRR, or hit-rate of the correct chunk in the final context — while
recording p50 and p95 end-to-end latency for each. That is precisely why the
`noop` provider exists and why PLAN section 6 makes "noop visibly degrades
quality" a done-when criterion: the null implementation makes the reranker's
contribution measurable rather than assumed. If the quality gain is large, keep
it and attack latency (smaller model, fewer candidates, shorter max_length, GPU).
If Recall@6 barely moves, the honest engineering answer is to drop it — and
having the number means you can defend either decision.

---

## 6. `backend/app/providers/rerank/__init__.py`

**Purpose** — Declares `app/providers/rerank/` a Python package so
`app.providers.rerank.bge_local` is importable and setuptools ships the
directory. Without it, section 5's reranker is unreachable code.

**Analogy** — The second nameplate on the second office door. Same building, same
rule.

**Mechanism is identical to section 3** (`embeddings/__init__.py`) — regular vs
namespace packages, why `packages.find` needs the file, why side effects in
`__init__.py` are invisible. That is all documented there and is not repeated
here; read section 3 first if this file looks arbitrary.

### What is different here

One thing genuinely differs, and it is a judgement call worth recording.

The `rerank` package contains **two** implementations with very different
weights: `bge_local` (drags in torch, ~2GB) and `noop` (pure Python, zero
dependencies). So a tempting middle path exists — re-export only `NoopReranker`
here, since it is free, and leave `bge_local` lazy.

That was rejected. A rule of the form "no imports in package `__init__`" is
checkable by anyone at a glance; a rule of the form "no *heavy* imports in
package `__init__`" requires every future contributor to correctly judge what
counts as heavy, and to re-judge it whenever a dependency grows. The second rule
gets broken quietly. Consistency here is worth more than the tiny convenience of
one shorter import path.

### Flow of the code

Identical to section 3: `pip install -e .` collects the directory via
`packages.find`, and at runtime `import app.providers.rerank.bge_local` imports
`app` then `app.providers` then `app.providers.rerank` then the module. Because
this file is empty of imports, steps 1 through 3 cost nothing.

### Math
None.

### Common gotchas / traps

- All of section 3's traps apply unchanged (namespace-package false success,
  re-export cost, invisible side effects, circular imports).
- **Package-specific:** the temptation described above. If you ever see
  `from .noop import NoopReranker` appear in this file, the next edit after it is
  almost always someone adding `from .bge_local import BGELocalReranker` "for
  symmetry" — and then every test in the project pays a torch import.

### Interview questions

**Q: When is a consistent rule better than a locally optimal one?**
When the rule has to survive people who lack your context. Re-exporting the cheap
provider and not the expensive one is *locally* optimal — it costs nothing today
and shortens one import. But it encodes a distinction ("heavy" vs "light") that
lives only in the author's head, so the next contributor sees an asymmetry with
no visible justification and resolves it the obvious way: by making it
symmetric. The cost of the inconsistent-but-optimal rule is paid later, by
someone else, in a form that is hard to trace back. This is the same reasoning
behind formatters like black: a uniformly-applied rule that is occasionally
suboptimal beats a case-by-case judgement that is occasionally wrong, because the
first one is enforceable and the second one is an argument.

---

## 7. `backend/app/providers/rerank/noop.py`

**Purpose** — A `RerankProvider` that does nothing: it returns the retriever's
own top-`k` in the order vector search produced them. Selecting it with
`RERANK_PROVIDER=noop` switches stage 2 of the pipeline off without touching a
line of `retrieval.py`. Without this file, "is the reranker worth its latency?"
is answerable only by editing code, which means the two things you compare are
never quite the same system.

**Analogy** — The placebo arm of a drug trial. Identical pill, identical
schedule, no active ingredient. Its entire value is that it makes the difference
attributable to the drug rather than to everything else that changed.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `name = "noop"` | Identifies the provider in logs and in `ProviderError`. | Always. |
| `warmup()` | Deliberately empty. Exists so callers can warm *any* provider without type-checking it first. | App startup. |
| `rerank()` | Validates, truncates to `top_k`, returns positions with descending synthetic scores. | Every chat request under this config. |
| Validation block | Byte-for-byte the same guards as `bge_local`. | Start of every call. |
| Score expression | `1.0 - i / len(docs)` — encodes position, not relevance. | End of every call. |

### Flow of the code

1. `retrieval.search()` holds 40 candidates from Qdrant, already sorted by cosine
   similarity (best first).
2. It calls `rerank(query, texts, top_k=6)` — exactly the same call it would make
   to `bge_local`. The call site has no idea which provider it holds.
3. Guard clauses run: empty `docs` returns `[]`; non-positive `top_k` or blank
   query raises `ProviderInvalidRequestError`.
4. `kept = min(top_k, len(docs))` — clamps so 6 requested from 3 candidates
   returns 3 rather than raising.
5. A list comprehension emits `[(0, 1.0), (1, 0.975), (2, 0.95), ...]`.
6. The caller maps those indexes back to its chunk objects, exactly as before.

No model, no thread offload, no I/O. The method is `async` purely to satisfy the
Protocol — an `async def` with no `await` is a legitimate and common way to
implement an async interface synchronously.

### Math

**The score formula.** For position *i* out of *n* documents:

    score(i) = 1.0 - i/n

This descends from `1.0` at position 0 and stays positive for all `i < n`. Two
properties are deliberate:

- **Monotonically decreasing**, so the contract "highest score first" remains
  true. A caller who re-sorts by score gets the same order back, rather than
  scrambling it.
- **Position-derived, not relevance-derived.** These numbers carry no information
  the retriever did not already provide by ordering.

The alternative — returning `0.0` for everything — was rejected because a caller
applying any `score > threshold` filter would silently discard every result. All
scores being equal also makes correctness depend on Python's sort being stable,
which is true but is a fragile thing to rely on.

**What you actually measure with this.** Run a fixed question set twice, once per
provider, and compare:

    Recall@6 = (# questions whose correct chunk appears in the final 6) / (# questions)

    MRR      = (1/Q) * sum over questions of 1/rank_of_first_correct_chunk

If `bge_local` and `noop` produce the same Recall@6, stage 1 was already
sufficient and the reranker is pure cost. The gap between the two numbers *is*
the reranker's value, expressed in a unit you can defend.

### Common gotchas / traps

- **Mistaking it for dead code and deleting it.** It is referenced by config, not
  by any import that a "find usages" search will surface. The docstring exists to
  survive that moment.
- **Comparing noop scores to bge scores.** One is a position ramp in (0, 1], the
  other is a model relevance score whose range depends on the library version
  (0…1 under sentence-transformers 5.x). They happen to overlap numerically,
  which makes the mistake *easier*, not harder. Any code comparing scores
  *across*
  providers is broken by construction — the general rule from section 5.
- **Letting validation drift from `bge_local`.** If this file stops rejecting a
  blank query, then swapping providers changes error behaviour too, and your
  clean A/B becomes a comparison of two variables. Keeping the guards identical
  is the point, not duplication to be refactored away.
- **Assuming noop means "no truncation".** It still applies `top_k`. Disabling
  reranking must not also change how much context reaches the LLM, or the
  comparison is again confounded — the LLM would be seeing 40 chunks in one arm
  and 6 in the other.
- **Expecting `warmup()` to be removable.** It is empty, but its presence is what
  lets startup code call `warmup()` on whatever provider the factory returned
  without an `isinstance` check.

### Interview questions

**Q: What is the Null Object pattern, and what does it buy you over a null
check?**
A Null Object is an implementation of an interface whose methods do nothing
meaningful, supplied in place of `None`. The benefit is that the calling code
loses a branch: instead of `if self.reranker: ranked = await
self.reranker.rerank(...) else: ranked = candidates[:k]`, there is one
unconditional call. That matters beyond tidiness — the two branches of a null
check are two different code paths, and in a measurement context they are two
different systems, so any quality difference you observe is partly attributable
to the branch itself. With a Null Object, the enabled and disabled configurations
execute identical control flow, so the only variable is the ranking. The cost is
one small class and the risk that someone deletes it as pointless.

**Q: How would you prove to a skeptical tech lead that reranking is worth 2
seconds of p95 latency?**
Not with an argument — with a table. Build a fixed eval set of real questions
where the correct source chunk is known, then run the pipeline twice, changing
only `RERANK_PROVIDER`. Report Recall@6 and MRR alongside p50/p95 latency for
each arm. That produces a statement like "reranking moves Recall@6 from 0.62 to
0.89 and costs 1.9s at p95," which converts an opinion into a tradeoff the lead
can actually decide on. If the gain is small, the honest conclusion is to ship
`noop` and spend the latency budget elsewhere — the value of the experiment is
that it can tell you to remove your own work.

---

## 8. `backend/app/providers/llm/__init__.py`

Package marker for the LLM implementations. Mechanism and reasoning are
identical to sections 3 and 6 and are not repeated: kept import-free, because
each module below imports a *different* vendor SDK and re-exporting them would
make every process pay for all five vendors regardless of which one is
configured. See section 3 for the packaging details.

---

## 9. `backend/app/providers/llm/gemini.py`

**Purpose** — The default `LLMProvider`: streams generated text from Gemini
Flash. This is the component that turns retrieved chunks into an answer, so
without it the pipeline can find perfect context and do nothing with it. It is
also the only file permitted to import `google.genai` (Hard Rule #2).

**Analogy** — A narrator handed a stack of research notes and told to answer
using only those notes, speaking aloud as they think rather than silently
composing a finished essay and reading it out at the end.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `DEFAULT_MODEL` | `gemini-2.0-flash` — the free, fast tier. | Import time. |
| `DEFAULT_TEMPERATURE` | 0.2. Low, because grounded answers should not be creative. | Every request. |
| `_ROLE_MAP` | `{"user": "user", "assistant": "model"}`. Note the rename. | Every message conversion. |
| `__init__` | Validates the key is non-empty; stores config. Builds no client. | Factory construction (startup). |
| `_get_client()` | Lazily constructs `genai.Client`. No lock — see below. | First request. |
| `stream()` | The Protocol method. Validates, builds config, awaits the stream, yields text. | Every chat request. |
| `_to_gemini_contents()` | `Message` → `types.Content`; rejects the `system` role. | Inside `stream`. |
| `_translate()` | SDK exception → our `ProviderError` family, keyed on HTTP status. | On any failure. |

### Flow of the code

1. Startup: the factory constructs `GeminiLLM(api_key=settings.gemini_api_key)`.
   An empty key raises `ProviderAuthError` **here** — a boot failure, not a
   user's chat request dying mid-sentence.
2. A request calls `llm.stream(system, messages, max_tokens)`. **Nothing runs.**
   A coroutine-generator object is returned.
3. `core/generation.py` starts `async for token in stream_obj:` — only now does
   the body begin.
4. Guard clauses: empty `messages` or non-positive `max_tokens` raise.
5. `_get_client()` constructs the SDK client on first use.
6. `_to_gemini_contents()` maps roles; `assistant` becomes `model`; a `system`
   role raises rather than being silently folded into the conversation.
7. `GenerateContentConfig` carries `system_instruction`, `max_output_tokens` and
   `temperature` — the system prompt travels out-of-band, not as a message.
8. `await client.aio.models.generate_content_stream(...)` returns an
   `AsyncIterator`. **The await is required before iterating.**
9. `async for chunk in stream:` — each chunk's `.text` is yielded if non-empty.
10. Any SDK exception is converted by `_translate()` and re-raised as a
    `ProviderError`; our own errors pass through untouched.

### Math
None. The only tunable number is `DEFAULT_TEMPERATURE = 0.2`, and the reasoning
is qualitative: in grounded RAG the facts are supplied in the context, so
sampling randomness has nothing useful to contribute and can only push the model
away from the source text. It is not set to 0.0 because a little variation
produces more natural phrasing without meaningfully raising drift risk.

### Common gotchas / traps

- **`async for` directly on `generate_content_stream(...)`.** Verified against
  google-genai 2.12.1: it is a *coroutine returning an AsyncIterator*, not an
  async generator, so it must be awaited first. This was checked with
  `inspect.iscoroutinefunction` rather than assumed — the reranker's score-range
  error in section 5 came from assuming an SDK's behaviour, and the lesson was
  applied here.
- **Async generators are lazy.** Calling `stream()` executes none of the body.
  A wrong `max_tokens` or a bad key produces no error at the call site; it
  surfaces on the first `async for`. Wrap the iteration in `try`, never the call.
  Demonstrated in this project's own smoke test: `stream('sys', [], 100)`
  returned normally and only raised when iterated.
- **Assuming every chunk has text.** Gemini emits chunks carrying safety
  verdicts or usage metadata with `text = None`. Yielding those would break
  string concatenation downstream, hence the `if text:` guard.
- **The `assistant` → `model` rename.** Sending `role="assistant"` is rejected by
  the API. This is exactly the kind of vendor quirk the provider layer exists to
  absorb; nothing above this file should ever learn the word "model" as a role.
- **Folding the system prompt into the messages.** Gemini takes it via
  `system_instruction`. Passing it as a user turn "works" and quietly weakens
  instruction-following — a silent quality bug, so this file raises instead.
- **Mapping errors on message text.** Message strings change between SDK
  releases; HTTP status codes do not. `_translate` keys on `code`.
- **Marking 4xx retryable.** A 400 means *we* sent something invalid. Failing
  over to Groq would resend the same invalid request and burn the backup's quota
  for nothing, which is exactly what the `retryable` flag from section 1 exists
  to prevent.
- **No lock in `_get_client()`.** Deliberate asymmetry with the embedder: this
  client is a cheap config-and-session object, so a rare double-construction is
  harmless. Two concurrent *model* loads would mean two copies of the weights in
  RAM, which is why that one is locked and this one is not.

### Interview questions

**Q: What is an async generator, and what is the practical consequence of its
laziness?**
An async generator is an `async def` containing `yield`. Calling it returns a
generator object without executing any of the body; the code runs incrementally
as the consumer drives it with `async for`, suspending at each `yield`. The
practical consequence is that errors and side effects relocate from call time to
iteration time — so `try: gen = provider.stream(...) except ProviderError:` is
dead code that can never fire, while the exception it was meant to catch escapes
from the `async for` somewhere else entirely. It also means a consumer that
abandons the loop early leaves the generator suspended, so cleanup belongs in
`finally` or an `async with`. For streaming LLM output this laziness is the whole
point: tokens are produced and forwarded as they arrive, so the user sees text
after ~200ms instead of after the full generation.

**Q: Why translate vendor exceptions instead of letting them propagate?**
Three reasons, in increasing order of importance. First, coupling: if
`google.genai.errors.ClientError` reaches the API layer, then every error handler
in the application depends on Google's SDK, and adding Groq means touching all of
them. Second, completeness: each vendor has its own exception taxonomy, so
without translation the handling code grows one branch per provider per error
type. Third, and most importantly, policy: the retry decision needs a single
consistent answer to "is this worth trying elsewhere?", which only exists if all
providers report failures in one vocabulary. Translating at the boundary means
the fallback logic in Phase 3 reads `if e.retryable` and works identically no
matter which of the five providers failed — the abstraction is only worth having
if failures respect it too.

---

## 10. `backend/app/providers/llm/groq_llm.py`

**Purpose** — The fallback `LLMProvider`: streams generated text from Groq's
hosted Llama 3.3. It exists so that a Gemini free-tier 429 is a hiccup rather
than an outage — Phase 3's automatic failover has nowhere to go without it — and
so that `LLM_PROVIDER=groq` is a config edit rather than a code change.

**Analogy** — A second checkout lane in a shop. Same till, same receipt format,
a different cashier. When lane one queues up you are waved to lane two, and the
receipt in your hand looks identical either way.

### New concept — the OpenAI-compatible chat API

A de-facto wire format that Groq, Ollama, vLLM, Together and most others copied
from OpenAI: messages are `[{"role": ..., "content": ...}]`, the system prompt is
simply the first message with `role: "system"`, and `stream=True` returns
incremental *delta* frames rather than one finished response. It matters here
because it makes Groq and (next) Ollama share a message shape, so the awkward
translation work is confined to Gemini, which is the format outlier.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `DEFAULT_MODEL` | `llama-3.3-70b-versatile` — Groq's strongest free-tier instruction follower. | Import time. |
| `DEFAULT_TEMPERATURE` | 0.2, same reasoning as Gemini: grounded answers should not be creative. | Every request. |
| `_ALLOWED_ROLES` | `{"user", "assistant"}`. Not a map — no renaming is needed. Its only job is to reject `system` arriving in the wrong place. | Every message conversion. |
| `__init__` | Validates the key is non-empty; stores config. Builds no client. | Factory construction (startup). |
| `_get_client()` | Lazily constructs `AsyncGroq`. No lock — cheap object. | First request. |
| `stream()` | The Protocol method. Validates, builds the payload, awaits the stream, yields deltas. | Every chat request. |
| `_to_chat_messages()` | Prepends the system prompt, then copies `Message` → dict. | Inside `stream`. |
| `_translate()` | SDK exception → our `ProviderError` family, keyed on HTTP status. | On any failure. |

### Flow of the code

1. The factory calls `GroqLLM(api_key=settings.groq_api_key)` at startup. A blank
   key raises `ProviderAuthError` immediately — boot fails loudly rather than a
   user's request failing quietly later.
2. A caller calls `provider.stream(system, messages, max_tokens)`. **Nothing
   runs.** An async generator object is returned.
3. The caller starts `async for chunk in ...`. Only now does the body execute:
   argument validation, then `_get_client()` (first call constructs `AsyncGroq`
   and imports the SDK).
4. `_to_chat_messages()` builds the payload: system prompt first if non-empty,
   then each turn, rejecting any stray `system` role.
5. `create(..., stream=True)` is **awaited** to get an `AsyncStream`, then
   iterated. Each frame is a delta; empty ones are skipped.
6. Any SDK exception is funnelled through `_translate()` into our hierarchy.
   `ProviderError` is re-raised untouched so validation errors are not
   re-wrapped as vendor failures.

### Math

None. This file is protocol plumbing.

### Common gotchas / traps

- **`await` then iterate.** `async for c in client.chat.completions.create(...)`
  fails: `create` is a coroutine that *returns* an `AsyncStream`. Identical trap
  to Gemini's `generate_content_stream`, different SDK.
- **Exception check order.** `groq.APITimeoutError` is a *subclass* of
  `groq.APIConnectionError`. Checking the parent first would silently classify
  every timeout as unavailable. Both are retryable so the bug would not change
  behaviour today — it would change it the moment the two are treated
  differently, which is exactly the kind of latent bug that is hard to find.
- **`delta.content` is often `None`.** The first frame announces the role, the
  last frame carries `finish_reason`, and the usage frame has no `choices` at
  all. Yielding `None` breaks string concatenation downstream.
- **Two system prompts.** Because Groq takes the system prompt in-band, a
  `system` role slipping into `messages` would produce two system turns and
  undefined precedence. Hence the explicit rejection.
- **`max_tokens` is the OUTPUT budget**, not the context window. Setting it
  small does not protect you from a too-long prompt — that returns 400, mapped
  to the non-retryable `ProviderInvalidRequestError`.

### Interview questions

**Q: You have two LLM providers with almost identical structure. Why not extract
a shared base class?**
Because they are not actually the same underneath, and the similarity is
superficial enough that a base class would leak. Gemini takes the system prompt
out-of-band and renames `assistant` to `model`; Groq takes it in-band and needs
no renaming. Ollama, next, shares Groq's format but has a different failure mode
entirely (model-still-loading). A base class would end up with hooks for each of
those differences — more moving parts than the ~40 duplicated lines it saves.
The rule of thumb: two similar implementations are a coincidence, three are a
pattern. The contract that genuinely matters is already enforced — the
`LLMProvider` Protocol — and it is enforced structurally, so nothing is lost.

**Q: Why is a 429 retryable but a 400 not, when both are 4xx?**
Because `retryable` encodes "would a *different provider* succeed?", not "is this
our fault?". A 429 is about *this* account's quota against *this* vendor, so the
identical request against Groq will likely succeed — retrying is worth a second
round-trip. A 400 means the request itself is malformed: context too long, bad
parameter, unsupported model. Replaying it elsewhere reproduces the same failure,
burns the fallback's quota, and doubles the user's wait before showing the same
error. Getting this wrong in the pessimistic direction (marking a 400 retryable)
is the more expensive mistake, so the default for *unknown* exceptions is
deliberately retryable while known client errors are explicitly not.

---

## 11. `backend/app/providers/llm/ollama.py`

**Purpose** — The local/offline `LLMProvider`: streams from an Ollama daemon
running on your own machine. Without it there is no zero-cost, zero-network path
through the pipeline — every test run and every experiment spends free-tier
quota, and nothing works on a plane.

**Analogy** — Cooking at home instead of ordering delivery. Slower, and you have
to stock the kitchen yourself, but there is no bill, no queue, and it still works
when the internet is down.

### New concept — NDJSON streaming

Newline-delimited JSON: the response body is a sequence of complete JSON objects,
one per line, rather than a single JSON array. A reader can parse each line the
moment it arrives instead of waiting for a closing `]`, which is what makes it a
streaming format. Ollama chose it over SSE because it is trivial for a local
server to emit — no `data:` prefixes, no blank-line frame separators, no
`event:` types. The practical consequence for us: there is no SDK doing the
framing, so `_extract_text` parses lines by hand.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `DEFAULT_MODEL` | `llama3.2` — small enough for a laptop without a GPU. | Import time. |
| `DEFAULT_TIMEOUT_SECONDS` | 300s. Deliberately huge; a cold model load is slow. | Every request. |
| `_ALLOWED_ROLES` | `{"user", "assistant"}`. Rejects a stray `system`. | Every message conversion. |
| `__init__` | Validates and normalises `base_url`. No client, no credentials. | Factory construction (startup). |
| `stream()` | The Protocol method. Builds the payload, opens the stream, yields deltas. | Every chat request. |
| `_to_chat_messages()` | Prepends the system prompt, copies `Message` → dict. | Inside `stream`. |
| `_extract_text()` | One NDJSON line → text delta or `None`. Raises on in-band errors. | Per line of the stream. |
| `_translate_status()` | HTTP error response → our `ProviderError` family. | On a 4xx/5xx response. |
| `_translate()` | Transport exception (refused, timeout) → our family. | On a connection failure. |

### Flow of the code

1. The factory calls `OllamaLLM(base_url=settings.ollama_base_url)` at startup.
   No credential check — Ollama has none. An empty URL is a configuration bug and
   raises `ProviderInvalidRequestError` immediately.
2. A caller calls `provider.stream(...)`. **Nothing runs**; an async generator
   object is returned.
3. First `async for` executes the body: validation, then payload assembly. Note
   `max_tokens` becomes `options.num_predict` — Ollama's spelling.
4. An `AsyncClient` is created *inside* the request and torn down with it.
5. `client.stream("POST", ...)` sends headers and returns as soon as the
   response *headers* arrive — the body is still being generated. A `>= 400`
   status is detected here, before any tokens, and needs an explicit
   `await response.aread()` because streaming mode does not fetch the body.
6. `aiter_lines()` yields NDJSON lines as they arrive; each goes through
   `_extract_text`, and non-empty deltas are yielded to the caller.
7. Transport failures funnel through `_translate`. `ProviderError` is re-raised
   untouched so our own validation errors are not re-wrapped.

### Math

None. This file is protocol plumbing.

### Common gotchas / traps

- **`AsyncClient` is loop-bound.** Caching one on `self` (the way the Gemini and
  Groq clients are cached) works until the provider outlives the loop that built
  it — pytest creating a loop per test, or a Celery worker — and then fails with
  an opaque "event loop is closed". Per-request construction over a local socket
  is cheap; that trade is why this file breaks the pattern the other two follow.
- **`response.text` is empty in streaming mode** until you `aread()`. Forgetting
  it produces an error message with no error in it.
- **Errors arrive mid-stream with HTTP 200.** The status line is sent before
  generation begins, so a later failure can only be reported as an `error` key
  inside a frame. Ignoring it truncates the answer silently — the worst failure
  mode, because it looks like a complete answer.
- **`num_predict`, not `max_tokens`.** Silently ignored if misspelled: the key
  lands in `options`, Ollama does not validate unknown keys, and the model
  generates until it stops on its own. No error, just a much longer answer than
  requested.
- **The first request is not representative.** Cold-loading weights dominates,
  so timing anything on run one gives a number that says nothing about steady
  state.

### Interview questions

**Q: The other two providers cache their HTTP client; this one builds a new one
per request. Isn't that exactly the connection-pooling mistake?**
It would be against a remote API, where TCP and TLS setup per request is real
latency. Here it is a loopback connection with no TLS, so the setup cost is
sub-millisecond against a generation that takes seconds — the pool would be
optimising something that is already ~0.01% of the request. Against that, an
`httpx.AsyncClient` binds to the event loop that created it, and this provider is
constructed once at startup while requests can run on different loops in tests
and in Celery workers. So the cached version buys nothing measurable and risks a
class of bug that is genuinely hard to diagnose. If Ollama were ever pointed at a
remote host over TLS, the calculation flips and the client should be cached
per-loop.

**Q: Why is a 404 from Ollama non-retryable when a 429 from Gemini is?**
Both questions reduce to "would another provider succeed?" — but for a 404 the
honest answer is that succeeding elsewhere is the *wrong* outcome. A 404 here
means the model was never pulled: a setup mistake with a one-line fix
(`ollama pull llama3.2`). If it silently failed over to Gemini, the developer
would get a perfectly good answer and never learn that their local path is
broken — and would discover it later, offline, when the fallback is unreachable.
Failing over is right when the failure is transient and external; it is wrong
when it masks a misconfiguration the operator needs to see.

---

## 12. `backend/app/providers/llm/anthropic_llm.py` (premium stub)

**Purpose** — Reserves the `anthropic` slot in the provider registry so PLAN §3
rule 4 — a premium client running Claude while everyone else runs Gemini, one
codebase — is a fill-in-the-blank later rather than a redesign. Without it,
adding Claude means editing the factory, config, and `.env.example` at the exact
moment there is a customer waiting.

**Analogy** — A labelled empty shelf in a new shop. Nothing on it yet, but the
aisle plan already accounts for it, so stocking it later does not mean moving
everything else.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| Module docstring | Carries the completion checklist: the dependency to add, the API shape, the error mapping to copy. | Read by whoever finishes it. |
| `DEFAULT_MODEL` | `claude-sonnet-5`. | Import time. |
| `__init__` | Validates the key. **Succeeds** — deliberately does not raise. | Factory construction. |
| `stream()` | Raises `NotImplementedError` with an actionable message. | If someone sets `LLM_PROVIDER=anthropic`. |
| The unreachable `yield` | Forces Python to compile `stream` as an async generator. | Never executes. |

### Flow of the code

There is no flow. The interesting decisions are all about *where* the failure
happens:

1. `__init__` succeeds so that a factory test can instantiate every provider in
   the registry without an `if provider in UNFINISHED` branch. Such a branch is
   written as temporary and is never removed.
2. `stream()` raises, and the message names the three providers that do work —
   an error that tells you what to do instead is worth the extra line.
3. The `yield` after the `raise` is unreachable but load-bearing. Without a
   `yield` anywhere in the body, `stream` is a plain coroutine, and the error
   surfaces when the caller *calls* it. The real implementation is an async
   generator, which raises only on first iteration. Keeping the `yield` makes
   the stub fail at the same moment the finished version would, so code written
   against the stub does not need rearranging when the body lands.

### Math

None.

### Common gotchas / traps

- **A stub that lies is worse than no stub.** Returning an empty string or a
  canned answer would let the pipeline "work" while producing nothing —
  ungrounded silence that looks like a retrieval bug. Raising is the honest
  failure.
- **`anthropic` is not in pyproject on purpose.** No one should install and
  import ~10MB of SDK for a code path that raises. Hence the import instruction
  lives in the docstring, not at the top of the file.
- **Anthropic's system prompt is out-of-band** (a top-level `system=` parameter),
  like Gemini and unlike Groq — so the finished version copies `gemini.py`'s
  message handling and `groq_llm.py`'s error handling, not one file wholesale.
- **`max_tokens` is required** by the Messages API, not optional. Our Protocol
  already demands it, which is one small piece of evidence the contract was
  drawn at the right level.

### Interview questions

**Q: Isn't writing stubs for features you don't have speculative generality —
the thing YAGNI warns about?**
YAGNI warns against building *machinery* you don't need: config knobs, plugin
layers, abstractions with one implementation. This is the opposite trade. The
abstraction (`LLMProvider`) already exists and is already justified by three
working providers; the stub adds no machinery at all — it is a name in a registry
and a raise. What it buys is validation: writing it proved the Protocol survives
a vendor with a different system-prompt convention without changing `base.py`.
A contract implemented only by providers written the same afternoon has not
really been tested. The line I would not cross is a stub with invented behaviour
— that is speculative design, and it rots.

**Q: Why does `__init__` succeed if the provider cannot work?**
To keep the failure in one place and keep tests honest. If construction raised,
every test that walks the registry would need to skip the unfinished providers,
and skip-lists written as temporary outlive the thing they were working around —
so a real bug in a real provider could later hide behind that same branch. By
constructing cleanly and failing at use, the registry stays uniform, the factory
needs no special case, and the only code path that fails is the one a person
actually selected. The cost is that a bad `LLM_PROVIDER=anthropic` is caught at
first request rather than at boot; acceptable here because the message names the
fix, and because it will stop being true the moment the file is finished.

---

## 13. `backend/app/providers/llm/openai_llm.py` (premium stub)

**Purpose** — Reserves the `openai` LLM slot: the second premium generator, and
the partner to `embeddings/openai_embed.py` for a client standardising on OpenAI.
Without it that client's onboarding starts with a code change.

**Analogy** — The second labelled shelf beside the first. Same aisle plan,
different supplier.

Mechanism is identical to section 12 and is not repeated: constructs
successfully, `stream()` raises, unreachable `yield` keeps it an async generator,
SDK deliberately absent from pyproject. Only the differences are documented
below.

### What is different here

- **The completion template is `groq_llm.py`, not `gemini.py`.** Groq copied
  OpenAI's wire format, so this file is nearly a rename of that one — same
  in-band system message, same `await`-then-iterate trap, same
  `APITimeoutError`-before-`APIConnectionError` ordering. Anthropic's stub points
  at `gemini.py` instead, because both take the system prompt out-of-band. Two
  premium stubs, two different templates: worth noticing, because it is the
  clearest evidence that "they all look the same" is false.
- **It is deliberately not merged with the Anthropic stub.** `LLM_PROVIDER`
  selects exactly one, so there is no shared runtime path to factor out, and a
  common `_StubLLM` base would need per-vendor hooks for the name, model, env var
  and message — more machinery than the ~20 duplicated lines it replaces.
- **Its env var is independent of the embedder's.** `LLM_PROVIDER=openai` and
  `EMBEDDING_PROVIDER=openai` are separate switches on purpose: changing the
  generator is free, while changing the embedder invalidates every stored vector
  and forces a re-embed (PLAN §3 rule 1). Coupling them behind one setting would
  make a cheap change silently trigger an expensive one.

### Flow of the code

None — see section 12.

### Math

None.

### Common gotchas / traps

- **`gpt-4o-mini` is a placeholder**, not a recommendation. Model names churn
  faster than this file will be touched; whoever implements it should check the
  current lineup rather than trusting a constant written months earlier.
- Everything in section 12's trap list applies unchanged.

### Interview questions

**Q: Your LLM and embedding providers are both selectable by env var. Why are
they not one setting, given a client usually wants the same vendor for both?**
Because the two changes have wildly different costs, and one setting would hide
that. Swapping the generator takes effect on the next request and is reversible
in seconds. Swapping the embedder changes the vector dimension and the embedding
space, which invalidates every vector already in Qdrant — the old and new vectors
are not comparable, so search silently returns nonsense rather than failing. That
is why the dimension is stored per-bot in Postgres and why `scripts/reembed.py`
exists as a planned command. Merging the switches would mean a developer changing
`AI_PROVIDER=openai` to get a better answer quality also destroys their retrieval
until a re-embed finishes. Config should make expensive operations look
expensive.

---

## 14. `backend/app/providers/embeddings/openai_embed.py` (premium stub)

**Purpose** — Reserves the `openai` slot for `EMBEDDING_PROVIDER`: the paid
upgrade from local bge-small, for a client who needs better retrieval than a
384-dim CPU model gives. Without it, that upgrade is a code change *plus* a
re-embed instead of just a re-embed.

**Analogy** — Wiring a house for a second circuit before the appliance exists.
The socket is capped and dead, but the cable gauge is already correct, so
installing later does not mean opening the walls.

General stub mechanism is section 12's and is not repeated. Two things are
specific to this file.

### New concept — symmetric vs asymmetric embedders

An **asymmetric** model is trained so that queries and passages are encoded
*differently* — bge glues the instruction `"Represent this sentence for searching
relevant passages: "` onto queries and nothing onto passages. Embedding both
sides identically lands them in slightly different regions of the vector space
and measurably hurts recall on short queries, with no error and no crash. A
**symmetric** model (OpenAI's, and most sentence-similarity models) encodes both
sides the same way, so `embed_query(t)` is literally `embed_documents([t])[0]`.

This is exactly why `EmbeddingProvider` has two methods rather than one. The
contract costs the symmetric provider a single trivial line; collapsing it to one
method would cost the asymmetric provider a silent quality regression. When a
contract's cost is that lopsided, pay it at the cheap end.

### What is different here — `dimension` cannot be a placeholder

Every other value in a stub is deferrable. `dimension` is not, because it is read
at *configuration* time rather than at call time: it sizes the Qdrant collection
and is written to the per-bot row in Postgres (PLAN §3 rule 1). A placeholder
would create a collection the finished provider cannot write into — a 1536-dim
vector rejected by a 384-dim collection — and the failure would appear during a
migration, far from this file, long after whoever typed the placeholder forgot
about it. So `DIMENSION = 1536` is the real number.

### Flow of the code

None — both methods raise. Note the absence of the unreachable `yield` from
sections 12 and 13: these are plain coroutines, not async generators, so the
raise already fires at the same moment the finished version would.

### Math

None here, but the relevant number is storage: a vector costs
`dimension x 4 bytes` at float32. Moving bge-small (384) to
text-embedding-3-small (1536) is 1.5KB per chunk instead of 1.5KB per four
chunks — a 4x increase in vector storage for the same corpus, before payload.

### Common gotchas / traps

- **Changing the embedder is never an in-place edit.** Old and new vectors are
  not comparable, and mixing them does not error — search just returns
  plausible-looking nonsense. This is the single most dangerous silent failure in
  the whole retrieval path, and the reason `scripts/reembed.py` is a planned
  Phase 2 deliverable rather than an improvisation.
- **Matryoshka truncation is a re-embed too.** `text-embedding-3` supports a
  `dimensions` parameter to shorten vectors, which is tempting to treat as a
  config tweak. It changes the space. Same rule applies.
- **Do not send one request per chunk.** The endpoint takes batched input with a
  large token cap; per-chunk requests turn a 500-page PDF into thousands of calls
  and a rate-limit incident.
- **Do not trust response ordering implicitly.** It is documented as preserved,
  but a length assertion is one line and turns a silent mis-zip — every chunk
  labelled with the wrong text — into a loud failure.

### Interview questions

**Q: Why does `EmbeddingProvider` have separate `embed_documents` and
`embed_query` methods when for most models they would be identical?**
Because the models where they differ fail *silently* when you conflate them. bge
and e5 are asymmetric: they are trained with an instruction prefix on the query
side only, and embedding a query as though it were a passage still produces a
perfectly valid vector — it just sits slightly off in the space, costing recall
on exactly the short queries users actually type. There is no exception and no
log line; retrieval simply gets worse, and you would go looking in the chunker.
Splitting the method makes the distinction structural, so a new provider author
has to make a decision about it rather than inherit a default. The cost is one
delegating line for symmetric providers like OpenAI's. That is a good trade
whenever the failure mode on one side is silent and the cost on the other is
trivial.

**Q: What actually breaks if you change the embedding model but don't re-embed?**
Nothing breaks loudly, which is the problem. If the dimension changes, Qdrant
rejects the writes and you find out immediately — that is the lucky case. If the
dimension happens to match (say, two different 384-dim models), every insert and
every search succeeds and returns results that look entirely reasonable. But the
new query vectors live in a different learned space than the stored ones, so
cosine similarity between them is meaningless — you are ranking by an
uninterpretable number. Answers get subtly worse across the board, citations
point at near-random chunks, and there is no error anywhere to trace. That is why
the dimension and model name are stored per-bot in Postgres rather than living in
config: the row is the record of which space a bot's vectors are in, and
`scripts/reembed.py` is the only sanctioned way to move a bot between spaces.

---

## 15. `backend/app/providers/rerank/cohere_rerank.py` (premium stub)

**Purpose** — Reserves the `cohere` slot for `RERANK_PROVIDER`: a hosted
cross-encoder for a client who wants stage-two reranking without the local
bge-reranker burning CPU on every query. Without it, moving reranking off-box is
a code change rather than an env edit.

**Analogy** — Keeping a specialist's number on the fridge. You handle it yourself
today, but when the volume grows you make a call rather than reorganising the
whole household.

General stub mechanism is section 12's. Two things are specific here.

### What is different — the contract was vindicated, not accommodated

Cohere's API returns `(index, relevance_score)` pairs natively, which is exactly
the `ScoredIndex` shape `base.py` defines. The finished body is close to a
passthrough.

That is worth pausing on. When `base.py` chose to return *indexes into the input
list* rather than reordered text, the justification was that it lets the caller
map results straight back to its own chunk objects — with `document_id`, `page`
and `source` attached — without string matching. It would have been easy to
suspect that shape was really just "whatever bge-reranker happened to produce".
A second, independent, commercial implementation arriving at the identical shape
is evidence the abstraction describes the *problem* rather than the first
solution to it. That is the difference between an interface and a wrapper.

### New concept — score scale is provider-private

bge's cross-encoder emits **unbounded logits** (roughly -11..+11; sigmoid-able,
but not probabilities). Cohere returns scores **normalised to 0..1**. Both rank
identically well. Neither number means anything to the other.

The practical consequence is a rule: scores are for *ordering*, never for
*thresholding*. The moment someone adds `if score < 0.5: drop` — tuned by eye
against bge — that filter discards everything or nothing under Cohere, silently,
because both behaviours are plausible outputs of a working reranker. If a
relevance cutoff is genuinely needed later, it must be calibrated per provider
and stored beside the provider name, never as a shared constant.

### Flow of the code

None — the method raises. Like section 14 and unlike sections 12–13, there is no
unreachable `yield`: this is a plain coroutine, so the raise already fires where
the finished version would.

### Math

None in this file. The relevant number is economic rather than mathematical:
reranking runs over ~40 candidates per query, so a hosted reranker is ~40
cross-encoder inferences billed per user question — which is exactly why the
local provider is the default and this one is the premium option, rather than the
other way round.

### Common gotchas / traps

- **Empty `docs` is a 400 from Cohere** but an empty result under our contract.
  Guard before the call, or an unlucky query with no candidates becomes a
  provider error instead of "I don't have that information."
- **Do not trust returned ordering implicitly.** It is sorted descending in
  practice; the contract *promises* it. One defensive `sort` is cheaper than
  discovering a vendor changed an undocumented behaviour.
- **`top_n` is a server-side cut.** Passing `top_k` means you are billed for
  scoring all documents regardless — the parameter reduces payload, not cost.
- **Never persist raw scores for cross-provider comparison** — see above. If they
  are logged for debugging, log the provider name in the same record.

### Interview questions

**Q: How would you know whether an abstraction you designed is any good?**
The test I trust most is whether a second implementation fits without bending it.
When I wrote the rerank contract, the only implementation was a local
cross-encoder, so the interface could easily have been a description of that one
library rather than of reranking. Two things later provided evidence: the noop
provider fits (it returns the first `top_k` indexes with flat scores, requiring
no contract change), and Cohere's hosted API returns literally the same
`(index, score)` pair shape natively. An interface that a pass-through, a local
model, and a commercial API all satisfy without special-casing is describing the
problem. The counter-signal I watch for is a contract that grows an optional
parameter every time a provider is added — that means it was a wrapper around the
first vendor, and each new one is being accommodated rather than fitted.

**Q: Two rerankers both work correctly but return scores on totally different
scales. Is that a design flaw in your interface?**
No — it is an honest interface, and the alternative is worse. I could force
normalisation to 0..1, but that would be a lie: normalising bge's logits requires
picking a squashing function and a range, which invents a calibration the model
never had, and the resulting numbers would look comparable across providers while
still not being so. The contract instead states what is actually guaranteed —
ordering within a single call from a single provider — and explicitly says scores
are not comparable across providers. That pushes the constraint into the one
place it can be handled correctly: any code wanting a relevance cutoff has to
calibrate it per provider and store it alongside the provider name. Interfaces
should promise exactly what they can deliver; a promise of comparability that
does not survive a provider swap is the kind of thing that fails silently in
production six months later.

---

## 16. `backend/app/providers/factory.py`

**Purpose** — The single place where env configuration becomes live provider
objects. It is what makes Hard Rule #2 enforceable rather than aspirational:
`app/core/` calls `get_llm()` and never learns which vendor answered. Without it
every call site would import a vendor directly, and `LLM_PROVIDER=groq` would
stop being a config change.

**Analogy** — A power socket. Appliances plug into one standard shape; whether
the electricity came from coal, wind, or a generator in the garden is the grid's
problem, not the kettle's.

### New concept — the factory pattern

A function that decides *which* implementation to construct, so callers depend on
the interface and never on a constructor. The textbook motivation is decoupling;
the motivation *here* is concrete and narrower: the choice lives in `.env` and
has to change without touching anyone's imports. Note what this factory
deliberately is not — there is no plugin discovery, no entry-point scanning, no
dynamic `importlib` on a config string. Those are the versions of this pattern
that become undebuggable. A dict of five named functions is the whole mechanism.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `_require()` | `str \| None` credential → `str`, or `ProviderAuthError` naming the env var. | Inside each builder. |
| `_build_*()` (9 of them) | Import the vendor module, construct one provider. | First request for that provider. |
| `_LLM_BUILDERS` / `_EMBEDDING_BUILDERS` / `_RERANK_BUILDERS` | Name → builder. The authoritative list of what exists. | Import time. |
| `_unknown()` | Builds the "no such provider, valid options are…" error. | On a bad name. |
| `_llm` / `_embedder` / `_reranker` | `@cache`d builders keyed on the RESOLVED name. Hold the instances. | Once per name per process. |
| `get_llm()` / `get_embedder()` / `get_reranker()` | Public API. Resolve `None` → configured default, delegate to the cached builder. | Everywhere in `core/`. |
| `available_*()` | The registry keys, sorted. | Phase 4's settings dropdown. |
| `reset_provider_cache()` | Drops every instance. | Tests only. |

### Flow of the code

1. `core/` calls `get_llm()` with no argument.
2. `get_llm` resolves `None` to `settings.llm_provider` — this is the step that
   makes caching correct, see below.
3. `_llm("gemini")` is called. On a cache miss it looks up the builder, logs, and
   runs it.
4. The builder imports `app.providers.llm.gemini` **inside the function body**
   and constructs `GeminiLLM`, pulling the credential through `_require`.
5. The instance is cached under the key `"gemini"` and returned. Every later
   call — with `None` or with `"gemini"` — gets that same object.

### Math

None. The only quantity that matters is memory: the local embedder and reranker
hold roughly 130MB and 1.1GB of weights respectively, which is the entire reason
step 2 exists.

### Common gotchas / traps

- **Resolve the name BEFORE caching, not after.** If `@cache` sat on the public
  `get_embedder(name=None)`, then `get_embedder()` and `get_embedder("bge_local")`
  would be two distinct cache keys and would load the weights **twice** — same
  model, double the RAM, and no error to notice. Splitting into a resolving
  public function plus a cached private one keyed on the resolved name is the
  whole fix, and it is the single most important line-level decision in this file.
- **`functools.cache` memoizes return values only.** A raise stores nothing, so a
  build that failed on a missing credential retries on the next call instead of
  poisoning the process. Good behaviour, but the opposite is the natural
  assumption — hence the comment in the code.
- **Vendor imports must stay inside the builders.** Hoisting them to the top of
  the file for tidiness would make `import app.providers.factory` pull all of
  torch, taxing every process regardless of configuration. This is the kind of
  "cleanup" a well-meaning reviewer suggests; the docstring exists to answer it.
- **Never construct a provider directly in `core/`.** Doing so bypasses the cache
  and, for the local models, silently duplicates the weights. The getters are the
  only sanctioned path.
- **`reset_provider_cache()` is test-only.** Calling it in request code would
  discard a loaded model mid-flight and make the next request pay a multi-second
  reload.

### Interview questions

**Q: Your factory caches provider instances. Walk me through how you'd get that
caching wrong.**
The subtle way is caching at the wrong level. The natural first attempt is to put
`@cache` straight on the public `get_embedder(name: str | None = None)`. It looks
right and it even works — but `None` and `"bge_local"` are different arguments, so
they are different cache keys, and the same model gets loaded twice into memory.
There is no error and no wrong answer; you just quietly use double the RAM, and
you would probably discover it as an OOM in a container weeks later. The fix is to
split the responsibility: a public function that resolves the default, and a
private cached function keyed on the already-resolved name, so there is exactly
one key per real provider. The general lesson is that a cache key must be derived
from the *identity of the thing*, not from the shape of the call. A second, milder
trap is assuming the cache memoizes exceptions — it does not, which here happens
to be what we want.

**Q: Why a factory at all, rather than just importing the provider you need?**
Because the decision has to be made at deploy time from an env var, and a direct
import makes it at author time. Concretely, `LLM_PROVIDER=groq` has to work with
zero code edits — that is a stated acceptance criterion for this phase, and it is
also what lets Phase 3 fail over from Gemini to Groq mid-request by asking the
factory for a second provider. But the deeper reason is that the factory is the
enforcement point for the project's second hard rule: no vendor SDK outside
`app/providers/`. If `core/retrieval.py` imported `google.genai` directly, the
rule would be a convention that decays under deadline. With a factory, the rule is
checkable — you can grep for vendor imports outside one directory — and violating
it requires visibly working around the abstraction rather than just forgetting it
exists. An architectural rule nobody can verify is a wish.

---

## 17. Amendment to `base.py` — `EmbeddingProvider.count_tokens`

Not a new file: a single method added to an existing contract, recorded here
because widening a Protocol after five implementations exist is exactly the kind
of change worth being able to justify later.

**Purpose** — Closes the loose end left by `core/chunking.py`, which needs to
size chunks in *tokens* but is forbidden from importing a tokenizer (Hard Rule
#2) and so falls back on a ~4-chars-per-token estimate. This method lets
ingestion hand the chunker the embedder's real tokenizer.

**Analogy** — Asking the tailor, not a ruler, whether the cloth will fit. Only
they know their own measuring conventions.

### Why it belongs on the embedder

The token count is a property of the *model*, not of the text. bge-small and
text-embedding-3 tokenize the same string into different counts, so any generic
tokenizer is confidently wrong the moment `EMBEDDING_PROVIDER` changes — and
wrong in the silent direction: text past the model's window is truncated during
encoding with no error at all. You embed a chunk, store the vector, and the tail
of the text simply is not in it. Nothing downstream can detect this.

### Why it is sync when every sibling is async

This is the first sync method on any of the three Protocols, and the asymmetry
is deliberate. `async` marks **I/O**, not expense. `embed_documents` is async
because it waits on a network call or a GPU; `count_tokens` is microseconds of
pure CPU with neither. More practically, `chunking._split_recursive` calls it
thousands of times inside a tight recursive loop — making it a coroutine would
force `await` through that entire call path, including the chunker's public API,
and buy nothing, because there is no I/O to overlap with.

The type also lines up exactly with `chunking.TokenCounter`
(`Callable[[str], int]`), so `chunk_blocks(blocks, count_tokens=embedder.count_tokens)`
just works — no adapter, no lambda.

### What this broke, immediately and on purpose

Adding a required method to a Protocol invalidates every implementation that
lacks it. Verified straight after the edit:

```
BGELocalEmbedder     satisfies protocol: False   has count_tokens: False
OpenAIEmbedder       satisfies protocol: False   has count_tokens: False
```

This is the structural typing from section 1 earning its place. Because
`EmbeddingProvider` is `@runtime_checkable`, the contract change is *detectable*
rather than something discovered later as a wrong answer. Both implementations
are updated next: `bge_local.py` gets a real implementation, `openai_embed.py`
gets the stub treatment consistent with its other methods.

### Common gotchas / traps

- **A Protocol is not enforced at class-definition time.** Nothing fails when
  you widen the contract — `BGELocalEmbedder` still imports and still runs. The
  breakage only shows up at an `isinstance` check or, worse, at the call site.
  Widening a Protocol therefore has to be followed immediately by updating every
  implementation, not scheduled for later.
- **`runtime_checkable` only checks method *presence*, not signatures.** An
  implementation with `count_tokens(self, text: bytes) -> str` passes
  `isinstance` and fails at runtime. The check is a smoke alarm, not a type
  checker.
- **Do not "fix" this by giving the Protocol a default implementation.** A
  Protocol with a body becomes a mixin, implementations start inheriting from it
  instead of satisfying it structurally, and the one-way dependency arrow that
  keeps vendor code a leaf of the tree (section 1) is lost.

### Interview questions

**Q: You added a required method to an interface with five existing
implementations. How do you do that safely?**
The dangerous version of this change is the one that fails silently, so the first
question is whether the language will tell me. With a Python `Protocol` the
answer is "only if I ask" — nothing breaks at import, and a stale implementation
keeps working until something actually calls the missing method. So the sequence
matters: widen the contract, immediately run an `isinstance` check across every
implementation to get the real list of what broke, then fix each one before
moving on. That check is why the Protocols are `runtime_checkable` at all. The
alternative approach — adding a default implementation on the Protocol so
nothing breaks — is tempting and wrong here: it would turn the contract into a
mixin, and every provider silently inheriting a *generic* token count is exactly
the confidently-wrong behaviour the method exists to prevent. A loud break
across five files is cheaper than a quiet wrong answer in one.

**Q: Why is one method on this interface synchronous when the rest are async?**
Because `async` in this codebase means "this waits on something", not "this is
slow". `embed_documents` waits on a network round-trip or a GPU; `count_tokens`
does neither — it is a pure CPU function measured in microseconds. Making it
async would be actively harmful rather than merely redundant: the chunker calls
it thousands of times inside a recursive splitting loop, so a coroutine would
force `await` through that whole call path and turn a synchronous, easily-tested
pure function into an async one, with no concurrency gained because there is no
I/O to interleave. The general rule I would defend is that the async boundary
should sit exactly where the I/O is. Marking CPU work async makes the codebase
look concurrent while giving up the ability to call it from ordinary code.

---

## 18. `bge_local.py` — implementing `count_tokens`

**Purpose** — Restores `BGELocalEmbedder` to the contract widened in section 17,
and gives the chunker a real ruler. Without it, chunk sizing is a
4-characters-per-token guess that is fine on English prose and badly wrong on
the technical documents this product exists to ingest.

**Analogy** — The tailor finally measures the cloth with their own tape instead
of the customer eyeballing it.

### New concept — subword tokenization

A tokenizer turns a string into the integer IDs the model consumes. Not words,
not characters: a *subword* vocabulary of ~30k entries where frequent words are
one token and unfamiliar ones are broken into known pieces. This matters here
because every limit in the system — the 512-token encoder window, the LLM's
context, the per-call bill — is denominated in tokens, and only the tokenizer
can say how many a given string costs. Full entry in LEARNING.md.

### The problem this file had to solve

`count_tokens` is sync, by contract. `BGELocalEmbedder` loads its model behind
an `asyncio.Lock` in an async double-checked `_get_model`. **Sync code cannot
await that**, so the existing load path is simply unreachable from the new
method. Three options:

| Option | Verdict |
|---|---|
| Require `warmup()` first, raise otherwise | Rejected — makes the chunker's correctness depend on call ordering in unrelated code, and fails at the worst time (mid-ingestion). |
| `asyncio.run()` / block on the async loader | Rejected — deadlocks if called from inside a running event loop, which is where ingestion lives. |
| **Load the tokenizer alone, behind a `threading.Lock`** | **Chosen** — a tokenizer is a vocabulary file, not weights. |

That last point is what makes the whole thing cheap: the tokenizer is *data*
shipped alongside the model, a few hundred KB with no torch and no forward pass.
Loading it costs nothing like the ~130MB model load, so a sync path is honest
rather than a workaround.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `MAX_SEQUENCE_TOKENS = 512` | The encoder's hard window. Exported so ingestion caps `max_tokens` without hardcoding a model detail. | Read at config time. |
| `self._tokenizer` / `_tokenizer_lock` | Tracked separately from the model. **`threading.Lock`**, not `asyncio.Lock` — it guards a blocking call made from sync code. | Per instance. |
| `count_tokens(text)` | `len(tokenize(text)) + num_special_tokens_to_add()`. | Thousands of times per document. |
| `_get_tokenizer()` | Double-checked load; steals `self._model.tokenizer` if warmup already ran. | Once per instance. |

### Flow of the code

1. `count_tokens` calls `_get_tokenizer()`.
2. Fast path: `self._tokenizer` is already set — return it, no lock taken.
3. Otherwise take the `threading.Lock` and re-check (another thread may have
   loaded it while we waited — the same double-checked shape as `_get_model`,
   in its threading dialect).
4. **If `self._model` is already loaded, take its tokenizer for free.** The
   model carries the very same one; loading a second would waste memory and
   risk the two drifting apart.
5. Otherwise `AutoTokenizer.from_pretrained(MODEL_NAME)`, translating any
   failure into `ProviderUnavailableError` — same failure modes as the model
   load, namely a cold cache with no network, or no disk.
6. Back in `count_tokens`: tokenize, add the special-token count, return.

### Math

```
tokens = len(tokenize(text)) + num_special_tokens_to_add()
```

The second term is 2 for BERT-family encoders, which wrap every input as
`[CLS] … [SEP]`. Those are real slots in the 512-token window, so omitting them
makes every full chunk two tokens optimistic — invisible until a boundary case
truncates.

For contrast, the fallback this replaces is `estimate_tokens = len(text) // 4`.
The 4 is a tokens-per-character ratio measured on **English prose**. It is not a
universal constant: identifiers, URLs and non-English text fragment into far more
pieces, so the estimate under-reports exactly the documents most worth getting
right.

### Common gotchas / traps

- **`.tokenize()`, not `.encode()`.** `encode()` checks its result against
  `model_max_length` and logs a "sequence length is longer than" warning for
  every oversized input. Measuring text that does not fit is this method's whole
  job, so that warning would fire constantly and mean nothing. `tokenize()` plus
  the special-token count gives the same number without the noise.
- **`threading.Lock`, not `asyncio.Lock`.** An `asyncio.Lock` can only be
  acquired with `await`. Using one here would not compile into sync code; using
  the threading one from async code is fine because the guarded work is
  microseconds and never yields.
- **Don't load the model to get the tokenizer.** Tempting one-liner, ~130MB and
  several seconds of cost on a path that needs neither.
- **The special tokens are easy to forget** and the resulting overflow is
  silent. This is the same class of bug as the 0-indexed page number: correct-
  looking and wrong.

### Verification

Protocol conformance restored, and the method confirmed sync:

```
satisfies protocol: True
sync              : True
model loaded?     : False        <- counted without ever loading the weights
```

The accuracy comparison against `estimate_tokens` could not be run in this
environment: `sentence-transformers`, `transformers` and `torch` are not
installed in the active interpreter, so the tokenizer load raised — correctly,
and as a typed `ProviderUnavailableError` naming the provider, which is at least
free evidence that the error translation works. Re-run once the ML dependencies
are installed.

### Interview questions

**Q: You need a cheap synchronous method on a class whose expensive resource is
loaded asynchronously. How do you approach that?**
The instinct is to bridge — run the async loader from sync code with
`asyncio.run` or a nested loop — and that is the one answer I would avoid, because
it deadlocks the moment it is called from inside a running event loop, which is
exactly where the caller lives. The better question is whether the sync method
really needs the expensive resource at all. Here it did not: counting tokens
needs the tokenizer, and a tokenizer is a vocabulary file rather than model
weights, so it has a genuinely cheap load path of its own. The fix was to notice
that the expensive thing and the needed thing were separable. I still kept a
double-checked lock around it, in the threading dialect rather than the asyncio
one, and reused the model's tokenizer when warmup had already run so the two can
never diverge. The general lesson is that "how do I call async from sync" is
usually the wrong question, and "what does this actually need" is the right one.

**Q: Why not just estimate tokens as characters ÷ 4 and skip all of this?**
Because the failure it causes is silent and unrecoverable. Text beyond the
embedder's 512-token window is truncated during encoding with no exception — you
store a vector that looks completely normal and simply does not contain the tail
of the chunk, and nothing downstream can detect it. So the cost of the estimate
being wrong is not a warning, it is a retrieval result that quietly never
matches. And it is wrong in a biased way: the 4 is a ratio measured on English
prose, so it under-counts code, URLs, identifiers and non-English text, which is
to say it is least accurate on precisely the technical documents this product
ingests. The estimate is still worth keeping as the chunker's default, because
`chunking.py` must stay pure and importable without a model — but ingestion
passes the real counter whenever there is an embedder to ask.

---

## 19. `openai_embed.py` — `count_tokens` stub, and closing the breakage

**Purpose** — Restores the second and last implementation invalidated by the
Protocol widening in section 17. With this, `EmbeddingProvider` has no
non-conforming implementations and ingestion can depend on `count_tokens`
existing.

**Analogy** — The second tailor admitting they don't own a tape measure, rather
than borrowing the first tailor's and quoting you their neighbour's sizes.

### The interesting decision: this stub could have been finished

Every other stub in this project is blocked on something real — an absent SDK, a
missing API key, a paid account. This method is not. OpenAI's token counting uses
`tiktoken`, a **local** BPE tokenizer with no network call and no key, so
`count_tokens` could have been implemented today while `embed_documents` and
`embed_query` stayed stubs.

It was left raising anyway, and the reason is worth keeping:

> A provider where two methods raise and a third quietly works is a worse
> contract than one that fails uniformly.

The failure mode is concrete. A half-working provider passes construction, passes
whatever smoke test only counts tokens, and then dies partway through a long
ingestion run — after extraction, after chunking, at the first embed call. Every
method raising means `EMBEDDING_PROVIDER=openai` fails immediately and
unambiguously. Uniform failure is a feature of a stub, not a limitation.

### The other decision: raise rather than approximate

Two "helpful" alternatives were available and both are traps.

| Tempting fallback | Why it is worse than raising |
|---|---|
| Return `len(text) // 4` | Plausible number, silently wrong. Sizes chunks against a prose constant that has nothing to do with OpenAI's vocabulary. |
| Borrow bge's tokenizer | Worse still — looks *principled*. It is measuring with a different model's ruler; the counts are simply a different model's counts. |

Both produce mis-sized chunks, and the resulting overflow is truncated during
encoding **with no error**. The vector is stored, looks normal, and is missing
the tail of its text. Nothing downstream can detect it. Against that, a
`NotImplementedError` at startup is a gift.

This is the same principle as section 17's refusal to give the Protocol a default
implementation: a generic token count is confidently-wrong behaviour, and the
method exists specifically to prevent it.

### Note for whoever finishes this file

`tiktoken` differs from the BERT-family tokenizer used by bge in one way that
matters to the arithmetic: **there are no special tokens to add.** bge wraps
every input in `[CLS] … [SEP]` and so adds 2; OpenAI's embedding endpoint does
not, so the encoded length *is* the answer. Copying bge's `+ num_special_tokens`
line into this file would over-count by 2 on every chunk — harmless in isolation,
but it is a needless inaccuracy in exactly the value everything else is sized
against. Cache the encoding object too; constructing it per call is slow enough
to notice in a chunking loop.

### Verification

The breakage opened in section 17 is now closed:

```
BGELocalEmbedder     protocol: True   sync: True   dim: 384
OpenAIEmbedder       protocol: True   sync: True   dim: 1536
stub raises correctly: NotImplementedError
```

Both conform, both counters are sync, and the two `dimension` values differ —
which is the reminder that switching embedder is a re-embed and a new Qdrant
collection, never an in-place config edit.

### Common gotchas / traps

- **A stub that partially works is harder to debug than one that does not
  work at all.** The failure moves from startup to the middle of a long job.
- **`isinstance` conformance says nothing about behaviour.** `OpenAIEmbedder`
  now satisfies `EmbeddingProvider` while raising on every call. The Protocol
  check confirms the shape is right, not that the thing functions — which is
  precisely why the factory tests need no skip-list, but also why passing that
  check is not evidence of a working provider.

### Interview question

**Q: You can implement part of an unimplemented interface cheaply. Should you?**
My default is no, and the reasoning is about *when* failures surface rather than
how many features exist. A class where some methods work and others raise will
construct fine, survive a shallow health check, and then fail partway through
real work — so the error arrives at the least useful moment, with the most state
already built up and the least obvious cause. A class that fails uniformly fails
at the boundary, where the message can name the config value that is wrong. There
is a second reason specific to this case: the cheap-to-implement method is a
*measurement*, and a measurement that is available while the thing it measures is
not is an invitation to size real work against a provider that cannot do it. So
the partial implementation would not just be incomplete, it would be actively
misleading. I would rather ship an honest raise and a docstring explaining the
four steps to finish it.

---

# Phase 1 — Core RAG library: file-by-file explanation

Companion to `phase_1_explanation.md`, which covers the provider layer
(contracts, factory, nine vendor implementations). This file covers `app/core/`:
the vendor-agnostic RAG pipeline that sits on top of those providers.

The split matters. Everything in the provider doc is about *translation* —
turning one vendor's shape into ours. Everything here is about *decisions* —
how to cut a document, what to retrieve, how to ground an answer. The provider
layer is replaceable plumbing; this is the part that determines whether the
product actually works.

Pipeline order, which is also the order these files were built:

```
chunking.py    document -> retrieval-sized, page-tagged pieces
ingestion.py   file -> text -> chunks -> vectors -> Qdrant
retrieval.py   question -> top-40 candidates -> rerank -> top-6
generation.py  top-6 + question -> grounded, cited answer
```

---

## 1. `backend/app/core/chunking.py`

**Purpose** — Turns an extracted document into retrieval-sized pieces that each
carry the page they came from. Everything downstream inherits its mistakes: if a
claim lands in one chunk and its subject in another, no embedder, reranker,
prompt or model recovers the connection. Retrieval quality is capped here, not
by the model choice.

**Analogy** — Cutting a long interview transcript into index cards. Cut at
speaker changes and you can find anything later; cut every 500 characters and
half your cards begin mid-word, saying nothing usable on their own.

### New concept — chunk overlap

Consecutive chunks deliberately repeat text: here the last ~80 tokens of one
chunk are also the first ~80 of the next. The reason is that a chunk boundary is
a guess about where meaning ends, and a wrong guess is unrecoverable — a fact
split across the boundary exists *intact in neither chunk*, so no amount of
retrieval skill finds it. Overlap buys a second chance: if the boundary falls
badly, the following chunk still holds the whole fact.

The cost is index inflation. At 400/80 the corpus produces roughly 20% more
vectors than it strictly needs, which is 20% more storage and marginally slower
search. That is a cheap insurance premium against a silent, unfixable failure.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `DEFAULT_MAX_TOKENS = 400` | Target chunk size, under bge-small's 512 limit. | Default argument. |
| `DEFAULT_OVERLAP_TOKENS = 80` | 20% repeat between neighbours. | Default argument. |
| `MIN_CHUNK_TOKENS = 48` | Below this a trailing chunk is a "runt" and gets merged. | In `_merge_runt`. |
| `_SEPARATORS` | Cut points, best to worst: paragraph, line, sentence, clause, word. | In `_split_recursive`. |
| `estimate_tokens()` | Fallback counter, ~4 chars/token. Tests only — see traps. | If no counter is injected. |
| `TextBlock` | Input: a span of text + its page (`None` if unpaginated). | Built by ingestion. |
| `Chunk` | Output: text, index, token count, `page_start`/`page_end`. | Returned to ingestion. |
| `chunk_blocks()` | Primary entry point. Validates, segments, packs. | Once per document. |
| `chunk_text()` | Wrapper for sources with no pages (.txt, .docx). | Once per document. |
| `_segment()` | Stage 1: blocks → atomic segments that each fit. | Inside `chunk_blocks`. |
| `_split_recursive()` | Descends the separator hierarchy until a piece fits. | Per block. |
| `_split_by_length()` | Last resort: hard character cut for separator-less text. | Rarely. |
| `_pack()` | Stage 2: greedily fill chunks, carry the overlapping tail. | Inside `chunk_blocks`. |
| `_overlap_tail()` | Picks the trailing segments to repeat. | At each chunk boundary. |
| `_emit()` | Joins segments into a `Chunk`, computes the page span. | Per chunk. |
| `_merge_runt()` | Folds an undersized *final* chunk into its predecessor. | Once, at the end. |

### Flow of the code

1. Ingestion calls `chunk_blocks(blocks, count_tokens=<real tokenizer>)`. A PDF
   arrives as one `TextBlock` per page.
2. Guard: `overlap_tokens >= max_tokens` raises immediately. That configuration
   would make the carried tail as big as the chunk, so the packer could never
   advance — a hang, not a wrong answer, which is why it is checked up front.
3. **Stage 1, `_segment`.** Each block is broken down by `_split_recursive`
   until every piece fits inside `max_tokens`. Blank blocks are dropped (scanned
   PDFs are full of blank pages). Each resulting segment remembers its page.
4. `_split_recursive` tries `"\n\n"` first. If the piece still doesn't fit, it
   descends to `"\n"`, then sentences, then clauses, then words. It stops
   descending the moment a piece fits — so most cuts land on paragraph breaks
   and a mid-word cut requires every level above to have failed.
5. **Stage 2, `_pack`.** Segments are added in document order until the next one
   would overflow. The chunk is emitted, `_overlap_tail` selects the trailing
   segments worth ~80 tokens, and those become the start of the next chunk.
6. `_emit` joins the segments and records `page_start`/`page_end` as the min and
   max page present — which is how a chunk spanning a page break honestly
   reports both.
7. `_merge_runt` folds an undersized final chunk into its predecessor.

### Math

**Chunk count.** For a document of `T` tokens with chunk size `C` and overlap
`V`, each chunk after the first advances by only `C - V` tokens, so:

```
n ≈ (T - V) / (C - V)
```

At `T=100,000`, `C=400`, `V=80`: `n ≈ 99,920 / 320 ≈ 312` chunks. With no
overlap it would be 250. That difference — 25% more vectors — is the price of
overlap, and the formula shows why overlap gets expensive fast: as `V` → `C`,
the denominator → 0 and the count explodes. At `V = 0.5C` you have doubled your
index for diminishing protection. 20% is the usual sweet spot.

**Why 400 and not 512.** bge-small's context window is 512 tokens; text beyond
that is silently truncated by the model — embedded away into nothing, with no
error. The 112-token margin covers three things: the gap between an estimated
and a real token count, the slight overshoot `_merge_runt` is permitted, and any
future prefix the embedder might prepend. Running at 500 would mean occasionally
discovering that the end of a chunk was never actually embedded.

### Common gotchas / traps

- **`estimate_tokens` is a stopgap, not a default to rely on.** At ~4 chars per
  token it is roughly right for English prose and badly wrong for code, URLs,
  tables and non-English text, all of which tokenize far denser. Ingestion must
  pass the embedder's real tokenizer. The consequence of forgetting is silent
  truncation, not a crash.
- **Zero overlap between two chunks is possible and is not a bug.** If a chunk
  consists of a single segment — an entire page or paragraph that already fits —
  `_overlap_tail` returns nothing, because it never consumes the whole chunk.
  That case only arises when the boundary lands exactly on a paragraph break,
  which is precisely where overlap matters least. Verified in practice: prose
  split into sentence segments overlaps by 54–56 words (~78 of the 80 target);
  a page that fits in one chunk overlaps by zero.
- **`_overlap_tail` must never return every segment.** If it did, the next chunk
  would start already holding everything the last one did, and `_pack` would
  loop forever. Hence `reversed(segments[1:])` — one short of the whole list.
- **Sentence splitting on `[.!?]` is wrong for "Dr.", "3.14", "e.g."** It is
  kept anyway, because a slightly wrong sentence boundary costs far less than
  not attempting sentence boundaries at all, and a real sentence tokenizer is a
  dependency this file is better off without.
- **Chunks are frozen on purpose.** Once ingestion embeds a chunk, the vector
  *means* that exact text. Mutating the text afterwards silently decouples the
  two, and search starts returning text that isn't what matched.
- **Blocks are chunked as a continuous stream, not page by page.** Chunking each
  page independently would guarantee a boundary at every page break — the single
  most common place for a sentence to continue.

### Interview questions

**Q: How do you choose a chunk size, and what goes wrong at each extreme?**
It is a recall/precision trade and both ends fail, differently. Chunks that are
too small hold a fact without its context — "it must be returned within 14 days"
retrieves nothing useful when the "it" was named a paragraph earlier — and they
fragment the index so the top-k fills with pieces of one section. Chunks that
are too large dilute the embedding: a single vector averaging four topics is
close to nothing in particular, so it loses to a tighter chunk even when it
contains the answer, and it wastes the LLM's context on irrelevant text. The
hard ceiling is the embedding model — bge-small truncates past 512 tokens, so
anything larger is partly not embedded at all, silently. I chose 400 to sit
under that limit with margin for token-count estimation error. The honest answer
though is that the size should be tuned against an eval set for the actual
corpus, because the right value depends on how the documents are written:
dense technical PDFs and chatty support articles do not want the same number.

**Q: Your chunker takes a token-counting function as a parameter instead of
importing a tokenizer. Why?**
Three reasons, escalating. The mechanical one is that the project forbids vendor
and model libraries outside `app/providers/`, and a tokenizer is part of the
embedding model — the correct count depends on which embedder is configured, so
importing one here would hardcode an assumption that breaks the moment someone
sets `EMBEDDING_PROVIDER=openai`. The practical one is testability: the module
has no torch import and no model load, so its tests run in milliseconds and the
chunking logic can be exercised with a trivial counter. The design one is that
it names the real dependency honestly — the chunker does not need a tokenizer,
it needs to know how long a string is in tokens, and those are different things.
The cost is that a caller can forget to pass the real counter and silently get a
crude estimate, which is why the fallback is documented as test-only and why
there is 112 tokens of headroom to absorb the error.

---

## 2. `backend/app/core/extraction.py`

**Purpose** — The front door of ingestion: turns an uploaded PDF, DOCX, TXT or
Markdown file into the page-tagged `TextBlock` list the chunker consumes.
Without it the pipeline only accepts strings, and "upload a PDF" — which is the
actual product — does not exist.

**Analogy** — A translator who reads three languages and always hands you the
same plain notepad, noting which page each passage came from.

### New concept — text-layer vs scanned PDFs

A PDF either stores real character data (a *text layer*) or is a photograph of a
page with no characters in it at all. Both open fine in any viewer and look
identical to a human. `pypdf` reads the first correctly and returns an **empty
string** for the second — no exception, no warning.

That makes it the most dangerous input this module handles. Ingesting a 200-page
scanned contract would produce zero chunks, report success, and leave a chatbot
that answers "I don't have that information" to every question about a document
the user watched upload successfully. `_extract_pdf` therefore measures how many
pages yielded usable text and raises `NoTextLayerError` below a threshold. OCR
is explicitly out of scope for Phase 1; detecting the need for it is not.

### A note on Hard Rule #2

This file imports `pypdf` and `python-docx` directly, inside `core/`. The rule
bans vendor SDKs outside `app/providers/`, and this is a deliberate reading of it
by intent rather than by letter: the rule exists so *swappable AI providers* stay
behind the factory and `LLM_PROVIDER=groq` remains a config change. Nothing about
PDF parsing is selected by an env var, so wrapping one PDF library in a provider
factory would be ceremony around a single implementation. AI vendors are
swappable and go behind the factory; file-format libraries are not and do not.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `SUPPORTED_SUFFIXES` | `.pdf .docx .txt .md`. The authoritative list. | On every call. |
| `_MIN_CHARS_PER_PAGE = 16` | Below this, a page counts as having no text layer. | Per PDF page. |
| `_MIN_TEXT_PAGE_RATIO = 0.2` | Fraction of pages needing text before we accept the doc. | Once per PDF. |
| `_TEXT_ENCODINGS` | utf-8, then cp1252. | Per plain-text file. |
| `ExtractionError` + 5 subclasses | Typed failures the API layer maps to distinct user messages. | On failure. |
| `extract()` | Dispatch on suffix, validate, log. The only public function. | Once per upload. |
| `_extract_pdf()` | One block per page, 1-indexed. Handles encryption and scans. | For `.pdf`. |
| `_extract_docx()` | Paragraphs **and tables** into one block, `page=None`. | For `.docx`. |
| `_extract_plain()` | Encoding-tolerant read into one block, `page=None`. | For `.txt`/`.md`. |

### Flow of the code

1. `extract(path)` checks the file exists, then that the suffix is supported —
   both before opening anything, so a bad upload fails cheaply.
2. Dispatch to the format handler.
3. **PDF:** open with `PdfReader`. If encrypted, try the empty password first
   (many "protected" PDFs use an empty owner password and open fine); only raise
   `EncryptedDocumentError` if that fails. Then walk pages with
   `enumerate(..., start=1)`, collecting non-empty text and counting how many
   pages carried a real text layer. After the loop, if too few pages had text,
   raise `NoTextLayerError`.
4. **DOCX:** collect paragraph text, then walk `document.tables` separately and
   pipe-join each row. Tables are *not* in `document.paragraphs`, and in real
   business documents they hold exactly what users ask about.
5. **Plain:** try utf-8, then cp1252; if both fail, decode lossily with a
   warning rather than rejecting the upload.
6. An empty result raises `EmptyDocumentError` — "no content" is a condition the
   caller must handle now, not discover later as zero chunks.

### Math

Only the scan heuristic. A PDF is rejected when

```
pages_with_text / total_pages < 0.2
```

where a page "has text" if it yields at least 16 non-whitespace characters. Both
numbers are deliberately loose in the same direction — toward accepting. 16
rather than 0 because pypdf often scrapes a stray page number or ligature off a
scanned page, so a strict `== ""` test would miss the exact case this exists to
catch. 20% rather than 100% because real PDFs legitimately contain full-page
images, cover pages and blank separators; rejecting a valid document is a worse
error than accepting a marginal one, since the user can see and act on a bad
answer but cannot act on a rejected upload.

### Common gotchas / traps

- **Page numbers must be 1-indexed.** `enumerate(reader.pages, start=1)`. A
  0-indexed citation is off by one on every single answer — correct-looking and
  wrong, which is the hardest kind of bug to notice and the easiest to ship.
- **DOCX has no page numbers, and inventing one is worse than omitting it.**
  Pagination is computed by the renderer from fonts, margins and paper size;
  Word itself does not know where breaks fall until layout. Hence `page=None`
  and a citation that honestly says "no page".
- **`document.paragraphs` silently excludes tables.** Pricing tiers, SLA
  numbers and contact details all live in tables. Skipping them drops the
  highest-value content in the document without any sign that it happened.
- **A renamed `.doc` is not a `.docx`.** It passes the suffix check and fails in
  the parser, which is why `PackageNotFoundError` gets its own message naming
  the likely cause.
- **The bare `except Exception` in `_extract_pdf` is load-bearing.** pypdf
  raises assorted untyped exceptions on malformed files rather than one family,
  so a narrow catch would let a corrupt upload crash the worker.
- **Lossy decoding is a choice, not an accident.** A document with a few mangled
  characters is still worth searching; the warning records that it happened.

### Interview questions

**Q: A user uploads a 200-page PDF and the chatbot then answers "I don't have
that information" about everything in it. How do you debug that?**
I would work backwards along the pipeline, because the symptom is identical at
every stage. First: how many chunks does the document have in the database? If
it is zero, the failure is before retrieval and the likely cause is a scanned
PDF — an image-only document where the parser returned empty strings without
erroring. That is exactly why extraction raises `NoTextLayerError` rather than
returning nothing: it converts a silent, hard-to-trace failure into a specific
message at upload time, when the user still has the file open and can act on it.
If chunks do exist, I would check whether their text looks like prose or like
garbled ligatures, then whether the query embeds sensibly, then whether the
Qdrant filter is over-restrictive. The general principle is that "no answer" is
the same observable for a parsing bug, an embedding bug and a filter bug, so the
pipeline has to fail loudly and specifically at each stage or you are left
bisecting.

**Q: Why give this module its own exception hierarchy instead of raising
ValueError?**
Because the API layer has to turn each failure into a *different instruction for
a human*, and that distinction only survives if the type carries it. A
password-protected file, an unsupported format, a scanned PDF and a corrupt file
are all "your upload failed" to a generic handler, but they mean "remove the
password and retry", "convert to PDF first", "this needs OCR, which we do not
support", and "this file is damaged" respectively. Encoding that in the type
rather than by string-matching an error message means the mapping is checkable
and does not break when someone rewords a message. It also keeps the decision
where the knowledge is: this module knows *why* extraction failed, the HTTP layer
knows what status code and copy that deserves, and neither needs to know the
other's business.

---

## 3. `backend/app/core/vectorstore.py`

**Purpose** — The only module that talks to Qdrant. Turns `Chunk` + vector into
stored points, and a query vector into `SearchHit`s that already carry their
citation. Without it there is nowhere to put embeddings and nothing to search.

**Analogy** — A librarian who files every book with the borrower's name on the
spine, and who physically cannot fetch you a book belonging to someone else.

### New concept — payload filtering as multi-tenancy

Qdrant calls the JSON metadata stored beside each vector its **payload**. It is
not searched semantically; it is filtered on. Our entire tenant isolation model
is a payload filter: one collection holds every organisation's chunks, and
`org_id` + `chatbot_id` conditions decide what a query can see.

The alternative — a Qdrant collection per tenant — sounds safer and is worse in
practice. Each collection carries its own HNSW graph and segment files, so
per-collection overhead is paid thousands of times, and creating one per signup
turns onboarding into a schema operation. Payload filtering on an indexed field
is the documented pattern.

The honest cost: **isolation becomes application logic rather than physical
separation.** A forgotten filter is a data breach, not a bug. That single
sentence is the justification for everything defensive in this file, and for why
Phase 2's isolation test is described in PLAN as sacred.

### How Hard Rule #3 is enforced structurally

The rule says every Qdrant search must filter by org_id + chatbot_id, with no
unfiltered path. A comment saying so would be worth nothing. Instead:

- `search()` takes `org_id` and `chatbot_id` as **required keyword-only**
  arguments. Omitting one is a `TypeError` at the call site, not a subtle
  runtime behaviour.
- **There is no `filter` parameter** for callers to pass. The filter is built
  inside `_scope_filter`, from arguments the caller cannot omit. Callers cannot
  supply a filter, therefore cannot supply an empty one.
- `_scope_filter` appends both conditions **unconditionally, before** any
  optional condition. There is no branch in which the condition list is empty.
- `_require_scope` rejects empty strings, because `org_id=""` does not bypass
  the filter — it matches points whose org_id is literally empty and returns
  nothing. That is the dangerous shape: the caller reads an empty result as "no
  relevant documents" rather than "my scoping is broken".
- `delete_document` is filtered too. Deleting by document_id alone is correct
  today and a cross-tenant deletion the moment ids are reused or guessable.

Phase 1 is single-tenant, so all of this is currently unused. That is the point:
adding a filter argument in Phase 2 to a function that already has callers means
auditing every call site and hoping.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `_POINT_NAMESPACE` | Fixed UUID that seeds every point ID. **Never change it** — a new namespace orphans the whole collection. | Import time. |
| `PAYLOAD_*` constants | Payload keys as constants. A typo in a filter string silently matches nothing, which looks exactly like "no relevant documents". | Everywhere. |
| `_INDEXED_FIELDS` | org/chatbot/document get payload indexes. | On collection create. |
| `SearchHit` + `.citation` | Flattened result carrying text AND provenance. | Per search result. |
| `_require_scope` | Rejects empty tenant ids. | Every read and write. |
| `_point_id` | Deterministic UUIDv5 → idempotent writes. | Per chunk. |
| `ensure_collection` | Creates collection + indexes, or verifies the existing one. | Worker/API boot. |
| `upsert_chunks` | Batched write, `wait=True`. | Per document. |
| `search` | Filtered ANN query, wide limit. | Per chat message. |
| `_to_hit` | Qdrant's `ScoredPoint` → our `SearchHit`. | Per result. |

### Flow of the code

**Write path.** `ensure_collection()` at boot → if the collection exists,
`_verify_dimension` compares its vector size against the live embedder's and
raises `CollectionMismatchError` on disagreement; if absent, create it with
COSINE distance and build a keyword payload index per filtered field. Then
`upsert_chunks` validates scope, checks `len(chunks) == len(vectors)`, builds
`PointStruct`s with deterministic ids, and writes in batches of 128 with
`wait=True`.

**Read path.** `search()` validates scope → `_scope_filter` builds a `must`
(AND) filter → `query_points` with `with_vectors=False` → `_to_hit` flattens
each result. `limit` here is the *wide* stage (40), not what the model sees; the
reranker narrows to 6.

### Math

Idempotent ids:

```
point_id = uuid5(_POINT_NAMESPACE, "{org}:{bot}:{doc}:{chunk_index}")
```

UUIDv5 is a SHA-1 hash of namespace + name, truncated and version-tagged. Same
inputs, same output, on any machine, in any process, forever — no coordination
and no lookup table. That property is what makes re-ingestion overwrite rather
than duplicate.

Why that matters concretely: Celery retries failed ingestion in Phase 2. With
random ids, a task that dies at chunk 900 of 1000 and retries writes 900
duplicates. Those duplicates are not merely wasted storage — near-identical
chunks crowd out genuinely different passages in the top-40 candidate list, so
retrieval quality degrades in proportion to how often ingestion failed.

Distance is cosine: `cos(θ) = (A·B)/(||A||·||B||)`. bge already returns unit
vectors, where cosine and dot product are identical, so this costs nothing today
and protects against a future embedder that does not normalize — where the
failure would be a silent ranking regression rather than an error.

### Common gotchas / traps

- **`chunks` and `vectors` are matched positionally.** A misalignment attaches
  each chunk's text to its neighbour's vector and produces a searchable index
  that returns confidently wrong passages with no error anywhere. Hence the
  length check and `zip(..., strict=True)`.
- **`wait=True` on writes is not optional here.** Returning success while points
  are still being indexed means marking a document "ready" and then answering
  "I don't have that information" about it.
- **Filtering without a payload index still returns correct results**, just
  slowly and worse as tenants multiply — so the problem never announces itself
  during Phase 1 testing with one tenant.
- **`dimension` has no default** and must come from `embedder.dimension`.
  Hardcoding 384 makes swapping to a 1536-dim provider a silent corruption.
- **Never return `ScoredPoint` upward.** Retrieval and generation would inherit
  the Qdrant SDK's shapes, and a later move to pgvector would reach into files
  that have nothing to do with storage.

### Verification

`qdrant_client` is not installed in this environment, so nothing that touches a
live server could run. The pure logic was exercised against a stubbed SDK:

```
deterministic       : True   cb3425cb-9df8-57ce-9d06-3a95631c9af9
differs by chunk    : True
differs by org      : True      <- same doc, different tenant => different point
differs by bot      : True
rejected scope      : ('', 'b') / ('o', '') / ('', '')
citation single page: handbook.pdf p.12
citation span       : handbook.pdf pp.12-13
citation same page  : handbook.pdf p.12
citation no page    : handbook.pdf
```

The org/bot lines matter beyond determinism: they prove the tenant is part of
the identity of a point, so two organisations ingesting the *same file* get
distinct points rather than silently overwriting each other.

### Interview questions

**Q: How do you isolate tenants in a shared vector database, and how do you know
it works?**
The mechanism is a mandatory payload filter — every point carries its org and
chatbot id, and every query filters on both against an indexed field. I chose a
shared collection over collection-per-tenant because Qdrant carries real
per-collection overhead in graph and segment files, and because creating a
collection per signup makes onboarding a schema operation. The honest trade is
that isolation becomes application logic rather than physical separation, so a
single forgotten filter is a breach rather than a bug. That is why I would not
rely on remembering: the filter is not an optional parameter, it is built inside
the store from keyword-only arguments the caller cannot omit, so there is no
call that compiles and skips it. Knowing it works is a separate question from
building it, and the answer is an adversarial test — two orgs, different
documents, then attempts to reach across with a forged chatbot id and a missing
filter — running in CI forever.

**Q: Why are your point IDs derived from a hash instead of generated randomly?**
Because ingestion gets retried, and retries with random ids duplicate data. A
deterministic UUIDv5 over org, chatbot, document and chunk index means writing
the same chunk twice is an overwrite rather than an insert, which makes the whole
operation idempotent without any bookkeeping — no lookup table, no coordination
between workers, and it holds across processes and machines. The failure it
prevents is subtle: duplicate chunks are not just wasted storage, they are
near-identical passages competing for slots in the top-40 candidate list, so
retrieval quality quietly degrades in proportion to how often ingestion failed.
The alternative would be deleting a document's points before rewriting them,
which is more moving parts and leaves a window where the bot has no data.

---

## 4. `backend/app/core/ingestion.py`

**Purpose** — The orchestrator: a file on disk becomes searchable chunks. It
owns no algorithm of its own, only the order in which the other four modules run
and the arguments they get. Without it, extraction, chunking, embedding and
storage are four things that have never met.

**Analogy** — A kitchen pass. It cooks nothing; it makes sure the plate leaves in
the right order with the right things on it, and it is where you notice that the
sauce went out before the steak.

### Why an orchestrator earns its own file

It would be possible to call these four modules from the CLI, and later from the
Celery task, and later again from an API route. That is how a pipeline ends up
subtly different in three places — one caller forgets the real tokenizer, another
forgets `ensure_collection`, a third batches differently. The sequence *is* the
logic, so it gets one home and every entry point calls it.

### The two decisions that carry the file

**1. The real tokenizer is passed to the chunker.**

```python
chunk_blocks(blocks, count_tokens=embedder.count_tokens, ...)
```

`chunk_blocks` defaults to the 4-chars-per-token estimate so it can stay pure and
dependency-free for tests. That default is *for tests*. This one line is what
sections 17-19 existed to make possible, and omitting it is invisible on English
prose while silently truncating exactly the technical documents this product is
for. It is a one-line difference between correct and quietly wrong.

**2. Embed and upsert in interleaved batches**, not embed-everything then
store-everything.

| | All-then-all | Interleaved (chosen) |
|---|---|---|
| Peak memory | Every vector for the document at once | One batch |
| Crash at 90% | Nothing written, all work lost | 90% written and queryable |
| Round trips | Fewest | More |

The second row is the real argument, and it only works because of the previous
file. Deterministic point ids mean the retry *overwrites* the batches that
already landed instead of duplicating them, so partial progress is durable rather
than corrupt. That is what makes Celery's retry in Phase 2 safe, and it is a
direct payoff of the idempotency decision in `vectorstore.py` — a good example of
one design choice paying for another module's simplicity.

### Deliberately not caught

This module lets `ExtractionError`, `ProviderError` and `VectorStoreError`
propagate untouched. They are already typed, already carry the provider or the
filename, and wrapping them in an `IngestionError` would bury the cause one level
deeper for no gain. The decision of what becomes a retry and what becomes a
`failed` document status is *policy*, and it belongs to Phase 2's Celery task,
which is the only layer that knows how many attempts have already happened.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `DEFAULT_INGEST_BATCH = 64` | Chunks per embed+write round trip. Distinct from the embedder's internal batch size, which is about throughput; this one bounds crash loss and peak memory. | Per document. |
| `IngestionResult` | Counts returned rather than logged-and-discarded, so the CLI prints them and Phase 2 writes them to the documents row. | Per document. |
| `content_document_id` | SHA-256 of the file bytes, first 32 hex chars. | When no id is supplied. |
| `ingest_file` | The pipeline. | Per upload. |
| `delete_document` | Thin pass-through, so callers have one ingestion-facing module. | On delete. |
| `_batched` | Fixed-size batches, order preserved. | Per document. |
| `_page_count` | Highest page seen, or None. | Per document. |

### Flow of the code

1. Resolve defaults: embedder from the factory, source from the filename,
   document id from the content hash.
2. `extract(path)` — a scanned PDF fails **here**, loudly.
3. `chunk_blocks(...)` with the real `count_tokens`.
4. Guard: zero chunks from non-empty blocks is a sizing bug, not an empty file,
   so it raises rather than writing nothing successfully.
5. `store.ensure_collection()` — **before** embedding, deliberately. Discovering
   a dimension mismatch after computing several thousand vectors throws all of
   that work away.
6. For each batch: `embed_documents` then `upsert_chunks`.
7. Return `IngestionResult`.

### Math

Content-addressed ids:

```
document_id = sha256(file_bytes).hexdigest()[:32]
```

Content-addressed rather than path-addressed, and the two failure modes it picks
between are worth stating: renaming a file must NOT create a second copy in the
index, while editing a file MUST create a new document. Hashing the bytes gives
both behaviours for free. Truncating to 32 hex characters is 128 bits, which is
collision-proof at any corpus size this product will ever see.

Combined with `vectorstore._point_id`, the full identity of a stored chunk is:

```
uuid5(NAMESPACE, "{org}:{bot}:{sha256(bytes)[:32]}:{chunk_index}")
```

So re-uploading an identical file is a no-op that overwrites itself, and every
level of the pipeline is idempotent end to end.

### Common gotchas / traps

- **Forgetting `count_tokens=` is the highest-cost silent bug in this file.**
  Everything still runs; chunks are simply sized with a ruler that does not
  match the model, and the overflow is truncated during encoding with no error.
- **`ensure_collection` before embedding, not after.** Ordering here is an
  economic decision, not a stylistic one.
- **Batch order is load-bearing.** `upsert_chunks` zips chunks against vectors
  positionally, so a batch must reach the embedder and the store in the same
  sequence. `_batched` preserving order is a correctness property.
- **`store` is injected, not constructed.** One client and one connection pool
  shared by CLI, API and worker — and tests can pass a fake with no Qdrant.
- **`pages=None` for .docx and .txt** rather than 1. Claiming a one-page
  document would be a fact we invented.

### Verification

This file is fully testable without any infrastructure, because the embedder and
store are injected and everything below them is pure. With a fake embedder and a
recording fake store:

```
handbook.md: 4 chunks, 1254 tokens -> 4 points
ensure_collection calls : 1
tenant on every write   : {('org_a', 'bot_1')}
pages (md => None)      : None
doc id stable           : True
doc id content-bound    : True
empty file              : EmptyDocumentError
```

The first run was **not** a valid test of batching: 4 chunks against
`batch_size=5` is a single batch, so the loop never iterated. Re-run against a
200-section document:

```
big.md: 20 chunks, 7901 tokens -> 20 points
batches        : 4 [5, 5, 5, 5]
order preserved: True
all chunks sent: True
```

That is the check that matters — order intact *across* batch boundaries, which
is the property `upsert_chunks`'s positional zip depends on. Worth recording the
mistake as much as the result: a test that exercises one iteration of a loop
tells you nothing about the loop.

### Interview questions

**Q: Ingestion of a large document fails halfway. What state is the system in,
and what happens on retry?**
Whatever was already written is valid and queryable, because embedding and
writing are interleaved in batches rather than done in two big phases. The
document is incomplete, but it is not corrupt. On retry the pipeline runs from
the beginning, and the batches that already landed are *overwritten* rather than
duplicated, because point ids are a deterministic hash of org, chatbot, document
and chunk index rather than random values. So the retry converges on the correct
final state no matter how many times it happens or where it failed. That is a
deliberate pairing of two decisions in different modules: interleaved batching
would be dangerous with random ids, since every failure would leave duplicate
near-identical chunks competing for slots in the candidate list, and idempotent
ids are what make it safe. The remaining gap is that a document can sit
permanently half-ingested if retries are exhausted, which is why Phase 2 tracks a
status on the document row rather than inferring health from the index.

**Q: Why does this module not catch the exceptions raised beneath it?**
Because it does not know the policy, and catching would destroy information it
cannot replace. The errors underneath are already typed and already specific — a
scanned PDF, a rate-limited provider, a dimension mismatch — and wrapping them in
a generic `IngestionError` would leave every caller string-matching to recover
what the type already said. More importantly, the decision that actually needs
making is "retry or fail permanently", and this module cannot make it: it does
not know which attempt this is, or whether the provider error was the retryable
kind. The Celery task in Phase 2 knows both. So the rule I would state is that a
layer should only catch an exception if it can either fix it or add information,
and pure orchestration can do neither.

---

## 5. `backend/app/core/retrieval.py`

**Purpose** — Turns a question into the handful of chunks worth answering from:
embed the query, search wide, rerank narrow. Without it the LLM either sees
nothing or sees the whole document, and neither produces a grounded answer.

**Analogy** — A librarian who pulls forty plausible books off the shelves in
thirty seconds, then actually opens six of them before handing you the best one.
Two different skills, deliberately done by two different stages.

### Why two stages, stated precisely

The stages fail in opposite directions, and each is asked only for what it is
good at.

| | Stage 1 — vector search | Stage 2 — rerank |
|---|---|---|
| Model | **Bi-encoder**: query and document embedded separately | **Cross-encoder**: both read together in one pass |
| Cost | Dot products against an HNSW index, milliseconds | One model pass per candidate |
| Accuracy | Approximate — crude comparison, approximate index | High: every query token attends to every document token |
| Asked for | **Recall** — right chunk somewhere in 40 | **Precision** — right chunk at rank 1 |

The document side of stage 1 was embedded at ingestion, which is what makes it
cheap: nothing about the query was known then, so the comparison happens between
two vectors that never saw each other. Stage 2 gives up that precomputation in
exchange for the query and document being processed *jointly* — which is exactly
why it is accurate and why it can never run over a whole collection.

Why not one stage? Vector search alone reliably puts the right chunk in the
top-40 and frequently at rank 11 — and only 6 reach the model. Reranking alone
would be a cross-encoder pass per chunk in the collection: seconds per query.

### The decision NOT made: no score threshold

There is no minimum-relevance cutoff anywhere in this file, and that is
deliberate. Rerank scores are **provider-private**: bge returns unbounded logits
(8.4 is a strong match), Cohere returns a calibrated 0..1. A `score > 0.5` filter
hardcoded here would mean one thing today, something entirely different after
`RERANK_PROVIDER` changes, and the symptom would be an empty result set that is
indistinguishable from "this bot has no relevant documents".

So the scores are used for **ordering only, never for thresholding** — the same
rule stated in the `RerankProvider` contract. Deciding that the retrieved context
does not answer the question is delegated to `generation.py`, where a model that
can actually read the passages says so in words. A number that cannot read them
is the wrong tool for that judgement.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `DEFAULT_CANDIDATES = 40` | Wide stage. Wide enough that a poorly-ranked chunk survives, narrow enough for one cross-encoder batch. | Per query. |
| `DEFAULT_TOP_K = 6` | What reaches the model. | Per query. |
| `RetrievedChunk` | Hit + `vector_rank` + `rerank_score`. | Per result. |
| `retrieve()` | The pipeline; the only public function. | Per chat message. |
| `_rerank()` | Maps `(index, score)` pairs back onto our own hits. | Per query. |
| `_elapsed_ms()` | Monotonic timing for the three stages. | Per stage. |

### Flow of the code

1. Reject an empty query, and reject `top_k > candidates` — the reranker can
   only narrow a candidate list, never extend it, so that config is a mistake
   whose only symptom would be quietly fewer chunks than asked for.
2. `embed_query` — **not** `embed_documents`. bge is asymmetric and needs its
   instruction prefix on the query side; the provider applies it, this layer only
   has to call the right method.
3. `store.search(...)` with `limit=candidates`, tenant arguments passed straight
   through. There is no unfiltered path here because this function cannot
   construct one.
4. No hits → return `[]`. A real answer, not an error: a bot whose documents are
   still processing genuinely has nothing to say.
5. `_rerank` → `RetrievedChunk` list, best first.
6. Log per-stage timings and `top_moved_from_rank`.

### Why the reranker returns indexes, not documents

`RerankProvider.rerank` returns `(index, score)` pairs pointing into the list we
gave it. That contract choice pays off here: we map straight back onto our own
`SearchHit` objects and keep `document_id`, `source` and `page` intact. The
alternative — matching returned text against our text — would mean trusting a
vendor not to normalise whitespace or truncate, and a failed match would silently
drop a chunk. Indexes cannot be subtly altered; they are either in range or an
error, which is why the range check raises instead of skipping.

### Math

Cost per query, with N chunks in the collection, C candidates, d = 384:

```
stage 1  ≈ O(log N · d)     HNSW graph walk
stage 2  = C forward passes of a cross-encoder
```

Stage 2 dominates and is **linear in C**, which is the entire reason C is capped
at 40 rather than 200. The tuning question is where the recall curve flattens:
going 40 → 80 doubles rerank latency, and buys nothing if the right chunk was
already inside the top 40 in ~all cases. That is measured against an eval set,
not guessed — `Recall@40` is the number to watch (the fraction of questions whose
answer chunk appears anywhere in the 40), because it is the ceiling stage 2 can
never exceed.

### Common gotchas / traps

- **`top_k > candidates` is a config bug, not something to clamp.** Clamping
  hides it; the only symptom would be fewer chunks than requested.
- **Never threshold on rerank score.** Provider-private scales; see above.
- **An out-of-range index from a provider must raise.** Skipping it returns
  fewer chunks with no indication why.
- **`perf_counter`, not `time()`.** Monotonic, so an NTP correction mid-request
  cannot produce a negative duration in the logs.
- **More context is not better.** Six passages with four off-topic produce worse
  answers than two good ones — the model has to decide what to trust, and
  irrelevant text is an active distractor rather than neutral filler.
- **Empty results are not an error.** Conflating "no documents" with "failure"
  turns a normal state (ingestion still running) into a 500.

### Verification

Constructed so the answering chunk sits at **vector rank 11** — inside the
top-40, outside the top-6, which is precisely the case the two-stage design
exists for:

```
searched wide           : 40 candidates
tenant passed through   : org_a / bot_1
returned                : 6 chunks

cross-encoder top result: Refunds are issued within 30 days. | from vector rank 11
noop top result         : Filler 0.                          | from vector rank 0
answer survives noop?   : False        <- the reranker earning its place
citation                : handbook.pdf p.12
empty query             : ValueError
top_k > candidates      : ValueError
bad rerank index        : IndexError
no chunks at all        : [] (empty list, not an error)
```

**An honest caveat about what this proves.** The scenario is constructed: the
fake cross-encoder finds the answer because it greps for the word, and the answer
was placed at rank 11 by hand. This demonstrates that the *mechanism* routes
correctly — wide search, rerank, remap, degrade visibly under `noop` — and says
nothing about whether real bge behaves this way on real documents. PLAN §6's
acceptance criterion needs actual models over an actual PDF. This is the unit
test; that is the acceptance test, and they are not substitutes.

### Interview questions

**Q: Why not just retrieve the top 6 from the vector search and skip reranking?**
Because the two models are good at different things and the cheap one is good at
the wrong one. A bi-encoder embeds query and document separately, so it is
comparing two vectors that never saw each other, and on top of that HNSW is an
approximate index — it is strong at getting the right chunk *somewhere* in a wide
list and unreliable about exactly where. In practice the answering chunk often
lands around rank 11, which is fine if you take 40 and fatal if you take 6. The
cross-encoder reads the query and the passage together in one pass, so it can
tell that a chunk mentioning "30 days" answers a question about refund timing;
that costs a model pass per candidate, which is affordable over 40 and impossible
over a collection. So the pipeline is shaped by the cost asymmetry: use the cheap
approximate model for recall, spend the expensive accurate one only on a short
list. The observable payoff is the Phase 1 acceptance criterion — switching the
reranker to noop should visibly degrade answers, and if it does not, either the
reranker is not earning its latency or the retrieval was never the bottleneck.

**Q: Would you drop chunks whose relevance score is below some threshold?**
Not in this layer, and the reason is that the score has no portable meaning. bge
returns unbounded logits while Cohere returns a calibrated zero-to-one, so any
constant I write is tied to one provider and silently changes meaning the moment
the provider is swapped — and the failure mode is an empty result set, which
looks exactly like a bot with no relevant documents rather than like a
misconfiguration. There is a deeper reason too: "does this context actually
answer the question" is a judgement about meaning, and a scalar produced by a
ranking model is not that judgement — it is only calibrated to compare passages
*against each other* within one call. So I pass the top passages to the model and
let the prompt require it to say "I don't have that information" when they do not
support an answer. That puts the decision where there is enough information to
make it, and it degrades gracefully rather than silently.

---

## 6. `backend/app/prompts/grounded_answer.txt`

**Purpose** — The system prompt that makes this a RAG product rather than a
chatbot with extra steps: answer only from the excerpts, cite every claim, and
refuse in a fixed sentence when the excerpts do not contain the answer. Without
it, retrieval feeds a model that will happily answer from memory instead.

**Analogy** — An open-book exam where "not in the book" earns full credit, and
every claim must carry a page reference. Both halves are needed: without the
first, guessing beats admitting; without the second, guessing is undetectable.

### New concept — grounded prompting

An LLM asked something it cannot answer from context will answer anyway,
fluently and wrongly. For a support bot on a company's own documents that is the
core failure mode — an invented refund window is worse than no bot at all,
because the user cannot tell it apart from a correct one. Grounded prompting is
the set of instructions that makes refusal an acceptable outcome and makes
unsupported claims visible. Full entry in LEARNING.md.

### Why a file and not a string in `generation.py`

PLAN requires it, and the reasons are worth stating:

- **Prompts are edited far more often than the code around them.** A prompt
  change should be a diff of prose, reviewable by someone who does not read
  Python.
- **`git diff` on a `.txt` is readable.** The same change inside a triple-quoted
  Python string arrives as an unreadable blob with escaping noise.
- **Phase 4 makes the system prompt editable per chatbot.** The file becomes the
  default that a database column overrides — a shape that only works if the
  prompt is already data rather than code.

### The trap specific to this file

**Every byte in it is sent to the model, on every request.** So:

- There are **no comments**. A `#` explaining a rule would be read *as* a rule.
  This is why all explanation lives in this document instead.
- The template is ~1470 characters, roughly 370 tokens, paid on every single
  message. Adding a rule is a permanent bill, which is why overlapping rules are
  merged rather than accumulated.

### Templating: `string.Template`, not `str.format`

`$context` rather than `{context}`. Prompt text accumulates literal braces over
time — JSON examples, code snippets, output formats — and `str.format` raises on
every one of them, turning a prose edit into a crash. `$` collides with far less.

Both are safe against hostile *content*, because both scan the template and not
the substituted values, but only one is safe against the prompt author writing a
brace. Verified:

```
placeholders     : ['context']
template chars   : 1473 (~ 368 tokens of every request)
braces survive   : True     <- document text "{not a placeholder}" passes through
dollars survive  : True     <- document text "$40" passes through
```

That check matters because document text is arbitrary: a pricing PDF containing
`${weird}` must not break rendering, and it does not.

### The rules, and why each one exists

| Rule | Why it is there |
|---|---|
| Only the context; no general knowledge | The model has plenty and will use it silently. Must be forbidden explicitly. |
| Exact refusal string | So the application can *detect* it — count refusals, avoid billing, trigger fallback. Fifty phrasings of "I'm not sure" are unparseable. |
| Answer the supported part, name the gap | The common real case is partial coverage. Without this the model picks all-or-nothing, usually all. |
| Fixed citation format `[source p.12]` | A loose spec means the model invents its own and parsing fails. |
| Never invent a source, page, quote or figure | Fabricated *citations* are the worst outcome — they make a wrong answer look verified. |
| Say when excerpts contradict | Chunks from different document versions genuinely conflict; silently choosing one hides it. |
| Answer in the question's language | Documents are often English while users are not. |
| Be concise | Long answers dilute citations and cost tokens. |
| Do not mention the rules or "context" | "Based on the provided context..." is a tell that breaks the illusion and wastes tokens. |

### Structure, and why the context comes first

Rules → `CONTEXT` → `END OF CONTEXT` → restated instruction. Three reasons for
that ordering:

1. **The end marker fences the data.** Document text is arbitrary and may contain
   "ignore previous instructions"; an explicit terminator makes it far harder for
   an excerpt to be read as part of the request.
2. **Restating the refusal instruction after the context** puts it in the
   position the model attends to most — models weight the beginning and end of a
   long context more heavily than the middle.
3. **The question is NOT in this file.** It arrives as a separate user message,
   so instructions live in the system role where they carry more weight and
   cannot be overwritten by user text — which matters when the "user" is a
   stranger on a customer's website.

This is defence in depth against prompt injection, not a guarantee. A
sufficiently crafted document can still influence the answer; what it should not
be able to do is silently rewrite the rules.

### Common gotchas / traps

- **Comments in a prompt file are sent to the model.**
- **Specifying the refusal loosely makes it undetectable downstream.** The exact
  string is a contract with `generation.py`, not a stylistic preference.
- **Rules accumulate.** Each is billed forever; merge instead of appending.
- **Do not put the question in the system message.** It belongs in the user turn.
- **Never expect the prompt alone to fix bad retrieval.** If the answering chunk
  is not in the context, a perfect prompt produces a correct refusal — which is
  the right behaviour and still a failed answer. The prompt caps honesty, not
  quality.

### Interview questions

**Q: How do you stop a RAG system from hallucinating?**
You do not stop it, you make it unnecessary and then make it detectable. Most
hallucination in RAG is really a retrieval failure — the model invents because the
answer was not in front of it — so the first fix is upstream, in chunking and
two-stage retrieval. What the prompt adds is a licence to refuse, in an exact
fixed sentence, because a model with no acceptable way to say "I don't know" will
always produce its most probable guess. Then citations make what remains visible:
if every claim must carry a source and page, an unsupported claim is something a
reviewer or a downstream check can catch, which turns a silent failure into a
loud one. Two things I would explicitly *not* rely on are lowering the
temperature, which controls randomness rather than truthfulness and just makes
the invention deterministic, and thresholding on a retrieval score, which is a
number that cannot read the passage. Beyond Phase 1 the real upgrade is a second
model verifying the answer against the context — better, and it doubles cost and
latency.

**Q: Why is the prompt a separate file, and why does it contain no comments?**
It is a file because prompts change on a different rhythm from code and by
different people: a prompt edit should be a readable prose diff, not a blob
inside a triple-quoted string, and Phase 4 makes the system prompt editable per
chatbot, so it needs to already be data rather than code — the file becomes the
default that a database column overrides. No comments because there is no comment
syntax: every byte of the file goes into the request, so a `#` explaining a rule
is read by the model as part of the rule. That also makes prompt length a real
cost — this one is about 370 tokens paid on every single message — which is why
overlapping rules get merged rather than appended. The explanation lives in the
docs precisely because it cannot live in the file.

---

## 7. `backend/app/prompts/__init__.py`

**Purpose** — Turns the prompt `.txt` files into strings: loads them once per
process, renders the context into the template, and detects a refusal in the
model's reply. Without it the template from section 6 is inert data.

**Analogy** — The person who fetches the standard form from the filing cabinet,
fills in the blanks, and afterwards recognises "application declined" when it
comes back.

### `REFUSAL` is a contract between a `.txt` file and this code

The prompt tells the model to reply with an exact sentence when the context does
not answer the question. The application needs to *detect* that sentence — to
count refusals as a retrieval-quality signal, to avoid billing a message that
answered nothing, and in Phase 3 to decide whether a provider fallback is worth
trying.

So the same string exists in two files, and strings that exist in two files
drift. Someone rewords the prompt to be friendlier, `is_refusal` stops matching,
and nothing breaks loudly — refusals are silently counted as answers, and the
metric that would have revealed bad retrieval is the one that quietly stopped
working. The `_load` check turns that into a startup failure:

```
drift guard: fired -> 'grounded_answer.txt' no longer contains the exact
             refusal string 'I do not possess that information.' ...
```

That is the whole reason the load function does more than read a file.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `REFUSAL` | The exact refusal sentence, mirrored from the prompt. | Import time. |
| `PromptError` | Typed failure: missing, empty, or contract-violating template. | On failure. |
| `load_prompt` | Cached read + validation. | Once per process, per name. |
| `render_grounded_answer` | Fills `$context`. | Per chat message. |
| `is_refusal` | Normalised prefix match on the reply. | Per answer. |

### Flow of the code

1. `load_prompt(name)` rejects a name containing `/`, `\` or `..`.
2. Reads via `importlib.resources.files(__package__)`.
3. Rejects an empty file.
4. For the grounded template, asserts `REFUSAL` still appears in it.
5. `@cache` keeps the result, so subsequent calls never touch the disk.
6. `render_grounded_answer` rejects empty context, then substitutes.

### Why `importlib.resources`, not `Path(__file__).parent`

Both work today. `importlib.resources` also works when the application is
packaged as a zip or a wheel, where `__file__` does not point at a real
directory on disk. It is the supported way to read data files that ship inside a
package, and choosing it now costs nothing.

### Why the empty-context guard exists

Rendering the grounded prompt with no context produces a prompt that says
"answer only from the context" followed by nothing — and the model correctly
replies with the refusal. The result is a *confident, well-formed refusal that
looks like the system working properly*, while the actual bug is upstream in
retrieval. Raising instead means an empty context is impossible to mistake for a
question the documents did not cover.

### `is_refusal`: prefix match, and the tradeoff it accepts

Not exact equality. The model reliably *starts* with the refusal sentence but may
append a clarifying one, and treating

> I don't have that information. The documents cover shipping instead.

as an answer would inflate the answered-question rate with non-answers.

The accepted false positive: a reply that opens with the refusal and then answers
anyway is counted as a refusal. That is the safer direction — **over-counting
refusals makes retrieval look worse than it is and prompts investigation, while
under-counting hides the failure completely.** When a detector must be wrong,
prefer the error that raises an alarm over the one that silences it.

### Verification

```
loaded            : 1473 chars
cached (same obj) : True
refusal in file   : True
rendered          : True | no leftover placeholder: True

exact refusal      : is_refusal=True
with trailing text : is_refusal=True
whitespace/case    : is_refusal=True
real answer        : is_refusal=False
near-miss phrasing : is_refusal=False      <- "I'm not sure about that."

empty context: PromptError
bad name     : PromptError
missing file : PromptError
```

Then, separately, the drift guard — simulated by changing `REFUSAL` and
reloading, since the guard cannot fire while the two files agree:

```
drift guard: fired
```

**The known limitation is the `near-miss phrasing` line.** If the model ignores
the instruction and says "I'm not sure about that", this returns `False` and the
non-answer is counted as an answer. No string matcher fixes that; the mitigations
are that the prompt specifies the sentence exactly and that a model reliably
follows it, plus watching the refusal rate for a sudden drop to zero, which is
what a provider swap breaking the instruction would look like.

### Common gotchas / traps

- **Caching means editing a prompt requires a restart.** Correct for production,
  surprising during development — the file looks changed and the behaviour is
  not.
- **Mirroring a string across two files needs a guard**, or it drifts.
- **`Template.substitute` raises `KeyError` on an unknown placeholder** —
  translated here, because the raw message names a variable and not the file.
- **Empty context must never be rendered.**

### Interview question

**Q: The same magic string has to exist in a prompt file and in your code. How do
you keep them in sync?**
The first thing I would do is question whether it has to exist twice, and here it
genuinely does: the model needs the sentence as an instruction and the
application needs it as a pattern to detect, and there is no single place that
serves both. Given that, the risk is drift — someone rewords the prompt to be
friendlier and refusal detection silently stops matching. What makes it dangerous
is the failure being *silent and in the wrong direction*: refusals get counted as
successful answers, so the metric that would have revealed bad retrieval is
exactly the one that broke. So I made the loader assert the contract, meaning the
application refuses to start if the prompt no longer contains the string the code
looks for. That converts a quiet metric corruption into a loud startup failure,
which is the trade I want whenever the two options are "fails immediately" and
"reports good numbers forever". The same reasoning drove the matcher being a
prefix rather than exact equality — when a detector has to be wrong, I want it
wrong in the direction that raises an alarm.

---

## 8. `backend/app/core/generation.py`

**Purpose** — The last stage: format retrieved chunks into a context block,
render the grounded prompt around it, and stream the model's reply. Without it
retrieval produces a list of passages nobody turns into an answer.

**Analogy** — The briefing pack handed to a spokesperson: the relevant pages,
each stamped with where it came from, plus standing instructions about what may
and may not be said.

### Two behaviours worth more than the code implementing them

**1. No chunks means no LLM call.** If retrieval found nothing, `stream_answer`
yields `REFUSAL` and returns. Two payoffs: a paid round-trip is not spent on a
request that cannot succeed, and the refusal string is *guaranteed* rather than
hoped for. For the one case we know with certainty is a refusal, detection is
exact rather than probabilistic — which directly shores up the limitation
admitted in section 7, where a model improvising "I'm not sure" would go
undetected.

**2. Excerpt headers ARE the citation string.** Each chunk is introduced by
`[handbook.pdf p.12]` — exactly the form the prompt asks for. The model copies
rather than composes.

Numbered headers (`[1]`, `[2]`) were deliberately rejected. The model would
faithfully cite `[1]`, and `[1]` is meaningless to the person reading the
answer — it would have to be mapped back through state the UI does not have. A
citation that is already human-readable at the point of generation needs no
resolution step and cannot be resolved *wrongly*.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `DEFAULT_MAX_ANSWER_TOKENS = 800` | Caps the answer, so a runaway generation cannot bill unbounded. | Per request. |
| `DEFAULT_MAX_CONTEXT_TOKENS = 3000` | Caps the context. Bounds the pathological case, not normal traffic. | Per request. |
| `ContextBlock` | Formatted text + which chunks survived + how many were dropped. | Per request. |
| `build_context` | Dedupe, budget, format. Pure and synchronous. | Per request. |
| `stream_answer` | The generator callers consume. | Per request. |
| `answer` | Joins the stream, for tests and scripts. | Tests, CLI. |

### Why `ContextBlock` returns `used` and not just text

The citations shown to a user must reflect what the model **actually saw**. If
the budget trimmed the last two chunks, citing them would be a fabrication
produced by our own code rather than by the model. Returning the surviving
chunks alongside the text makes citing a trimmed chunk impossible rather than
merely unlikely.

### The token budget is approximate, and that is a considered choice

`build_context` counts with the character heuristic, not a real tokenizer.
`LLMProvider` has no `count_tokens` and deliberately so: unlike the embedder,
there is no cheap local tokenizer to ask. Gemini's is a **network call**, and
every vendor's differs, so adding the method to the contract would force each
provider to either ship a vendor tokenizer or make an HTTP request in order to
answer "how long is this string".

Given an imprecise number, the budget is set **conservatively low**, because the
two errors are not symmetric:

| Budget too small | Budget too large |
|---|---|
| One fewer chunk; marginally worse answer | Hard provider error, mid-stream, after the user has seen tokens |

Six chunks of ~400 tokens is ~2400, so 3000 does not trim normal traffic at all.
It exists to bound the pathological case.

### Flow of the code

1. Reject an empty question.
2. **No chunks → yield `REFUSAL`, return.** No provider is constructed.
3. `build_context`: skip blanks, skip byte-identical duplicates, format each as
   `[citation]\ntext`, accumulate until the budget is hit.
4. Still nothing usable → refusal.
5. `render_grounded_answer(context.text)` → system prompt.
6. Question goes in a **user** message, never the system message.
7. Stream fragments through unchanged.

### Ordering, and "lost in the middle"

Retrieval order is preserved — best first. Models attend most strongly to the
beginning and end of a long context and least to the middle, so the reranker's
top result must not be buried. Best-first also means a truncated context loses
the *least* relevant chunks, which is the only acceptable direction to lose them
in.

### The `and used` condition is load-bearing

```python
if total + cost > max_tokens and used:
    break
```

Without `and used`, a single chunk larger than the budget would produce an empty
context — and an empty context is a *guaranteed* refusal. Admitting one
oversized chunk that the provider may well accept is strictly better than
manufacturing a certain failure. The guard turns "definitely no answer" into
"probably an answer".

### Common gotchas / traps

- **Duplicate chunk text is not harmless.** Overlapping chunks can be identical
  when a short document splits at a boundary; sending the same passage twice
  spends budget and reads to the model as corroboration it has not got.
- **Never put the question in the system message.** It belongs in the user turn,
  where it cannot rewrite the rules.
- **Citing a trimmed chunk is a fabrication by our code**, not the model's.
- **`answer()` exists only because the contract is stream-only.** A whole answer
  is trivially built from a stream; the reverse is impossible.

### Verification

With a fake LLM that records what it was handed:

```
--- context block ---
[handbook.pdf p.12]
Refunds are issued within 30 days.

[terms.pdf p.3]
Shipping takes 3-5 days.
---
citations        : ['handbook.pdf p.12', 'terms.pdf p.3']

question in USER role, not system: user | True
context reached the model        : True

no chunks -> answer : "I don't have that information."
is_refusal          : True
LLM never called    : True      <- no paid round-trip
exact string        : True

dedupe: 2 in -> 1 used, 1 dropped
tiny budget: used  : 1 | dropped: 1
never empty        : True       <- the `and used` guard
fragments received : 3 ['Refunds are issued ', 'within 30 days '] ...
empty question     : ValueError
```

**What this does not prove.** The test also reported that the citation format
survived into the answer — but the fake LLM was written to emit that string, so
that line demonstrates the plumbing carries it, not that a real model copies the
header format. Whether the prompt actually produces well-formed citations is a
question only a real provider can answer, and it belongs to the CLI acceptance
run. The distinction matters because a fake that is written to pass is not
evidence about the thing being faked.

### Interview questions

**Q: How do you make a RAG system's citations trustworthy?**
The main idea is to give the model as little opportunity to compose a citation as
possible. Each excerpt is labelled with the exact string the answer should
contain — `[handbook.pdf p.12]` — so the model's job is to copy a token sequence
sitting right in front of it rather than to remember or construct one, and
copying is something these models are extremely reliable at. I deliberately did
not number the excerpts, because `[1]` is meaningless to a reader and would have
to be resolved back through state that the interface may not have, which is an
extra step that can be done wrongly. The other half is on our side: the code
returns which chunks actually made it into the context, so if the token budget
trimmed some, we cannot display citations for passages the model never saw —
otherwise our own code becomes the source of a fabricated citation, which is
worse than the model's, because it looks systematic. What none of this gives you
is a guarantee that a cited passage really supports the sentence; that needs a
verification pass, and it is a later phase.

**Q: Why does an empty retrieval result skip the model call entirely?**
Because the outcome is already known, and calling anyway is worse in three
separate ways. It costs a paid round-trip and several hundred milliseconds on a
request that cannot succeed. It burns rate limit that a request which *can*
succeed might need. And most importantly it makes the refusal probabilistic
rather than certain: the prompt asks for an exact sentence, and models mostly
comply, but "mostly" means the application's refusal detection is doing fuzzy
matching on the one case where it could have been exact. Returning the constant
directly means that for the case we are certain about, detection is certain too,
which shrinks the surface where the fuzzy matcher has to be right. The general
shape is that when you already know the answer, an external call is not caution,
it is an opportunity to be wrong.

---

## 9. `backend/app/cli.py`

**Purpose** — The harness that makes Phase 1 checkable: ingest a file, ask a
question, watch a cited answer stream into the terminal. Without it every module
built so far is a library nobody has ever run end to end.

**Analogy** — The ignition and dashboard. It does not make the car go; it is how
you find out whether it does, and how fast.

### Deliberately a thin shell

Every decision lives in `core/`. This file parses arguments, wires objects
together, prints, and turns exceptions into exit codes. The failure mode it is
written against is a test harness accumulating logic — retrieve a bit
differently here, format context slightly differently there — until the CLI
works and the API does not, because the two have quietly stopped sharing a path.
The CLI must exercise the *same* functions Phase 3's `POST /chat` will call, or
it stops being evidence about the product.

### No `--llm` flag, on purpose

The acceptance criterion in PLAN §6 is that switching provider takes **zero code
edits** — an env var and nothing else. A `--llm groq` flag would let me
demonstrate the swap in a terminal without ever proving the property, because
the flag would be a second code path that the API will never use. Provider
selection stays where it will live in production: the environment. The
`providers` command exists to *observe* that, showing what is configured and
what is available.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `LOCAL_ORG_ID` / `LOCAL_CHATBOT_ID` | Single-tenant placeholders. Phase 2 replaces them with real Clerk ids. | Every command. |
| `_configure_output` | Forces UTF-8 on the console. | Startup. |
| `_build_store` | `VectorStore(dimension=embedder.dimension)`. | Per command. |
| `cmd_providers` | Prints configured + available providers. | `providers`. |
| `cmd_ingest` | Ingest and report. | `ingest`. |
| `cmd_ask` | Retrieve, stream, cite, time. | `ask`. |
| `_print_chunks` | `--show-chunks`: how far the reranker moved each result. | Optional. |
| `_print_timings` | Retrieval / time-to-first-token / total. | `ask`. |

### Why `_configure_output` exists

Windows terminals default to a legacy code page. Printing a chunk containing an
en dash or a curly quote — routine in any real PDF — raises `UnicodeEncodeError`.
The pipeline would have worked perfectly and the CLI would crash *while
displaying its own success*, which is a maximally confusing failure. Forcing
UTF-8 with `errors="replace"` means the worst case is a replacement character
rather than a traceback.

### Why time-to-first-token is printed separately

Total latency is the number people quote and the wrong one to optimise. **Time
to first token (TTFT)** is when the first character appears; total time is when
the last one does. A four-second answer that starts in 400ms reads as
responsive, and the identical answer delivered whole at four seconds reads as
broken. Since the whole `LLMProvider` contract is stream-only precisely to make
TTFT low, the harness has to measure it or the design decision is unverifiable.

Printing retrieval time separately splits the budget into the two halves that
have different fixes: slow retrieval is an embedding or index problem, slow TTFT
is a provider or prompt-size problem.

### `--show-chunks` and the reranker column

`_print_chunks` prints how far each result moved:

```
1. [handbook.pdf p.12] score=8.4  (up 11 from vector rank 11)
```

That column is the human-readable version of PLAN §6's third criterion. If every
result shows no movement, the reranker is spending latency to confirm what the
vector search already said, and the honest conclusion is either that it is not
earning its place on this corpus or that retrieval was never the bottleneck.

### Flow of the code

1. `_configure_output`, parse args, configure logging (INFO only with `-v`, so
   normal runs are not buried in structured log lines).
2. Dispatch through `_COMMANDS`.
3. `ask`: retrieve → optionally print chunks → stream with `write`+`flush` per
   fragment → print sources → print timings.
4. Typed errors are caught in `_run` and printed as one line.
5. Store closed in a `finally`, always.

### Why `sys.stdout.write` + `flush`, not `print`

`print` appends a newline per call, which would put every token on its own line.
And without the explicit flush the terminal buffers until a newline arrives, so
the answer appears in one lump at the end — the output would look identical to a
non-streaming implementation, and the harness would be unable to show the very
property it exists to demonstrate.

### Error handling

`ExtractionError`, `VectorStoreError` and `ProviderError` are caught and printed
as a single line. They already carry what the user needs — which file, which
provider, what to do — so a traceback would be noise around a message that was
already written for a human. This is the payoff for the typed hierarchies in
`extraction.py` and `base.py`: the CLI's error handling is four lines because
the work was done where the knowledge was.

### Verification

Argument parsing and provider reporting run:

```
ask       : {'command': 'ask', 'org': 'local_org', 'bot': 'local_bot',
             'question': 'how long do refunds take?', 'top_k': 6,
             'candidates': 40, 'show_chunks': True}

configured:
  LLM_PROVIDER        = gemini
  EMBEDDING_PROVIDER  = bge_local
  RERANK_PROVIDER     = bge_local
available:
  llm        : anthropic, gemini, groq, ollama, openai
  embedding  : bge_local, openai
  rerank     : bge_local, cohere, noop
```

**`ingest` and `ask` have never been run.** `python -m app.cli --help` fails at
import with `ModuleNotFoundError: No module named 'qdrant_client'`; the checks
above required stubbing that module. `sentence-transformers`, `transformers`,
`torch` and `qdrant_client` are all absent from this environment, and a running
Qdrant is also required.

So **Phase 1's "done when" is not met**, and nothing above should be read as
evidence that it is. Every module has been exercised against fakes, which
verifies wiring and logic; not one has been run against a real model, a real
PDF, or a real database. The remaining work is not more code — it is:

```
cd backend && pip install -e .     # or: uv sync
docker compose up -d qdrant
python -m app.cli providers
python -m app.cli ingest <a real pdf>
python -m app.cli ask "<something the pdf answers>"
```

Then the three acceptance criteria, in order: a fast cited streamed answer;
`LLM_PROVIDER=groq` working with no code edits; `RERANK_PROVIDER=noop` visibly
degrading quality.

### Common gotchas / traps

- **`print` per fragment would fake non-streaming output**, hiding the property
  the harness exists to show.
- **Logging defaults to WARNING.** INFO would bury a streamed answer in
  structured log lines mid-sentence.
- **The store must be closed in `finally`.** An unclosed `AsyncClient` prints an
  event-loop warning at exit that reads like a pipeline bug.
- **A CLI-only convenience flag is a second code path.** It makes the demo
  easier and the evidence weaker.

### Interview question

**Q: You have built a pipeline but the dependencies are not installed, so you
cannot run it. Is the work done?**
No, and I would be precise about what is and is not established. Every module has
been exercised against fakes — a fake embedder, a recording fake vector store, a
fake LLM — which is real evidence about wiring, ordering, batching and error
paths, and it caught genuine problems: a batching loop I had verified with only
one iteration, and a refusal-drift guard that had never actually fired. What it
cannot tell me is anything about quality, because every fake was written by the
same person who wrote the code, so they agree by construction. Whether chunks are
sized correctly against a real tokenizer, whether the reranker actually promotes
the right passage on a real corpus, whether the model copies the citation format
instead of inventing one — none of that is knowable without the real models. So I
would describe the state as "the pipeline is complete and the acceptance criteria
are unverified", and I would not describe it as done. The distinction matters
because the failure modes designed against here — silent truncation, silent
scan-only PDFs, silent duplicate chunks — are precisely the ones that unit tests
with fakes cannot see.

---

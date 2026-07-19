# Phase 2 — Multi-tenancy & async ingestion

Per-file build depth for Phase 2 of PLAN.md. Concepts live in LEARNING.md; this
doc is about *these files*, in the order they were built.

Phase 2's whole job is to turn `org_id` from a string we pass around into a
boundary that holds under attack, and to move ingestion off the request path.

**Build order and why:** the isolation test came first (Hard Rule #4 — tests
first for security-critical code), then the enabling change it needed, then data,
auth, API, worker, and finally the migration script.

---

## 1. `backend/tests/test_isolation.py`

**Purpose** — Proves Org A can never see Org B's chunks, including under forged
input. The one test whose failure means a breach rather than a bug. PLAN §7 runs
it in CI on every push, forever.

**Analogy** — A bank vault where every safe-deposit box shares one room. This is
the auditor who walks in with the wrong key, the right key to the wrong box, and
a blank key, and confirms all three open nothing.

### Main pieces

| Piece | What it does | When it runs |
|---|---|---|
| `StubEmbedder` | Deterministic hash-based vectors, no torch | Every test |
| `PassthroughReranker` | Returns input order unchanged | The two `retrieve()` tests |
| `store` fixture | In-memory Qdrant seeded with 3 tenants | Function-scoped, per test |
| `_search` helper | Search with `limit=50` | Most tests |
| Tests 1–2 | Filter holds at the org boundary | — |
| Test 3 | Filter is complete, not org-only | — |
| Tests 4–5 | Forged scope values return nothing | — |
| Tests 6–7 | Empty scope raises, on read and write | — |
| Tests 8–9 | Deletes are scoped, and actually work | — |
| Tests 10–11 | Isolation survives `retrieve()` | — |
| Tests 12–13 | **Structural**: signature + source scan | — |

### Flow

1. Fixture builds a `VectorStore` at `:memory:` and seeds three scopes —
   `(A, bot_a1)`, `(A, bot_a2)`, `(B, bot_b1)` — all holding the **same
   sentence**.
2. Each test searches with one scope and asserts on what comes back.
3. The structural tests never touch the store: one reads
   `inspect.signature`, the other reads `vectorstore.py` as text.

### Design decisions worth remembering

**Identical text in every tenant.** If each org had distinctive text, the test
would pass because of vector similarity rather than because of the filter — it
would be green with the filter deleted. Identical text means identical vectors,
so only the filter can separate them. This is the difference between a test and
a decoration.

**`limit=50` against 3 seeded chunks.** To prove absence you must ask for
everything. `limit=1` returns one correct chunk and hides two leaking ones
behind it.

**Function-scoped fixture.** Two tests delete points. A session-scoped store
would let one test's deletion leak into another's assertions — an isolation
suite that is not itself isolated proves nothing.

**Structural tests.** `test_scope_arguments_are_required_and_keyword_only`
asserts `org_id` is `KEYWORD_ONLY` with no default;
`test_every_qdrant_query_passes_a_filter` scans the source for a
`query_points(` call without `query_filter=`. Behavioural tests catch a leak
once somebody writes the leaking call. Structural tests fail at the moment the
signature grows a default that would *permit* it — earlier, and cheaper.

### Verified, not assumed

The test was mutation-checked: deleting the `chatbot_id` condition from
`_scope_filter` produced **9 failures**, including the cross-tenant delete. A
sacred test that has never been seen to fail is a sacred test nobody has
checked. Do this again any time the filter changes.

### Common gotchas

- Qdrant local mode warns that payload indexes have no effect. Expected and
  harmless — indexes are a performance feature; the filter semantics under test
  are identical.
- `asyncio_mode = "auto"` in pyproject.toml is why `async def test_*` works with
  no decorator.
- An empty `org_id` does **not** bypass a filter. It matches points whose
  `org_id` is literally `""` — i.e. none — so the caller sees an empty list and
  reads it as "no documents". That silent-wrong-answer path is why
  `_require_scope` raises.

### Interview questions

**Q: One collection with payload filters, or one collection per tenant?**
One collection. Per-collection overhead in Qdrant (an HNSW graph, segment files)
would be paid thousands of times, and creating a collection per signup makes
onboarding a schema operation. The cost is that isolation becomes application
logic rather than physical separation — which is exactly why the filter is
mandatory in the type signature and why this test exists. At very large
per-tenant volumes, or under a regulatory requirement for physical separation,
the trade flips.

**Q: Why assert on a function signature in a test?**
Because the failure mode is a *future* change, not current behaviour. If
`org_id` ever gains a default, every behavioural test still passes — they all
pass it explicitly — while an unscoped call becomes writable for the first time.
The structural test fails on the signature change itself, at review time.

---

## 2. `backend/app/core/vectorstore.py` (edit)

**Purpose** — Adds a `location` parameter so `VectorStore` can point at Qdrant's
in-process mode. A security test that needs Docker is a test CI eventually skips.

**Analogy** — A car given a dynamometer mount. Same engine, no road needed.

### What changed

- `__init__` takes `location: str | None`. When set, builds
  `AsyncQdrantClient(location=location)` and **ignores** `url`/`api_key` — a
  client cannot be both embedded and remote, and silently preferring one would
  make a misconfigured test pass against an empty local store.
- Later in the phase, `delete_chatbot(org_id, chatbot_id)` was added for the
  bot-deletion cleanup task. Same mandatory scoping as everything else.

### Gotcha

Qdrant local mode is a real implementation, not a mock — but it is
single-process and its payload indexes are inert. Use it for correctness tests
only, never for performance measurement.

---

## 3. `backend/app/db/tables.py` + `0001_phase2_tenancy_tables.py`

**Purpose** — The four Phase-2 tables. Without them `org_id` is just a string;
this makes it an entity with an owner, a plan, and a lifecycle.

**Analogy** — A building's tenancy register. Qdrant is the storage room full of
labelled boxes; this is the ledger of which company rents which unit.

### Main pieces

- `Organization` — the tenant root. `clerk_org_id` unique+indexed; it is what
  auth looks up on every request.
- `User` — one org per user (PLAN lists multi-org as a stretch goal).
- `Chatbot` — `embedding_model` + `dimension` stored **per bot**; `public_key`
  for the widget; `model_overrides` JSONB.
- `Document` — the ingestion lifecycle row, carrying `org_id` **and**
  `chatbot_id`.
- `PLAN_LIMITS` — plan caps in code, not in a table.
- `generate_public_key()` — `secrets.token_urlsafe(32)`.

### Flow

Models are declared here → imported by `db/models.py` (the registry) →
`migrations/env.py` imports the registry → Alembic diffs `Base.metadata`.
Adding a model means adding one import line to the registry.

### Design decisions

**`documents.org_id` is denormalised on purpose.** A document reaches its org
through its chatbot, so the column is redundant. Storing it means every
tenant-scoped query filters on a column physically present on the row, with no
join to forget — and it mirrors the Qdrant payload exactly. A security filter
you can omit by forgetting a join is the bug class Hard Rule #3 exists to
prevent.

**UUID primary keys.** Sequential ids are enumerable, and an enumerable tenant
id turns every missing authorization check into a walk through the customer
list. Also: generatable without a DB round trip (needed for Qdrant point ids and
file paths), and collision-free across environments.

**`secrets`, not `random`.** `random` is a Mersenne Twister seeded from the
clock; a few observed outputs predict the rest. `public_key` is a bearer
credential, so it needs a CSPRNG. Most security-sensitive line in the file.

**The `error`-only-when-failed CHECK.** Without it, a retry that succeeds but
forgets to clear `error` leaves a READY document showing a stale failure
forever. The worker relies on this: it clears `error` when moving to PROCESSING
because the constraint requires it.

**Deviation from PLAN §4:** PLAN names the column `model_config`. Pydantic v2
reserves that attribute, and the API schemas mirror these names — a Pydantic
model with a `model_config` *field* raises at class-definition time. Renamed to
`model_overrides`.

### Common gotchas

- The migration is **hand-written** (Docker was down). Run `alembic upgrade
  head` then `alembic check` — a clean `check` is the proof it matches the
  models. **This has not been run yet.**
- PostgreSQL enum types are schema objects that survive their table being
  dropped. `downgrade()` drops them explicitly; without that, downgrade →
  upgrade fails with "type plan_tier already exists".
- `sqlalchemy.Enum` (a column type) and stdlib `enum.Enum` are unrelated things
  with the same name. Aliased to `SAEnum` here.

### Interview questions

**Q: When is denormalising a foreign key the right call?**
When the duplicated column is a *filter* rather than data, when the relationship
is immutable (a document never changes chatbot), and when forgetting the join
has an outsized cost. All three hold here: it is a security filter, the link
never moves, and the failure mode is a cross-tenant leak. The usual objection —
two places can disagree — is bounded by there being exactly one INSERT path.

**Q: Why store `dimension` per chatbot instead of reading config?**
Because the vectors in Qdrant were produced by whatever embedder ran at
ingestion time, and config describes *now*. If `EMBEDDING_PROVIDER` changes, the
bot's existing vectors do not. Recording the model that actually produced them
is what lets `reembed.py` detect the mismatch instead of the bot silently
comparing 384-dim queries against 1536-dim history.

---

## 4. `backend/app/security/clerk.py`

**Purpose** — Turns a Clerk session token into an `org_id` that came from a
signature, not from a request body. PLAN's "resolve org_id SERVER-SIDE (always)"
is implemented here and nowhere else.

**Analogy** — The doorman who checks your pass under UV light and writes the
flat number on your badge himself. You never write the flat number.

### Main pieces

| Piece | Role |
|---|---|
| `Principal` | Frozen, carries **our** UUID `org_id` |
| `_JWKSCache` | Holds a `PyJWKClient`, rebuilt on TTL |
| `verify_token` | Cryptography only, no DB |
| `extract_clerk_org_id` | Handles both Clerk claim shapes (`o.id`, `org_id`) |
| `extract_role` | Token role, defaulting to **least** privilege |
| `resolve_principal` | Maps verified claims onto our rows |
| `require_org` / `require_owner` | FastAPI dependencies |

### Flow

`Bearer <jwt>` → read `kid` from header → find key in JWKS → verify RS256 +
`exp`/`nbf`/`iat`/`iss` → read org claim → look up `Organization` by
`clerk_org_id` → return `Principal`.

The split between `verify_token` (pure crypto) and `resolve_principal` (DB) is
deliberate: the crypto is testable without a database, and "valid token, unknown
tenant" stays distinguishable from "bad token".

### The security decisions

**`algorithms=["RS256"]`, passed explicitly.** The classic JWT attack: rewrite
the header to `alg: none` (no signature needed) or to `alg: HS256` and sign with
the RSA *public* key — which is public. A library that reads the algorithm from
the token will verify both. **Never let a token choose its own verification
algorithm.**

**Issuer verification.** Anyone can create a free Clerk instance, and its tokens
are validly signed against *its* JWKS. Without an `iss` check the only thing
stopping a stranger's token is that we happen to fetch keys from a different
URL — true today, not a guarantee. `CLERK_ISSUER` must be set.

**Uniform error messages.** Every crypto failure becomes the same
`InvalidTokenError` string. Distinguishing "expired" from "bad signature" tells
an attacker which part of the forgery to fix; the detail goes to the log.

**Status codes carry meaning.** 401 = bad credential, retry with a fresh token.
403 = authenticated but not authorized, a refresh will not help (a 401 here
sends clients into a refresh loop). 500 for unconfigured auth — a misconfigured
server must never look like a rejected credential.

**Database beats token for role.** The token is authentic but reflects Clerk's
view; our row reflects decisions made here. Unrecognised role strings default to
MEMBER, so a Clerk-side rename cannot silently promote everyone.

**Deviation from PLAN §6:** PLAN says middleware. This is a router-level
dependency, because middleware runs on every path and needs a growing exempt
list (`/health`, `/docs`, Phase 6's signature-authenticated Stripe webhook), and
path-prefix exemption lists are a known source of auth bypasses. The honest
cost: with middleware, forgetting leaves a route *protected*; with dependencies,
forgetting leaves it *public*. Mitigated by attaching to
`APIRouter(dependencies=[...])`, so the forgettable unit is a whole router.

### Common gotchas

- JWKS TTL vs unknown-`kid` refetch: an unknown kid refetches immediately (this
  is how rotation works with no downtime). The TTL exists for the opposite case
  — a *revoked* key, which stays known, and without a ceiling would be accepted
  for the process lifetime.
- Clock skew: without `leeway`, a server two seconds fast rejects fresh tokens
  as "not yet valid" — a maddening intermittent failure.
- Ruff's B008 flags `Depends()` in argument defaults. The fix is
  `Annotated[T, Depends(...)]`, not a suppression.

### Interview questions

**Q: Walk through verifying a JWT.**
Split on dots: header, payload, signature. Base64url-decode the header to get
`alg` and `kid`. Fetch the matching public key from the issuer's JWKS. Verify
the signature over `header.payload` **using an algorithm you chose, not the one
the token names**. Then validate claims: `exp`, `nbf`/`iat` with small leeway,
`iss`, and `aud` if the issuer sets one. Only then read the payload as fact.

**Q: Why not just trust `org_id` from the request body?**
Because it is attacker-controlled. Every Qdrant filter, every SQL `WHERE`, and
the entire isolation test would then be filtering on a value the attacker chose
— the strongest possible boundary applied to a lie. The token is signed by
Clerk's private key, which we do not hold and neither does the attacker.

---

## 5. The API layer — `schemas.py`, `chatbots.py`, `documents.py`

**Purpose** — CRUD for bots and documents, every query scoped to the
`Principal`'s org, extending the isolation guarantee from Qdrant up into SQL.

**Analogy** — A hotel front desk that only hands you keys to rooms registered in
your name. Ask for 402; if it isn't yours, the answer is "no such room".

### Endpoints

```
GET    /api/bots                                  list (org-scoped)
POST   /api/bots                                  create (plan cap)
GET    /api/bots/usage                            usage vs caps
GET    /api/bots/{id}                             get
PATCH  /api/bots/{id}                             update
DELETE /api/bots/{id}                             delete (owner only)
GET    /api/bots/{id}/documents                   list
POST   /api/bots/{id}/documents                   upload (plan cap)
DELETE /api/bots/{id}/documents/{doc_id}          delete (owner only)
```

### Design decisions

**404, not 403, for another tenant's row.** 403 is the honest code and the wrong
one: it confirms the resource exists, and across many ids becomes an enumeration
oracle that leaks the customer list. `owned_chatbot()` is the single place this
decision is made, and it is shared by both routers — two implementations would
be two chances to get it wrong, only one of them tested.

**Scoping in the WHERE clause, never a post-fetch `if`.** Fetch-then-check
leaves another tenant's row loaded in memory, ready to be logged, returned in an
error, or acted on by whatever gets added below the check later.

**No `org_id` field on any request model, and `extra="forbid"`.** A body
containing `org_id` is rejected with 422 rather than ignored. A rejected request
is visible in logs; an ignored field is one careless `**payload` away from being
accepted.

**Handlers are sync `def`.** SQLAlchemy's `Session` blocks. In `async def` it
blocks the event loop and stalls every other in-flight request; in plain `def`
FastAPI runs it in a threadpool, which is what threads are for. Consequence: we
cannot `await` Qdrant here, so vector cleanup is enqueued to Celery — the right
shape anyway.

**`exclude_unset=True` on PATCH.** Distinguishes "client didn't mention
`system_prompt`" from "client sent `null` to clear it". Without it every omitted
field is treated as an explicit null and wipes settings nobody touched.

**Route order.** `/usage` is declared before `/{chatbot_id}` — FastAPI matches
in declaration order, so the path parameter would otherwise swallow it and fail
to parse "usage" as a UUID.

**`embedding_model`/`dimension` come from the live embedder, never the request.**
A client-supplied dimension creates a bot whose vectors the collection can never
accept.

### Upload safety (three rules)

1. **The client's filename never touches the filesystem.** `../../etc/cron.d/x`
   is a filename too. The path is built from server-generated UUIDs plus an
   allow-listed suffix; the original name is a display string in the DB.
2. **Size enforced while streaming.** `Content-Length` is a header the client
   wrote; checking after the read means the disk is already full. A running
   total aborts mid-copy, and the partial file is deleted — otherwise the limit
   is enforced on paper and defeated in practice.
3. **Content-hash dedupe.** Re-uploading the same bytes returns the existing
   row. Point ids derive from `document_id`, so a second row would produce a
   full duplicate set of vectors and let one passage occupy several top-40
   slots.

**Commit before publish.** The Celery task is enqueued *after* `db.commit()`. A
worker is fast enough to pick the job up before an uncommitted transaction is
visible and would then fail to find the row. Worst case this way is a document
stuck in PENDING — visible and re-queueable.

### Interview questions

**Q: Why 404 instead of 403 for another tenant's resource?**
403 confirms existence. Iterate over ids and you have a census of the customer
base — and often, from names, a competitor's client list. 404 makes "not yours"
and "never existed" indistinguishable. The trade is debuggability: a legitimate
user with a stale link sees "not found" rather than "not allowed", which is why
the real reason is logged server-side.

**Q: Why is a blocking DB call in an `async def` handler a problem?**
The event loop is a single thread running every concurrent request. An `await`
yields control; a blocking call does not — it holds the thread, so every other
in-flight request stalls for its duration. One 200ms query becomes 200ms of
latency for everyone. A plain `def` handler is dispatched to a threadpool where
blocking is expected and isolated.

---

## 6. `celery_app.py` + `ingest_tasks.py`

**Purpose** — Runs ingestion outside the request. A 500-page PDF takes minutes;
in-request it would hold a connection open and lose everything on disconnect.

**Analogy** — A restaurant's kitchen ticket rail. The waiter leaves in seconds;
cooking runs on its own clock, and the ticket survives the waiter's break.

### Configuration decisions (`celery_app.py`)

| Setting | Why |
|---|---|
| `task_serializer="json"` | Pickle deserialization executes arbitrary code — a writable broker becomes RCE on the worker |
| `task_acks_late=True` | Ack after completion, so a killed worker's task is redelivered instead of vanishing |
| `worker_prefetch_multiplier=1` | The local embedder is CPU-bound and holds model weights; parallel tasks multiply memory, not throughput |
| `task_soft_time_limit=25m` | Fires *inside* the task, so it can mark the document FAILED with a real reason |
| `--pool=solo` (Windows) | Celery's prefork pool needs `fork()`, which Windows lacks |

**Messages carry an id and nothing else.** A message is a snapshot — sending the
row means acting on state that may be stale by execution time. Broker messages
are also not private (Redis, logs, monitoring tools), and must stay small.
The worker re-reads current truth from Postgres.

### Retry classification (`ingest_tasks.py`)

```
ProviderRateLimit / Timeout / Unavailable  -> retry (transient)
VectorStoreUnavailableError                -> retry (Qdrant restarting)
ExtractionError (scanned, encrypted)       -> FAIL NOW, tell the user
ProviderAuth / InvalidRequest              -> FAIL NOW, our bug
SoftTimeLimitExceeded                      -> FAIL NOW (25 more minutes won't help)
```

Retrying the wrong failures is worse than not retrying: it delays the honest
error and burns queue capacity on work that cannot succeed. Phase 1's
`ProviderError.retryable` flag and typed `ExtractionError` hierarchy are what
make this a lookup instead of a guess — this module is their first consumer.

**Backoff with jitter.** 10s → 20s → 40s, ±25%. Doubling gives a struggling
dependency room to recover. The jitter matters when an outage fails fifty
documents at once: without it all fifty retry at the same instant and knock the
recovering service over again — the *thundering herd*.

**PENDING between retries, not PROCESSING.** PROCESSING means "a worker has this
right now". A document waiting 40 seconds does not. The distinction is what lets
an operator tell a queued document from a stuck one.

**Every exit path writes a terminal status.** A task that raises without doing so
leaves a document spinning in the dashboard forever — the most common way async
ingestion goes wrong.

**`asyncio.run` per task, and a `VectorStore` per task.** The store owns an HTTP
pool bound to an event loop; a module-level client would be bound to whichever
loop created it and fail on every subsequent task.

**`session.close()` in `finally`.** The worker is long-lived. Leaking one pooled
connection per task exhausts the pool hours later, under load, looking like a
database problem.

### Common gotchas

- "Received unregistered task" is not a broker problem — it means the module
  isn't in `include=[...]`.
- `dimension` for ingestion comes from the **chatbot row**, not the live
  embedder. If `EMBEDDING_PROVIDER` changed, that must surface as a loud
  `CollectionMismatchError`, not as vectors written in a width nothing queries.
- `requeue_stuck_documents` exists because `acks_late` covers a crashing worker,
  not a machine losing power. Not scheduled until Phase 7 brings Celery Beat.

### Interview questions

**Q: Why `acks_late`, and what does it require of the task?**
With early ack, a worker killed mid-task loses the message and the document
hangs in PROCESSING forever. Late ack redelivers it. The requirement is
idempotency — the task must tolerate running twice. Ours does: point ids are
deterministic, so a re-run overwrites its own partial work instead of appending
a duplicate set of chunks.

**Q: Why not pickle for task messages?**
Pickle deserialization can execute arbitrary code, so anyone who can write to
the broker gets code execution on the worker. JSON cannot express arbitrary
objects — a limitation that is also the security property, and it forces the
good habit of sending ids rather than objects.

---

## 7. `backend/scripts/reembed.py`

**Purpose** — Migrates a chatbot's documents to a different embedding model.
PLAN §3 rule 1: "a command, not a crisis."

**Analogy** — Re-cataloguing a library after switching classification systems.
The books don't move; every index card is rewritten, and the old catalogue stays
usable until the new one is complete.

### Usage

```bash
python -m scripts.reembed --chatbot <uuid> --dry-run
python -m scripts.reembed --chatbot <uuid> --collection chunks_v2
python -m scripts.reembed --org <uuid> --collection chunks_v2
```

### Why re-embedding is unavoidable

Two models produce vectors that are **not comparable, even at the same width**.
Each learned its own coordinate space, so the bge-small vector for "refund
policy" and the OpenAI vector for the same phrase point in unrelated directions.
Query with the new model against old vectors and the neighbours are effectively
random — with no error anywhere. Qdrant compares the numbers it is given, the
reranker orders nonsense, the LLM answers from it. The only symptom is a bot
that has quietly become bad at its job.

### Design decisions

**A new collection, not an in-place rewrite.** A collection has a fixed vector
size, so 384 → 1536 cannot be done in place at all. Even at matching widths, an
in-place rewrite means the collection holds two incomparable spaces for the
duration and every query in that window is degraded. Writing fresh keeps the
live collection correct until the switch, which is one env var:
`QDRANT_COLLECTION=chunks_v2`. Writing to the live collection requires an
explicit `--in-place`.

**DB rows updated last.** An interrupted run leaves the bot pointing at its old
model and old collection — both still valid and still consistent with each other.

**Same `document_id` in the new collection.** Point ids stay deterministic, so
re-running the script is idempotent.

**Only READY documents.** A FAILED document has no vectors to migrate;
re-ingesting it here would silently repair it in a script whose job is
migration, not repair.

**Continue past failures, exit non-zero.** One unreadable file should not
abandon ninety-nine others. But the exit code is 1 if anything was skipped or
failed, so a deployment script can gate on it — and the operator knows before
flipping `QDRANT_COLLECTION`.

### Known wart

`_stored_path` is now duplicated in three files (`api/documents.py`,
`workers/ingest_tasks.py`, here). Worth consolidating next time any of the three
is touched. It was not imported from the API module so that a maintenance script
does not pull FastAPI into its import graph.

### Interview question

**Q: A customer wants to upgrade to a better embedding model. What happens?**
Every stored vector becomes unusable, silently — old and new vectors are not
comparable, and nothing errors. So: point `EMBEDDING_PROVIDER` at the new model,
run `reembed.py` into a *new* collection while the old one keeps serving live
traffic, verify on a sample of known questions, then switch `QDRANT_COLLECTION`
and restart. Rollback is switching the env var back, because the old collection
was never touched. The DB's per-bot `embedding_model`/`dimension` columns are
what make the mismatch detectable in the first place.

---

## Status against PLAN §6 "done when"

> `test_isolation.py` passes: create Org A and Org B, ingest different docs,
> prove Org A's queries can never return B's chunks — including adversarial
> attempts. This test is sacred; it runs in CI forever.

**The vector-store half is done and mutation-verified** — 17 tests, covering
org, bot, forged ids, empty scope, deletes, and `retrieve()`.

**Not yet done, and needed before Phase 2 can be called complete:**

1. **Run the migration.** `docker compose up -d postgres && alembic upgrade head
   && alembic check`. The migration is hand-written and unverified.
2. **API-level isolation tests.** The current suite proves the *store* is
   isolated. It does not yet prove `GET /api/bots/{B's id}` returns 404 with A's
   token, or that a forged `org_id` in a body is rejected. That is the
   adversarial half of PLAN's criterion.
3. **An end-to-end run.** Upload → Celery → READY → answer, with a real worker.
4. **CI.** PLAN §7 requires this test on every push; there is no workflow yet.

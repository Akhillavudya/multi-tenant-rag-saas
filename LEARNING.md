
# LEARNING.md

My revision notebook for this project. Every important concept, library, and
design decision gets ONE entry here (max ~1 page). By the end, this file is my
interview prep for RAG / backend / SaaS engineering.

Rules: no duplicates (extend existing entries), keep the template order,
math only where it truly applies, always fill "Why not alternatives".

---

## TEMPLATE (copy for each new concept)

### <Concept / Library Name>            [Phase N]
**What it is** — 2–4 lines.
**Problem it solves** — what breaks without it, in THIS project.
**Intuition** — one real-world analogy.
**How it works** — 3–6 steps.
**Math (if any)** — formulas + what each symbol means.
**Complexity** — time / space (if algorithmic).
**Why we chose it / Why not alternatives** — vs 2–3 real options, with tradeoffs.
**Internal working** — high-level, no source dumps.
**Common bugs** — mistakes + how I'd debug them. (Grows as I hit real bugs.)
**Interview questions** — 5–8.
**Mini exercise** — one small task I can do by hand.
**5 takeaways.**

---

## EXAMPLE ENTRY (the quality bar — Claude Code: match this)

### Vector Embeddings & Cosine Similarity            [Phase 1]
**What it is** — An embedding model maps text to a vector (list of numbers,
here 384-dim from bge-small) so that similar meanings land close together in
space. Search becomes geometry.
**Problem it solves** — Keyword search can't tell that "refund policy" and
"money back rules" mean the same thing. We need meaning-based retrieval to find
the right chunks for the LLM.
**Intuition** — A map of ideas: every sentence gets GPS coordinates; related
sentences are neighbors, unrelated ones are far apart.
**How it works** — 1) Tokenize text. 2) Transformer encoder produces token
vectors. 3) Pool (mean/CLS) into one sentence vector. 4) Normalize. 5) Compare
vectors with cosine similarity; nearest = most relevant.
**Math** — cos(θ) = (A·B)/(||A||·||B||), where A·B = Σ aᵢbᵢ. Ranges −1…1;
higher = more similar. If vectors are L2-normalized (||A||=1), cosine equals
the dot product — which is why vector DBs normalize once at insert and then do
cheap dot products. Contrast: Euclidean distance measures length AND direction;
cosine ignores magnitude, which suits text (a long and short sentence about the
same topic should still match).
**Complexity** — one comparison O(d); brute-force search O(N·d). Qdrant's HNSW
index makes it ~O(log N) per query (see HNSW entry).
**Why bge-small / why not alternatives** — Free + local + better than MiniLM
(what Sage used). vs OpenAI text-embedding-3-large: higher quality but paid and
data leaves our machine — our provider factory lets us swap later by env var.
vs plain BM25 keyword search: catches exact terms but misses synonyms; hybrid
is a later upgrade.
**Internal working** — bge-small is a small BERT-style encoder fine-tuned with
contrastive learning: pull (query, relevant passage) pairs together, push
irrelevant apart in the vector space.
**Common bugs** — (1) Mixing vectors from two different models in one
collection → nonsense results; that's why dimension + model name live in the DB
per bot. (2) Forgetting to use the query-prefix some BGE models expect →
silently worse retrieval. (3) Comparing unnormalized vectors with dot product →
long texts win unfairly.
**Interview Qs** — Why cosine over Euclidean for text? What does dimensionality
trade off? How is an embedding model trained (contrastive loss)? Why can't you
compare embeddings from different models? Dense vs sparse retrieval? What
happens to recall as chunks get longer?
**Mini exercise** — Compute cosine similarity of A=[1,2,2], B=[2,1,2] by hand;
then verify with numpy.
**Takeaways** — meaning→geometry · cosine ignores length · normalize once, dot
product forever · never mix models in one index · quality of embeddings caps
quality of RAG.

---

## CONCEPT QUEUE (Claude Code: add each as we reach it; tick when written)

**Phase 0** — [x] Docker & docker-compose · [x] env-based config
(pydantic-settings) · [x] Alembic migrations · [x] pre-commit/linting
**Phase 1** — [x] Embeddings & cosine similarity · [ ] Chunking & overlap
(why size matters for recall vs precision) · [ ] HNSW / vector indexes ·
[ ] Two-stage retrieval & rerankers (cross-encoder vs bi-encoder — the math of
why rerank wins) · [ ] Grounded prompting & hallucination control ·
[x] Protocols/interfaces (structural typing) · [ ] the factory pattern ·
[x] async/await in Python
**Phase 2** — [ ] Multi-tenancy & payload-filter isolation · [ ] JWT & Clerk
auth (what's inside a JWT, signature verification) · [ ] Celery task queues
(broker, worker, ack, retry) · [ ] SQLAlchemy ORM & sessions
**Phase 3** — [ ] SSE streaming (vs WebSockets vs polling) · [ ] Rate limiting
& metering (token bucket) · [ ] Idempotency
**Phase 4** — [ ] React/Next.js mental model (server vs client components) ·
[ ] Tailwind basics
**Phase 5** — [ ] iframes, postMessage & CORS (why the widget is sandboxed)
**Phase 6** — [ ] Stripe subscriptions & webhooks (why webhooks, signature
verification, replay attacks)
**Phase 7** — [ ] Reverse proxy & HTTPS (Caddy) · [ ] p50/p95 latency & basic
observability

---

(new entries below)

---

### Docker & docker-compose            [Phase 0]
**What it is** — Docker packages a program together with its OS-level
dependencies into an *image*; running that image gives an isolated *container*.
Compose is a YAML file declaring several containers and their wiring so one
command starts a whole environment.
**Problem it solves** — This project needs Postgres, Redis, and Qdrant at
specific versions. Installing them natively on Windows means version drift, no
clean reset, and a painful migration when the app moves to the Oracle ARM box.
**Intuition** — An image is a recipe; a container is one meal cooked from it.
Throwing away the meal doesn't harm the recipe, and everyone cooking from the
same recipe gets the same dish.
**How it works** — 1) An image is built as read-only layers, each a filesystem
diff. 2) Running it adds a thin writable layer on top. 3) The kernel isolates
the process via namespaces (PID, network, mount) and limits it via cgroups.
4) Compose creates a private network where services resolve each other by
service name. 5) Named volumes hold data outside the container's writable
layer, so it survives the container's death.
**Math** — None. Only `interval × retries` = worst-case healthcheck detection
time (5s × 10 = 50s in our compose file).
**Why we chose it / alternatives** — vs native installs: reproducible and
disposable, at the cost of a background daemon and some RAM. vs full VMs (Vagrant):
containers boot in ~0.1s versus ~30s and use megabytes not gigabytes, but share
the host kernel so isolation is weaker — fine for local infrastructure, not for
running untrusted tenant code. vs cloud-only dev (Neon + Qdrant Cloud from day
one): no local setup at all, but you can't work offline and free tiers throttle.
**Internal working** — Containers are ordinary Linux processes with restricted
views of the system. Namespaces control *what a process can see* (its own PID
tree, network stack, filesystem root); cgroups control *how much it can use*
(CPU, memory). There is no hardware virtualization. On Windows, Docker Desktop
runs a Linux VM underneath, which is why the daemon must be running first.
**Common bugs** — (1) Changing `POSTGRES_PASSWORD` after first boot appears to
do nothing — the init script only runs on an empty data directory; fix with
`down -v`, which destroys local data. (2) `docker compose down -v` wipes named
volumes; `down` alone preserves them. (3) **Silent port hijack on Windows (hit
this for real, Phase 0).** A natively-installed PostgreSQL Windows service
already held 5432. On Linux a second `bind()` to a taken port is refused; on
Windows it is *allowed* unless the first socket set `SO_EXCLUSIVEADDRUSE`, which
the Postgres installer does not. So Docker bound 5432 too, reported success, and
`docker compose ps` showed `healthy` — while the app's connections reached the
*native* server, which has no `ragsaas` role. Postgres reports a missing role as
`password authentication failed` (to prevent username enumeration), which sends
you hunting for a credential bug that doesn't exist. Diagnose with
`netstat -ano | findstr :5432` — **two PIDs is the tell** — then `tasklist /FI
"PID eq <pid>"`. Fix: move the *host* side only (`"5433:5432"`). Also: verify
with `select version()`; the container reports `linux-musl`, a native Windows
server reports `windows`. (4) Using
`localhost` in a *container's* config to reach another container; inside the
compose network the hostname is the service name. (5) `:latest` tags making a
rebuild silently upgrade your database — always pin.
**Interview Qs** — Container vs VM? What do namespaces and cgroups each do?
Image vs container? Why are images layered, and how does that help caching?
Why is `down -v` dangerous? How do containers on a compose network find each
other? Why should logs go to stdout rather than a file inside the container?
**Mini exercise** — Run `docker compose up -d`, write a row into Postgres via
`docker exec`, then `docker compose down` followed by `up` and confirm the row
survived. Then `down -v`, `up`, and confirm it's gone.
**Takeaways** — image = recipe, container = meal · volumes are the only durable
part · service name is the hostname · pin every tag · healthcheck ≠ started ·
shares the host kernel, so isolation is cheaper but weaker than a VM.

---

### Env-based config & pydantic-settings            [Phase 0]
**What it is** — A library that loads environment variables and `.env` files
into a typed Python class, coercing and validating each value, and refusing to
start the app when something is missing or malformed.
**Problem it solves** — Secrets must never be in code (Hard Rule #1), and the
same codebase must run locally, in CI, and in production with different
databases and providers. Scattered `os.getenv` calls give untyped strings,
silent `None`s, and no way to enumerate what config the app even has.
**Intuition** — A building's electrical panel: every circuit terminates in one
labelled box instead of random junctions hidden behind the walls.
**How it works** — 1) Declare a `Settings(BaseSettings)` class, one field per
variable. 2) At instantiation, each field is resolved by precedence: real env
var → `.env` file → default. 3) The raw string is coerced to the declared type
(`int`, `Literal[...]`). 4) Validation failures raise `ValidationError` listing
every bad field at once. 5) `@lru_cache` makes it a per-process singleton.
**Math** — None.
**Why we chose it / alternatives** — vs raw `os.getenv`: free validation, typing,
and a single discoverable schema, at the cost of a dependency. vs
`python-dotenv` alone: that only loads `.env` into `os.environ` and still leaves
you with untyped strings. vs a config file (YAML/TOML) committed to the repo:
easy to read but tempts you into committing secrets and doesn't match how
containers inject config. The Twelve-Factor App principle is that config lives
in the environment precisely because that's what differs between deployments.
**Internal working** — Built on pydantic v2, whose validation core is compiled
Rust (`pydantic-core`). `BaseSettings` subclasses the normal model and adds
settings *sources*, tried in priority order. Field names map to env vars
case-insensitively.
**Common bugs** — (1) A stale exported shell variable silently beating your
`.env`, because real env vars have higher precedence — check `echo $VAR` first.
(2) `extra="ignore"` hiding typos: `DATABSE_URL` in `.env` is not an error, the
field just uses its default and you connect to the wrong database. (3) Import-time
instantiation means *any* import of the app requires valid config, including in
tests — expose `get_settings()` so tests can `cache_clear()`. (4) Adding a
required field without updating `.env.example`, breaking everyone else's startup.
**Interview Qs** — Why validate config at startup instead of on first use? What
is the precedence order of config sources and why does the env var win? Why is
`.env` inappropriate for production secrets? What does the Twelve-Factor App say
about config? How would you handle a secret that must be rotated without a
restart? Why cache the settings object?
**Mini exercise** — Set `APP_ENV=prod` (invalid — the `Literal` expects
`production`) and run `pytest`. Read the `ValidationError` carefully: note that
it names the field, the invalid value, and the permitted set.
**Takeaways** — one panel, not many junctions · fail fast at boot beats failing
at 2am · env var > .env > default · types catch what strings hide · `.env` is a
dev convenience, secrets managers are the production answer.

---

### Alembic migrations            [Phase 0]
**What it is** — A schema migration tool for SQLAlchemy. It compares your ORM
models against the live database and generates a numbered, reversible Python
script containing the SQL needed to close the gap.
**Problem it solves** — Databases hold state, so you can't just redeploy a new
schema the way you redeploy code. Without migrations, your laptop's schema and
production's drift apart with no record of why, and Hard Rule #5 exists to stop
exactly that.
**Intuition** — Git for table structure: every schema change is a commit you can
replay forward or roll back.
**How it works** — 1) `env.py` imports your models so `Base.metadata` holds every
table. 2) `alembic revision --autogenerate` reflects the live schema and diffs
it against that metadata. 3) It writes a script with `upgrade()` and
`downgrade()`. 4) You read and correct it. 5) `alembic upgrade head` runs
pending scripts in order and records each in the `alembic_version` table, so
they never run twice.
**Math** — Not maths, but the structure matters: migrations form a singly linked
list via each script's `down_revision` pointer, becoming a DAG when branches
merge. `head` is the tail; `base` is the start.
**Why we chose it / alternatives** — vs `Base.metadata.create_all()`: that only
creates *missing* tables and never alters existing ones, so it silently does
nothing the moment you add a column — acceptable for throwaway test DBs, unsafe
for real data. vs hand-written SQL files: total control but no autogeneration,
no ordering guarantees, and no rollback. vs Django migrations: excellent, but
tied to Django's ORM. Alembic is written by SQLAlchemy's author and is the
standard for this stack.
**Internal working** — The `alembic_version` table stores exactly one row: the
current revision ID. On `upgrade`, Alembic reads it, walks the script chain from
there to `head`, and executes each `upgrade()` inside a transaction — Postgres
supports transactional DDL, so a failed migration rolls back cleanly rather than
leaving a half-changed schema.
**Common bugs** — (1) **Autogenerate cannot detect renames** — it sees a drop
plus an add, which destroys that column's data; hand-edit to
`op.alter_column(..., new_column_name=...)`. (2) A model not imported into the
registry is absent from the metadata, so Alembic proposes `DROP TABLE` for it —
always read generated migrations before running them. (3) Type changes are
ignored without `compare_type=True`. (4) Editing an already-applied migration
instead of writing a new one, so machines that ran the old version never get the
fix. (5) Running Alembic from the wrong directory, breaking `import app`.
**Interview Qs** — Why not `create_all()`? How do you rename a column with zero
downtime? What's in `alembic_version` and why? How does Alembic know which
migrations are pending? What can autogenerate not detect? Why does putting the
DB URL in `alembic.ini` matter? How would you roll back a bad migration in
production?
**Mini exercise** — Once Phase 2 adds a model, run `--autogenerate`, then read
the generated script line by line and predict the exact SQL before running
`upgrade head --sql` to check yourself.
**Takeaways** — schema needs version control as much as code does · autogenerate
is a draft, not an authority · renames are the trap · never edit an applied
migration · expand/contract is how you change schema without downtime.

---

### Linting, formatting & pre-commit hooks            [Phase 0]
**What it is** — A linter (ruff) analyses code without running it to find bugs
and style violations; a formatter (black) rewrites code into one canonical
style; pre-commit runs both automatically before each commit and blocks it on
failure.
**Problem it solves** — This project has hard architectural rules (no vendor SDK
outside `app/providers/`, no secrets in code). Automated checks enforce what
discipline forgets, and they catch real bugs — unused variables, shadowed names,
mutable default arguments — before the code ever runs.
**Intuition** — A spellchecker that won't let you hit send on the email.
**How it works** — 1) `pre-commit install` writes a script into
`.git/hooks/pre-commit`. 2) On `git commit`, git runs it. 3) pre-commit builds
each tool in its own isolated environment at the pinned `rev`. 4) Each hook runs
against **staged files only**. 5) A hook that modifies a file fails the commit
so you can review the change, re-stage, and commit again.
**Math** — None.
**Why we chose it / alternatives** — ruff vs flake8+isort+pyupgrade+bugbear: ruff
is one Rust binary replacing all of them at roughly 100× the speed, so it stays
fast enough to run on every save. black vs manual style review: black has
essentially no options, which sounds restrictive but ends all formatting debate —
its real value is social, not technical. Hooks vs CI-only: hooks fail in ~2s
locally, CI fails minutes later after a push; but hooks are skippable with
`--no-verify`, so CI must remain the actual enforcement point. Use both.
**Internal working** — ruff parses source into an AST and matches rule patterns
against it, which is why it finds semantic issues (an unused import) rather than
just textual ones. black parses to an AST, discards your formatting entirely,
and re-prints from the tree — the reason its output is deterministic regardless
of input formatting.
**Common bugs** — (1) Enabling two formatters (we briefly had `ruff-format` *and*
black) so each commit churns files back and forth. (2) Forgetting
`pre-commit install` — the config file alone does nothing, and everyone assumes
the hooks are running. (3) Hook order: `ruff --fix` must precede black, or ruff
can rewrite an import and leave it unformatted. (4) Both E501 (line length) and
black enabled, producing contradictory demands — hence `ignore = ["E501"]`.
(5) Hooks see only staged files, so partial staging can commit unchecked code.
**Interview Qs** — Linter vs formatter vs type checker? Why run checks in a hook
if CI already does? What class of bugs can a linter catch that tests can't, and
vice versa? Why is black deliberately non-configurable? How do you introduce a
formatter to a large existing codebase without one giant diff? Why pin hook
versions?
**Mini exercise** — Write a function with a mutable default argument
(`def f(x, acc=[])`), then run `ruff check` and read rule B006. Work out for
yourself why that default is shared across calls.
**Takeaways** — automate what discipline forgets · one formatter, never two ·
lint before format · hooks for speed, CI for authority · black's value is that
it ends the argument.
---

### Interfaces, Protocols & structural typing            [Phase 1]
**What it is** — An *interface* is a promised set of methods, separated from any
particular implementation. Python offers two ways to express one: `ABC` (nominal
— you must inherit) and `typing.Protocol` (structural — you must merely *match
the shape*). We use `Protocol` in `app/providers/base.py` for the three contracts
`LLMProvider`, `EmbeddingProvider`, `RerankProvider`.
**Problem it solves** — Without an interface, `core/retrieval.py` would call
`google.generativeai` directly and "swap to Groq" would mean rewriting business
logic in a dozen places. PLAN §3's whole selling point — swap a provider by
editing one env var — only exists because the code above the boundary knows a
*shape*, not a vendor. It also makes tests possible: a fake embedder that returns
fixed vectors satisfies the Protocol, so `core/` can be tested with no model
downloaded and no network.
**Intuition** — A wall socket. It defines the plug shape, not the power station.
Coal, solar, nuclear behind it — your lamp is unchanged. And notably the *lamp*
manufacturer defined the plug, not the power station: the consumer owns the
interface.
**How it works** — 1) Declare a class inheriting `Protocol` with method
signatures and `...` bodies. 2) Consumers annotate parameters with that Protocol
type. 3) Implementations define matching methods, *without importing or
inheriting* the Protocol. 4) A static checker (mypy/Pyright) verifies the shapes
line up. 5) At runtime nothing happens — Python just calls the method. 6)
Optionally `@runtime_checkable` enables `isinstance`, which checks method *names
only*.
**Math (if any)** — None.
**Complexity** — None at runtime; Protocols vanish after type-check.
**Why we chose it / Why not alternatives** — *vs `ABC`*: ABC would force
`class GeminiLLM(LLMProvider)`, making the vendor file import the abstraction —
the dependency arrow points both ways, and the base class becomes a magnet for
"shared" logic that quietly turns vendor-specific. ABC's win is a real runtime
error if a method is missing; we trade that for one-way dependencies. *vs plain
duck typing (no annotations)*: works until someone renames a method and nothing
catches it until production. *vs a single big `LLMClient` class with `if
provider == "gemini"` branches*: every new provider edits the same file, the
class grows all vendors' quirks, and testing needs the whole thing — this is the
exact code smell the factory pattern exists to remove.
**Internal working** — `Protocol` subclasses are ordinary classes with
`_is_protocol = True` and a `__subclasshook__` that (when `@runtime_checkable`)
compares attribute *names* via `hasattr`. Type checkers do the real work
statically, comparing parameter types, count, and return type — and they check
methods **contravariantly** in arguments, covariantly in returns, which is why an
implementation may widen a parameter type but not narrow it.
**Common bugs** — (1) Assuming `Protocol` enforces anything at runtime: a
missing method is an `AttributeError` at first call, not at class definition.
(2) Trusting `isinstance` with `@runtime_checkable` — it ignores signatures
entirely, so a wrong-arity `stream()` passes. (3) `async def` + `yield` makes an
async *generator*: the body doesn't run until the first `async for`, so a
try/except around the *call* catches nothing — wrap the iteration.
(4) Adding a method to the Protocol and forgetting an implementation: silent
until that provider is selected, which is why every provider needs at least a
smoke test.
**Interview questions** — Protocol vs ABC, and when would you pick each? What is
structural vs nominal typing? Does Python enforce Protocols at runtime? What does
`@runtime_checkable` actually check? Why is it valuable that the implementation
doesn't import the interface? How would you unit-test code that depends on a paid
LLM API? What is the Dependency Inversion Principle and how does this file show
it? Why do we return `AsyncIterator[str]` instead of `str`?
**Mini exercise** — Write a `Protocol` with one method, then a class that
implements it with a *different* parameter type. Confirm the code runs fine, then
run mypy and read the error. That gap between "runs" and "type-checks" is the
whole lesson.
**Takeaways** — interfaces belong to the consumer, not the vendor · structural
typing keeps the dependency arrow one-way · Protocols are compile-time only ·
`isinstance` on a Protocol is a smoke test, not validation · the interface is
what makes the code testable without the vendor.

---

### async/await, the event loop & blocking work            [Phase 1]
**What it is** — `async def` defines a *coroutine*: a function that can pause at
an `await` and hand control back to an **event loop**, which then runs some other
ready coroutine. One thread, many in-flight operations. `asyncio.to_thread(fn)`
is the escape hatch for code that can't pause — it runs `fn` on a worker thread
and gives you something awaitable.
**Problem it solves** — A chat request spends almost all its time *waiting*:
waiting on Qdrant, waiting on the LLM's token stream. With ordinary blocking
code, one user waiting 4 seconds for Gemini occupies a whole thread; 100
concurrent chats need 100 threads. With async, those 100 waits cost almost
nothing and one process serves them all. That is why FastAPI is async and why
`EmbeddingProvider.embed_query` is `async` even though bge runs locally.
**Intuition** — A waiter in a restaurant. They don't stand at your table while
the kitchen cooks — they take your order and serve four other tables while
waiting. `await` is "put this order in and go do something useful." A blocking
call is a waiter who stands frozen at one table; the other tables starve.
**How it works** — 1) Calling a coroutine returns a coroutine object; nothing
runs yet. 2) `await`ing it (or wrapping in a Task) hands it to the event loop.
3) The loop runs it until the next `await` on something not-yet-ready. 4) The
coroutine yields control; the loop runs other ready tasks. 5) When the awaited
thing completes (socket readable, thread finished), the loop resumes the
coroutine exactly where it paused. 6) Cooperative, not preemptive — a coroutine
that never awaits **never gives the loop a turn**.
**Math (if any)** — None. But the intuition is Little's Law: throughput =
concurrency ÷ latency. Async raises achievable concurrency for I/O-bound work
without raising thread count.
**Complexity** — Task switch is a function return, ~microseconds; a thread
context switch is ~10–100× costlier and each thread costs ~8MB of stack.
**Why we chose it / Why not alternatives** — *vs threads-per-request*: memory
and context-switching cost, plus real race conditions on shared state; async has
no preemption, so between two `await`s your code is atomic. *vs processes*: the
right answer for CPU-bound work (and what Celery gives us in Phase 2), but far
too heavy per request. *vs sync FastAPI endpoints*: FastAPI will run a `def`
endpoint in a threadpool automatically, which is fine — but SSE streaming in
Phase 3 fundamentally wants an async generator, so async is the through-line.
**Internal working** — Coroutines are built on generators: `await` compiles to a
`yield` that bubbles a "not ready yet" signal up to the loop, carrying a future.
The loop keeps a ready queue plus an OS readiness poller (`epoll`/IOCP). Native
extensions like torch and numpy release the GIL inside their C kernels, which is
exactly why `asyncio.to_thread` works for embedding: the math genuinely runs in
parallel on another core. Pure-Python CPU work does not benefit — that needs a
process.
**Common bugs** — (1) A blocking call inside `async def` (`requests.get`,
`time.sleep`, `model.encode`) freezes *every* request in the process; symptom is
"everything is slow when one thing is slow." Fix: async client, or
`asyncio.to_thread`. (2) Forgetting `await` — you get a coroutine object,
often silently truthy, and the work never runs; ruff/mypy catch it. (3)
Expecting a lazy `async def ... yield` generator to run at call time — its body
starts on first `async for`, so exceptions surface at iteration, not at the call.
(4) `asyncio.Lock` is not thread-safe and `threading.Lock` blocks the loop —
they are not interchangeable. (5) Assuming `async` makes CPU-bound code faster;
it makes it *worse* by adding overhead to something that still blocks.
**Interview questions** — What exactly does `await` do? Difference between
concurrency and parallelism, and which does asyncio give you? Why doesn't the GIL
stop `asyncio.to_thread` from helping with torch? What happens if you call
`time.sleep(5)` in an async endpoint? When would you choose a process pool over a
thread pool? Why is `async` a good fit for an LLM proxy specifically? How do you
run several provider calls concurrently (`asyncio.gather`)? What is a coroutine
object before it's awaited?
**Mini exercise** — Write two async functions that each `time.sleep(1)`, gather
them, time it (≈2s). Swap to `asyncio.sleep(1)`, time it again (≈1s). The
difference *is* the event loop, felt directly.
**Takeaways** — async wins on waiting, never on computing · one blocking call
poisons the whole loop · `to_thread` works because C extensions drop the GIL ·
cooperative means you must yield · CPU-bound work belongs in a process/Celery.

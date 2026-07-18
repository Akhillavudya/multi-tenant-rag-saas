# Phase 0 — Foundations: file-by-file explanation

Build depth for every file created in Phase 0, in the order it was created.
Concept-level revision notes live in `LEARNING.md`; this file is about *these
specific files* and why they look the way they do.

**Phase 0 goal (PLAN.md §6):** repo init, `docker compose up` gives Qdrant +
Redis (+ Postgres), FastAPI skeleton with `/health`, Alembic wired,
pydantic-settings config, pre-commit (ruff, black).
**Done when:** `uvicorn` runs, `/health` returns ok, `pytest` runs,
`.env.example` exists and no secret is committed.

---

## 1. `.gitignore`

**Purpose** — Tells git which files to pretend don't exist, so secrets and junk
never get committed. Without it, your first `git add .` commits `.env`, and a
key pushed to GitHub is compromised forever even if the next commit deletes it.

**Analogy** — A bouncer at the door of your repo holding a list of who is never
allowed in. It doesn't remove people already inside — it only stops new ones at
the door.

### Main pieces

| Entry | What it does |
|---|---|
| `.env` | The one that actually matters. Real secrets live here; this line is what keeps Hard Rule #1 true. |
| `__pycache__/`, `*.py[cod]` | Python's compiled bytecode cache. Machine-specific, auto-regenerated, pure diff noise. |
| `.venv/` | The virtual environment — hundreds of MB of libraries, reproduced from `pyproject.toml` instead. |
| `models/`, `.cache/` | Where bge-small's weights land in Phase 1. ~130MB of binary; git handles binaries badly. |
| `.data/` | Docker bind-mount data, if we ever use one instead of named volumes. |
| `node_modules/`, `.next/` | Phase 4's frontend equivalent of `.venv`. |

### Flow

Not executable — a config file git consults. On every `git add` / `git status`,
git walks the working tree and checks each path against these patterns. A match
means the file is untracked and invisible. Trailing `/` = directory only;
`*` matches within a single path segment; a leading `!` re-includes something a
previous pattern excluded.

### Math
None.

### Gotchas

- **`.gitignore` only affects *untracked* files.** If a file is already tracked,
  adding it here changes nothing — git keeps versioning it. You need
  `git rm --cached <file>` to untrack it. This is exactly why `.gitignore` must
  exist *before* the first `git add`.
- Ignoring a directory (`models/`) is cheaper than ignoring its contents
  (`models/**`) — git stops descending into it entirely.

### Interview questions

**Q: You accidentally committed a secret and pushed it. Is adding it to
`.gitignore` enough?**
No, on two counts. `.gitignore` doesn't untrack an already-tracked file, and the
secret still exists in every historical commit, readable by anyone who clones or
by anyone who already forked. The only real remedy is **rotating the key**.
History rewriting (`git filter-repo`, BFG) cleans the repo afterwards but cannot
un-leak what was publicly visible.

**Q: Difference between `.gitignore`, `.git/info/exclude`, and
`--assume-unchanged`?**
`.gitignore` is committed and shared with the team — project-wide rules.
`.git/info/exclude` is identical in behaviour but local and uncommitted — good
for your personal editor droppings that shouldn't be imposed on others.
`--assume-unchanged` applies to *tracked* files, telling git to stop watching
them locally; it's a performance/convenience flag, not an ignore mechanism, and
it silently causes "why isn't my change showing up" confusion.

---

## 2. `.env.example`

**Purpose** — Documents every environment variable the app needs without
containing a single real value, so it's safe to commit. Without it, a new
developer (or you on a new machine) clones the repo and can't tell which of ~25
keys must be set before anything will start.

**Analogy** — A blank form with every field labelled but no answers filled in.
`.env` is your completed copy; `.env.example` is the blank one you hand over.

### Main pieces

| Section | Why it's there |
|---|---|
| `DATABASE_URL` | Connection string, format `postgresql+psycopg://user:pass@host:port/dbname`. The `+psycopg` suffix names the *driver* — that's SQLAlchemy syntax, not standard Postgres URL syntax. |
| `REDIS_URL` | The trailing `/0` is the Redis database number; Redis ships 16 numbered logical DBs on one server. |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` | Local Qdrant needs no key; Qdrant Cloud does. One collection (`chunks`) for all tenants — see PLAN §4. |
| `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `RERANK_PROVIDER` | **The swappability contract (PLAN §3).** These three lines are the entire "upgrade to premium" mechanism. `gemini` → `groq` must need zero code edits. |
| Credential blocks | Blank on purpose. A provider's key is only required when that provider is selected. |
| Phase-tagged blocks | Clerk (Phase 2), Stripe (Phase 6) listed early so this file stays the single source of truth for config. |

### Flow

`cp .env.example .env` → fill in real values → `pydantic-settings` reads `.env`
at import time inside `config.py` → every module imports `settings` from there.
The `.example` file itself is never read by the running app; it is documentation
that happens to be machine-shaped.

### Math
None.

### Gotchas

- **Drift is the real enemy.** Every key here must have a matching field in
  `config.py` and vice versa. A key in `.env` with no field in `Settings` is
  silently ignored — you think it's configured, it isn't. `extra="ignore"` in
  the config makes this *deliberately* tolerated for shared keys, which raises
  the stakes on keeping the two files in sync.
- Never put a real value here "just as an example." Placeholder-looking secrets
  get committed and then used.

### Interview questions

**Q: Why commit `.env.example` instead of documenting config in the README?**
It's colocated with the code, diffable, and drift becomes visible in review — a
PR that adds a config field without the example line is an obvious omission.
README prose rots silently. It also copy-pastes straight into a working `.env`.

**Q: Your app needs 20 secrets in production. Is `.env` the right mechanism?**
No. `.env` files are a development convenience. Production wants a secrets
manager (Vault, AWS Secrets Manager, Doppler) or the platform's encrypted env
vars, because those provide rotation, audit logging, and access control that a
plaintext file on a box cannot. The saving grace is that application code
doesn't change: `pydantic-settings` reads real environment variables
identically, so only the delivery mechanism differs.

---

## 3. `docker-compose.yml`

**New concept — Docker & Compose.** Docker packages a program plus its whole
OS-level environment into an *image*, which runs as an isolated *container*, so
"works on my machine" becomes "works on every machine." Compose is a YAML file
declaring several containers and their wiring, so one command starts the entire
infrastructure. We use it here because this project needs three servers
(Postgres, Redis, Qdrant) at pinned versions on a Windows laptop that will later
be redeployed to an Oracle ARM box — installing them natively would mean version
drift and a painful migration.

**Purpose** — Gives you Postgres + Redis + Qdrant running locally via
`docker compose up -d`, at pinned versions, isolated from your OS. Without it
you'd install three databases natively, fight version drift, and have no clean
reset when local state gets corrupted.

**Analogy** — An image is a recipe; a container is one cooked meal from it, and
throwing a meal away doesn't harm the recipe. Compose is the dinner order for
the whole table: three dishes and which must be ready before the others.

### Main pieces

| Key | What it does |
|---|---|
| `services:` | Each entry is one container. Compose also creates a private network where services reach each other **by service name** — from inside a container, Postgres is at `postgres:5432`, not `localhost`. |
| `image:` | Which prebuilt image to pull. Every tag is pinned (`postgres:16-alpine`, `qdrant/qdrant:v1.12.4`), never `:latest`, or a rebuild months later silently upgrades your database. `alpine` = minimal Linux base, roughly 10× smaller. |
| `container_name:` | Fixed, readable name for `docker logs ragsaas-postgres`. Without it Docker generates names like `project-postgres-1`. |
| `ports: "5432:5432"` | `HOST:CONTAINER`. Publishes the container port onto the Windows host — this is why host-run FastAPI can reach `localhost:5432`. Without it the service is reachable only from inside the compose network. |
| `environment:` | Config injected into the container. The Postgres image reads `POSTGRES_USER/PASSWORD/DB` **on first boot only**, to initialize the data directory. Changing them later does nothing until the volume is wiped. |
| `volumes:` | Named volumes (`postgres_data:`) are Docker-managed disk that outlives the container. Containers are ephemeral — no volume means `docker compose down` destroys your database. |
| `healthcheck:` | A command Docker loops to decide whether a service is *ready*, not merely started. Postgres accepts connections several seconds after its process launches. |

### Flow

1. `docker compose up -d` reads this file.
2. Docker pulls any missing images from Docker Hub.
3. Creates the private network and the three named volumes if they don't exist.
4. Starts all three containers — no `depends_on` here, so they boot in parallel.
5. Each healthcheck starts polling on its `interval`; `docker compose ps` shows
   `healthy` once `retries` consecutive checks pass.
6. `-d` detaches, so containers keep running after the terminal closes.

### Math

Only trivial arithmetic: `interval × retries` is the worst-case time before a
service is declared unhealthy — here `5s × 10 = 50s`.

### Gotchas

- **The Qdrant healthcheck is unusual.** Most images have `curl`, so you'd write
  `curl -f http://localhost:6333/healthz`. The Qdrant image ships neither `curl`
  nor `wget`, so we use bash's built-in `/dev/tcp` pseudo-device to test whether
  the port accepts a TCP connection. Weaker (it proves the port is open, not
  that the API is healthy) but it needs no extra tooling in the image.
- **Changing `POSTGRES_PASSWORD` after first boot appears to do nothing.** The
  init script only runs on an empty data directory. Fix: `docker compose down -v`
  then `up` — destroying local data.
- Port conflicts: if you already run Postgres natively on 5432, the container
  fails to bind. Change the *host* side only: `"5433:5432"`.

### Interview questions

**Q: Container vs virtual machine?**
A VM virtualizes hardware and runs a full guest kernel — gigabytes, ~30s boot.
A container shares the host kernel and isolates only process namespace,
filesystem, and network — megabytes, ~0.1s boot. Containers are far cheaper but
provide weaker isolation: a kernel exploit escapes a container but not a VM.
That's why untrusted multi-tenant workloads often still run in VMs or
microVMs (Firecracker, gVisor).

**Q: What's the difference between `docker compose down` and `down -v`?**
`down` removes containers and the network but keeps named volumes, so data
survives a restart. `-v` additionally deletes the volumes, wiping every
database. It's the "clean slate" button — useful when a bad local migration
corrupts state, catastrophic if aimed at anything you care about.

---

## 4. `backend/pyproject.toml`

**New concepts.**
*`pyproject.toml`* is the modern, standardized Python project file (PEP 518/621).
It replaced the old scatter of `setup.py` + `requirements.txt` + `setup.cfg` +
`.flake8` + `pytest.ini` with one declarative file. We use it because this
project will accumulate five or six tools, and one config file beats six.
*ruff* is a Rust-written linter that runs the checks of flake8 + isort +
pyupgrade + bugbear roughly 100× faster; we use it because automated checks
enforce our hard rules better than discipline does.
*black* is an opinionated auto-formatter with essentially no options; we use it
so code formatting never becomes a review discussion.

**Purpose** — Declares what the backend *is*: dependencies, required Python
version, and the configuration for every tool. Without it there's no
reproducible install and each tool needs its own separate config file.

**Analogy** — A recipe card listing ingredients and oven settings in one place.
Anyone with the card reproduces the same dish; without it they're guessing.

### Main pieces

| Section | What it does |
|---|---|
| `[project]` | Package metadata. `requires-python = ">=3.11"` makes pip refuse to install on an older interpreter rather than failing mysteriously later — we rely on 3.10+ syntax like `str \| None`. |
| `dependencies` | What the app needs to *run*. Pinned with `>=` (floors, not exact pins) — a lockfile would pin exactly, but for a solo learning project floors keep upgrades easy. |
| `[project.optional-dependencies] dev` | Test/lint-only packages, installed via `pip install -e ".[dev]"`. Kept separate so a production image doesn't ship pytest. |
| `[build-system]` | Tells pip *how* to build this package — setuptools here. Required for `pip install -e .` (editable install) to work at all. |
| `[tool.setuptools.packages.find] include = ["app*"]` | Without this, setuptools also tries to package the `tests` directory and errors on ambiguity. |
| `[tool.black]` | `line-length = 88` (black's default, chosen so two files sit side by side on a normal screen). |
| `[tool.ruff.lint] select` | Which rule families are active — see table below. |
| `[tool.pytest.ini_options]` | `testpaths`, plus `asyncio_mode = "auto"` so async tests need no decorator, plus `filterwarnings = ["error::DeprecationWarning"]` which turns deprecation warnings into test failures so we find out about breaking library changes early instead of on upgrade day. |

**Ruff rule families we selected:**

| Code | Family | Catches |
|---|---|---|
| `E`,`W` | pycodestyle | Style/whitespace issues |
| `F` | pyflakes | Real bugs: unused imports/variables, undefined names |
| `I` | isort | Import ordering — keeps diffs small |
| `B` | flake8-bugbear | Genuine footguns, e.g. mutable default arguments |
| `UP` | pyupgrade | Old syntax that 3.11 improves on |
| `N` | pep8-naming | Naming conventions |

We `ignore = ["E501"]` (line too long) because black already handles line length,
and having both complain produces contradictory demands.

### Flow

1. `pip install -e ".[dev]"` reads `[build-system]`, uses setuptools to install
   the package in *editable* mode — your source directory is linked, not copied,
   so edits take effect without reinstalling.
2. `dependencies` + the `dev` extra are resolved and installed.
3. Later, `ruff check .`, `black .`, and `pytest` each read their own `[tool.*]`
   section from this same file. No tool needs a CLI flag.

### Math
None.

### Gotchas

- **Editable install (`-e`) matters more than it looks.** Without it, `import app`
  resolves to a *copy* in site-packages, and your edits appear to do nothing —
  a genuinely maddening bug to hit as a beginner.
- `filterwarnings = ["error::DeprecationWarning"]` is strict on purpose. It will
  occasionally fail a test because a *library* is deprecated internally; the fix
  is to add a targeted `ignore::` entry, not to delete the rule.
- Dependency floors (`>=`) mean two installs months apart can differ. If builds
  ever become non-reproducible, that's the moment to add a lockfile (`uv lock`
  or pip-tools), not before.

### Interview questions

**Q: What does `pip install -e .` actually do, and when would you not want it?**
It installs the project in editable/development mode: rather than copying files
into site-packages, it adds a link back to your source tree, so code changes are
live immediately. You don't want it in a production container image, where you
want an immutable, copied install with no dependence on a source directory that
may not exist.

**Q: Why separate `dependencies` from an optional `dev` extra?**
Attack surface, image size, and honesty about what the software requires. A
production deployment installing pytest, linters, and their transitive
dependencies ships hundreds of megabytes of code it never runs — every one of
which is a potential CVE to patch. The split also documents the boundary: if
something is truly needed at runtime, it belongs in `dependencies`.

---

## 5. `backend/app/config.py`

**New concept — pydantic-settings.** A library that reads environment variables
and `.env` files into a typed Python class, coercing strings to the declared
types and refusing to start when something is invalid. We use it because
`DATABASE_URL` being absent or `API_PORT` being `"eight thousand"` should crash
the process at import time with a clear message — not surface as a `None` or a
`TypeError` 200ms into the first user request.

**Purpose** — The single place environment variables are read and validated, so
config errors fail loudly at startup. Without it, `os.getenv` calls scatter
across the codebase and nobody can enumerate what config the app has.

**Analogy** — A building's electrical panel: every circuit terminates in one
labelled box, not in random junctions hidden behind the walls.

### Main pieces

| Piece | What it does / when it runs |
|---|---|
| `class Settings(BaseSettings)` | The schema. Every field is one env var. Fields with a default are optional; fields without one are **required** and their absence aborts startup. |
| `model_config = SettingsConfigDict(...)` | Configures the loader itself (pydantic v2 style). |
| `env_file=(".env", "../.env")` | Tries both paths, so the app works whether you run from the repo root or from `backend/`. Later entries do not override earlier ones. |
| `case_sensitive=False` | `DATABASE_URL` in `.env` maps to field `database_url`. Env vars are conventionally uppercase; Python attributes are lowercase. |
| `extra="ignore"` | Unknown keys in `.env` are tolerated rather than fatal — necessary because the frontend and worker share one `.env`. **This is the tradeoff that makes a typo silent**, so keeping `.env.example` in sync matters. |
| `Literal["development","production"]` | Constrains a value to an exact set. `APP_ENV=prod` (instead of `production`) is rejected at startup, rather than making `is_production` silently `False` in production. Used the same way for the three provider selectors. |
| `str \| None = None` | The "optional secret" pattern: a provider's key is only needed when that provider is chosen. |
| `@property is_production` | Derived config. One place decides what "production" means, so no module hand-compares strings. |
| `@lru_cache get_settings()` | Caches the instance so `.env` is parsed exactly once per process. |
| `settings = get_settings()` | The module-level singleton everything imports. |

### Flow

1. Something imports `app.config` (usually `main.py`, transitively).
2. Python executes the module top to bottom, reaching `settings = get_settings()`.
3. `get_settings()` constructs `Settings()`.
4. pydantic-settings resolves each field by precedence: **real environment
   variable → `.env` file → field default**. Real env vars win, which is what
   lets production and Docker override without touching files.
5. Each raw string is coerced to the declared type and validated
   (`Literal` membership, `int` parsing).
6. On any failure it raises `ValidationError` listing *every* bad field at once,
   and the process dies before serving traffic.
7. `lru_cache` stores the result; later `get_settings()` calls are free.

### Math
None.

### Gotchas

- **Precedence surprises.** A stale `DATABASE_URL` exported in your shell
  silently beats your `.env`. When config "isn't taking effect," check the real
  environment first (`echo $DATABASE_URL`).
- **`extra="ignore"` hides typos.** `DATABSE_URL=...` in `.env` is not an error;
  the field just falls back to its default and you connect to the wrong DB.
- **Import-time side effects.** Because `settings` is created at import,
  importing anything that transitively imports config requires valid config —
  including in tests. That's why `get_settings()` is exposed separately:
  tests can `get_settings.cache_clear()` and rebuild with patched env.
- A field with no default is required. Adding one later without adding it to
  `.env.example` breaks every teammate's startup.

### Interview questions

**Q: Why validate configuration at startup instead of when it's first used?**
This is fail-fast versus fail-late. Validating at boot converts a class of
production incidents — a deploy with a missing key that only breaks the payment
path at 2am — into a deploy that refuses to start, which your orchestrator
catches and rolls back automatically. The cost is that the app can't run in a
degraded partial mode; the benefit is that if it's running, its config is known
good.

**Q: What's the precedence order for configuration sources, and why does real
env var beat `.env`?**
Environment variable → `.env` file → default. Real env vars win because that's
how containers and orchestrators inject config: the image bakes in an `.env`
with sane development defaults, and Kubernetes/Docker/Vercel override specific
values per environment without rebuilding the image.

---

## 6. `backend/app/core/logging.py`

**New concept — contextvars.** A `ContextVar` holds a value scoped to the
current execution context, and asyncio gives each task its own copy that
survives across `await`. We use them because `threading.local` — the classic
answer — is *wrong* under asyncio: many requests interleave on a single thread,
so they'd overwrite each other's request IDs and you'd attribute one tenant's
logs to another.

**Purpose** — Emits logs as one JSON object per line, automatically stamped with
`request_id` / `org_id` / `bot_id`, so a single request is traceable through an
interleaved system. Without it, PLAN §7's "context on every line" would require
threading those values through every function signature in the codebase.

**Analogy** — A hospital wristband: attached once at admission, and every test
result afterwards carries the patient ID without anyone re-asking their name.

### Main pieces

| Piece | What it does / when it runs |
|---|---|
| `request_id_var`, `org_id_var`, `bot_id_var` | The three context slots. Set by middleware at request start; `None` outside a request (startup, Celery workers). |
| `_STANDARD_ATTRS` | The set of attributes every `LogRecord` natively carries. Computed once by constructing a throwaway record and reading its `__dict__`. Used to distinguish built-in fields from ones the caller supplied. |
| `class JsonFormatter(logging.Formatter)` | Converts a `LogRecord` into a JSON line. Runs once per emitted log record. |
| `JsonFormatter.format()` | Builds the dict: timestamp, level, logger name, message; then contextvars (omitted when `None`); then caller extras; then exception traceback if present. |
| `configure_logging()` | Installs the handler on the root logger. Called **once**, from `main.py`'s lifespan startup. |
| `root.handlers.clear()` | Removes uvicorn's default handler — otherwise every line prints twice, once plain and once JSON. |
| The uvicorn loop at the end | Clears uvicorn's own logger handlers and sets `propagate = True`, so its access/error logs flow up into our formatter instead of using its format. |

### Flow

1. `configure_logging(level, json_output=...)` runs at startup.
2. It builds a `StreamHandler` writing to stdout (never a file — in containers,
   stdout is the log transport; Docker/systemd captures it).
3. Attaches `JsonFormatter` in production, a plain readable format in dev.
4. Later, any module calls `logger.info("request", extra={...})`.
5. Python's logging machinery creates a `LogRecord`, and `extra` keys are set as
   attributes directly on that record.
6. `format()` runs: reads the contextvars *at emit time* — which is why the
   values must still be set when the log call happens.
7. Anything on the record not in `_STANDARD_ATTRS` is merged into the JSON, so
   `extra={"duration_ms": 12}` becomes a top-level queryable field.
8. `json.dumps(..., default=str)` serializes; `default=str` prevents a crash on
   non-serializable values like `UUID` or `datetime`.

### Math
None.

### Gotchas

- **Structured vs formatted logging.** `logger.info(f"user {uid} did {action}")`
  produces a string you must regex later. `logger.info("action", extra={...})`
  produces fields you can filter and aggregate. Prefer the latter.
- **`extra` key collisions crash.** Passing `extra={"message": ...}` or
  `{"name": ...}` collides with a real `LogRecord` attribute and raises
  `KeyError`. Avoid the reserved names.
- **Reading a contextvar too late returns `None`.** If the log call happens after
  the middleware resets the token, the ID silently vanishes — this exact bug is
  why `middleware.py` uses `try/except/else/finally` rather than logging after
  the `finally`.
- Logging to a file inside a container is an anti-pattern: the file dies with
  the container and nothing collects it.

### Interview questions

**Q: Why `contextvars` instead of `threading.local` for request context?**
Because asyncio multiplexes many concurrent requests onto one thread. With
`threading.local`, all of them share one storage slot, so request B overwrites
request A's ID and logs get cross-attributed — in a multi-tenant app that means
one tenant's `org_id` on another tenant's log line. `ContextVar` is copied per
asyncio task and propagates correctly across `await` boundaries.

**Q: Why JSON logs instead of human-readable text?**
Because logs are queried by machines at scale. JSON makes every field
independently filterable and aggregatable ("p95 `duration_ms` for `org_id=X`")
without brittle regex over prose. The tradeoff is human readability at the
terminal, which is why this module keeps a plain formatter for development and
only switches to JSON in production.

---

## 7. `backend/app/core/middleware.py`

**New concept — ASGI middleware.** A layer wrapping every request: it runs code
before the route handler, calls the handler, then runs code after. We use it
because cross-cutting concerns (request IDs, timing, later auth and rate
limiting) must apply to *every* endpoint — putting them in each route handler
guarantees someone eventually forgets one.

**Purpose** — Runs on every HTTP request: assigns a request ID, binds it to the
log context, times the request, and emits one access-log line. Without it,
nothing ever *sets* the contextvars from `logging.py`, so they'd always be empty.

**Analogy** — A receptionist who tags you on the way in, notes your arrival and
departure times, and removes the tag on the way out.

### Main pieces

| Piece | What it does / when it runs |
|---|---|
| `REQUEST_ID_HEADER = "X-Request-ID"` | Named constant so the header string isn't duplicated between read and write. |
| `class RequestContextMiddleware(BaseHTTPMiddleware)` | Starlette's convenience base for middleware written as a single async function. |
| `dispatch(request, call_next)` | **The whole contract.** Code before `call_next` runs on the way in; `call_next(request)` invokes the rest of the app; code after runs on the way out. |
| `request.headers.get(...) or uuid4().hex[:16]` | Honours an inbound ID (so a trace can span widget → API → worker later), generates one otherwise. |
| `request_id_var.set(request_id)` | Binds the ID; returns a **token** used to restore the previous value. |
| `time.perf_counter()` | Monotonic high-resolution clock. Correct for durations because, unlike `time.time()`, it can't jump backwards when the system clock syncs. |
| `_elapsed_ms(start)` | Small helper — the duration calculation appears on both the success and failure paths. |
| `finally: request_id_var.reset(token)` | Restores the prior context value on every exit path. |

### Flow

1. Request arrives; `dispatch` runs.
2. Resolve or generate `request_id`; bind it to the contextvar; record `start`.
3. `await call_next(request)` — routing, dependencies, and the route handler all
   run inside this call, so any log they emit is automatically stamped.
4. **Success path (`else`)**: set the response header, log one `request` line
   with method/path/status/duration.
5. **Failure path (`except`)**: `logger.exception` records the traceback *with
   context still bound*, then `raise` re-raises so FastAPI's exception handlers
   still produce a proper 500 response.
6. **Always (`finally`)**: reset the contextvar.

The `try/except/else/finally` shape is deliberate: `finally` runs *before* code
placed after the block, so logging after the block would read an already-reset
contextvar and lose the request ID. Putting the success log in `else` keeps it
inside the bound window.

### Math

Only unit conversion: `perf_counter()` returns fractional seconds, so
`(end - start) * 1000` yields milliseconds, rounded to 2dp.

### Gotchas

- **`BaseHTTPMiddleware` has real costs.** It wraps the request in an anyio task
  and can interfere with streaming responses and background tasks. Since Phase 3
  streams SSE, we may need to rewrite this as pure ASGI middleware — worth
  watching for once `/chat` streams.
- **Middleware order is reverse of registration.** `add_middleware` puts each new
  layer *outermost*, so the last registered runs first on the way in. When auth
  arrives in Phase 2, ordering will matter.
- **Forgetting `reset(token)` leaks context** into whatever task reuses the
  context next, causing wrong IDs on unrelated log lines — a genuinely hard bug
  to track down.
- Re-raising is not optional: swallowing the exception here would turn every
  crash into a hang or an empty response.

### Interview questions

**Q: Why accept an inbound `X-Request-ID` instead of always generating one?**
For distributed tracing. In a system where a request crosses several services,
each hop generating a fresh ID means you can't stitch the story together.
Honouring an inbound ID lets one identifier follow the request through widget →
API → Celery worker, so a single query returns every log for that user action.
The caveat is trust: an inbound header is user-controlled input, so it should be
length-limited and sanitized before being logged, and never used for anything
security-relevant.

**Q: Why `time.perf_counter()` rather than `time.time()` for measuring duration?**
`time.time()` is wall-clock and can jump — NTP corrections, DST, manual clock
changes — which can produce negative or wildly wrong durations. `perf_counter()`
is monotonic and higher resolution; it only ever moves forward, which is exactly
what an elapsed-time measurement needs. Its absolute value is meaningless, so
it's only valid for differences.

---

## 8. `backend/app/api/health.py`

**Purpose** — The `/health` liveness endpoint PLAN §6 requires, and the first
proof that routing plus response validation work end to end. Without it, neither
you nor Docker nor a load balancer can ask "is this process alive?"

**Analogy** — A pulse check, not a full physical: it answers "alive?" and
deliberately declines to answer "healthy in every respect?"

### Main pieces

| Piece | What it does / when it runs |
|---|---|
| `APIRouter(tags=["health"])` | A mountable group of routes. Lets each feature own a file and `main.py` just include them, instead of one giant `main.py`. `tags` groups them in the auto-generated `/docs`. |
| `class HealthResponse(BaseModel)` | The response *schema*. Not decoration: it's the contract. |
| `@router.get("/health", response_model=HealthResponse)` | Registers the route and declares the output type. FastAPI validates the returned object against it and generates OpenAPI docs from it. |
| `async def health()` | The handler. `async` because it does no blocking work — under FastAPI, a plain `def` handler is pushed to a threadpool instead. |

### Flow

1. At import, the decorator registers `GET /health` on the router.
2. `main.py` calls `include_router`, attaching it to the app.
3. A request arrives → middleware binds the request ID → FastAPI matches the path.
4. The handler returns a `HealthResponse`.
5. FastAPI serializes it to JSON against `response_model` and returns 200.
6. Middleware logs the line and resets context.

### Math
None.

### Gotchas

- **Liveness must not check dependencies.** If `/health` queried Postgres, a slow
  database would make Kubernetes conclude the app is dead and restart it —
  removing capacity at exactly the moment the system is already struggling, and
  potentially cascading into an outage. Readiness is the probe that checks
  dependencies; that's Phase 2.
- `response_model` also *filters* output: fields not in the model are stripped.
  That's a useful accidental safety net against leaking internal fields, and
  will matter once responses carry tenant data.
- The hardcoded `"0.1.0"` duplicates `pyproject.toml`'s version — a small,
  known drift risk worth wiring to package metadata later.

### Interview questions

**Q: What's the difference between a liveness and a readiness probe?**
Liveness answers "is this process broken beyond recovery?" — failing it triggers
a *restart*. Readiness answers "can this instance serve traffic right now?" —
failing it removes the instance from the load balancer but leaves it running.
The distinction matters because they have opposite correct responses to a
dependency outage: if the database is down, restarting the app doesn't help
(liveness should still pass) but sending it traffic is pointless (readiness
should fail).

**Q: Why declare a `response_model` instead of just returning a dict?**
Three benefits from one declaration: runtime validation that the endpoint
actually returns its documented shape, automatic OpenAPI schema generation, and
response filtering that strips undeclared fields. The third is a security
property — it's how you stop an internal field like a password hash from
escaping when someone returns an ORM object directly.

---

## 9. `backend/app/main.py`

**New concept — lifespan.** An async context manager where everything before
`yield` runs once at startup and everything after runs once at shutdown. We use
it because Phase 1 must load the bge-small embedding model into memory *once* at
boot — doing it per request would add seconds of latency to every question.

**Purpose** — The assembly point: creates the FastAPI app, configures logging at
startup, attaches middleware, mounts routers. Without it there's no `app` object
for uvicorn to serve — this is the name in `uvicorn app.main:app`.

**Analogy** — A power strip: the components exist independently, and this is
where they get plugged into each other and switched on.

### Main pieces

| Piece | What it does / when it runs |
|---|---|
| `@asynccontextmanager async def lifespan(app)` | Startup/shutdown hook. Everything before `yield` runs once as the server boots; everything after runs on graceful shutdown. |
| `configure_logging(...)` inside lifespan | Logging is configured **first**, before anything else can log, so no line escapes unformatted. `json_output=settings.is_production` gives readable dev logs and machine-parseable production logs. |
| `app = FastAPI(title=..., version=..., lifespan=lifespan)` | The ASGI application object. `title`/`version` populate `/docs`. |
| `app.add_middleware(RequestContextMiddleware)` | Wraps every request. Registered before routers for clarity, though FastAPI doesn't require that ordering. |
| `app.include_router(health_router)` | Mounts the routes. Each new feature area adds exactly one line here. |

### Flow

1. `uvicorn app.main:app` imports this module.
2. Import executes: `app.config` loads and validates settings (crashing here if
   `.env` is wrong), then the router module registers its routes.
3. The `FastAPI(...)` object is constructed; middleware and routers attach.
4. Uvicorn starts the server and enters the lifespan context: logging is
   configured, `"app starting"` is logged.
5. The server serves requests until interrupted.
6. On SIGINT/SIGTERM, execution resumes after `yield`, logs `"app stopped"`, and
   the process exits.

### Math
None.

### Gotchas

- **`lifespan` replaced `@app.on_event("startup")`**, which is deprecated. Most
  tutorials online still show the old decorators.
- **Import-time work is invisible until it fails.** Config validation happens on
  import, so a bad `.env` produces a traceback *before* any "starting" log line
  appears — confusing until you know the ordering.
- **The uvicorn `--reload` flag re-imports this module on every file save**,
  re-running lifespan. Anything expensive there (Phase 1's model load) makes
  reload painfully slow; that's the moment to consider lazy loading.
- Startup failures inside lifespan prevent the server from accepting connections
  at all — which is the desired fail-fast behaviour, matching `config.py`.

### Interview questions

**Q: Why load expensive resources in lifespan instead of at module import or per
request?**
Per request is obviously wrong — it repeats seconds of work on every call.
Module import is subtler: it runs during *collection* too, so tests, CLI tools,
and Alembic all pay the cost merely for importing the app, and there's no
corresponding cleanup hook. Lifespan is scoped exactly to "the server is
serving," gives a symmetric shutdown path for closing connection pools and
flushing buffers, and is skipped entirely when the module is imported for other
reasons.

**Q: What is ASGI and why does FastAPI use it instead of WSGI?**
WSGI is the older synchronous Python web interface: one request occupies one
worker thread from start to finish. ASGI is its async successor, letting a
single worker handle many concurrent requests by suspending them at `await`
points. That's decisive for this project — a RAG chat request spends most of its
life *waiting* on network I/O (embedding, Qdrant, the LLM). Under WSGI each of
those waits would pin a whole thread; under ASGI the worker serves other
requests while waiting. ASGI is also what makes SSE streaming in Phase 3
practical.

---

## 10. `backend/app/db/base.py`

**New concept — SQLAlchemy ORM.** An Object-Relational Mapper maps Python
classes to SQL tables, so you write `session.get(Chatbot, id)` instead of raw
SQL strings. We use it here for two reasons: it prevents SQL injection by
parameterizing everything, and Alembic reads its metadata to *generate*
migrations — which is what makes Hard Rule #5 practical rather than tedious.

**Purpose** — Defines the `Base` every model inherits from, plus the engine and
session factory. Without it there's no metadata for Alembic to diff, so
migrations can't be autogenerated.

**Analogy** — The blank family tree every table hangs off, plus the phone line
to the database.

### Main pieces

| Piece | What it does / when it runs |
|---|---|
| `class Base(DeclarativeBase)` | Parent of every model. Collects each subclass's table definition into `Base.metadata` at class-definition time. |
| `engine = create_engine(...)` | The connection pool + dialect. Created once at import. Note it does **not** connect yet — SQLAlchemy connects lazily on first use, which is why importing this module works with the database down. |
| `pool_pre_ping=True` | Issues a cheap `SELECT 1` before lending out a pooled connection. Essential for Neon, which drops idle connections; without it the first query after idle fails. |
| `echo=False` | Flip to `True` to print every SQL statement — genuinely useful while learning the ORM. |
| `SessionLocal = sessionmaker(...)` | A factory producing `Session` objects. The session is the ORM's unit of work: it tracks changed objects and flushes them as one transaction. |
| `autoflush=False` | Stops SQLAlchemy from silently issuing SQL mid-query-construction. Makes behaviour predictable at the cost of requiring explicit `flush()`. |
| `expire_on_commit=False` | Keeps ORM objects readable after `commit()`. Default `True` marks attributes stale, so touching one post-commit triggers a fresh SELECT — or a `DetachedInstanceError` once the session closes. |
| `get_db()` | FastAPI dependency yielding one session per request, closed in `finally` no matter what. |

### Flow

1. Import creates `Base`, `engine`, `SessionLocal`. No network I/O yet.
2. A request arrives; FastAPI calls `get_db()`.
3. `SessionLocal()` creates a session; the engine leases a pooled connection on
   first actual query.
4. The handler uses the session; `yield` hands it over.
5. When the response is done, `finally: db.close()` returns the connection to
   the pool — it is not destroyed, just released.

### Math
None. (Pool sizing is the closest thing: max concurrent DB work equals
`pool_size + max_overflow`, both left at SQLAlchemy defaults for now.)

### Gotchas

- **Engine vs Session confusion.** One engine per application (it owns the
  pool); one session per request (it owns a transaction). Creating an engine per
  request destroys pooling and exhausts connections.
- **`expire_on_commit=True` bites beginners hard**: you commit, return the
  object, and FastAPI's serializer raises `DetachedInstanceError` because the
  session already closed. Turned off deliberately here.
- Sessions are **not thread-safe or task-safe**. Never share one across
  requests — which is exactly what the per-request dependency prevents.

### Interview questions

**Q: What problem does connection pooling solve?**
Opening a Postgres connection is expensive — a TCP handshake, TLS negotiation,
authentication, and a new backend process forked server-side, typically tens of
milliseconds. On a per-request basis that dwarfs the query itself. A pool opens
a set of connections once and lends them out, so a request pays microseconds
instead. It also caps total connections, protecting the database from being
overwhelmed — important on Neon's free tier, where the connection limit is low.

**Q: What is the "unit of work" pattern that a Session implements?**
The session tracks every object you've loaded or modified and defers the writes,
then flushes them all in a single transaction at commit. The benefits are fewer
round trips, correct ordering of inserts respecting foreign keys, and
all-or-nothing atomicity. The cost is a mental model where an in-memory object
and its database row can diverge until you flush — the source of most confusing
ORM behaviour.

---

## 11. `backend/app/db/models.py`

**Purpose** — One module importing every ORM model, so Alembic sees them all
through a single import. Without it you must remember to import each new model
into Alembic's config, and a forgotten import means Alembic generates a
`DROP TABLE` for it.

**Analogy** — The class register the teacher reads from: a student not on the
list is marked absent, even if they're sitting right there.

### Main pieces

Currently just re-exports `Base`. As Phase 2 lands, each model gets one import
line with `# noqa: F401` — the linter thinks the import is unused, but the
import *is* the point: it executes the class definition, which registers the
table on `Base.metadata`.

### Flow

`migrations/env.py` imports this module → each import statement executes a model
module → each `class X(Base)` registers its table → `Base.metadata` is complete
→ `target_metadata = Base.metadata` gives autogenerate the full picture.

### Math
None.

### Gotchas

- **The failure mode here is genuinely dangerous.** Alembic autogenerate works by
  diffing metadata against the live schema. A model that isn't imported is
  absent from the metadata, so Alembic concludes the table "shouldn't exist" and
  writes `op.drop_table(...)`. Running that migration deletes real data. Always
  read a generated migration before applying it.
- `__all__` doesn't affect Alembic; only the import side effect matters.

### Interview question

**Q: Why does Alembic need models imported, rather than scanning the filesystem?**
Because Python has no reliable way to know a class exists until its module is
executed — a class definition is a runtime statement, not a static declaration.
Alembic inspects `Base.metadata`, an object populated as a side effect of class
creation. Scanning files would mean guessing at import paths and executing
arbitrary modules, so Alembic requires the explicit import instead. The
practical consequence is the single-registry-module convention used here.

---

## 12. Alembic wiring — `alembic.ini` + `migrations/env.py`

**New concept — Alembic.** A migration tool that compares your ORM models
against the live database and generates the SQL to close the gap, storing each
change as a numbered, reversible script. We use it because Hard Rule #5 forbids
hand-editing the database: without migrations, your laptop's schema and
production's schema drift apart with no record of why.

**Purpose** — Version control for database schema; each change is a reviewable,
replayable script. Without it, schema changes are hand-typed SQL that works
locally and silently differs in production.

**Analogy** — Git for table structure: every schema change is a commit you can
replay forward or roll back.

### Main pieces

| Piece | What it does |
|---|---|
| `alembic.ini` | Static config: `script_location` (where migrations live), `prepend_sys_path = .` (so `import app` resolves when Alembic runs). |
| **`sqlalchemy.url` removed from the ini** | The generated default is `driver://user:pass@localhost/dbname`. This file is committed, so a real URL here means a password in git. We deleted it and inject at runtime instead. |
| `migrations/versions/` | The generated migration scripts, each with a revision ID and a `down_revision` pointer forming a linked list. |
| `migrations/env.py` | The script Alembic executes for every command. |
| `config.set_main_option("sqlalchemy.url", settings.database_url)` | Injects the real URL from `.env` at runtime. Uses `set_main_option` rather than direct assignment because it escapes `%` — and URL-encoded passwords contain `%`, which ConfigParser would otherwise read as interpolation syntax and crash on. |
| `target_metadata = Base.metadata` | What autogenerate diffs against. `None` (the default) means autogenerate produces empty migrations. |
| `compare_type=True` | Without it, changing `VARCHAR(50)` → `VARCHAR(200)` is silently ignored. Off by default because type comparison across dialects is imperfect. |
| `compare_server_default=True` | Same, for server-side column defaults. |
| `run_migrations_offline()` | Emits SQL to stdout instead of executing (`alembic upgrade head --sql`), for environments where a DBA applies changes by hand. |
| `run_migrations_online()` | The normal path: connect and execute. |
| `poolclass=pool.NullPool` | A migration is a short-lived process; pooling would hold connections open for a script about to exit. |

### Flow

1. `alembic revision --autogenerate -m "add orgs"` runs `env.py`.
2. `env.py` imports `app.config` (validating settings) and `app.db.models`
   (populating `Base.metadata`).
3. Alembic connects, reflects the live schema, and diffs it against the metadata.
4. It writes a new script in `versions/` with `upgrade()` and `downgrade()`.
5. **You read it** — autogenerate is a draft, not an authority.
6. `alembic upgrade head` executes pending scripts in order, recording each in
   the `alembic_version` table so they never run twice.

### Math

Not mathematics, but the data structure is worth naming: migrations form a
**singly linked list** (each script's `down_revision` points at its parent),
which becomes a **DAG** when branches merge. `head` is the tail of that list.

### Gotchas

- **Autogenerate is a draft.** It reliably detects added/dropped tables and
  columns. It does *not* detect renames — it sees a drop plus an add, which
  destroys the data in that column. Renames must be hand-edited into
  `op.alter_column(..., new_column_name=...)`.
- **It also misses** server default changes and constraint tweaks unless the
  `compare_*` flags above are on.
- Never edit an already-applied migration. Write a new one — teammates and
  production have already run the old one.
- `prepend_sys_path = .` means Alembic must be run **from `backend/`**, or
  `import app` fails.

### Interview questions

**Q: Why not just let the ORM create tables with `Base.metadata.create_all()`?**
`create_all` only creates missing tables — it never alters existing ones. The
moment you add a column to a model, it does nothing, and your code expects a
column the database lacks. It also gives no history, no rollback, and no review
point. It's acceptable for throwaway test databases (and is genuinely used
there) but not for anything holding real data.

**Q: How do you safely rename a column in a zero-downtime deployment?**
Not in one migration — during a rolling deploy, old and new code run
simultaneously, so any single-step rename breaks one of them. The standard
expand/contract pattern: (1) add the new column, (2) deploy code that writes
both and reads the new one with a fallback, (3) backfill existing rows, (4) once
no code reads the old column, drop it in a later migration. Four steps, each
independently safe to roll back.

---

## 13. `.pre-commit-config.yaml`

**New concept — git hooks + pre-commit.** Git runs scripts at defined moments in
its lifecycle; a `pre-commit` hook runs before a commit is created and can abort
it. The `pre-commit` *tool* manages these hooks in isolated environments so you
don't hand-write shell scripts into `.git/hooks/`. We use it so lint and format
rules are enforced automatically rather than depending on memory.

**Purpose** — Runs ruff and black on every `git commit`, blocking the commit if
they fail. Without it, code style depends on you remembering, and CI catches
problems minutes later instead of instantly.

**Analogy** — A spellchecker that won't let you hit send on the email.

### Main pieces

| Hook | What it catches |
|---|---|
| `trailing-whitespace`, `end-of-file-fixer` | Invisible diff noise. |
| `check-yaml`, `check-toml` | Syntax errors in `docker-compose.yml` and `pyproject.toml` *before* they break a run. |
| `check-merge-conflict` | Committed `<<<<<<<` markers. |
| `check-added-large-files --maxkb=1000` | Stops bge-small's weights ever entering git history. |
| `detect-private-key` | A backstop for Hard Rule #1. |
| `ruff --fix` | Lints and auto-fixes (imports, unused variables). |
| `black` | Formats. |
| `files: ^backend/` | Scopes Python hooks to the backend so Phase 4's frontend isn't affected. |

### Flow

1. `pre-commit install` writes a script into `.git/hooks/pre-commit`. **One-time
   setup — the config file alone does nothing.**
2. On `git commit`, the hook runs each repo's tools against **staged files only**.
3. A hook that rewrites a file (ruff --fix, black) causes the commit to **fail**
   with the file modified. This is intended: you review the change, `git add`
   it, and commit again.
4. All hooks pass → the commit is created.

### Math
None.

### Gotchas

- **Order matters.** `ruff --fix` runs before `black`, so formatting always has
  the final say. Reversed, ruff could rewrite an import and leave it unformatted.
- **Two formatters must never both run.** The first draft of this file enabled
  `ruff-format` *and* `black`; they'd fight over the same files. `ruff-format`
  was removed — PLAN §7 specifies black.
- **`rev:` pins are not automatic.** They're pinned so every machine runs
  identical tool versions; `pre-commit autoupdate` bumps them deliberately.
- Hooks only see *staged* files, so `git commit -a` and partial staging can
  behave surprisingly.

### Interview question

**Q: If CI already runs your linters, why run them in a pre-commit hook too?**
Feedback latency and cost. A hook fails in ~2 seconds on your machine; CI fails
minutes later, after a push, and leaves a red build in shared history that
others may branch from. But hooks are local and skippable (`--no-verify`), so
they're a convenience, not a guarantee — CI remains the enforcement point.
The correct arrangement is both: hooks for speed, CI for authority.

---

## 14. `backend/tests/conftest.py`

**New concept — pytest fixtures.** Reusable setup functions that pytest injects
into tests by matching the test's *parameter name* to a fixture name. We use
them because Phase 2's isolation tests need per-test orgs and databases
constructed identically every time, and duplicating that setup is how tests
drift apart and stop testing what you think.

**Purpose** — Shared fixtures, chiefly an in-process test client that calls the
API without a running server. Without it, every test file rebuilds identical
setup.

**Analogy** — The stage crew who set up the set before each scene, so the actors
just perform.

### Main pieces

| Piece | What it does |
|---|---|
| The filename `conftest.py` | Magic name. pytest auto-discovers it; tests never import it. Fixtures apply to every test in this directory and below. |
| `@pytest.fixture(scope="session")` | Declares the fixture and how long it lives. `session` = built once per test run. Alternatives: `function` (default, fresh per test), `module`, `class`. |
| `TestClient(app)` | Starlette's client; sends requests through the ASGI app **in-process** — no socket, no port, no server. Fast and free of port conflicts. |
| `with TestClient(app) as c` | The context manager is what triggers **lifespan**. Constructing without `with` skips startup/shutdown entirely. |
| `yield test_client` | Everything before `yield` is setup, after is teardown. |

### Flow

1. pytest collects tests and sees `test_health_returns_ok(client)`.
2. It matches the parameter name `client` to this fixture.
3. First use builds it: enters the context manager, which runs FastAPI's
   lifespan (configuring logging, logging "app starting").
4. The client is injected; the test runs.
5. Because scope is `session`, later tests reuse the same instance.
6. At end of run, teardown resumes after `yield`, exiting the context and
   running lifespan shutdown.

### Math
None.

### Gotchas

- **Forgetting `with` is the classic bug.** Without it lifespan never runs, so
  anything initialized at startup is missing — tests fail mysteriously, or worse,
  pass locally and fail in production where startup does happen.
- **`scope="session"` is only safe while tests don't mutate shared state.** Once
  Phase 2 adds a database, most fixtures need `function` scope so one test's data
  can't leak into another's — that's precisely the leakage `test_isolation.py`
  exists to catch.
- Fixture injection is by name, so renaming a fixture silently breaks every test
  using it with an unhelpful "fixture not found" error.

### Interview question

**Q: What's the difference between `TestClient` and hitting a real running server
in tests?**
`TestClient` runs the ASGI app in-process: no network, no port binding,
millisecond round trips, and tracebacks surface directly in the test output.
That makes it right for the vast majority of tests. What it *doesn't* exercise
is the real deployment path — the actual server, its worker configuration,
timeouts, proxies, and TLS. So unit and integration tests use `TestClient`,
while a small number of smoke tests should hit a genuinely deployed instance.

---

## 15. `backend/tests/test_health.py`

**Purpose** — Proves the app boots, routes resolve, and `/health` returns the
documented shape — satisfying PLAN §6's "pytest runs (even 1 test)". Without it
nothing verifies the wiring; you'd merely be trusting it works.

**Analogy** — The smoke test on new wiring: flip the switch and confirm the bulb
lights before finishing the rest of the house.

### Main pieces

| Test | What it actually proves |
|---|---|
| `test_health_returns_ok` | The whole stack: import, config validation, lifespan, middleware, routing, serialization. |
| `test_health_reports_env_and_version` | The `response_model` contract. `set(body) == {...}` asserts **exactly** these keys — so an accidentally leaked field fails the test. |
| `test_response_carries_request_id` | Middleware stamps every response even with no inbound header. |
| `test_inbound_request_id_is_preserved` | The distributed-tracing behaviour: a caller-supplied ID survives. |
| `test_unknown_route_returns_404` | A negative test — proves the router isn't matching everything. |

### Flow

pytest reads `[tool.pytest.ini_options]` from `pyproject.toml` → collects
`tests/` → builds the `client` fixture (running lifespan) → executes each test →
reports.

### Math
None.

### Gotchas

- **Asserting `set(body) == {...}` rather than `>=`** is deliberate. Subset
  assertions pass when the endpoint leaks extra fields; exact assertions catch it.
  For an endpoint returning tenant data, that difference is a security control.
- Testing the *implementation* (mocking the handler) instead of the *behaviour*
  (calling the endpoint) produces tests that pass while the app is broken.
- The `StarletteDeprecationWarning` about httpx currently appears; our
  `filterwarnings = ["error::DeprecationWarning"]` doesn't fail on it because it
  is a `StarletteDeprecationWarning` subclass raised at import, not collection.
  Worth revisiting if it becomes noise.

### Interview question

**Q: What makes a test valuable versus merely present?**
A valuable test fails when the behaviour users depend on breaks, and doesn't
fail when only implementation details change. Tests coupled to internals — heavy
mocking, asserting on private methods — invert this: they break during safe
refactors and stay green during real regressions, so they cost maintenance while
providing false confidence. The tests here call the endpoint the way a client
does, so they survive any refactor that preserves the contract.

---

## Phase 0 status — COMPLETE

Verified against PLAN §6 "done when":

| Criterion | Result |
|---|---|
| `uvicorn` runs | ✅ starts, logs "app starting" |
| `/health` returns ok | ✅ `200 {"status":"ok","env":"development","version":"0.1.0"}` + `x-request-id` |
| `pytest` runs | ✅ 5 passed |
| `.env.example` exists, no secret committed | ✅ `.env` is gitignored |
| Alembic wired | ✅ connects to Postgres, `Context impl PostgresqlImpl` |
| pre-commit configured | ✅ config written (needs `pre-commit install` after `git init`) |
| docker compose gives Qdrant + Redis + Postgres | ✅ all three `healthy`, all three reachable |
| ruff + black | ✅ clean across 19 files |

Connectivity proven: Postgres `select version()` via our own engine, Qdrant
`/healthz` + `/collections`, Redis `PING` → `PONG`.

### Bugs

Full post-mortems for all six Phase 0 bugs — symptom, wrong hypotheses, root
cause, fix, and interview angles — live in
**[`docs/bugs_explanations/phase0_bugs.md`](../bugs_explanations/phase0_bugs.md)**.

Summary: global `pip install` broke dbt · two formatters configured at once ·
dead import + unformatted code · access log would have lost the request ID ·
**Postgres port hijack** (the big one) · nine files written with no explanation.

<details>
<summary>Kept inline: the port-hijack story, since it shaped the compose file</summary>

#### Silent port hijack

**Symptom.** `alembic upgrade head` failed with
`FATAL: password authentication failed for user "ragsaas"` — while
`docker compose ps` showed Postgres `healthy` and `psql` inside the container
worked fine.

**Four hypotheses, three wrong:**

1. *Stale volume with old credentials* — disproved by container logs showing a
   fresh `running bootstrap script ... ok` / `Success.`, not the
   "Skipping initialization" line that a reused data directory produces.
2. *Hidden characters in `.env`* (a Windows `\r` in the password is a classic) —
   disproved with `grep DATABASE_URL .env | cat -A`, which shows line endings
   and trailing whitespace explicitly. Clean.
3. *Credential mismatch* — disproved by `docker exec ragsaas-postgres printenv
   POSTGRES_USER POSTGRES_PASSWORD`. Identical to `.env`.
4. *We aren't talking to that container at all* — **correct.**

**A false positive that cost time.** An early `psql` check inside the container
succeeded, and I treated that as proof the password worked. It wasn't: local
connections in that image use `trust` auth over a Unix socket, so **no password
is ever checked**. The container's own log even says so:
`initdb: warning: enabling "trust" authentication for local connections`.
Re-running it over TCP with `-h 127.0.0.1 -e PGPASSWORD=...` was the test that
actually proved something.

**The decisive command:**
```
netstat -ano | findstr :5432
```
It returned **two different PIDs** listening on `0.0.0.0:5432`. `tasklist /FI
"PID eq <pid>"` identified them as `postgres.exe` (a native Windows PostgreSQL
service) and `com.docker.backend.exe` (our container's proxy).

**Root cause.** A natively-installed PostgreSQL service already owned 5432. On
Linux, a second `bind()` to a bound port is refused with `EADDRINUSE`. On
**Windows**, a second bind is permitted unless the first socket set
`SO_EXCLUSIVEADDRUSE` — which the PostgreSQL installer does not. So Docker bound
the same port, reported success, and health checks passed, while the app's
connections were served by the *native* Postgres, which has no `ragsaas` role.

**Why the error message misleads.** Postgres reports a *nonexistent role* as
`password authentication failed`, not "role does not exist" — deliberately, so
attackers cannot enumerate valid usernames. So an auth error means "user or
password is wrong," and you cannot tell which. That ambiguity is what sent the
investigation toward credentials for three hypotheses.

**Fix.** Map the host side only: `"5433:5432"` in `docker-compose.yml`, and
`localhost:5433` in `.env` / `.env.example`. Inside the container and inside the
compose network, Postgres is still on 5432 — so nothing about the eventual
deployment changes.

**Transferable lessons.**
- A healthcheck proves the *container* is fine, never that *you are reaching it*.
- When two components each look correct, question the assumption that they're
  connected to each other.
- `select version()` identifies which server you actually reached: the container
  reports `linux-musl`, a native Windows install reports `windows`.
- Auth errors from Postgres are deliberately ambiguous — don't over-read them.

</details>

### Bug log

Moved to **[`docs/bugs_explanations/phase0_bugs.md`](../bugs_explanations/phase0_bugs.md)**
so bugs live in one place across phases.

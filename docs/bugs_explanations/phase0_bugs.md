# Phase 0 — Bug post-mortems

Every bug hit while building Phase 0, with the reasoning, not just the fix.
Each entry follows the same shape:

**Symptom → Hypotheses (including wrong ones) → Root cause → Fix → Lesson →
Interview angle**

The wrong hypotheses are kept deliberately. Knowing *why* a plausible theory was
wrong is what makes the next bug faster to solve.

Severity key: 🔴 broke the build · 🟠 broke something else · 🟡 caught before it bit

---

## Bug 1 🟠 — `pip install` into the global Python broke dbt

**Symptom**
```
ERROR: pip's dependency resolver does not currently take into account all the
packages that are installed...
dbt-common 1.38.0 requires pathspec<1.1,>=0.9, but you have pathspec 1.1.1
dbt-core 1.11.12 requires pathspec<0.13,>=0.9, but you have pathspec 1.1.1
```
The install of *this* project's dependencies reported success, but broke two
unrelated packages already on the machine.

**Root cause**
`pip install -e ".[dev]"` was run before creating a virtual environment, so it
installed into the system Python. Our `black` dependency required
`pathspec>=1.0.0`; pip upgraded `pathspec` to satisfy it, and that upgrade
violated the constraint `dbt-core` had already been installed under.

This is **dependency hell**: two packages in one environment requiring mutually
exclusive versions of a shared transitive dependency. Only one version of a
package can exist per environment, so one of them must lose.

**Fix**
1. `python -m venv .venv` inside `backend/`
2. Reinstalled the project there — an isolated `site-packages` where our
   `pathspec` version affects nothing else
3. Restored the global one: `pip install "pathspec<0.13,>=0.9"`
4. Uninstalled the packages that had leaked in: `pip uninstall black ragsaas-backend`
5. Verified with `pip check` → `No broken requirements found`, and
   `python -c "import dbt.version"` → ok

**Lesson**
**Create the virtualenv before the first `pip install`, every single time.** The
mistake is invisible at the moment you make it — the install succeeds, and the
damage lands on software you weren't even working on. A global Python should
hold almost nothing.

`pip check` is the tool that tells you whether an environment is internally
consistent. Run it after any messy install.

**Debug next time**
- `python -c "import sys; print(sys.prefix)"` — is a venv active? (Inside one,
  this points at the venv, not `C:\Python311`.)
- `pip -V` prints the path of the pip you're actually running.
- Look at your shell prompt: `(.venv)` prefix means active.

**Interview angle**
*What problem do virtual environments solve?*
Python installs one version of a package per environment, so two projects
needing different versions of the same library cannot share one. A venv gives
each project its own `site-packages`, making dependency sets independent. It's
the same isolation principle as containers, one layer up: containers isolate the
OS, venvs isolate the interpreter's packages.

---

## Bug 2 🟡 — Two formatters configured at once

**Symptom**
Caught by review before running. The first `.pre-commit-config.yaml` enabled
both `ruff-format` and `black`.

**Root cause**
Both are *formatters* — they rewrite files into their own canonical style. Ruff
aims for black compatibility but is not byte-identical. With both hooks enabled,
each commit could see one tool reformat what the other just wrote, producing
endless churn and diffs full of changes nobody made.

**Fix**
Removed `ruff-format`. Ruff now lints only (`--fix` for safe auto-fixes); black
formats. PLAN §7 specifies ruff + black, so black wins.

**Lesson**
**One formatter, always.** Linters and formatters have different jobs and can
coexist happily — but two tools with authority over the same bytes will fight.
The same reasoning explains `ignore = ["E501"]` in `pyproject.toml`: black owns
line length, so ruff must not also enforce it.

Hook *order* follows from this: `ruff --fix` runs first (it may rewrite imports),
black runs last so formatting has the final say.

**Interview angle**
*Difference between a linter, a formatter, and a type checker?*
A formatter changes only appearance and never semantics (black). A linter
analyses code without running it and flags likely bugs and style violations,
some auto-fixable (ruff). A type checker verifies type consistency across the
program (mypy). They overlap slightly, so you configure them not to contradict
each other.

---

## Bug 3 🟡 — Dead import, then lint/format failures on first run

**Symptom**
```
ruff:  I001 Import block is un-sorted or un-formatted (app/main.py)
black: would reformat app/core/logging.py, app/db/base.py
```
Also, an early draft of `config.py` imported `Field` from pydantic without using
it, then "justified" it with a fake `__all__` entry.

**Root cause**
Code written by hand without the tools run against it. The `Field` case is worse
than untidiness: it was a rationalization — writing a bogus `__all__` to silence
a linter rather than deleting the unused import.

**Fix**
Deleted the unused import. Ran `ruff check --fix .` and `black .`, then re-ran
the tests to confirm formatting changed nothing behavioural. Clean: 19 files.

**Lesson**
**Never work around a linter — either fix the code or the rule is wrong.** A
suppression you can't justify in one sentence is a code smell. And run the tools
*before* claiming work is done: "it looks right" is not a verification.

**Interview angle**
*When is it legitimate to suppress a lint rule?*
When the rule is genuinely wrong for that line and you can say why in the
`noqa` comment — e.g. `from app.db.tables import Organization  # noqa: F401`,
where the import looks unused but its side effect (registering the model on
`Base.metadata`) is the entire point. Blanket suppressions, or ones added to
make a message disappear without understanding it, are how a codebase's checks
gradually stop meaning anything.

---

## Bug 4 🟡 — Access log would have lost the request ID

**Symptom**
Caught while re-reading the first draft of `middleware.py`; never shipped.

The draft used `try/except/finally` and logged the completed request *after* the
block:

```python
try:
    response = await call_next(request)
except Exception:
    ...
    raise
finally:
    request_id_var.reset(token)   # runs BEFORE the code below

logger.info("request", ...)       # contextvar already reset here
```

**Root cause**
A control-flow misunderstanding. `finally` executes when the `try` block is
exited — *before* any statement following the whole construct. So by the time
the access log ran, the contextvar was already reset and every access-log line
would have carried no `request_id`. The irony: the one line whose entire purpose
is correlation would be the line missing the correlation ID.

**Fix**
Moved the success log into an `else:` block, which runs after the `try` body
succeeds but still *inside* the construct, before `finally`:

```python
try:
    response = await call_next(request)
except Exception:
    logger.exception(...)   # context still bound
    raise
else:
    logger.info("request", ...)   # context still bound
finally:
    request_id_var.reset(token)   # always last
```

**Lesson**
`try/except/else/finally` execution order: `try` → (`except` on error | `else`
on success) → `finally` → code after the block. The `else` clause is
underused; it means "the try body succeeded" and is exactly right when you need
work done inside the guarded region but only on the happy path.

Also: **resetting the token is not optional.** Skipping it leaks the ID into
whatever task reuses that context, silently mislabelling unrelated log lines.

**Interview angle**
*What's the difference between putting code in `else` versus after the whole
`try` block?*
`else` runs before `finally` and only when no exception occurred; code after the
block runs after `finally` and also runs when an exception was caught and
handled. When `finally` performs cleanup that later code depends on — releasing
a lock, resetting a context, closing a resource — the distinction decides
whether your code sees the pre- or post-cleanup state.

---

## Bug 5 🔴 — Postgres "password authentication failed" — the silent port hijack

**The big one.** Longest to diagnose, most transferable lesson.

**Symptom**
```
sqlalchemy.exc.OperationalError: (psycopg.OperationalError) connection failed:
FATAL: password authentication failed for user "ragsaas"
```
Yet `docker compose ps` reported Postgres `healthy`, and `psql` inside the
container connected fine.

**The contradiction:** the container is provably working; the app provably can't
authenticate to it; and the credentials in `.env` and `docker-compose.yml` are
visibly identical.

### Hypotheses

**H1 — Stale volume with old credentials.** ❌
Postgres reads `POSTGRES_USER`/`POSTGRES_PASSWORD` **only on first boot of an
empty data directory**. A reused volume keeps the *old* credentials while the
YAML advertises new ones — a classic and very plausible cause.

*Disproved by:* `docker logs ragsaas-postgres` showed
`running bootstrap script ... ok` and `Success.` — a **fresh** initdb. A reused
directory instead prints *"Database directory appears to contain a database;
Skipping initialization."*

**H2 — Hidden characters in `.env`.** ❌
On Windows, a CRLF line ending can leave a trailing `\r` on the parsed value, so
the password becomes `ragsaas\r` and fails — invisible in any normal editor.

*Disproved by:* `grep DATABASE_URL .env | cat -A`, which renders line endings as
`$` and would show `^M$` for CRLF. Clean.

**H3 — Credential mismatch between file and container.** ❌
*Disproved by:* `docker exec ragsaas-postgres printenv POSTGRES_USER
POSTGRES_PASSWORD | cat -A` → `ragsaas$` / `ragsaas$`. Identical.

**H4 — We are not talking to that container at all.** ✅

### The false positive that cost time

An early check — `docker exec ragsaas-postgres psql -U ragsaas -d ragsaas` —
succeeded, and that was read as "the password works."

**It proved nothing.** That connection goes over a **Unix socket** inside the
container, where the image configures **`trust` authentication**: no password is
checked at all. The container's own startup log says so:

```
initdb: warning: enabling "trust" authentication for local connections
```

The test that actually proved something forced TCP *and* a password:
```bash
docker exec -e PGPASSWORD=ragsaas ragsaas-postgres \
  psql -h 127.0.0.1 -U ragsaas -d ragsaas -c "select 1"
```
→ succeeded. So the container accepts these credentials over TCP. Therefore the
failing connection was never reaching this container.

### The decisive command

```bash
netstat -ano | findstr :5432
```
```
TCP 0.0.0.0:5432 ... LISTENING 7464
TCP 0.0.0.0:5432 ... LISTENING 14760      ← TWO different PIDs
```
```bash
tasklist /FI "PID eq 7464"    →  postgres.exe            (native Windows service)
tasklist /FI "PID eq 14760"   →  com.docker.backend.exe  (our container)
```

### Root cause

A **natively installed PostgreSQL Windows service** already owned port 5432.

On **Linux**, a second `bind()` to a bound port fails with `EADDRINUSE`. On
**Windows**, it is *allowed* unless the first socket set `SO_EXCLUSIVEADDRUSE` —
which the PostgreSQL installer does not. So Docker bound the same port,
reported success, and every healthcheck passed, while the app's connections were
served by the **native** Postgres — which has no `ragsaas` role.

```
  app ──connect localhost:5432──┐
                                ▼
        ┌────────────────────────────────────────┐
        │ TWO listeners on 0.0.0.0:5432          │
        ├────────────────────────────────────────┤
 7464   │ postgres.exe (native service)          │ ← served the connection
 14760  │ com.docker.backend.exe (our container) │ ← never reached
        └────────────────────────────────────────┘
```

### Why the error message misled

Postgres reports a **nonexistent role** as `password authentication failed`, not
"role does not exist" — deliberately, so an attacker cannot enumerate valid
usernames by comparing error messages. So the error means "user *or* password is
wrong, and I won't tell you which." That ambiguity is precisely what pushed three
consecutive hypotheses toward credentials.

A security feature made debugging harder. That tradeoff is real and intentional.

### Fix

Move the **host side only**:

```yaml
ports:
  - "5433:5432"     # host 5433 → container 5432
```
```bash
DATABASE_URL=postgresql+psycopg://ragsaas:ragsaas@localhost:5433/ragsaas
```

Inside the container and inside the compose network, Postgres is still on 5432 —
so nothing about the eventual deployment changes. The native Postgres was left
running and untouched.

**Verified:**
```
connected as : ragsaas
server port  : 5432
server       : PostgreSQL 16.14 on x86_64-pc-linux-musl
```
`linux-musl` = the Alpine container. A native Windows install reports `windows`.

### Lessons

1. **A healthcheck proves the container is fine. It never proves you're reaching
   it.** The check runs *inside* the container and knows nothing about host
   networking. Green health plus a failing app is a routing problem.
2. **When two components each look correct, question the assumption that they're
   connected to each other.** Three hypotheses assumed the endpoint was right and
   the credentials wrong. The real fault was in the arrow between them.
3. **Verify identity, not just reachability.** `select version()` says *which*
   server answered. Fingerprint the thing you connected to.
4. **Distinguish "it connected" from "it authenticated."** Unix socket + `trust`
   auth answers a different question than TCP + password.
5. **Error messages can be deliberately vague for security reasons.** Don't
   over-read them.

### Debug next time

| Command | Answers |
|---|---|
| `netstat -ano \| findstr :PORT` | Who is listening? **More than one PID is the tell.** |
| `tasklist /FI "PID eq <pid>"` | What is that process? |
| `docker logs <container>` | Did it initialize fresh, or skip initialization? |
| `docker exec <c> printenv VAR` | What does the container actually think its config is? |
| `grep VAR .env \| cat -A` | Hidden `\r` / trailing whitespace? |
| `select version()` | Which server actually answered? |
| `docker exec -e PGPASSWORD=x <c> psql -h 127.0.0.1 ...` | Does auth work over **TCP**, not just the socket? |

### Interview angles

*A service reports healthy but clients can't connect. How do you investigate?*
Establish whether clients are reaching the service at all before touching its
configuration. Healthchecks usually run inside the container, so they validate
the process, not the network path. Check what's listening on the port from the
client's perspective, confirm the port mapping, then fingerprint the responder
(a version query, a banner) to confirm it's the process you think. Only then
look at credentials and permissions.

*Why does Postgres return the same error for a wrong password and a nonexistent
user?*
To prevent username enumeration. Distinguishable errors let an attacker discover
valid accounts before attempting any password, turning one unknown into two
independent, easier problems. The same reasoning drives generic "invalid email
or password" messages in login forms. The cost is genuinely worse
diagnosability, which is why servers log the specific reason internally while
returning the vague message externally.

---

## Meta-bug 🔴 — Nine files written in one turn with no explanation

**Symptom**
Phase 0 was started by generating nine files back-to-back. The code was fine.
The process was wrong: CLAUDE.md states the goal is "production-quality software
**AND maximum learning**," and a wall of finished files teaches nothing.

**Root cause**
Optimizing for task completion over the actual objective. The instruction to
explain design decisions was followed shallowly — a brief note up front — rather
than as the point of the exercise.

**Fix**
Added the **ONE FILE AT A TIME** rule to CLAUDE.md: purpose + analogy + new
concepts in chat, full depth (functions, flow, math, gotchas, interview Q&A) in
`docs/explanation/phase_N_explanation.md`, then one file, then stop. All nine
files were retroactively documented.

**Lesson**
Speed is not the objective unless the objective is speed. When someone states
what they're optimizing for, that governs — even when a faster path exists and
would produce identical artifacts.

---

## Phase 0 scoreboard

| # | Bug | Severity | Caught by |
|---|---|---|---|
| 1 | Global pip install broke dbt | 🟠 | pip's own resolver warning |
| 2 | Two formatters configured | 🟡 | review before running |
| 3 | Dead import + unformatted code | 🟡 | ruff / black |
| 4 | Access log would lose request ID | 🟡 | re-reading the draft |
| 5 | Postgres port hijack | 🔴 | `netstat` — after 3 wrong theories |
| — | Nine files, no explanation | 🔴 | the user |

**Pattern worth noticing:** the two most expensive problems (#1, #5) were both
*environment* issues, not code issues — and neither would have been caught by
any test. The cheap ones were caught by tooling and review, which is exactly what
tooling and review are for.

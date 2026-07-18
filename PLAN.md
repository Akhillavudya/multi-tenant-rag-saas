# Multi-Tenant RAG SaaS — Architecture & Build Plan

Production-quality architecture on a $0 stack, with every AI component swappable to premium providers by changing config only.

---

## 1. System overview

```
                        ┌─────────────────────────────────────────┐
                        │              CLIENT SIDE                │
                        │                                         │
   Org admin ──────────►│  Next.js Dashboard (Vercel free)        │
                        │  - auth (Clerk free tier)               │
                        │  - create bots, upload docs, view chats │
                        │                                         │
   End user on ────────►│  Embeddable Widget (vanilla JS + iframe)│
   client's website     │  - talks to API with public bot key     │
                        └───────────────┬─────────────────────────┘
                                        │ HTTPS (SSE streaming for chat)
                        ┌───────────────▼─────────────────────────┐
                        │        FastAPI Backend (API)            │
                        │  - verifies Clerk JWT / bot key         │
                        │  - resolves org_id SERVER-SIDE (always) │
                        │  - chat endpoint: retrieve→rerank→LLM   │
                        │  - usage metering + plan limits         │
                        └──────┬──────────────┬───────────────────┘
                               │              │ enqueue ingestion jobs
                    ┌──────────▼───┐   ┌──────▼──────────────────┐
                    │ Postgres     │   │ Worker (Celery + Redis) │
                    │ (Neon free)  │   │ - parse → chunk → embed │
                    │ orgs, bots,  │   │ - one-time, async       │
                    │ docs, usage  │   └──────┬──────────────────┘
                    └──────────────┘          │ upsert vectors
                                       ┌──────▼──────────────────┐
                                       │ Qdrant (free tier/local)│
                                       │ ONE collection, every   │
                                       │ point tagged org_id +   │
                                       │ chatbot_id (indexed)    │
                                       └─────────────────────────┘

              PROVIDER LAYER (swappable — see section 3)
              ├─ LLM:       Gemini Flash free tier  → Claude/GPT/Ollama
              ├─ Embedder:  bge-small (local)       → OpenAI-3-large/Voyage/BGE-M3
              └─ Reranker:  bge-reranker-v2 (local) → Cohere Rerank
```

**The chat request path (the thing that must be fast):**
1. Widget/dashboard calls `POST /chat` with bot key or session JWT.
2. Backend resolves `org_id` + `chatbot_id` from the key — never from request body.
3. Embed the question (one small call).
4. Qdrant search: top-40 candidates, **mandatory filter** `org_id + chatbot_id`.
5. Rerank 40 → keep top 6.
6. Stream LLM answer (grounded prompt + citations) via SSE.
7. Log conversation, increment usage, enforce plan cap.

**The ingestion path (heavy work, done once, async):**
upload → store file → Celery job → parse → clean → chunk (~600 tokens, 15% overlap) → embed in batches → upsert to Qdrant with payload `{org_id, chatbot_id, document_id, source, page}` → mark document `ready`.

---

## 2. Free stack (dev + demo) and its paid upgrade path

| Layer | Free (build with this) | Paid upgrade (later) | Swap cost |
|---|---|---|---|
| Generation LLM | Gemini Flash free tier (default), Groq free (fallback), Ollama (offline) | Claude Haiku/Sonnet, GPT | env var |
| Embeddings | `BAAI/bge-small-en-v1.5` local (384-dim) | OpenAI `text-embedding-3-large`, Voyage, BGE-M3 | env var + re-embed script |
| Reranker | `BAAI/bge-reranker-v2-m3` local | Cohere Rerank | env var |
| Vector DB | Qdrant local (Docker) / Qdrant Cloud free | Qdrant Cloud paid | connection string |
| SQL DB | Neon free tier | Neon paid | connection string |
| Queue | Redis local (Docker) / Upstash free | managed Redis | connection string |
| Auth | Clerk free tier | Clerk paid | nothing |
| Billing | **Stripe TEST MODE** (fake cards, $0, zero risk) | Stripe live mode | flip one key |
| Frontend hosting | Vercel free | Vercel paid | nothing |
| Backend hosting | Oracle Cloud Always-Free ARM (4 OCPU / 24GB — enough RAM for local embedder+reranker) | any VPS | nothing |

Notes:
- Stripe test mode means you build the **entire** billing system — checkout, webhooks, plans, caps — with fake cards and zero money. Perfect for you.
- Oracle free ARM box: you already planned this for Materia; the same box hosts this backend + worker + local models.
- Free LLM tiers can rate-limit or change; that's fine for a demo. The provider layer makes outages a one-line switch.

---

## 3. The provider abstraction layer (the "little change to upgrade" design)

This is the heart of the swappability requirement. Three interfaces, one factory, all selection via env vars. **No business logic ever imports a vendor SDK directly.**

```
app/providers/
├── base.py          # the three interfaces (contracts)
├── factory.py       # reads settings → returns concrete provider
├── llm/
│   ├── gemini.py    # default (free)
│   ├── groq_llm.py  # fallback (free)
│   ├── ollama.py    # local/offline
│   ├── anthropic_llm.py   # premium (stub now, fill later)
│   └── openai_llm.py      # premium (stub now, fill later)
├── embeddings/
│   ├── bge_local.py       # default (free, local)
│   └── openai_embed.py    # premium
└── rerank/
    ├── bge_local.py       # default (free, local)
    ├── cohere_rerank.py   # premium
    └── noop.py            # pass-through (lets you A/B the reranker's value)
```

```python
# base.py — the contracts
class LLMProvider(Protocol):
    async def stream(self, system: str, messages: list[Message],
                     max_tokens: int) -> AsyncIterator[str]: ...

class EmbeddingProvider(Protocol):
    dimension: int
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...

class RerankProvider(Protocol):
    async def rerank(self, query: str, docs: list[str],
                     top_k: int) -> list[tuple[int, float]]: ...
```

```bash
# .env — swapping = editing these lines, nothing else
LLM_PROVIDER=gemini            # gemini | groq | ollama | anthropic | openai
EMBEDDING_PROVIDER=bge_local   # bge_local | openai
RERANK_PROVIDER=bge_local      # bge_local | cohere | noop
```

**Rules that keep swaps trivial:**
1. Vector dimension lives in DB per-bot (`embedding_model`, `dimension` columns). Changing embedders requires re-embedding — write `scripts/reembed.py` in Phase 2 so it's a command, not a crisis.
2. All prompts live in `app/prompts/` as templates, not inline strings.
3. Providers raise a common `ProviderError`; the API layer maps it to clean HTTP errors and (for LLMs) can fall back Gemini→Groq automatically.
4. Per-bot model override column (`model_config` JSON) so a future premium client can run Claude while everyone else runs Gemini — same codebase.

---

## 4. Data model (Postgres)

```
organizations   id, name, clerk_org_id, plan (free|pro|business), created_at
users           id, clerk_user_id, org_id, role
chatbots        id, org_id, name, system_prompt, model_config JSON,
                embedding_model, dimension, status, public_key (widget key)
documents       id, org_id, chatbot_id, filename, source_type (file|url),
                source_url, status (processing|ready|failed), error, pages,
                chunk_count, created_at
conversations   id, org_id, chatbot_id, started_at, channel (widget|playground)
messages        id, conversation_id, role, content, citations JSON,
                tokens_in, tokens_out, latency_ms, created_at
usage_records   id, org_id, period (YYYY-MM), messages_used, docs_ingested
subscriptions   id, org_id, stripe_customer_id, stripe_sub_id, plan,
                status, current_period_end
plans (seed)    free: 1 bot, 100 msg/mo, 5 docs
                pro:  5 bots, 2000 msg/mo, 100 docs
                business: 20 bots, 10000 msg/mo, 1000 docs
```

Qdrant: **one collection** (`chunks`), payload `{org_id, chatbot_id, document_id, text, source, page}`, **keyword indexes on org_id and chatbot_id**. Same pattern as Sage's thread_id filter, promoted to a security boundary.

---

## 5. Repo structure

```
ragsaas/
├── CLAUDE.md                  # instructions for Claude Code (see §8)
├── PLAN.md                    # this file
├── docker-compose.yml         # qdrant + redis (+ postgres optional) for dev
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app, routers, middleware
│   │   ├── config.py          # pydantic-settings, reads .env
│   │   ├── providers/         # §3
│   │   ├── core/
│   │   │   ├── chunking.py    # token-aware splitter w/ overlap
│   │   │   ├── ingestion.py   # parse→clean→chunk→embed→upsert
│   │   │   ├── retrieval.py   # search(org_id, bot_id, query) → reranked chunks
│   │   │   └── generation.py  # grounded prompt build + stream
│   │   ├── api/               # routers: auth, bots, documents, chat, billing, usage
│   │   ├── db/                # SQLAlchemy models + Alembic migrations
│   │   ├── workers/           # celery_app.py, ingest_tasks.py
│   │   ├── security/          # clerk JWT verify, bot-key resolve, rate limit
│   │   └── prompts/
│   ├── tests/
│   │   ├── test_isolation.py  # THE test — cross-org leakage must be impossible
│   │   ├── test_chunking.py
│   │   ├── test_retrieval.py
│   │   └── test_limits.py
│   └── pyproject.toml
├── frontend/                  # Next.js 14 (App Router) + Tailwind + Clerk
│   └── app/ (dashboard)/bots, /bots/[id]/{sources,playground,conversations,settings}, /billing
├── widget/
│   ├── loader.js              # <script> clients paste; injects bubble + iframe
│   └── frame/                 # tiny chat UI served in the iframe
└── scripts/
    ├── seed_demo.py           # demo org + sample docs for showing clients
    └── reembed.py             # migrate a bot to a new embedding model
```

---

## 6. Phase-by-phase build plan

Work strictly in order. Each phase ends with something running + tested. Do not start a phase until the previous phase's "done when" passes.

### Phase 0 — Foundations (1–2 days)
Repo init, `docker-compose up` gives Qdrant + Redis; Neon DB created; FastAPI skeleton with `/health`; Alembic wired; pydantic-settings config; pre-commit (ruff, black); CLAUDE.md written.
**Done when:** `uvicorn` runs, `/health` returns ok, `pytest` runs (even 1 test), `.env.example` exists and no secret is committed.

### Phase 1 — Provider layer + core RAG library (3–5 days)
Build §3 exactly: interfaces, factory, Gemini + Groq + Ollama LLMs, bge_local embedder, bge_local + noop rerankers, premium stubs. Then `core/`: chunking, ingestion (files: PDF/DOCX/TXT via pypdf + python-docx), retrieval (wide top-40 → rerank → top-6), generation (grounded prompt: answer only from context, cite source+page, say "I don't have that information" when absent). A CLI (`python -m app.cli ask ...`) to test end-to-end single-tenant.
**Done when:** you can ingest a PDF and get a fast, cited, accurate streamed answer in the terminal; swapping `LLM_PROVIDER=groq` works with zero code edits; `RERANK_PROVIDER=noop` visibly degrades answer quality (proves the reranker earns its place).

### Phase 2 — Multi-tenancy + async ingestion (4–6 days)
Clerk auth (JWT verified in FastAPI middleware → `request.state.org_id`); orgs/bots/documents CRUD; Celery ingestion worker with status transitions (processing→ready/failed); Qdrant payload filters mandatory in `retrieval.search()` — the function signature **requires** org_id and bot_id, there is no unfiltered path; `scripts/reembed.py`.
**Done when:** `test_isolation.py` passes: create Org A and Org B, ingest different docs, prove Org A's queries can never return B's chunks — including adversarial attempts (forged bot_id in request body, missing filter). This test is sacred; it runs in CI forever.

### Phase 3 — Chat API + metering (3–4 days)
`POST /chat` with SSE streaming; bot-key auth for widget traffic (public_key → org/bot lookup, per-key rate limit); conversation + message logging with citations, tokens, latency; usage increment per message; plan-cap enforcement (429 + friendly "limit reached" payload when over); Gemini→Groq automatic fallback on provider errors.
**Done when:** curl streams an answer token-by-token; the 101st free-tier message is politely refused; usage numbers in DB match messages sent.

### Phase 4 — Dashboard (5–8 days; your biggest new-skill phase)
Next.js + Clerk + Tailwind. Pages: bot list/create; sources tab (upload w/ drag-drop, live status via polling, delete); playground (streamed chat to test the bot); conversations viewer; settings (system prompt, model — reads provider list from API); usage bar vs plan.
**Done when:** a non-technical person could sign up, create a bot, upload a PDF, and chat with it — without you explaining anything.

### Phase 5 — Embeddable widget (2–4 days)
`loader.js` (paste-able `<script data-bot-key="...">`) injects a chat bubble; iframe chat UI (isolated CSS); streams from `/chat` with the public bot key; "Powered by ..." footer; embed-code snippet shown in dashboard settings.
**Done when:** you paste the snippet into a plain HTML file and get a working chatbot on a "client site" you never touched.

### Phase 6 — Billing in Stripe TEST MODE (3–5 days)
Products/prices for free/pro/business; Checkout session from `/billing`; webhook handler (`checkout.session.completed`, `customer.subscription.updated/deleted`) flips org plan; customer portal link for cancel/upgrade; limits already enforced from Phase 3 now vary by plan.
**Done when:** with Stripe test card `4242...`, you can upgrade an org, watch caps rise, cancel, and watch it downgrade — end to end, $0 spent, zero financial risk.

### Phase 7 — Deploy + demo polish (3–5 days)
Frontend → Vercel; backend + worker + Qdrant + Redis → Oracle free ARM (docker compose, Caddy for HTTPS); `seed_demo.py` creates a demo org with sample docs; landing page with live widget on it; README with architecture diagram + a metrics section (answer latency p50/p95, retrieval hit-rate on a 20-question eval set — reuse your Sage Phase-C eval-harness approach); short Loom-style demo video.
**Done when:** you can send one URL to a stranger and they can try the bot in 30 seconds.

### Stretch (post-v1, in demand order)
Website-URL ingestion (trafilatura crawler, page/depth caps, robots.txt) → hybrid retrieval (Qdrant sparse+dense) → team members per org → BYOK per-org keys → conversation analytics.

**Total: roughly 4–7 weeks part-time.** Phases 1–3 are the résumé; Phases 4–6 are the product.

---

## 7. Production-quality bar (applies to every phase)
- No secrets in code, ever. `.env` + `.env.example`; `.gitignore` from Phase 0.
- Every provider call has a timeout and typed errors; API returns clean JSON errors.
- Structured logging (org_id, bot_id, request_id on every line).
- Migrations only via Alembic; never hand-edit the DB.
- `test_isolation.py` + `test_limits.py` run in CI (GitHub Actions) on every push.
- Type hints everywhere; ruff + black enforced by pre-commit.

## 8. Working with Claude Code (put this in CLAUDE.md)
- Reference this plan: "We are in Phase N of PLAN.md. Build only X."
- One phase-slice per session; commit small, run tests before committing.
- Ask for tests first on security-critical code (isolation, limits, webhooks).
- Never let it hardcode a key or bypass the provider factory "just to test."
- After each phase: "Review this phase against PLAN.md §6 'done when' criteria."
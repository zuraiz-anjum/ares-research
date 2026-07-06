# Ares — Autonomous Research & Evidence System — Full Project Brief

Hand this whole document to a new Claude conversation to bring it up to speed
on the project with zero prior context.

## 1. What it is

Ares is a multi-agent AI research assistant. You ask one question in a chat
UI; a LangGraph state machine routes it through one of 14 specialized
pipelines (research, report, comparison, debate, academic paper, PDF export,
data analysis, charting, code, email drafting, planning, plain chat, document
Q&A, survey/CSV analysis), runs whichever agents that pipeline needs, and
streams the answer back token-by-token over SSE.

It started as a take-home assignment scaffold and has since become the
owner's own project — all "this is an assignment" framing has been removed
from the docs (ASSIGNMENT.md was deleted; DECISIONS.md and JOURNAL.md were
rewritten to drop that language). It's deployed on Railway with a persistent
volume for SQLite data.

## 2. Architecture at a glance

```
START -> clarity -> intent_router -> [mode-specific pipeline] -> suggestions -> END
```

- **clarity** — decides if the question is answerable as-is or needs a
  clarifying question first (via LangGraph's `interrupt()`); also detects
  "write me an academic paper" requests and collects 4 requirements via an
  MCQ interrupt before continuing. As of the latest fix, it also looks at
  recent conversation history to resolve ambiguous follow-ups (see section 7).
- **intent_router** — classifies the query into one of 14 modes using a mix
  of fast keyword-signal shortcuts and an LLM call for ambiguous cases.
- **decomposer** — breaks compound questions into parallel sub-queries.
- **research** — runs Tavily web searches in parallel (ThreadPoolExecutor,
  not asyncio, since the Tavily client is synchronous) and scores confidence.
- **validator** — re-runs research if confidence is below threshold (max 3
  attempts).
- Mode-specific writer agents: **synthesis** (chat/research answers),
  **report_writer**, **comparison_matrix**, **data_analyst**,
  **debate_writer**, **email_drafter**, **academic_writer**, **code_writer**,
  **chart_writer**, **survey_analyst**, **doc_agent** (RAG/URL docs),
  **planner** + **dynamic_spawner** (plan mode spawns specialist agents).
- Self-correction layer: **fact_checker** (verifies claims), **critic**
  (scores research quality), **draft_critic** (scores writing quality, can
  loop back for revision), **challenger** (adversarial counter-evidence
  search + reconciliation), **voting_synthesis** (research mode only — 3
  independent LLM "perspectives" on the same findings, judged by a 4th call
  that picks the strongest and explains why).
- **data_extractor** / **data_visualizer** — pull quantitative data out of
  research findings and chart it (matplotlib, Agg backend).
- **pdf_generator** (ReportLab) / **pptx_generator** (python-pptx) — export.
- **suggestions** — universal terminal node, proposes 3 follow-up questions.

26 total LangGraph nodes (verified against `tests/test_smoke.py`'s expected
node set). The README.md in the repo is stale and undercounts this as
"20 agents, 12 modes" — the actual current numbers are 26 nodes / 14 modes.

## 3. Tech stack

- **LangGraph** (>=0.2.50) — the state machine / agent orchestration layer
- **LangChain** — LLM abstraction + structured output (`.with_structured_output()`)
- **FastAPI** + **uvicorn** — async HTTP server, SSE streaming
- **Pydantic v2** (`pydantic-settings`) — config from env vars
- **ChromaDB** + **FlashRank** — RAG vector store + cross-encoder reranking
- **Tavily** — web search API
- **Groq / Cerebras / Gemini / OpenRouter** — the 4 LLM providers in the
  fallback chain (all free-tier)
- **aiosqlite** + **langgraph-checkpoint-sqlite** — async SQLite for
  conversation checkpoints, session history, analytics, cost tracking
- **ReportLab** — PDF generation (pure Python, no system deps — chosen over
  WeasyPrint/Puppeteer specifically to avoid needing GTK/Chromium in Docker)
- **python-pptx** — PowerPoint export
- **matplotlib** + **seaborn** — chart rendering (Agg backend, no display needed)
- **slowapi** — per-IP rate limiting (20 req/min on chat endpoints)
- **tenacity** — retry logic for Tavily calls (3 attempts OR 30s wall clock,
  whichever hits first)
- **langsmith** (optional) — tracing, enabled via `LANGSMITH_API_KEY`
- Vanilla JS + SSE (`EventSource`) frontend, no framework — `static/index.html`
  is a single-file chat UI; `static/dashboard.html` is the analytics view.

## 4. Multi-provider LLM fallback (`app/llm.py`)

The core reliability mechanism, since every provider used is free-tier with
daily/rate limits:

- Chain order: **Groq** (llama-3.3-70b-versatile) → **Cerebras**
  (gpt-oss-120b, via `ChatOpenAI` pointed at Cerebras's OpenAI-compatible
  endpoint — the official `langchain-cerebras` integration mangles model
  names) → **Gemini** (gemini-2.0-flash) → **OpenRouter**
  (llama-3.3-70b-instruct:free).
- `_FallbackLLM` wraps all configured providers; on a rate-limit or known
  provider bug (Groq's structured-output quirks), it advances a module-level
  `_active_idx` so every subsequent call in the process skips the dead
  provider — no per-request retry-from-scratch overhead.
- `_active_idx` resets automatically on UTC day-rollover (daily quotas reset
  at midnight UTC), so a previously-exhausted provider isn't wedged forever.
- If the *entire* chain fails on the same call, it gets one bounded 3-second
  retry pass before giving up.
- `AllProvidersExhaustedError` — raised when the chain saw a genuine
  rate-limit anywhere along the way, even if the *final* exception was
  something unrelated (e.g. Groq/Cerebras/Gemini all rate-limited, then
  OpenRouter fails on an unrelated auth error). This lets the frontend show
  "today's free quota is used up" instead of a generic error in that case.
- **Live fallback status in the UI**: `_FallbackLLM` emits a
  `provider_switch` / `provider_retry` event (via a contextvar-scoped sink,
  same pattern as the thread-id contextvar) whenever it changes providers
  mid-request. `/chat/stream` merges these events with the graph's own
  `astream_events` output via a shared `asyncio.Queue`, so the frontend shows
  "Groq is busy — switching to Cerebras..." live instead of a silent spinner.

## 5. Conversation memory / state — IMPORTANT recent fix

`app/state.py`'s `AgentState.messages` field is the running conversation
transcript, persisted per `thread_id` via LangGraph's SQLite checkpointer.

**Until recently this was fundamentally broken**: the field had no
`add_messages` reducer annotation anywhere in the codebase, so LangGraph's
default "last write wins" merge silently **replaced** the entire persisted
transcript with just the current turn's new message on every single request
— across every mode, not an isolated edge case. Verified directly: told the
graph "my favorite color is blue" in turn 1, asked "what did I just tell
you?" in turn 2 on the same thread — state had exactly 1 message after turn
2, and the model had zero memory of turn 1.

**Fix**: annotated the field as
`messages: Annotated[list[AnyMessage], add_messages]` — the standard
LangGraph pattern — so new messages append/merge into history instead of
overwriting it.

**Follow-on fix**: even with history now available, `clarity_node` was still
only forwarding the isolated latest message to its own LLM call, so a bare
follow-up like "which benchmarks were used?" (referring to a paper discussed
earlier in the same thread) would still trigger an unnecessary clarification
interrupt. Added a short recent-history window (last 6 messages) to the
clarity prompt plus a new `standalone_query` field on `ClarityVerdict`, so
the LLM resolves pronouns/implicit references against history and rewrites
the follow-up into a self-contained query (e.g. "Which benchmarks were used
in the Attention Is All You Need paper?") before it reaches intent_router
and research/synthesis.

A regression test (`tests/test_eval.py::test_conversation_history_accumulates_across_turns`)
now guards this specifically — it runs two turns against the same
`thread_id` in mock mode and asserts both that message count grows and that
turn 1's content survives into turn 2's state. Verified the test actually
catches the regression by temporarily reverting the reducer locally before
committing.

## 6. Other production-hardening work done this session

- **Path/persistence consistency**: 6+ files (`main.py`, `checkpoint.py`,
  `calibration.py`, `search_cache.py`, `rag/store.py`, `memory/entity_store.py`)
  each independently hardcoded relative SQLite/file paths instead of using
  `config._data_path()`, which resolves against `DATA_DIR` (a Railway volume
  mount). Fixed so everything survives container restarts/redeploys.
- **PDF/chart/critique "state bleed" bug**: a PDF generated for one query
  was reappearing under a completely unrelated later query in the same
  thread. Root cause: the checkpointer persists `AgentState` across turns,
  but `main.py` only explicitly reset a handful of fields per turn. Fixed
  via a `_FRESH_TURN_STATE` dict (~28 fields: `pdf_url`, `chart_urls`,
  `critique`, `fact_check_results`, `sources`, etc.) spread into the inputs
  dict of every new turn on both `/chat` and `/chat/stream`.
- **Quota-exhaustion UX**: instead of a generic "An internal error
  occurred," the backend now shows "Today's free API quota has been reached
  across all providers... try again later" when `_is_quota_exhausted()`
  (checks both `AllProvidersExhaustedError` and direct rate-limit
  exceptions) fires.
- **Background task lifecycle**: replaced raw `asyncio.create_task(...)`
  calls (which can be silently garbage-collected if nothing holds a
  reference) with `app/utils/background.py`'s `spawn()` helper, which keeps
  tasks in a module-level set until they complete.
- **Event-loop blocking fixes**: `entity_store.recall()` (synchronous
  full-file read) is now offloaded via `loop.run_in_executor()` in both
  `synthesis.py` and `voting_synthesis.py` instead of blocking the event
  loop on every request.
- **SQLite retention/pruning**: added indexes + a 7-day retention policy on
  `node_checkpoints` (`app/utils/checkpoint.py`), and a 90-day retention
  policy on sessions/usage tables in `main.py`, so the DBs don't grow
  unbounded.
- **Entity memory cap**: `app/memory/entity_store.py` now caps at 500
  entities with eviction, guarded by a `threading.Lock()`.
- **`/share` path traversal hardening**: replaced a `str.startswith()` path
  check (bypassable) with proper `Path` containment checks against an
  allow-list of root directories.
- **Token/session cleanup**: `_active_tokens` changed from a bare set to a
  dict tracking expiry, with periodic pruning of expired sessions.

## 7. Full API surface (`app/main.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Chat UI (static/index.html) |
| GET | `/chat/stream` | SSE streaming chat — main endpoint |
| POST | `/chat` | Non-streaming chat |
| POST | `/resume` | Answer a clarification / MCQ interrupt |
| POST | `/upload` | Upload document (PDF/DOCX/TXT/MD) for RAG |
| POST | `/upload/image` | Upload image — Gemini Vision describes it, stored as RAG chunks |
| GET | `/history` | Recent session list |
| GET | `/analytics` | Usage analytics JSON |
| GET | `/dashboard` | Analytics dashboard UI |
| GET | `/calibration` | Confidence-threshold calibration report |
| GET | `/prompt-health` | Per-node quality metrics (draft scores, fact-check pass rate, etc.) |
| GET | `/cost/{thread_id}` / `/cost` | Token/cost tracking per thread or globally |
| GET | `/cache/stats` | Search-result cache stats |
| POST | `/export/pptx` | Export a report as PowerPoint |
| POST | `/share` / GET | `/r/{share_id}` | Shareable public report links |
| POST | `/schedule` / GET `/schedule` | Scheduled research tasks |
| POST | `/auth/login` / `/auth/logout` / GET `/auth/status` | Cookie-based password gate |
| GET | `/health` | Health check |
| GET | `/diagnostics`, `/errors` | Ops/debugging endpoints |

## 8. Config (`app/config.py`, all via env vars / `.env`)

- Provider keys: `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`,
  `CEREBRAS_API_KEY` (at least one required), `TAVILY_API_KEY` (required)
  — all stripped of whitespace at load time via a field validator.
- `MOCK_MODE` — when true, every agent short-circuits to a hardcoded
  response and Tavily returns fake results; zero external calls. This is
  how the test suite runs offline.
- `MAX_TOKEN_BUDGET` (24000 default), `SESSION_COST_BUDGET_USD` (0.50
  default) with automatic mode-downgrading under budget pressure
  (plan → research → chat).
- `SEARCH_CACHE_ENABLED` / `SEARCH_CACHE_TTL_SECONDS` — caches Tavily
  results for repeated queries.
- `DATA_DIR` — set this to a Railway/Fly volume mount path in production so
  SQLite DBs and generated files survive restarts; empty in local dev.
- `APP_PASSWORD` — optional cookie-based auth gate.
- `SLACK_WEBHOOK_URL`, `SMTP_*` — optional integrations.
- `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` — optional tracing.

## 9. Testing

5 test files, 123 tests total, all run offline via `MOCK_MODE` (no API
keys or network needed):

- `tests/test_smoke.py` — graph compilation, node presence, routing logic
  (`route_after_research`, `route_after_validation`), token budget helpers.
- `tests/test_eval.py` — full mock-mode end-to-end graph runs, groundedness
  checks, the new conversation-history-accumulation regression test,
  config/settings sanity checks.
- `tests/test_agent_communication.py`, `test_advanced_features.py`,
  `test_features.py` — feature-specific coverage.

Run with: `python -m pytest tests/ -q`

## 10. Deployment

- Deployed on **Railway**, GitHub-integrated auto-deploy (push to `master`
  triggers a redeploy).
- `Dockerfile` + `docker-compose.yml` for local/self-hosted Docker runs.
- Persistent state relies on `DATA_DIR` being set to a mounted volume in
  Railway — without it, every redeploy wipes SQLite data (history,
  checkpoints, analytics, cost tracking).
- Known infra caveat (was flagged and since fixed by the project owner):
  the OpenRouter API key was returning `401 Missing Authentication header`
  in Railway's production environment — a credential/env-var issue on
  Railway's side, not a code bug. Confirmed resolved (verified live with a
  real completion call) as of the OpenRouter key being re-added to both
  local `.env` and Railway's environment variables.

## 11. Engineering decisions worth knowing (see `DECISIONS.md` for full list)

- Groq over OpenAI (free tier, fast, structured-output support).
- Token counting via `len(text) // 4` character approximation (provider-
  agnostic, since tiktoken is OpenAI-specific and Groq uses LLaMA's tokenizer).
- `original_query` stored explicitly in state rather than reading
  `messages[-1]` or `messages[0]`, because clarification flows mean the last
  message isn't always the original question.
- ThreadPoolExecutor (not asyncio) for parallel Tavily search, since the
  Tavily client is synchronous.
- Confidence threshold is inclusive (`>=`), not exclusive.
- Mock mode is per-agent hardcoded returns rather than `FakeListChatModel`,
  because the latter doesn't support `.with_structured_output()`.
- SqliteSaver (not MemorySaver) for checkpointing, so conversations survive
  restarts — accepted trade-off: SQLite's limited write concurrency is fine
  at this traffic scale; Postgres would be the right call at real scale.
- ReportLab over WeasyPrint/Puppeteer for PDF generation — zero system
  dependencies (pure Python), simpler Docker images.

## 12. What to know if continuing work on this project

- This is the owner's personal project now, not a graded assignment — all
  take-home framing has been deliberately removed from the docs.
- Ares fast-iteration norm observed this session: commit + push
  immediately after each verified fix, don't wait for a big batch.
- Prefer verifying claims with a live local run (start uvicorn, curl the
  endpoint, or invoke the graph directly in a script) over reasoning about
  LangGraph/async behavior from memory — this is exactly how both the
  PDF-leak bug and the conversation-memory bug were root-caused correctly
  instead of guessed at.
- Test coverage gap worth watching: the 123 tests are strong on routing/unit
  logic and mock-mode smoke checks, but historically thin on genuine
  multi-turn/integration behavior — that's the exact gap that let the
  conversation-memory bug ship silently. Bias toward integration-style tests
  (real graph invocations across multiple turns) over more unit tests when
  adding coverage.

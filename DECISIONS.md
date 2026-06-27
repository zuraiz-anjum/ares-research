# Engineering Decisions

## 1. Provider swap: Groq instead of OpenAI

**Context:** The assignment says OpenAI API keys but says "you can change this to other keys you may have."

**Options:** Keep OpenAI, swap to Groq, swap to Gemini free tier.

**Choice:** Groq with llama-3.3-70b-versatile.

**Trade-off:** Groq's free tier is fast and supports structured output which all four agents rely on. The downside is tokenisation differences. Groq uses LLaMA's tokeniser, not OpenAI's cl100k_base, so the token budget approximation is slightly less accurate. The swap is a one-line change in llm.py since the LangChain abstraction handles provider differences cleanly.


## 2. Token counting: character approximation instead of tiktoken

**Context:** The assignment requires a configurable per-query token budget. tiktoken is OpenAI-specific and doesn't work with Groq/LLaMA.

**Options:** Use tiktoken with cl100k_base as an approximation, use LLaMA's tokeniser directly, use character count divided by 4.

**Choice:** `len(content) // 4` which is the industry-standard character-based approximation.

**Trade-off:** This is less accurate than a real tokeniser (roughly 10-15% error) but works across any provider, requires zero dependencies, and is good enough for a budget guard whose purpose is catching clearly oversized queries, not billing precision.


## 3. Storing original_query in state instead of reading messages[-1]

**Context:** After a clarification flow, `messages[-1]` points to the user's clarification answer, not the original question. Both research and validator were reading `messages[-1]` to get the question.

**Options:** Read `messages[0]` (first message), add an `original_query` field to state, pass the query explicitly through each node.

**Choice:** Added `original_query: str` to AgentState, populated by the clarity node before any interrupt.

**Trade-off:** Adding a field to shared state is slightly more explicit than reading `messages[0]`, but it makes the intent clear. This field is the canonical question for this turn. `messages[0]` would break if the conversation starts with a system message or if the graph is resumed mid-conversation. The explicit field is more robust.


## 4. Decomposer as a separate node instead of modifying the Research agent

**Context:** Query decomposition needs to happen before research runs.

**Options:** Add decomposition logic inside research_node, create a separate decomposer_node, add a pre-processing step in the router.

**Choice:** Separate decomposer_node sitting between clarity and research in the graph.

**Trade-off:** This adds one LLM call per query which adds latency and cost on every request. The benefit is clean separation. The research agent's job is to search and summarise, not plan queries. It also makes decomposition independently testable, which is how I was able to write unit tests for it without running the full graph.


## 5. ThreadPoolExecutor instead of asyncio for parallel search

**Context:** Multiple sub-queries need to run simultaneously after decomposition.

**Options:** Run sequentially in a loop, use asyncio with an async Tavily client, use ThreadPoolExecutor.

**Choice:** ThreadPoolExecutor.

**Trade-off:** The Tavily client is synchronous. Wrapping synchronous I/O in asyncio requires running it in a thread pool anyway, making asyncio's overhead pointless here. ThreadPoolExecutor is simpler, achieves true parallelism for I/O-bound tasks (which network calls are), and requires zero changes to the rest of the codebase. For 2-3 concurrent searches the difference is negligible. At scale, a proper async client would matter.


## 6. Bounded context: last 10 messages instead of full history

**Context:** The synthesis agent was passing the entire conversation history to the LLM on every call. Long conversations would eventually hit context limits and cost more.

**Options:** Last N messages, sliding window with token count, summarise old messages.

**Choice:** `messages[-10:]` which is the last 10 messages.

**Trade-off:** 10 is somewhat arbitrary but covers most realistic conversation depths while keeping context bounded. Summarising old messages would be more sophisticated but adds another LLM call and complexity for a problem that rarely occurs in a research assistant. The hard cutoff is simple and predictable.


## 7. Tenacity for retries with combined attempt and time limits

**Context:**  Network APIs fail, rate limits, timeouts, brief outages. Without retries, a single Tavily hiccup crashes the whole research pipeline and the user gets an error for no good reason.

**Options:** Manual try/except with sleep, tenacity, httpx built-in retry.

**Choice:** Tenacity with `stop=(stop_after_attempt(3) | stop_after_delay(30))`.

**Trade-off:** Tenacity's `|` operator stops on whichever condition fires first, either 3 attempts OR 30 seconds total. This prevents the worst case where three attempts that each take 15 seconds would block for 45 seconds. The 30-second wall clock limit keeps the user-facing latency bounded regardless of how slow Tavily is being.


## 8. Confidence threshold as inclusive (>=) not exclusive (>)

**Context:** The original code used `confidence_score > confidence_threshold`. A score of exactly 6 with a threshold of 6 would route to the validator unnecessarily.

**Options:** Keep `>`, change to `>=`, change the default threshold to 7.

**Choice:** Changed to `>=`.

**Trade-off:** The threshold is defined as "the minimum acceptable confidence." A score equal to the threshold means the research meets the bar and it should not be sent to the validator. Changing the threshold to 7 instead would silently shift the bar rather than fix the semantics. `>=` is the correct implementation of the stated intent.


## 9. Mock mode: full isolation via per-agent hardcoded returns

**Context:** Tests need to run without any network calls or API keys — not just Tavily, but the LLM too.

**Options:** Mock both Tavily and LLM using FakeListChatModel, mock Tavily only, add per-agent mock returns controlled by MOCK_MODE setting.

**Choice:** When `MOCK_MODE=true`, each agent checks the flag and returns a hardcoded response instead of calling the LLM. Tavily returns a fake result constant. Zero external calls.

**Trade-off:** FakeListChatModel doesn't support `.with_structured_output()` which all agents use, so it wasn't viable. Per-agent hardcoded returns are simple, reliable, and make the mock contract explicit. The downside is the mock responses are static — they don't adapt to the query. That's acceptable for testing routing and infrastructure. Groundedness testing against real LLM output still requires live API calls, but routing, cost, and latency are fully covered offline.


## 10. SqliteSaver instead of MemorySaver for conversation persistence

**Context:** The baseline used MemorySaver which stores all conversation state in memory. Every server restart wipes all active conversations, which would be embarrassing in any real deployment.

**Options:** Keep MemorySaver, use SqliteSaver (SQLite file), use RedisSaver or PostgresSaver.

**Choice:** SqliteSaver with a local `checkpoints.db` file.

**Trade-off:** SqliteSaver requires no external service — SQLite is built into Python. Conversations now survive server restarts and deployments. The downside is that the SQLite file grows indefinitely without a cleanup strategy, and SQLite has limited concurrency under high load. For a production system with real traffic, Postgres would be the right choice. For this project, SQLite is the right balance between persistence and simplicity.


## 11. Rate limiting: 20 requests per minute per IP

**Context:** The API has no protection against a single client hammering it with requests, which would burn through API quota and degrade performance for other users.

**Options:** No rate limiting, slowapi (per-IP), API key authentication, queue-based throttling.

**Choice:** slowapi with a 20 requests/minute limit per IP on `/chat` and `/resume`.

**Trade-off:** 20/min is generous for a research assistant — a real user typing questions won't hit it. It protects against accidental loops and basic abuse without adding authentication complexity. The limit is applied per IP which breaks in shared NAT environments (all users behind one IP share the limit), but that's acceptable for this scale.


## 13. Multi-provider LLM fallback chain

**Context:** Groq's free tier has a daily token limit. Once hit, all requests fail until midnight UTC.

**Options:** Notify the user and give up, retry with exponential backoff (still same provider), fall through to a secondary provider.

**Choice:** `_FallbackLLM` wrapper class with a module-level `_active_idx` integer tracking which provider to use. Chain: Groq (llama-3.3-70b) → Cerebras (gpt-oss-120b, via OpenAI-compatible endpoint) → Gemini (gemini-2.0-flash) → OpenRouter.

**Trade-off:** The global `_active_idx` persists across requests within a process — once Groq fails, every subsequent request starts at Cerebras without retrying Groq. This is correct behaviour for a daily rate limit (retrying Groq every request would add latency for no benefit). The downside is that if Groq resets at midnight, the fallback won't revert automatically until the server restarts. Cerebras is placed before Gemini because Gemini's `tenacity` retry logic blocks the thread for 30+ seconds on quota errors, making it a worse user experience even when Cerebras is available.


## 14. Cerebras via ChatOpenAI instead of ChatCerebras

**Context:** `langchain-cerebras` mangles model names — `ChatCerebras(model="llama3.3-70b")` sends `llama-3.3-70b` to the API which returns 404 because the model slug doesn't exist on this key.

**Options:** Use `ChatCerebras` and figure out the correct slug, use `ChatOpenAI` with `base_url` pointing to Cerebras's OpenAI-compatible endpoint.

**Choice:** `ChatOpenAI(base_url="https://api.cerebras.ai/v1", api_key=..., model="gpt-oss-120b")`.

**Trade-off:** This bypasses the Cerebras LangChain integration entirely and treats Cerebras as a generic OpenAI-compatible provider. It's more fragile if Cerebras changes their API contract, but it works reliably right now and avoids the model-name mangling bug in the official integration.


## 15. Matplotlib Agg backend for chart rendering

**Context:** Matplotlib's default backends (TkAgg, Qt5Agg) require a display/GUI environment. Servers typically have no display, causing `cannot connect to X server` errors.

**Options:** Use Agg (non-interactive raster backend), use Cairo, use Plotly (HTML/JS output).

**Choice:** `matplotlib.use("Agg")` set at module level before any plt import.

**Trade-off:** Agg is the standard non-interactive backend for server-side rendering. It produces PNG files suitable for web display. The downside is no interactivity (zoom, hover) — charts are static images. Plotly would give interactive charts but requires either a JS runtime in the backend or shipping raw JSON to the frontend and rendering there, which adds complexity. Static PNGs are sufficient for a research assistant where charts are generated on demand.


## 16. ReportLab for PDF generation instead of WeasyPrint or Puppeteer

**Context:** PDF generation from HTML (WeasyPrint, Puppeteer) typically requires system libraries (GTK, Chromium) that are painful to install in Docker and on Windows.

**Options:** WeasyPrint (CSS → PDF, needs libpango, libcairo), Puppeteer (headless Chrome), ReportLab (pure Python, programmatic PDF).

**Choice:** ReportLab with `SimpleDocTemplate` and custom `Paragraph` flowables.

**Trade-off:** ReportLab requires more code to lay out the document programmatically rather than writing CSS, but it has zero system dependencies — it's pure Python. This makes Docker images simpler and the setup portable across Windows, macOS, and Linux. The custom `_md_to_flowables()` function converts Markdown headings and bullets to ReportLab elements, which handles ~90% of what research reports need.


## 17. SQLite for session history persistence

**Context:** The in-memory `deque` for session history was lost on every server restart, making the history sidebar useless after a redeployment or crash.

**Options:** Keep in-memory only, write to SQLite via `aiosqlite`, use Redis, use PostgreSQL.

**Choice:** `aiosqlite` with a local `ares_history.db` file. On startup, the last 50 sessions are loaded into the deque. Each new session is written to SQLite asynchronously via `asyncio.create_task()` so it doesn't block the SSE response.

**Trade-off:** SQLite has limited write concurrency (single writer), but at the usage patterns of a personal research assistant this is never a bottleneck. The async write via `create_task` means there's a tiny window where the server could crash between a session completing and the DB write finishing, losing that one entry — acceptable for a history sidebar where losing one entry is not catastrophic. For billing-critical data, you'd want a synchronous write or a proper queue.


## 18. Source citations: pass-through to frontend, not embedded in answer text

**Context:** Tavily returns source URLs for each search result. We could embed them as `[1](url)` footnotes in the LLM's answer text, or pass them separately to the frontend.

**Options:** Instruct the LLM to cite sources inline, post-process the answer to append footnotes, pass sources as a separate SSE event and render them in the UI.

**Choice:** Collect all source `{title, url}` pairs in `research_node`, emit them as a `"sources"` SSE event, render as a collapsible "Sources" section below the answer.

**Trade-off:** Embedding citations in the LLM's answer text is unreliable — LLMs hallucinate URLs and format them inconsistently. Passing sources separately is 100% accurate (they come directly from Tavily, not the LLM) and keeps the answer text clean. The collapsible UI lets power users verify sources without cluttering the default view.


## 20. Spec ambiguities I noticed

**README says OPENAI_API_KEY, brief says you can swap providers.** I swapped to Groq and documented it here rather than modifying the original README, which is the baseline artifact.

**"Reproducible/mocked run mode" is undefined.** The spec doesn't say what mock mode should cover. I implemented full isolation — when `MOCK_MODE=true`, every agent returns hardcoded responses and Tavily returns fake results. Zero external API calls. The routing tests run entirely offline in under 2 seconds.

**Confidence threshold semantics are ambiguous.** The config field is named `confidence_threshold` but the original code used `>` not `>=`. Whether the threshold is inclusive or exclusive is unspecified. I treated it as inclusive (a score AT the threshold passes) and fixed the implementation to match.

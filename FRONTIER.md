# Frontier: Parallel Query Decomposition

## What I built

I added a query decomposition layer to the research pipeline. Before any search happens, a new Decomposer agent analyses the user's question and decides whether it needs to be split into multiple independent sub-queries.

Simple question gets one search query, same as before. Compound question gets 2-3 independent sub-queries, each searched in parallel, results merged before synthesis.

The new pipeline:
```
START -> Clarity -> Decomposer -> Research -> (Validator | Synthesis) -> END
```

The Research agent now runs all sub-queries simultaneously using ThreadPoolExecutor, then feeds the merged results into the LLM for summarisation. I also added a confidence penalty when any sub-query returns sparse results. If one search comes back nearly empty, the system is less confident in the overall answer and routes to the validator more aggressively.

## Where the idea came from

I studied parallel and distributed computing, and the problem immediately reminded me of embarrassingly parallel tasks, which are tasks with no data dependency that should never be serialised. The baseline system was serialising everything: one query, one search, one summary. For compound questions that's just unnecessary latency.

The academic work on this, Least-to-Most Prompting (Zhou et al., 2022) and Self-Ask (Press et al., 2022), both assume sequential execution where each sub-answer feeds into the next question. That makes sense for compositional reasoning chains. But company research queries don't work that way. "Compare Stripe and Brex's funding" doesn't require knowing Stripe's numbers before you can search for Brex's. They're completely independent.

LangChain's MultiQueryRetriever generates multiple queries but for a different purpose: rephrasing the same query to improve recall. That's not decomposition, that's query expansion. Different problem.

My novel twist was applying the embarrassingly parallel principle from distributed systems to LLM query planning, specifically for the case where sub-queries are independent by nature.

## Why this is non-obvious

The obvious approach when someone asks "Compare Stripe and Brex" is to send that exact string to a search engine and hope for the best. That's what the baseline did.

The slightly less obvious approach is to decompose it, which both academic papers do, but they assume you have to do it sequentially.

The non-obvious insight is that for information retrieval, sequential decomposition is a category error. You only need sequential execution when answers have data dependencies. Research sub-queries almost never do. Applying parallel execution here isn't just a performance optimisation, it's architecturally correct for the problem domain.

## Design decisions

Why a separate Decomposer node rather than modifying the Research agent? Separation of concerns. The Research agent's job is to search and summarise. Query planning is a different responsibility. Keeping them separate makes each node easier to test and reason about independently.

Why ThreadPoolExecutor over asyncio? The Tavily client is synchronous. Wrapping it in asyncio would require running it in an executor anyway. ThreadPoolExecutor is simpler, sufficient for the number of sub-queries we generate (max 3), and requires no changes to the rest of the codebase.

Why cap at 3 sub-queries? Each sub-query is a Tavily API call. Beyond 3, you're paying for searches that are unlikely to meaningfully improve the answer quality for a single user question. The cap keeps costs bounded and forces the decomposer to be selective.

Why penalise confidence on sparse results? If one sub-query returns no results, the merged findings are incomplete. Pretending to be highly confident about an answer that's missing half its evidence would be dishonest. The penalty routes incomplete research through the validator for another attempt rather than going straight to synthesis.

## What I'd build next

True async execution: replace ThreadPoolExecutor with asyncio and an async Tavily client. For 3 concurrent searches this probably doesn't matter much, but at scale it would matter a lot.

Per-query confidence scoring: instead of one confidence score for the merged result, run a quick LLM evaluation per sub-query result before merging. Aggregate by taking the minimum. This gives a more honest picture of which sub-queries actually found useful information.

Dependency detection: some compound questions DO have sequential dependencies. Right now the decomposer assumes independence. A smarter version would classify queries as parallel vs sequential and handle sequential ones differently.

Caching layer: if a sub-query for "Stripe recent funding" was run 10 minutes ago, return the cached result instead of hitting Tavily again. This is the natural next step after decomposition.

## Hardest thing I attempted

Getting the decomposer to reliably distinguish compound from simple queries without over-decomposing. If every question gets split, you waste API calls on searches that don't add value. If too few questions get split, the feature doesn't fire when it should.

The hardest parts were latency and dependency detection. I solved both.

For latency: the decomposer originally made an LLM call on every query including simple ones like "Tell me about Stripe". I added a fast-path heuristic — if the question contains no compound indicators (compare, vs, and, between, etc.), skip the LLM call entirely and return the query as-is. Simple queries now have zero decomposition overhead.

For dependency detection: I added `has_dependencies` to the decomposition result. When the LLM detects that sub-questions have sequential dependencies ("What did Stripe do after their last funding round?" requires finding the funding round before asking what happened after), it flags this and the system collapses back to a single query. Our architecture only handles parallel sub-queries, so sequential ones should never be decomposed. I fully pulled this off — both the fast path and dependency collapse are tested.

## Roads not taken

Multi-hop research: the idea was to build a system where the answer to one search informs the next question, like Self-Ask but as a proper LangGraph loop. I rejected this because it's solving a different problem (compositional reasoning) than what this assistant is actually used for (company research). Adding it would complicate the graph significantly for a use case that rarely comes up in practice.

Persistent company memory: cache research findings per company so repeated queries don't re-search. I spent time thinking about this seriously and it's a genuinely useful feature. I rejected it because it introduces a staleness problem. Cached data about a company's funding round might be outdated tomorrow. Solving staleness correctly would take more time than I had and a half-baked implementation would be worse than no cache at all.

Iterative query refinement: after getting initial results, have the system evaluate whether they actually answered the question and reformulate if not. I rejected this because it's architecturally similar to the existing validation loop. The validator already does a version of this. Adding a second refinement loop before the validator would be redundant and would increase latency on every query, not just bad ones.

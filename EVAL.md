# Evaluation

## Running the tests

```bash
python -m pytest tests/ -v
```

22 tests, under 1 second, no API keys needed, no network calls.

---

## Test files

| File | Focus | Tests |
|------|-------|-------|
| `tests/test_smoke.py` | Routing correctness, infrastructure | 15 |
| `tests/test_eval.py` | Labeled test set, groundedness, cost, latency | 7 |

---

## Labeled test set

### Routing correctness (routing_cases)

Every meaningful confidence score is covered, including the boundary case that caught Bug 7:

| Score | Expected route | Label |
|-------|---------------|-------|
| 9 | synthesis | High confidence skips validator |
| 7 | synthesis | Above threshold skips validator |
| 6 | synthesis | Exactly at threshold → synthesis (the `>=` fix) |
| 5 | validator | Below threshold needs validation |
| 1 | validator | Very low confidence needs validation |
| 0 | validator | Zero confidence needs validation |

### Validation loop correctness (validation_cases)

| Result | Attempts | Expected | Label |
|--------|----------|----------|-------|
| sufficient | 1 | synthesis | Sufficient always exits |
| sufficient | 3 | synthesis | Sufficient exits even at max |
| insufficient | 1 | research | Below max retries |
| insufficient | 2 | research | One below max retries |
| insufficient | 3 | synthesis | At max exits loop — the infinite loop fix |

---

## Groundedness

`test_synthesis_uses_findings_not_raw_research` patches the synthesis LLM call and inspects what gets passed into the prompt. It asserts:
- The processed `findings` are present in the system prompt
- The raw web scrape (`raw_research`) is not — confirming Bug 6 stays fixed

---

## Cost

Two cost tests check the token budget guard end-to-end:
- Short queries pass the guard cleanly
- Queries over budget return a graceful error response with `status: "error"`
- Token estimation formula (`len(content) // 4`) is verified directly

---

## Latency

`test_mock_search_latency` asserts mock search completes in under 100ms — confirming no network call fires when `MOCK_MODE=true`.

Full end-to-end latency depends on live API conditions and is better tracked through the structured logs than pinned in tests.

---

## What's not in the test suite (and why)

**Live groundedness / hallucination** — `test_groundedness_answer_references_mock_findings` covers groundedness in mock mode by verifying the answer references content from the findings. For live LLM output, full verification requires real API calls which are non-deterministic and expensive for CI. The right approach for production is a periodic offline eval against a golden set with semantic similarity scoring.

**Live latency and cost** — these fluctuate with API conditions. Tracked via observability logs, not tests.

---

## Regression coverage

| Bug fixed | Test that prevents regression |
|-----------|-------------------------------|
| Bug 2 — infinite validation loop | `test_max_attempts_reached_routes_to_synthesis` |
| Bug 6 — synthesis uses raw scrape | `test_synthesis_uses_findings_not_raw_research` |
| Bug 7 — off-by-one threshold | `test_confidence_at_threshold_routes_to_synthesis` |
| Token budget guard | `test_token_budget_guard_fires` |

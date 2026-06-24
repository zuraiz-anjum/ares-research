# Evaluation

## Running the tests

```bash
python -m pytest tests/ -v
```

That's it. 15 tests, under 1 second, no API keys needed, no network calls.

---

## What's being tested

The most important thing to get right in this system is routing — which agent runs next determines everything else. I focused the test suite on routing correctness, including the boundary cases that actually caught real bugs in this codebase.

### Routing tests

| Test | What it's checking |
|------|----------------|
| High confidence → synthesis | Score above threshold skips the validator |
| Low confidence → validator | Score below threshold gets validated first |
| Score exactly at threshold → synthesis | The `>=` fix — a score of 6 should pass, not get sent to validator |
| One below threshold → validator | Just below the line still validates |
| Zero confidence → validator | Lowest possible score still routes correctly |
| Max confidence → synthesis | Highest possible score routes correctly |
| Sufficient findings → synthesis | Good research proceeds to answer |
| Insufficient findings → research | Bad research loops back for another attempt |
| Max attempts reached → synthesis | Loop actually exits when it should — the infinite loop fix |
| One below max → research | Loop keeps going until max is genuinely hit |
| Sufficient at max attempts → synthesis | Sufficient always wins regardless of attempt count |

### Infrastructure tests

| Test | What it's checking |
|------|----------------|
| Graph compiles | The whole wiring comes together with no import errors |
| Mock mode works | Tavily returns fake data — no real API call fired |
| Token budget check | Short messages pass the guard correctly |
| Token estimation | Character // 4 approximation is correct |

---

## What I didn't test (and why)

**Groundedness / hallucination** — checking whether the final answer is actually grounded in the search results means running the full graph with a real LLM. That's non-deterministic, expensive, and too slow for a per-commit test. The right home for that is a periodic offline eval against a labeled golden set, not a test suite.

**Latency and cost** — these fluctuate with live API conditions. Better tracked through the observability logs than pinned in tests.

**Full end-to-end conversation** — manual testing covers this. The routing tests already give strong confidence that the wiring is correct.

---

## Why this catches real bugs

Two of the bugs found in this codebase would have been caught immediately by these tests:

- The off-by-one threshold bug (Bug 7) would have failed `test_confidence_at_threshold_routes_to_synthesis`
- The infinite validation loop (Bug 2) would have failed `test_max_attempts_reached_routes_to_synthesis`

Both are now regression-proof.

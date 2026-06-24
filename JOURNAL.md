# Journal

## Learning log

The biggest thing I hadn't touched before this project was LangGraph's interrupt/resume pattern. I understood multi-agent graphs conceptually but the human-in-the-loop mechanism was genuinely new.

What confused me at first was thinking of `interrupt()` like a regular function call that pauses and waits. It's not. It freezes the entire graph execution mid-node, saves the checkpoint, and surfaces a signal to the caller. The graph is literally suspended at that line. To resume it you don't call the function again with new arguments. You pass `Command(resume=value)` which injects the value as the return of the `interrupt()` call and continues from exactly where it paused.

The moment this clicked for me was when I saw the bug live. I sent "what is the company doing?", it asked which company, I answered "Stripe", and it asked the same question again. I knew immediately that the graph was restarting instead of resuming. But understanding WHY that happened required me to really understand what checkpointing means in LangGraph. Calling `graph.invoke(regular_dict, config)` is treated as a new execution, not a resume. The checkpoint captures the frozen state including where execution paused. `Command(resume=...)` is the only way to wake up that specific frozen execution.

The bug detection side of the project also taught me a lot. The bugs weren't obvious crashes. Most were silent failures. The validator's attempts counter mutating state directly and being silently discarded, the synthesis agent preferring raw web scrape over processed findings because of Python's `or` short-circuit evaluation. Finding these required reading the code carefully and thinking about what LangGraph actually does under the hood, not just what the code looks like it's doing.

## AI collaboration and overrides

AI helped in several ways. It explained the LangGraph interrupt/resume pattern with the `input()` analogy which was the explanation that made it click for me. It spotted that `synthesis.py` used `raw_research or findings` instead of `findings or raw_research` which I had missed. It wrote the tenacity retry decorator correctly on the first attempt since I didn't know the library at all. 

Where I overrode it:

When adding mock mode to `search.py`, Claude added the `@retry` decorator above the `MOCK_RESULTS` constant, not just above the function. A decorator on a list literal doesn't make sense and Python would throw an error. I caught it and corrected the placement.

When fixing the resume endpoint, Claude wrote `from langchain_protocol import Command` which is not a real module. The correct import is `from langgraph.types import Command`. The app would have crashed on startup with an ImportError. I caught it before running the server and corrected it. 

Claude also had path errors at one point, attempting to write a file to `app/.env.example` when the file is in the root directory, not inside `app/`. Small thing but it would have broken the commit if I hadn't caught it.

## Honest self-assessment

What's solid: the bug fixes are thorough and well-reasoned. I understand every one of them well enough to explain live on camera, not just what the fix is, but why the original code was wrong at a conceptual level. The eval harness covers routing correctness including the boundary cases that caught two of the bugs. The production hardening (retries, timeout, token budget, bounded context, logging) addresses real failure modes, not just checkboxes.

What's half-baked: the mock mode only mocks Tavily. The LLM still makes real API calls even when `MOCK_MODE=true`. I tried to mock the LLM with `FakeListChatModel` and it doesn't support `.with_structured_output()` which all four agents use. The workaround is fine for the test suite but the mock mode is not a complete isolation layer. I'd fix this by writing a custom fake LLM class that properly handles structured output. The decomposer edge cases are also not fully solved. For queries with hidden sequential dependencies, the decomposer might still mis-classify.

What I'd never ship as-is: the mock mode only partially isolates external dependencies. Tavily is mocked but the LLM still makes real API calls. For a proper test suite in CI, you'd want full isolation. I'd fix this by writing a custom LLM wrapper that handles `.with_structured_output()` and returns deterministic responses for each agent type.

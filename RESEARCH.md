# Research Brief — Parallel Query Decomposition for Multi-Agent Research Assistants

## The problem I was trying to solve

When I was testing the baseline system, I noticed something pretty obvious. For example, if you ask something like "Compare Stripe and Brex's recent funding", the system fires off one search query and gets back a messy mix of results about both companies. The LLM then has to make sense of all that noise. The answer you get is vague at best.

The fix seemed obvious to me: split the question into two focused searches, run them, and combine the results. But before building anything I wanted to see what research already exists on this and whether my approach was actually sound.

---

## What I read

### 1. Least-to-Most Prompting — Zhou et al. (2022)
This paper tackles the problem of complex reasoning in LLMs. Their argument is that models are bad at answering multi-step questions in one shot, but pretty good at each individual step if you isolate it. So they break a complex question into sub-problems and solve them one by one , each answer feeding into the next question.

**Citation:** Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., & Chi, E. (2022). *Least-to-Most Prompting Enables Complex Reasoning in Large Language Models.* arXiv preprint arXiv:2205.10625. https://arxiv.org/abs/2205.10625

### 2. Self-Ask — Press et al. (2022)
Self-Ask is similar but adds web search into the loop. The model asks itself follow-up questions, searches for each answer, then uses those answers to get to the final answer. It's sequential — question 1 gets answered, that answer shapes question 2, and so on.

**Citation:** Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N. A., & Lewis, M. (2022). *Measuring and Narrowing the Compositionality Gap in Language Models.* Findings of EMNLP 2023. ACL Anthology. https://aclanthology.org/2023.findings-emnlp.378

### 3. Demonstrate-Search-Predict — Khattab et al. (2022)
This paper proposes composing retrieval and LLMs for knowledge-intensive tasks. The key idea is a pipeline where you demonstrate what good retrieval looks like, search using that context, then predict the answer. It's one of the more direct pieces of work on multi-step retrieval with LLMs and it shows the same sequential assumption — each retrieval step builds on the last.

**Citation:** Khattab, O., Santhanam, K., Li, X. L., Hall, D., Liang, P., Potts, C., & Zaharia, M. (2022). *Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive NLP.* arXiv preprint arXiv:2212.14024. https://arxiv.org/pdf/2212.14024

### 4. LangChain MultiQueryRetriever
LangChain built a production component that generates multiple query phrasings from a single user question and retrieves results for all of them. The idea is that one phrasing might miss documents that a different phrasing would catch. It generates the queries upfront rather than sequentially.

**Citation:** LangChain. *MultiQueryRetriever.* LangChain Reference Documentation. https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever. Accessed June 2026.

---

## What I agree with

All four sources agree on the same fundamental point, one query is not always enough for a complex question. You get better results when you break things down. I saw this myself while testing, so the research lines up with what I observed.

I also think LangChain's decision to generate all queries upfront (rather than one at a time) is the right call. It's cleaner and faster.

---

## What I disagree with

Least-to-Most, Self-Ask, and Demonstrate-Search-Predict all assume you have to solve sub-problems **in order** , the answer to question 1 determines what question 2 even is. That makes sense for pure reasoning tasks where there's a dependency chain.

But that's not how company research questions work. "Compare Stripe and Brex's funding" breaks into:
- "Stripe recent funding"
- "Brex recent funding"

These have nothing to do with each other. You don't need to know Stripe's funding before you can search for Brex's. Running them one after the other is just wasting time.

LangChain's MultiQueryRetriever looks similar on the surface but solves a fundamentally different problem as it rephrases one query multiple ways to improve recall from a single knowledge source. My use case is genuine decomposition: the sub-queries retrieve completely different information from different searches. The distinction matters architecturally — rephrasing is about search coverage, decomposition is about query planning.

---

## My point of view

I studied parallel and distributed computing recently, and when I looked at this problem I immediately thought about **Embarrassingly parallel tasks** — tasks with no data dependency between them that should never be serialised. Serialising them doesn't make the result more correct, it just makes it slower.

Company research sub-queries are almost always embarrassingly parallel. "Compare X and Y" is two independent searches. "What are A, B, and C doing?" is three. There's no reason to wait for one before starting the other.

So my approach is:
1. Look at the question and decide if it's simple (one search) or compound (multiple independent searches)
2. If compound, generate all sub-queries upfront
3. Run them all in parallel
4. Merge the findings before synthesis

This is different from what the papers do. They both assume sequential dependencies. My use case doesn't have those dependencies, so I shouldn't pay the latency cost of serialisation. That's the core of my argument, and it came directly from applying a distributed systems concept to an LLM problem.

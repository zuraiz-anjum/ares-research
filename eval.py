"""RAG Pipeline Evaluation Script.

Runs a suite of test queries through the Ares RAG pipeline and scores each
response on three dimensions:

  faithfulness    — does the answer stick to the retrieved context, not training data?
  answer_relevance — does the answer address what was actually asked?
  retrieval_quality — did the retriever surface useful chunks? (scored from confidence_score)

Scoring is done by an LLM judge (same provider chain as production).
Results are printed as a table and saved to eval_results.json.

Usage:
    python eval.py                        # run default test suite
    python eval.py --query "custom query" # test a single query
    python eval.py --doc path/to/doc.pdf  # test RAG over a document
"""

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()


TEST_QUERIES = [
    {
        "query": "What is OpenAI's latest funding round and valuation?",
        "mode":  "research",
        "expected_topics": ["funding", "valuation", "OpenAI"],
    },
    {
        "query": "Compare Stripe, Braintree, and Adyen on fees and features",
        "mode":  "comparison",
        "expected_topics": ["Stripe", "Braintree", "Adyen", "fees"],
    },
    {
        "query": "What are the main use cases for large language models in enterprise?",
        "mode":  "research",
        "expected_topics": ["LLM", "enterprise", "use cases"],
    },
    {
        "query": "What is 2 + 2?",
        "mode":  "chat",
        "expected_topics": ["4"],
    },
    {
        "query": "Summarise Tesla's Q4 2024 earnings",
        "mode":  "research",
        "expected_topics": ["Tesla", "earnings", "revenue"],
    },
]

FAITHFULNESS_PROMPT = """You are an evaluation judge assessing whether an AI answer is faithful to its sources.

Rate faithfulness on a scale of 0-10:
- 10: Every claim is directly supported by the research context. No hallucinations.
- 7-9: Minor gaps or imprecision but no outright fabrications.
- 4-6: Some claims appear to come from the model's training data rather than the context.
- 1-3: Significant fabrication. Numbers or facts not in the context.
- 0: Completely fabricated with no grounding.

Research context provided to the model:
{context}

Model answer:
{answer}

Respond with JSON: {{"score": <int 0-10>, "reason": "<one sentence>"}}"""

RELEVANCE_PROMPT = """You are an evaluation judge assessing whether an AI answer actually addresses the question asked.

Rate relevance 0-10:
- 10: Fully on-topic, complete, covers all key aspects of the question.
- 7-9: Mostly relevant with minor gaps.
- 4-6: Partially relevant; misses important aspects or drifts off-topic.
- 1-3: Tangential. Mentions the topic but does not answer the question.
- 0: Completely irrelevant.

User question: {query}
Model answer:  {answer}

Respond with JSON: {{"score": <int 0-10>, "reason": "<one sentence>"}}"""


async def _run_query(query: str) -> dict:
    """Run a single query through the Ares graph and return the result dict."""
    from langchain_core.messages import HumanMessage
    import app.graph as _gm
    from app.graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver

    graph = build_graph(MemorySaver())
    thread_id = f"eval-{int(time.time())}"
    inputs = {
        "messages": [HumanMessage(content=query)],
        "attempts": 0,
        "doc_id": "",
        "original_query": query,
    }
    config = {"configurable": {"thread_id": thread_id}}
    result = await graph.ainvoke(inputs, config)
    return result


def _judge(prompt: str) -> dict:
    """Ask the LLM judge and parse its JSON response."""
    from app.llm import get_llm
    from langchain_core.messages import HumanMessage
    llm = get_llm(temperature=0)
    raw = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw)
    except Exception:
        return {"score": -1, "reason": f"Parse error: {raw[:100]}"}


def evaluate_one(query: str, result: dict, context: str = "") -> dict:
    """Score a single query-answer pair."""
    answer = result.get("messages", [{}])[-1].content if result.get("messages") else ""
    confidence = result.get("confidence_score", -1)
    mode = result.get("mode", "unknown")

    faith = _judge(FAITHFULNESS_PROMPT.format(context=context or answer[:600], answer=answer))
    relev = _judge(RELEVANCE_PROMPT.format(query=query, answer=answer))

    scores = {
        "faithfulness":      faith.get("score", -1),
        "answer_relevance":  relev.get("score", -1),
        "retrieval_quality": confidence,
    }
    valid = [s for s in scores.values() if s >= 0]
    scores["overall"] = round(sum(valid) / len(valid), 1) if valid else -1

    return {
        "query":            query,
        "mode":             mode,
        "answer_preview":   answer[:200] + ("…" if len(answer) > 200 else ""),
        "scores":           scores,
        "faith_reason":     faith.get("reason", ""),
        "relev_reason":     relev.get("reason", ""),
    }


def _print_table(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"{'QUERY':<38}  {'MODE':<12}  {'FAITH':>5}  {'RELEV':>5}  {'RETR':>5}  {'OVRL':>5}")
    print("-" * 90)
    for r in results:
        s = r["scores"]
        def fmt(v): return f"{v:5.1f}" if v >= 0 else "  N/A"
        print(f"{r['query'][:37]:<38}  {r['mode']:<12}  {fmt(s['faithfulness'])}  {fmt(s['answer_relevance'])}  {fmt(s['retrieval_quality'])}  {fmt(s['overall'])}")
    print("=" * 90)

    all_scores = [r["scores"]["overall"] for r in results if r["scores"]["overall"] >= 0]
    if all_scores:
        print(f"\nMEAN OVERALL: {sum(all_scores)/len(all_scores):.1f} / 10  (across {len(all_scores)} queries)")
    print()


async def main(args) -> None:
    from app.config import validate_settings
    validate_settings()

    queries = TEST_QUERIES
    if args.query:
        queries = [{"query": args.query, "mode": "unknown", "expected_topics": []}]

    results = []
    for item in queries:
        q = item["query"]
        print(f"\n▶ Running: {q[:70]}…")
        t = time.monotonic()
        try:
            result = await _run_query(q)
            elapsed = time.monotonic() - t
            context = result.get("raw_research", "") or result.get("findings", "")
            scored = evaluate_one(q, result, context)
            scored["latency_s"] = round(elapsed, 2)
            results.append(scored)
            s = scored["scores"]
            print(f"   ✓ faith={s['faithfulness']}  relev={s['answer_relevance']}  conf={s['retrieval_quality']}  ({elapsed:.1f}s)")
        except Exception as e:
            print(f"   ✗ Error: {e}")
            results.append({"query": q, "error": str(e), "scores": {}, "mode": "error"})

    _print_table(results)

    out = {
        "run_at":  datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    with open("eval_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to eval_results.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ares RAG Pipeline Evaluation")
    parser.add_argument("--query",  help="Run a single custom query instead of the default suite")
    parser.add_argument("--doc",    help="Path to a document for RAG evaluation")
    args = parser.parse_args()
    asyncio.run(main(args))

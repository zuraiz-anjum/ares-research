"""FastAPI application — Ares: Autonomous Research & Evidence System.

  POST /chat          -> start (or continue) a conversation turn (non-streaming)
  POST /resume        -> answer a clarifying question and continue
  GET  /chat/stream   -> SSE stream: node progress + synthesis tokens in real time
  GET  /history       -> recent session list (last 50 queries)
  GET  /              -> the chat UI
"""

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from langgraph.errors import GraphInterrupt

from app.config import settings, validate_settings
from app.graph import graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# In-memory session history (last 50 queries, survives only while server runs)
_history: deque = deque(maxlen=50)


@asynccontextmanager
async def lifespan(_: FastAPI):
    import os
    os.makedirs("static/charts", exist_ok=True)
    os.makedirs("static/reports", exist_ok=True)
    validate_settings()
    logger.info("startup validation passed")
    yield


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Ares — Autonomous Research & Evidence System", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_mode": settings.mock_mode}


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    clarification: str


def _budget_error(inputs: dict, thread_id: str) -> dict | None:
    messages = inputs.get("messages", [])
    total_tokens = sum(len(m.content) for m in messages) // 4
    if total_tokens > settings.max_token_budget:
        logging.warning(f"token_exceeded thread_id={thread_id} estimated_tokens={total_tokens}")
        return {
            "status": "error",
            "answer": "Your query is too long. Please shorten the message.",
            "thread_id": thread_id,
        }
    return None


async def _run_async(inputs: dict | Command, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    if isinstance(inputs, dict):
        err = _budget_error(inputs, thread_id)
        if err:
            return err
    try:
        # ainvoke instead of asyncio.to_thread(graph.invoke) because several
        # nodes (synthesis, report_writer, etc.) are async functions.
        result = await graph.ainvoke(inputs, config)
    except GraphInterrupt as exc:
        question = "Could you clarify your request?"
        if exc.args:
            val = exc.args[0]
            if isinstance(val, (list, tuple)) and val:
                try:
                    question = val[0].value.get("question", question)
                except Exception:
                    pass
        logger.info(f"clarification_needed thread={thread_id}")
        return {"status": "needs_clarification", "question": question, "thread_id": thread_id}
    except Exception:
        logger.exception(f"graph_error thread={thread_id}")
        return {"status": "error", "answer": "An internal error occurred. Please try again.", "thread_id": thread_id}

    # Fallback for older LangGraph versions that return __interrupt__ in the dict.
    if isinstance(result, dict) and result.get("__interrupt__"):
        interrupt = result["__interrupt__"][0]
        question = interrupt.value.get("question", "Could you clarify your request?")
        logger.info(f"clarification_needed thread={thread_id}")
        return {"status": "needs_clarification", "question": question, "thread_id": thread_id}

    answer = result["messages"][-1].content
    sub_queries = result.get("sub_queries", [])
    critique = result.get("critique", "")
    source_url = result.get("source_url", "")
    suggestions = result.get("suggestions", [])
    logger.info(f"request_complete thread={thread_id}")
    return {"status": "complete", "answer": answer, "thread_id": thread_id, "sub_queries": sub_queries, "critique": critique, "source_url": source_url, "suggestions": suggestions}


@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest) -> dict:
    thread_id = req.thread_id or str(uuid.uuid4())
    inputs = {"messages": [HumanMessage(content=req.message)], "attempts": 0}
    return await _run_async(inputs, thread_id)


@app.post("/resume")
@limiter.limit("20/minute")
async def resume(request: Request, req: ResumeRequest) -> dict:
    return await _run_async(Command(resume=req.clarification), req.thread_id)


@app.get("/history")
async def history() -> dict:
    return {"sessions": list(_history)}


# Graph nodes we want to surface as pipeline-step events in the SSE stream.
_GRAPH_NODES = {
    "clarity", "intent_router", "planner", "decomposer", "research", "validator",
    "doc_agent", "data_analyst", "debate_writer", "synthesis", "report_writer",
    "pdf_generator", "chart_writer", "code_writer", "fact_checker", "critic", "suggestions",
}
_STREAMING_NODES = {"synthesis", "report_writer", "data_analyst", "code_writer", "debate_writer"}


@app.get("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    message: str,
    thread_id: str | None = None,
) -> StreamingResponse:
    _thread_id = thread_id or str(uuid.uuid4())

    async def generate():
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        if len(message) // 4 > settings.max_token_budget:
            yield sse({"type": "error", "message": "Your query is too long. Please shorten it."})
            return

        # Fast mock path: simulate events with small delays so the UI is testable
        # without live API keys.
        if settings.mock_mode:
            for node in ("clarity", "decomposer", "research", "synthesis"):
                yield sse({"type": "node_start", "node": node})
                await asyncio.sleep(0.3)
                if node == "decomposer":
                    yield sse({"type": "decomposed", "queries": [message]})
            mock = "Mock answer: The company has strong revenue growth of 30% YoY with a recent $500 M funding round."
            for word in mock.split():
                yield sse({"type": "token", "content": word + " "})
                await asyncio.sleep(0.04)
            yield sse({"type": "complete", "answer": mock, "thread_id": _thread_id, "sub_queries": [message]})
            return

        inputs = {"messages": [HumanMessage(content=message)], "attempts": 0}
        config = {"configurable": {"thread_id": _thread_id}}

        full_answer = ""
        sub_queries: list[str] = []
        critique = ""
        suggestions: list[str] = []
        fact_check_results: list = []
        plan_steps: list[str] = []
        chart_url = ""
        pdf_url = ""
        detected_mode = "research"
        final_output: dict = {}
        t_start = time.monotonic()

        try:
            async for event in graph.astream_events(inputs, config, version="v2"):
                kind = event["event"]
                name = event.get("name", "")
                meta = event.get("metadata", {})

                if kind == "on_chain_start" and name in _GRAPH_NODES:
                    if meta.get("langgraph_node") == name:
                        yield sse({"type": "node_start", "node": name})

                elif kind == "on_chain_end":
                    node = meta.get("langgraph_node")
                    out = event.get("data", {}).get("output", {}) or {}
                    if name == "decomposer" and node == "decomposer":
                        sub_queries = out.get("sub_queries", [])
                        if sub_queries:
                            yield sse({"type": "decomposed", "queries": sub_queries})
                    elif name == "intent_router" and node == "intent_router":
                        detected_mode = out.get("mode", "research")
                        yield sse({"type": "mode", "mode": detected_mode})
                    elif name == "critic" and node == "critic":
                        critique = out.get("critique", "")
                        if critique:
                            yield sse({"type": "critique", "content": critique})
                    elif name == "planner" and node == "planner":
                        plan_steps = out.get("plan_steps", [])
                        if plan_steps:
                            yield sse({"type": "plan_revealed", "steps": plan_steps})
                    elif name == "fact_checker" and node == "fact_checker":
                        fact_check_results = out.get("fact_check_results", [])
                        if fact_check_results:
                            yield sse({"type": "fact_check", "results": fact_check_results})
                    elif name == "chart_writer" and node == "chart_writer":
                        chart_url = out.get("chart_url", "")
                        if chart_url:
                            yield sse({"type": "chart", "url": chart_url})
                    elif name == "pdf_generator" and node == "pdf_generator":
                        pdf_url = out.get("pdf_url", "")
                        if pdf_url:
                            yield sse({"type": "pdf_ready", "url": pdf_url})
                    elif name == "suggestions" and node == "suggestions":
                        suggestions = out.get("suggestions", [])
                        if suggestions:
                            yield sse({"type": "suggestions", "questions": suggestions})
                    elif name == "LangGraph":
                        final_output = event.get("data", {}).get("output") or {}

                elif kind == "on_chat_model_stream" and meta.get("langgraph_node") in _STREAMING_NODES:
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        full_answer += chunk.content
                        yield sse({"type": "token", "content": chunk.content})

        except Exception:
            logger.exception(f"stream_error thread={_thread_id}")
            yield sse({"type": "error", "message": "An internal error occurred."})
            return

        # Check for a pending interrupt (clarification request).
        try:
            snapshot = await graph.aget_state(config)
            pending = [ipt for task in snapshot.tasks for ipt in (task.interrupts or [])]
            if pending:
                question = pending[0].value.get("question", "Could you clarify your request?")
                logger.info(f"clarification_needed thread={_thread_id}")
                yield sse({"type": "needs_clarification", "question": question, "thread_id": _thread_id})
                return
        except Exception:
            logger.warning(f"get_state_failed thread={_thread_id}")

        # Fallback: pull answer from final graph output if streaming produced nothing.
        if not full_answer and final_output.get("messages"):
            full_answer = final_output["messages"][-1].content
        if not sub_queries:
            sub_queries = final_output.get("sub_queries", [])
        if not chart_url:
            chart_url = final_output.get("chart_url", "")
        if not pdf_url:
            pdf_url = final_output.get("pdf_url", "")

        latency_ms = int((time.monotonic() - t_start) * 1000)
        # Approximate token count: total chars / 4 across all content exchanged.
        all_text = message + full_answer + critique + " ".join(sub_queries)
        token_count = len(all_text) // 4

        import app.llm as _llm_mod
        provider_names = [n for n, _ in _llm_mod._build_llms(0.2, False)]
        provider_used = provider_names[min(_llm_mod._active_idx, len(provider_names)-1)] \
                        if provider_names else "unknown"

        # Record in session history
        _history.appendleft({
            "thread_id": _thread_id,
            "query": message,
            "mode": detected_mode,
            "preview": (full_answer[:120] + "…") if len(full_answer) > 120 else full_answer,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "token_count": token_count,
            "latency_ms": latency_ms,
            "provider": provider_used,
        })

        source_url = final_output.get("source_url", "")
        logger.info(f"stream_complete thread={_thread_id} tokens≈{token_count} provider={provider_used} ms={latency_ms}")
        yield sse({
            "type": "complete",
            "answer": full_answer,
            "thread_id": _thread_id,
            "sub_queries": sub_queries,
            "critique": critique,
            "source_url": source_url,
            "suggestions": suggestions,
            "fact_check_results": fact_check_results,
            "plan_steps": plan_steps,
            "chart_url": chart_url,
            "pdf_url": pdf_url,
            "telemetry": {
                "token_count": token_count,
                "provider": provider_used,
                "latency_ms": latency_ms,
            },
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

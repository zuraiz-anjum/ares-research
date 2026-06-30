"""FastAPI application — Ares: Autonomous Research & Evidence System.

  POST /chat          -> start (or continue) a conversation turn (non-streaming)
  POST /resume        -> answer a clarifying question and continue
  GET  /chat/stream   -> SSE stream: node progress + synthesis tokens in real time
  GET  /history       -> recent session list (last 50 queries, persisted to SQLite)
  GET  /              -> the chat UI
"""

import asyncio
import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import hashlib
import secrets

import aiosqlite
from fastapi import Cookie, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pathlib import Path
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from langgraph.errors import GraphInterrupt

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings, validate_settings
import app.graph as _graph_mod
from app.graph import build_graph
from app.utils.checkpoint import set_thread_id, _TABLE_DDL, ANALYTICS_DB as _CP_ANALYTICS_DB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = "ares_history.db"

# In-memory mirror of the last 50 sessions — seeded from SQLite on startup.
_history: deque = deque(maxlen=50)


async def _init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id  TEXT,
                query      TEXT,
                mode       TEXT,
                preview    TEXT,
                timestamp  TEXT,
                token_count INTEGER DEFAULT 0,
                latency_ms  INTEGER DEFAULT 0,
                provider    TEXT DEFAULT 'unknown'
            )
        """)
        await db.commit()
        cursor = await db.execute(
            "SELECT thread_id, query, mode, preview, timestamp, "
            "token_count, latency_ms, provider "
            "FROM sessions ORDER BY id DESC LIMIT 50"
        )
        rows = await cursor.fetchall()
    # Oldest first so appendleft makes the most-recent appear at top.
    for row in reversed(rows):
        _history.appendleft({
            "thread_id":   row[0],
            "query":       row[1],
            "mode":        row[2],
            "preview":     row[3],
            "timestamp":   row[4],
            "token_count": row[5],
            "latency_ms":  row[6],
            "provider":    row[7],
        })
    logger.info(f"history_loaded count={len(rows)}")


ANALYTICS_DB = "ares_analytics.db"


async def _init_analytics_db() -> None:
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                mode       TEXT,
                token_count INTEGER DEFAULT 0,
                latency_ms  INTEGER DEFAULT 0,
                provider    TEXT DEFAULT 'unknown',
                confidence  INTEGER DEFAULT -1
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS shared_reports (
                share_id   TEXT PRIMARY KEY,
                file_path  TEXT NOT NULL,
                title      TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                query       TEXT NOT NULL,
                cron        TEXT,
                next_run_at TEXT,
                email       TEXT,
                created_at  TEXT NOT NULL,
                last_run_at TEXT
            )
        """)
        await db.execute(_TABLE_DDL)
        await db.commit()


async def _save_session(session: dict) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO sessions "
            "(thread_id, query, mode, preview, timestamp, token_count, latency_ms, provider) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session["thread_id"], session["query"],  session["mode"],
                session["preview"],   session["timestamp"], session["token_count"],
                session["latency_ms"], session["provider"],
            ),
        )
        await db.commit()


CHECKPOINT_DB = "ares_checkpoints.db"


UPLOADS_DIR = "uploads"
_ALLOWED_EXTENSIONS       = {".pdf", ".txt", ".md", ".docx"}
_ALLOWED_SURVEY_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_ALLOWED_IMAGE_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(_: FastAPI):
    import os
    os.makedirs("static/charts", exist_ok=True)
    os.makedirs("static/reports", exist_ok=True)
    os.makedirs("static/shared", exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    await _init_db()
    await _init_analytics_db()
    validate_settings()

    # Activate LangSmith tracing when key is configured.
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
        logger.info(f"langsmith_tracing_enabled project={settings.langsmith_project}")
    else:
        logger.info("langsmith_tracing_disabled (set LANGSMITH_API_KEY to enable)")

    logger.info("startup validation passed")
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        _graph_mod.graph = build_graph(checkpointer)
        logger.info(f"checkpoint_db={CHECKPOINT_DB} persistent=true")
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


# ── Auth ─────────────────────────────────────────────────────────────────────
# Simple cookie-based password gate.  Set APP_PASSWORD in .env to activate.
# When APP_PASSWORD is empty, all routes are open (dev mode).

_SESSION_COOKIE = "ares_session"
_active_tokens: set[str] = set()


def _check_auth(request: Request) -> bool:
    """Return True if auth is disabled or the request carries a valid session."""
    if not settings.app_password:
        return True
    token = request.cookies.get(_SESSION_COOKIE, "")
    return token in _active_tokens


class LoginRequest(BaseModel):
    password: str


@app.post("/auth/login")
async def login(req: LoginRequest, response: Response) -> dict:
    if not settings.app_password:
        return {"status": "ok", "auth": False}
    expected = hashlib.sha256(settings.app_password.encode()).hexdigest()
    given    = hashlib.sha256(req.password.encode()).hexdigest()
    if not secrets.compare_digest(expected, given):
        raise HTTPException(401, "Invalid password")
    token = secrets.token_hex(32)
    _active_tokens.add(token)
    response.set_cookie(
        _SESSION_COOKIE, token,
        httponly=True, samesite="strict", max_age=86400 * 7,
    )
    return {"status": "ok"}


@app.post("/auth/logout")
async def logout(response: Response, ares_session: str = Cookie(default="")) -> dict:
    _active_tokens.discard(ares_session)
    response.delete_cookie(_SESSION_COOKIE)
    return {"status": "ok"}


@app.get("/auth/status")
def auth_status(request: Request) -> dict:
    return {"required": bool(settings.app_password), "authenticated": _check_auth(request)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_mode": settings.mock_mode}


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    doc_id: str | None = None     # set when the user has uploaded a document
    is_survey: bool = False       # True when the uploaded file is CSV/Excel


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
    set_thread_id(thread_id)
    if isinstance(inputs, dict):
        err = _budget_error(inputs, thread_id)
        if err:
            return err
    try:
        result = await _graph_mod.graph.ainvoke(inputs, config)
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

    if isinstance(result, dict) and result.get("__interrupt__"):
        interrupt = result["__interrupt__"][0]
        question = interrupt.value.get("question", "Could you clarify your request?")
        logger.info(f"clarification_needed thread={thread_id}")
        return {"status": "needs_clarification", "question": question, "thread_id": thread_id}

    answer = result["messages"][-1].content
    logger.info(f"request_complete thread={thread_id}")
    return {
        "status":             "complete",
        "answer":             answer,
        "thread_id":          thread_id,
        "sub_queries":        result.get("sub_queries", []),
        "critique":           result.get("critique", ""),
        "source_url":         result.get("source_url", ""),
        "suggestions":        result.get("suggestions", []),
        "pdf_url":            result.get("pdf_url", ""),
        "chart_url":          result.get("chart_url", ""),
        "chart_urls":         result.get("chart_urls") or [],
        "fact_check_results": result.get("fact_check_results", []),
        "plan_steps":         result.get("plan_steps", []),
        "confidence_score":   result.get("confidence_score", -1),
        "sources":            result.get("sources", []),
    }


@app.post("/upload")
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict:
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")
    ext = Path(file.filename or "").suffix.lower()

    _all_allowed = _ALLOWED_EXTENSIONS | _ALLOWED_SURVEY_EXTENSIONS
    if ext not in _all_allowed:
        raise HTTPException(
            400,
            f"Unsupported type '{ext}'. Allowed: {', '.join(sorted(_all_allowed))}",
        )
    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")

    doc_id    = uuid.uuid4().hex
    save_path = os.path.join(UPLOADS_DIR, f"{doc_id}{ext}")
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    with open(save_path, "wb") as fh:
        fh.write(contents)

    # CSV / Excel → survey mode: skip RAG chunking, the survey_analyst reads
    # the file directly with pandas.
    if ext in _ALLOWED_SURVEY_EXTENSIONS:
        try:
            import pandas as pd
            df = pd.read_csv(save_path) if ext == ".csv" else pd.read_excel(save_path)
            shape = f"{df.shape[0]} rows × {df.shape[1]} columns"
            columns = list(df.columns[:20])
        except Exception as exc:
            logger.warning(f"survey_preview_failed doc_id={doc_id}: {exc}")
            shape, columns = "unknown", []
        logger.info(f"survey_upload doc_id={doc_id} filename={file.filename!r} shape={shape!r}")
        return {
            "doc_id":    doc_id,
            "filename":  file.filename,
            "is_survey": True,
            "shape":     shape,
            "columns":   columns,
        }

    # Text/PDF/DOCX → standard RAG chunking.
    try:
        from app.rag.loader import load_and_chunk
        from app.rag.store import add_document
        chunks = load_and_chunk(save_path, file.filename)
        n      = add_document(doc_id, chunks, file.filename)
        logger.info(f"rag_upload doc_id={doc_id} filename={file.filename!r} chunks={n}")
    except Exception:
        logger.exception(f"rag_upload_failed doc_id={doc_id}")
        raise HTTPException(500, "Failed to process document. Check the file is not encrypted or empty.")

    return {"doc_id": doc_id, "filename": file.filename, "chunks": n, "is_survey": False}


@app.post("/upload/image")
@limiter.limit("10/minute")
async def upload_image(request: Request, file: UploadFile = File(...)) -> dict:
    """Upload an image for vision-model analysis in the next query."""
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(400, f"Unsupported image type '{ext}'. Allowed: {', '.join(_ALLOWED_IMAGE_EXTENSIONS)}")
    contents = await file.read()
    if len(contents) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")

    doc_id = uuid.uuid4().hex
    save_path = os.path.join(UPLOADS_DIR, f"{doc_id}{ext}")
    with open(save_path, "wb") as fh:
        fh.write(contents)

    # Convert image to text description via vision model and store as RAG chunks
    try:
        import base64
        from langchain_core.messages import HumanMessage as _HM
        from app.llm import get_llm
        b64 = base64.b64encode(contents).decode()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/png")
        vision_msg = _HM(content=[
            {"type": "text", "text": "Describe this image in full detail, including any text, charts, tables, numbers, logos, or important visual elements. Be thorough and specific."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ])
        # Use Gemini for vision (best free vision model available)
        from app.config import settings as _s
        if _s.gemini_api_key:
            from langchain_google_genai import ChatGoogleGenerativeAI
            vision_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=_s.gemini_api_key)
        elif _s.openrouter_api_key:
            from langchain_openai import ChatOpenAI
            vision_llm = ChatOpenAI(model="google/gemini-2.0-flash-exp:free", api_key=_s.openrouter_api_key, base_url="https://openrouter.ai/api/v1")
        else:
            raise ValueError("No vision-capable provider configured (need GEMINI_API_KEY or OPENROUTER_API_KEY)")

        description = (await vision_llm.ainvoke([vision_msg])).content
        logger.info(f"image_described doc_id={doc_id} chars={len(description)}")

        # Store description as a searchable RAG chunk so doc_agent can retrieve it
        from app.rag.loader import load_and_chunk as _lac
        from app.rag.store import add_document as _add
        image_text = f"[Image: {file.filename}]\n\n{description}"
        chunks = [image_text[i:i+800] for i in range(0, len(image_text), 800)]
        n = _add(doc_id, chunks, file.filename)
        logger.info(f"image_rag_stored doc_id={doc_id} chunks={n}")
    except Exception:
        logger.exception(f"image_analysis_failed doc_id={doc_id}")
        raise HTTPException(500, "Failed to analyse image. Ensure a vision-capable API key is configured.")

    return {"doc_id": doc_id, "filename": file.filename, "chunks": n, "type": "image"}


@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request, req: ChatRequest) -> dict:
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")
    thread_id = req.thread_id or str(uuid.uuid4())
    inputs: dict = {
        "messages":          [HumanMessage(content=req.message)],
        "attempts":          0,
        "doc_id":            req.doc_id or "",
        "is_survey":         req.is_survey,
        "original_query":    req.message,
        "revision_count":    0,
        "revision_feedback": "",
        "draft_score":       0,
    }
    return await _run_async(inputs, thread_id)


@app.post("/resume")
@limiter.limit("20/minute")
async def resume(request: Request, req: ResumeRequest) -> dict:
    return await _run_async(Command(resume=req.clarification), req.thread_id)


@app.get("/history")
async def history() -> dict:
    return {"sessions": list(_history)}


# Graph nodes to surface as pipeline-step events in the SSE stream.
_GRAPH_NODES = {
    "clarity", "intent_router", "planner", "decomposer", "research", "validator",
    "doc_agent", "data_analyst", "comparison_matrix", "debate_writer", "synthesis",
    "report_writer", "draft_critic", "academic_writer", "pdf_generator",
    "chart_writer", "code_writer", "email_drafter",
    "fact_checker", "critic", "suggestions",
}
_STREAMING_NODES = {
    "synthesis", "report_writer", "data_analyst", "code_writer",
    "debate_writer", "comparison_matrix", "email_drafter",
}


@app.get("/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(
    request: Request,
    message: str,
    thread_id: str | None = None,
    doc_id: str | None = None,
    is_survey: bool = False,
) -> StreamingResponse:
    if not _check_auth(request):
        async def _deny():
            yield f"data: {json.dumps({'type':'error','message':'Authentication required'})}\n\n"
        return StreamingResponse(_deny(), media_type="text/event-stream")

    _thread_id = thread_id or str(uuid.uuid4())

    async def generate():
        def sse(data: dict) -> str:
            return f"data: {json.dumps(data)}\n\n"

        if len(message) // 4 > settings.max_token_budget:
            yield sse({"type": "error", "message": "Your query is too long. Please shorten it."})
            return

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

        inputs: dict = {
            "messages":          [HumanMessage(content=message)],
            "attempts":          0,
            "doc_id":            doc_id or "",
            "is_survey":         is_survey,
            "original_query":    message,
            "revision_count":    0,
            "revision_feedback": "",
            "draft_score":       0,
        }
        config = {"configurable": {"thread_id": _thread_id}}
        set_thread_id(_thread_id)

        full_answer = ""
        sub_queries: list[str] = []
        critique = ""
        suggestions: list[str] = []
        sources: list[dict] = []
        fact_check_results: list = []
        plan_steps: list[str] = []
        chart_url  = ""
        chart_urls: list[str] = []
        pdf_url    = ""
        detected_mode = "research"
        confidence_score: int = -1
        final_output: dict = {}
        t_start = time.monotonic()

        try:
            async for event in _graph_mod.graph.astream_events(inputs, config, version="v2"):
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
                    elif name == "research" and node == "research":
                        sources = out.get("sources", [])
                        if sources:
                            yield sse({"type": "sources", "items": sources})
                        _conf = out.get("confidence_score")
                        if _conf is not None:
                            confidence_score = _conf
                            yield sse({"type": "confidence", "score": confidence_score})
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
                    elif name == "draft_critic" and node == "draft_critic":
                        d_score    = out.get("draft_score", -1)
                        d_feedback = out.get("revision_feedback", "")
                        d_revision = out.get("revision_count", 0)
                        if d_feedback:
                            yield sse({"type": "revision", "round": d_revision,
                                       "score": d_score, "feedback": d_feedback[:300]})
                        else:
                            yield sse({"type": "draft_accepted", "score": d_score})
                    elif name == "data_visualizer" and node == "data_visualizer":
                        chart_urls = out.get("chart_urls") or []
                        chart_url  = chart_urls[0] if chart_urls else ""
                        if chart_urls:
                            yield sse({"type": "chart", "urls": chart_urls, "url": chart_url})
                    elif name == "survey_analyst" and node == "survey_analyst":
                        chart_urls = out.get("chart_urls") or []
                        chart_url  = chart_urls[0] if chart_urls else ""
                        if chart_urls:
                            yield sse({"type": "chart", "urls": chart_urls, "url": chart_url})
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
            snapshot = await _graph_mod.graph.aget_state(config)
            pending = [ipt for task in snapshot.tasks for ipt in (task.interrupts or [])]
            if pending:
                val = pending[0].value
                logger.info(f"clarification_needed thread={_thread_id}")
                if isinstance(val, dict) and val.get("type") == "academic_mcq":
                    yield sse({
                        "type": "needs_clarification",
                        "question_type": "mcq",
                        "questions": val["questions"],
                        "thread_id": _thread_id,
                    })
                else:
                    question = val.get("question", "Could you clarify your request?") if isinstance(val, dict) else str(val)
                    yield sse({"type": "needs_clarification", "question_type": "text", "question": question, "thread_id": _thread_id})
                return
        except Exception:
            logger.warning(f"get_state_failed thread={_thread_id}")

        # Fallback: pull answer from final graph output if streaming produced nothing.
        if not full_answer and final_output.get("messages"):
            full_answer = final_output["messages"][-1].content
        if not sub_queries:
            sub_queries = final_output.get("sub_queries", [])
        if not sources:
            sources = final_output.get("sources", [])
        if not chart_urls:
            chart_urls = final_output.get("chart_urls") or []
        if not chart_url:
            chart_url = chart_urls[0] if chart_urls else final_output.get("chart_url", "")
        if not pdf_url:
            pdf_url = final_output.get("pdf_url", "")

        latency_ms = int((time.monotonic() - t_start) * 1000)
        all_text = message + full_answer + critique + " ".join(sub_queries)
        token_count = len(all_text) // 4

        import app.llm as _llm_mod
        provider_names = [n for n, _ in _llm_mod._build_llms(0.2, False)]
        provider_used = (
            provider_names[min(_llm_mod._active_idx, len(provider_names) - 1)]
            if provider_names else "unknown"
        )

        session = {
            "thread_id":   _thread_id,
            "query":       message,
            "mode":        detected_mode,
            "preview":     (full_answer[:120] + "…") if len(full_answer) > 120 else full_answer,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "token_count": token_count,
            "latency_ms":  latency_ms,
            "provider":    provider_used,
        }
        _history.appendleft(session)
        asyncio.create_task(_save_session(session))
        asyncio.create_task(_save_usage(detected_mode, token_count, latency_ms, provider_used, confidence_score))

        # Fire Slack webhook in background if configured and report-worthy.
        if settings.slack_webhook_url and detected_mode in ("research", "report", "pdf", "comparison"):
            asyncio.create_task(_notify_slack(message, full_answer, pdf_url, detected_mode))

        source_url = final_output.get("source_url", "")
        logger.info(
            f"stream_complete thread={_thread_id} "
            f"tokens≈{token_count} provider={provider_used} ms={latency_ms} "
            f"confidence={confidence_score}"
        )
        yield sse({
            "type":               "complete",
            "answer":             full_answer,
            "thread_id":          _thread_id,
            "sub_queries":        sub_queries,
            "critique":           critique,
            "source_url":         source_url,
            "suggestions":        suggestions,
            "sources":            sources,
            "fact_check_results": fact_check_results,
            "plan_steps":         plan_steps,
            "chart_url":          chart_url,
            "chart_urls":         chart_urls,
            "pdf_url":            pdf_url,
            "confidence_score":   confidence_score,
            "telemetry": {
                "token_count":      token_count,
                "provider":         provider_used,
                "latency_ms":       latency_ms,
                "confidence_score": confidence_score,
            },
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


async def _save_usage(mode: str, token_count: int, latency_ms: int, provider: str, confidence: int) -> None:
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        await db.execute(
            "INSERT INTO usage (timestamp, mode, token_count, latency_ms, provider, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), mode, token_count, latency_ms, provider, confidence),
        )
        await db.commit()


async def _notify_slack(query: str, answer: str, pdf_url: str, mode: str) -> None:
    """POST a summary to the configured Slack incoming-webhook URL."""
    try:
        import aiohttp
        preview = answer[:280] + ("…" if len(answer) > 280 else "")
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f"Ares • {mode.upper()} complete"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Query:* {query}\n\n{preview}"}},
        ]
        if pdf_url:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"<{pdf_url}|Download PDF>"}})
        async with aiohttp.ClientSession() as session:
            await session.post(settings.slack_webhook_url, json={"blocks": blocks})
    except Exception:
        logger.warning("slack_webhook_failed", exc_info=True)


# ── Shareable report links ─────────────────────────────────────────────────────


class ShareRequest(BaseModel):
    file_url: str   # e.g. /static/reports/report_abc.pdf
    title: str = ""


@app.post("/share")
@limiter.limit("30/minute")
async def create_share(request: Request, req: ShareRequest) -> dict:
    """Create a public shareable link for any generated report or chart."""
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")
    # Resolve file path from the URL
    rel = req.file_url.lstrip("/")
    file_path = Path(rel)
    if not file_path.exists():
        raise HTTPException(404, "File not found")
    share_id = secrets.token_urlsafe(10)
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        await db.execute(
            "INSERT INTO shared_reports (share_id, file_path, title, created_at) VALUES (?, ?, ?, ?)",
            (share_id, str(file_path), req.title, datetime.now(timezone.utc).isoformat()),
        )
        await db.commit()
    return {"share_id": share_id, "url": f"/r/{share_id}"}


@app.get("/r/{share_id}")
async def get_shared_report(share_id: str) -> FileResponse:
    """Serve a shared report publicly (no auth required)."""
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        cursor = await db.execute(
            "SELECT file_path, title FROM shared_reports WHERE share_id = ?", (share_id,)
        )
        row = await cursor.fetchone()
    if not row:
        raise HTTPException(404, "Shared report not found or expired")
    file_path = Path(row[0])
    if not file_path.exists():
        raise HTTPException(410, "Report file no longer available")
    return FileResponse(file_path, filename=file_path.name)


# ── Usage analytics ────────────────────────────────────────────────────────────


@app.get("/analytics")
async def analytics(request: Request) -> dict:
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        # Total requests
        cur = await db.execute("SELECT COUNT(*) FROM usage")
        total = (await cur.fetchone())[0]

        # Requests by mode
        cur = await db.execute(
            "SELECT mode, COUNT(*) as cnt FROM usage GROUP BY mode ORDER BY cnt DESC"
        )
        by_mode = [{"mode": r[0], "count": r[1]} for r in await cur.fetchall()]

        # Provider distribution
        cur = await db.execute(
            "SELECT provider, COUNT(*) as cnt FROM usage GROUP BY provider ORDER BY cnt DESC"
        )
        by_provider = [{"provider": r[0], "count": r[1]} for r in await cur.fetchall()]

        # Last 7 days daily counts
        cur = await db.execute("""
            SELECT substr(timestamp, 1, 10) as day, COUNT(*) as cnt
            FROM usage
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY day ORDER BY day
        """)
        daily = [{"day": r[0], "count": r[1]} for r in await cur.fetchall()]

        # Averages
        cur = await db.execute(
            "SELECT AVG(token_count), AVG(latency_ms), AVG(CASE WHEN confidence >= 0 THEN confidence END) FROM usage"
        )
        avgs = await cur.fetchone()
        avg_tokens   = round(avgs[0] or 0, 1)
        avg_latency  = round(avgs[1] or 0, 1)
        avg_confidence = round(avgs[2] or 0, 1)

    return {
        "total_requests": total,
        "by_mode":        by_mode,
        "by_provider":    by_provider,
        "daily_last_7":   daily,
        "averages": {
            "token_count":      avg_tokens,
            "latency_ms":       avg_latency,
            "confidence_score": avg_confidence,
        },
    }


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse("static/dashboard.html")


# ── Node diagnostics ───────────────────────────────────────────────────────────


@app.get("/diagnostics")
async def diagnostics(
    request: Request,
    thread_id: str | None = None,
    node: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> dict:
    """Return node checkpoint records for diagnosing pipeline failures.

    Query params (all optional):
      thread_id — filter to a specific conversation thread
      node      — filter to a specific node name (e.g. "research")
      status    — filter by status: started | completed | failed | interrupted
      limit     — max rows to return (default 100, max 500)
    """
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")

    limit = min(limit, 500)
    conditions = []
    params: list = []
    if thread_id:
        conditions.append("thread_id = ?")
        params.append(thread_id)
    if node:
        conditions.append("node = ?")
        params.append(node)
    if status:
        conditions.append("status = ?")
        params.append(status)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    async with aiosqlite.connect(ANALYTICS_DB) as db:
        cur = await db.execute(
            f"SELECT id, thread_id, node, status, input_summary, output_summary, "
            f"error, duration_ms, timestamp "
            f"FROM node_checkpoints {where} ORDER BY id DESC LIMIT ?",
            params,
        )
        rows = await cur.fetchall()

        # Aggregate: failure counts per node
        cur2 = await db.execute(
            "SELECT node, status, COUNT(*) FROM node_checkpoints GROUP BY node, status ORDER BY node"
        )
        agg_rows = await cur2.fetchall()

    checkpoints = []
    for r in rows:
        entry: dict = {
            "id":          r[0],
            "thread_id":   r[1],
            "node":        r[2],
            "status":      r[3],
            "duration_ms": r[7],
            "timestamp":   r[8],
        }
        if r[4]:
            try:
                entry["input_summary"] = json.loads(r[4])
            except Exception:
                entry["input_summary"] = r[4]
        if r[5]:
            try:
                entry["output_summary"] = json.loads(r[5])
            except Exception:
                entry["output_summary"] = r[5]
        if r[6]:
            entry["error"] = r[6]
        checkpoints.append(entry)

    # Build node health summary
    node_health: dict = {}
    for node_name, s, count in agg_rows:
        if node_name not in node_health:
            node_health[node_name] = {}
        node_health[node_name][s] = count

    return {
        "count":       len(checkpoints),
        "checkpoints": checkpoints,
        "node_health": node_health,
    }


# ── PPTX export ────────────────────────────────────────────────────────────────


class PptxRequest(BaseModel):
    title: str
    content: str   # markdown report content


@app.post("/export/pptx")
@limiter.limit("10/minute")
async def export_pptx(request: Request, req: PptxRequest) -> FileResponse:
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")
    try:
        from app.agents.pptx_generator import generate_pptx
        out_path = generate_pptx(req.title, req.content)
        return FileResponse(out_path, filename=Path(out_path).name, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")
    except ImportError:
        raise HTTPException(501, "python-pptx not installed. Run: pip install python-pptx")
    except Exception:
        logger.exception("pptx_export_failed")
        raise HTTPException(500, "Failed to generate PPTX")


# ── Scheduled research tasks ───────────────────────────────────────────────────


class ScheduleRequest(BaseModel):
    query: str
    email: str = ""
    run_now: bool = False


@app.post("/schedule")
@limiter.limit("10/minute")
async def schedule_task(request: Request, req: ScheduleRequest) -> dict:
    """Register a one-shot or recurring research task."""
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        await db.execute(
            "INSERT INTO scheduled_tasks (query, email, next_run_at, created_at) VALUES (?, ?, ?, ?)",
            (req.query, req.email, now, now),
        )
        await db.commit()
        cur = await db.execute("SELECT last_insert_rowid()")
        task_id = (await cur.fetchone())[0]
    if req.run_now:
        asyncio.create_task(_run_scheduled(task_id, req.query, req.email))
    return {"task_id": task_id, "status": "scheduled", "query": req.query}


@app.get("/schedule")
async def list_schedules(request: Request) -> dict:
    if not _check_auth(request):
        raise HTTPException(401, "Authentication required")
    async with aiosqlite.connect(ANALYTICS_DB) as db:
        cur = await db.execute(
            "SELECT id, query, email, next_run_at, created_at, last_run_at FROM scheduled_tasks ORDER BY id DESC"
        )
        rows = await cur.fetchall()
    return {"tasks": [{"id": r[0], "query": r[1], "email": r[2], "next_run_at": r[3], "created_at": r[4], "last_run_at": r[5]} for r in rows]}


async def _run_scheduled(task_id: int, query: str, email: str) -> None:
    """Run a scheduled research query and optionally email the result."""
    try:
        from langchain_core.messages import HumanMessage as HM
        thread_id = f"scheduled-{task_id}-{uuid.uuid4().hex[:8]}"
        inputs = {"messages": [HM(content=query)], "attempts": 0, "doc_id": "", "original_query": query}
        config = {"configurable": {"thread_id": thread_id}}
        result = await _graph_mod.graph.ainvoke(inputs, config)
        answer = result["messages"][-1].content if result.get("messages") else ""
        logger.info(f"scheduled_task_complete id={task_id}")
        if email and settings.smtp_host:
            asyncio.create_task(_send_email(email, f"Ares Report: {query[:60]}", answer))
        async with aiosqlite.connect(ANALYTICS_DB) as db:
            await db.execute(
                "UPDATE scheduled_tasks SET last_run_at = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), task_id),
            )
            await db.commit()
    except Exception:
        logger.exception(f"scheduled_task_failed id={task_id}")


async def _send_email(to: str, subject: str, body: str) -> None:
    """Send a plain-text email via configured SMTP."""
    if not settings.smtp_host:
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = settings.smtp_from
        msg["To"]      = to
        loop = asyncio.get_event_loop()
        def _send():
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as s:
                s.starttls()
                if settings.smtp_user:
                    s.login(settings.smtp_user, settings.smtp_password)
                s.sendmail(settings.smtp_from, [to], msg.as_string())
        await loop.run_in_executor(None, _send)
        logger.info(f"email_sent to={to} subject={subject!r}")
    except Exception:
        logger.warning("email_send_failed", exc_info=True)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

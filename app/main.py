"""FastAPI application.

Exposes a small JSON API plus a static chat UI:

  POST /chat    -> start (or continue) a conversation turn
  POST /resume  -> answer a clarifying question and continue
  GET  /        -> the chat UI
"""

import logging
import uuid

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel

from app.config import settings
from app.graph import graph

app = FastAPI(title="Multi-Agent Research Assistant")


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    clarification: str


def _run(inputs: dict | Command, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
     # Token budget guard
    if isinstance(inputs, dict):
        messages = inputs.get("messages", [])
        total_tokens = sum(len(m.content) for m in messages) // 4
        if total_tokens > settings.max_token_budget:
            logging.warning(f"Token budget exceeded: {total_tokens} tokens estimated")
            return {"status": "error", "answer": "Your query is too large to process. Please shorten your message.", "thread_id": thread_id}
        
    result = graph.invoke(inputs, config)

    # If the clarity agent paused for input, surface its question.
    if isinstance(result, dict) and result.get("__interrupt__"):
        interrupt = result["__interrupt__"][0]
        question = interrupt.value.get("question", "Could you clarify your request?")
        return {"status": "needs_clarification", "question": question, "thread_id": thread_id}

    answer = result["messages"][-1].content
    return {"status": "complete", "answer": answer, "thread_id": thread_id}


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    thread_id = req.thread_id or str(uuid.uuid4())
    inputs = {"messages": [HumanMessage(content=req.message)], "attempts": 0}
    return _run(inputs, thread_id)


@app.post("/resume")
def resume(req: ResumeRequest) -> dict:
    # Continue the conversation now that the user has clarified.
    
    return _run(Command(resume=req.clarification),req.thread_id)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)

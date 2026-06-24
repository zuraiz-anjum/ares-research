# Bug Report

---

## Bug 1 — API credentials leaked to stdout

**Location:** `app/config.py:34`

**Symptom:** The app prints all its settings to stdout on every startup. That includes the actual API keys. In a real deployment stdout goes into logs, and logs are often stored in plain text or shipped to third-party aggregators, so your keys end up somewhere you didn't intend.

**Root cause:** `settings.model_dump()` dumps every field in the Settings class as a dict, secrets included. Wrapping that in a `print()` call means the keys get logged unconditionally on import.

**Fix:** Deleted the print line. If something is misconfigured, the first real API call will fail with a clear error anyway and you don't need a startup log for that.

**Prevention:** Never pass a full settings object to a logger or print statement. If you genuinely need a startup diagnostic, be explicit about what you're logging and exclude any field that holds a secret.

---

## Bug 2 — Validator attempts counter never increments (infinite loop)

**Location:** `app/agents/validator.py:39`

**Symptom:** When research findings are repeatedly judged insufficient, the graph loops between research and validator forever. The request never returns an answer and hangs indefinitely.

**Root cause:** Line 39 mutates the local state copy directly (`state["attempts"] = ...`). LangGraph gives each node a snapshot of state and any mutations to it are silently thrown away. The only way to persist a change is to include it in the return dict. Since `attempts` was never returned, it stayed at 0 forever. The routing condition `state["attempts"] < settings.max_validation_attempts` was always `0 < 3` — always true , so the loop had no exit.

**Fix:** Removed the mutation line and included `attempts` in the return dict:
```python
return {
    "validation_result": verdict.validation_result,
    "attempts": state.get("attempts", 0) + 1,
}
```

**Prevention:** Treat `state` as read-only inside node functions. Every change must go through the return dict , that is the only mechanism LangGraph uses to persist updates.

---

## Bug 3 — Clarification loop never resolves (broken interrupt/resume)

**Location:** `app/main.py:58`

**Symptom:** When the clarity agent asks a clarifying question, answering it just triggers another clarifying question. The graph restarts from the beginning instead of resuming from where it paused, so the original question is forgotten and the loop never resolves.

**Root cause:** The `/resume` endpoint was building a regular state dict and passing it to `graph.invoke` — exactly the same way `/chat` works. LangGraph has no way to tell this apart from a brand new request, so it merges the dict into state and starts the graph from scratch again. The frozen `interrupt()` call inside the clarity node is abandoned, which is why clarity runs again and asks yet another question.

**Fix:** Use `Command(resume=...)` instead of a regular dict. This is LangGraph's specific signal to find the frozen execution in that thread and inject the value directly as the return of the `interrupt()` call:
```python
from langgraph.types import Command

@app.post("/resume")
def resume(req: ResumeRequest) -> dict:
    return _run(Command(resume=req.clarification), req.thread_id)
```

**Prevention:** Any `interrupt()` call inside a node must have a matching `Command(resume=...)` on the caller side. Never try to resume an interrupted thread by passing a regular state dict — LangGraph won't know the difference and will restart from scratch.

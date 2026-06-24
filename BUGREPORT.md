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

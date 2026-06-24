# Bug Report

---

## Bug 1 — API credentials leaked to stdout

**Location:** `app/config.py:34`

**Symptom:** The app prints all its settings to stdout on every startup. That includes the actual API keys. In a real deployment stdout goes into logs, and logs are often stored in plain text or shipped to third-party aggregators, so your keys end up somewhere you didn't intend.

**Root cause:** `settings.model_dump()` dumps every field in the Settings class as a dict, secrets included. Wrapping that in a `print()` call means the keys get logged unconditionally on import.

**Fix:** Deleted the print line. If something is misconfigured, the first real API call will fail with a clear error anyway and you don't need a startup log for that.

**Prevention:** Never pass a full settings object to a logger or print statement. If you genuinely need a startup diagnostic, be explicit about what you're logging and exclude any field that holds a secret.

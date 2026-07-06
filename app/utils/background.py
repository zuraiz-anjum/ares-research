"""Fire-and-forget task helper that keeps a strong reference.

`asyncio.create_task()` only holds a *weak* reference internally — if nothing
else references the returned Task object, the event loop is free to garbage
collect it before it finishes, silently dropping whatever it was doing
mid-flight. This is a documented asyncio footgun, not a hypothetical one.

`spawn()` keeps every task alive in a module-level set until it completes
(success or failure), then lets it go via the done-callback. Use it for any
"start this and don't wait" background write (DB save, webhook, calibration
record, etc.) instead of calling `asyncio.create_task()` directly.
"""

import asyncio
from typing import Coroutine

_background_tasks: set[asyncio.Task] = set()


def spawn(coro: Coroutine) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

"""Cross-session entity memory.

After each research run, an LLM extracts key facts (companies, funding, metrics)
and stores them to a JSON file on disk.  When the same entities appear in a new
query the relevant facts are injected into the synthesis prompt as prior context,
reducing hallucination and improving consistency across sessions.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_STORE_PATH = Path("ares_entity_memory.json")


def _load() -> dict:
    if _STORE_PATH.exists():
        try:
            return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    _STORE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class EntityFacts(BaseModel):
    entities: list[dict] = Field(
        description="List of {name: str, facts: list[str]} where facts are concrete, citable statements from the research."
    )


_EXTRACT_PROMPT = """You are an entity-fact extractor.

Given research findings, extract the KEY entities (companies, people, products)
and their most important, verifiable facts (funding amounts, valuations,
founding years, revenue, headcount, product names, notable events).

Return ONLY facts explicitly stated in the research — do not infer or guess.
Skip entities with fewer than 2 concrete facts."""


def extract_and_store(findings: str, query: str) -> int:
    """Extract entities from findings and persist them. Returns count of entities stored."""
    if not findings or len(findings) < 100:
        return 0
    try:
        from app.llm import get_llm
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = get_llm(temperature=0).with_structured_output(EntityFacts)
        result: EntityFacts = llm.invoke([
            SystemMessage(content=_EXTRACT_PROMPT),
            HumanMessage(content=f"Research findings:\n{findings[:4000]}"),
        ])
        if not result.entities:
            return 0

        store = _load()
        now = datetime.now(timezone.utc).isoformat()
        for entity in result.entities:
            name = entity.get("name", "").strip()
            facts = entity.get("facts", [])
            if not name or not facts:
                continue
            key = name.lower()
            if key not in store:
                store[key] = {"name": name, "facts": [], "updated_at": now, "source_queries": []}
            # Merge facts (avoid exact duplicates)
            existing = set(store[key]["facts"])
            new_facts = [f for f in facts if f not in existing]
            store[key]["facts"] = list(existing | set(new_facts))[-50:]  # cap at 50
            store[key]["updated_at"] = now
            store[key]["source_queries"] = (store[key].get("source_queries", []) + [query])[-5:]

        _save(store)
        logger.info(f"entity_memory_stored entities={len(result.entities)}")
        return len(result.entities)
    except Exception:
        logger.warning("entity_memory_extract_failed", exc_info=True)
        return 0


def recall(query: str, top_k: int = 3) -> str:
    """Return a formatted string of relevant entity facts for injection into prompts.

    Matches entities whose names appear in the query (case-insensitive).
    """
    store = _load()
    if not store:
        return ""

    q_lower = query.lower()
    matched = []
    for key, data in store.items():
        if key in q_lower or data["name"].lower() in q_lower:
            matched.append(data)

    if not matched:
        return ""

    lines = ["[From prior research sessions:]"]
    for data in matched[:top_k]:
        lines.append(f"\n## {data['name']}")
        for fact in data["facts"][:8]:
            lines.append(f"- {fact}")

    result = "\n".join(lines)
    logger.info(f"entity_memory_recalled entities={len(matched)}")
    return result

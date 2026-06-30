"""Dynamic Spawner agent — findings-aware parallel agent selection.

Used exclusively by 'plan' mode after research completes.

Unlike intent_router (which reads the user's query text and picks a static
pipeline), dynamic_spawner reads the ACTUAL research findings and decides
which specialist agents are needed — adapting to what was found, not just
what was asked.

Three specialist agents can be spawned in parallel:
  Narrative     — always runs; produces a clear, structured answer
  Quantitative  — runs if findings contain substantial numerical data
  Comparative   — runs if findings discuss multiple distinct entities

A Weaver LLM then combines all active outputs into one coherent response.

Pipeline position (plan mode only):
    planner → research → validator* → dynamic_spawner → challenger → fact_checker → ...
"""

import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)


# ── Spawn decision ─────────────────────────────────────────────────────────────

class SpawnDecision(BaseModel):
    spawn_quantitative: bool = Field(
        description="True if the findings contain significant numerical data "
                    "(financial figures, percentages, metrics, statistics) worth dedicated analysis"
    )
    spawn_comparative: bool = Field(
        description="True if the findings discuss 2 or more distinct named entities "
                    "(companies, products, people, technologies) that can be meaningfully compared"
    )
    rationale: str = Field(
        description="One sentence explaining which agents were selected and why"
    )


SPAWN_DECIDER_PROMPT = """\
You are the Spawn Decider. Read the research findings excerpt below and decide
which specialist analysis agents should be activated alongside the core Narrative agent.

Activate the Quantitative agent when:
- There are specific revenue, funding, valuation, or growth figures
- There are percentages, ratios, or other metrics that benefit from interpretation
- There are statistics comparing quantities across time periods or groups

Activate the Comparative agent when:
- Two or more distinct named companies, products, or entities appear in the findings
- The user's question explicitly or implicitly asks to compare or contrast
- The findings have enough data on each entity to draw meaningful distinctions

Be selective — not every query needs all agents. A single-entity narrative question
rarely needs the Comparative agent. A qualitative policy question rarely needs Quantitative.
"""


# ── Specialist agent prompts ───────────────────────────────────────────────────

NARRATIVE_PROMPT = """\
You are the Narrative Analyst in a multi-agent research pipeline.
Your job: produce a clear, well-structured answer to the user's question
using the research findings. This is the primary response.

Style: direct and comprehensive. Use ## headings for distinct themes.
Lead with the most important insight. Be specific with names, dates, facts.
Do NOT fill gaps with training knowledge. Say explicitly when data is missing.

Research findings:
{findings}"""

QUANTITATIVE_PROMPT = """\
You are the Quantitative Analyst in a multi-agent research pipeline.
Your job: extract, interpret, and contextualise ALL numerical data in the findings.

Output format:
## Key Metrics
[Table or bullet list of every important number, figure, or statistic found]

## What the Numbers Mean
[2-3 sentences interpreting trends, ratios, or patterns in the data]

## Data Gaps
[Any metrics that would be useful but were not found — list briefly or write "None"]

Research findings:
{findings}"""

COMPARATIVE_PROMPT = """\
You are the Comparative Analyst in a multi-agent research pipeline.
Your job: produce a structured comparison of ALL named entities in the findings.

Output format:
## Comparison Overview
[One sentence stating what is being compared]

| Dimension | {entities} |
|-----------|{dashes}|
[Fill rows for: scale/size, recent performance, strengths, key risks, notable facts]

## Verdict
[1-2 sentences on which entity leads in which dimension and why, based only on the data]

Research findings:
{findings}"""

WEAVER_PROMPT = """\
You are the Synthesis Weaver. Multiple specialist agents have analysed the same
research findings from different angles. Your job: weave their outputs into ONE
coherent, comprehensive response that feels natural — not a list of reports.

Rules:
- Integrate, don't just concatenate. Blend the narrative, data, and comparisons naturally.
- Lead with the most important insight.
- Keep quantitative data where it adds precision, but embed it in prose rather than isolated tables.
- If a Comparative table was generated, include it — tables are fine in the output.
- Total length should be similar to or slightly longer than the longest individual output.
- Do NOT add meta-commentary like "three agents were used" — just produce the answer.
- Use markdown (##, bullet points, tables) where it genuinely aids clarity.
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_entity_names(findings: str, n: int = 4) -> list[str]:
    """Cheaply extract up to n capitalised entity names for the comparison table header."""
    import re
    candidates = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', findings)
    seen, result = set(), []
    for c in candidates:
        if c not in seen and len(c) > 2:
            seen.add(c)
            result.append(c)
        if len(result) == n:
            break
    return result or ["Entity A", "Entity B"]


async def _call_agent(system: str, user_msg: str, label: str) -> str:
    from app.utils.checkpoint import save_sub_agent_error
    llm = get_llm(streaming=False)
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user_msg),
        ])
        logger.info(f"dynamic_spawner: {label} completed len={len(resp.content)}")
        return resp.content
    except Exception as exc:
        await save_sub_agent_error("dynamic_spawner", label, exc)
        return ""


# ── Node ───────────────────────────────────────────────────────────────────────

async def dynamic_spawner_node(state: AgentState) -> dict:
    query    = state.get("original_query", "")
    findings = state.get("findings", "") or state.get("raw_research", "")
    findings, _ = truncate_to_budget(findings, label="spawner_findings")

    if settings.mock_mode:
        mock = (
            f"## Dynamic Analysis: {query}\n\n"
            "**Narrative:** Strong momentum with 30% YoY growth.\n\n"
            "**Quantitative:** Revenue $2B | Growth 30% | Funding $500M\n\n"
            "**Verdict:** Well-positioned for continued expansion.\n\n"
            "*Agents spawned: Narrative, Quantitative*"
        )
        return {
            "messages":      [AIMessage(content=mock)],
            "spawned_agents": ["narrative", "quantitative"],
            "spawn_rationale": "Mock: narrative + quantitative",
        }

    # ── Step 1: decide which agents to spawn ─────────────────────────────────
    decider = get_llm(streaming=False).with_structured_output(SpawnDecision)
    try:
        decision: SpawnDecision = await decider.ainvoke([
            SystemMessage(content=SPAWN_DECIDER_PROMPT),
            HumanMessage(content=f"Question: {query}\n\nFindings excerpt:\n{findings[:2500]}"),
        ])
        logger.info(
            f"dynamic_spawner: quantitative={decision.spawn_quantitative} "
            f"comparative={decision.spawn_comparative} — {decision.rationale}"
        )
    except Exception as exc:
        logger.warning(f"dynamic_spawner: SpawnDecision failed ({exc}), using narrative only")
        decision = SpawnDecision(
            spawn_quantitative=False, spawn_comparative=False,
            rationale="fallback — only narrative spawned"
        )

    spawned = ["narrative"]
    if decision.spawn_quantitative:
        spawned.append("quantitative")
    if decision.spawn_comparative:
        spawned.append("comparative")

    # ── Step 2: build agent tasks ─────────────────────────────────────────────
    tasks: list = [
        _call_agent(
            NARRATIVE_PROMPT.format(findings=findings),
            f"Question: {query}",
            "narrative",
        )
    ]

    if decision.spawn_quantitative:
        tasks.append(_call_agent(
            QUANTITATIVE_PROMPT.format(findings=findings),
            f"Question: {query}",
            "quantitative",
        ))

    if decision.spawn_comparative:
        entities = _extract_entity_names(findings)
        entity_header = " | ".join(entities)
        dashes = "|".join(["---"] * len(entities))
        tasks.append(_call_agent(
            COMPARATIVE_PROMPT.format(
                findings=findings,
                entities=entity_header,
                dashes=dashes,
            ),
            f"Question: {query}",
            "comparative",
        ))

    # ── Step 3: run all agents in parallel ────────────────────────────────────
    results = await asyncio.gather(*tasks)
    outputs = {name: text for name, text in zip(spawned, results) if text}

    if not outputs:
        logger.error("dynamic_spawner: all agents failed, returning empty")
        return {"messages": [AIMessage(content="(analysis failed)")]}

    # ── Step 4: weave outputs together (streaming) ────────────────────────────
    if len(outputs) == 1:
        # Single agent — return directly, no weaving needed.
        final = next(iter(outputs.values())).rstrip()
    else:
        weaver_context = (
            f"User question: {query}\n\n"
            + "\n\n".join(
                f"=== {k.upper()} AGENT OUTPUT ===\n{v}"
                for k, v in outputs.items()
            )
        )
        weaver_llm = get_llm(streaming=True)
        weaved = await weaver_llm.ainvoke([
            SystemMessage(content=WEAVER_PROMPT),
            HumanMessage(content=weaver_context),
        ])
        final = weaved.content

    label_map = {
        "narrative": "Narrative", "quantitative": "Quantitative", "comparative": "Comparative"
    }
    agent_list = ", ".join(label_map[a] for a in spawned)
    final = final.rstrip() + f"\n\n---\n*Agents spawned: {agent_list}*"

    return {
        "messages":       [AIMessage(content=final)],
        "spawned_agents": spawned,
        "spawn_rationale": decision.rationale,
    }

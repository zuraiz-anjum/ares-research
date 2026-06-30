"""Academic Writer agent.

Converts research findings into a properly structured academic paper — the
same format you'd find on Google Scholar: Abstract, numbered sections,
inline [N] citations, and a numbered reference list.

Pipeline position:  research → academic_writer → chart_writer → pdf_generator
"""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings, truncate_to_budget
from app.llm import get_llm
from app.state import AgentState

logger = logging.getLogger(__name__)

ACADEMIC_SYSTEM_PROMPT = """You are an academic research writer. Convert the provided
research findings into a complete, publication-quality academic paper.

Output format — follow this EXACTLY, preserving all markers:

# [Write a clear, descriptive academic title — not a question, max 12 words]

**Abstract:** [Write 150-200 words: state the topic, the approach taken, the key findings, and their significance. Formal tone, third person, no citations.]

**Keywords:** [5-8 relevant terms separated by commas]

---

## 1. Introduction

[150 words. Provide background context, state why this topic matters, and outline what this paper covers. End with one sentence stating the paper's objective.]

## 2. Background

[150 words. Summarise what is already known. Reference key sources using [1], [2] notation where appropriate.]

## 3. Analysis & Findings

[300 words minimum. Present the main findings in detail. Use subsections ### 3.1 and ### 3.2 if there are distinct facets. Cite sources inline as [1], [2] etc. Include specific numbers, dates, and named entities.]

## 4. Discussion

[150 words. Interpret the findings. Identify patterns, compare with prior knowledge, and discuss implications. Acknowledge limitations where relevant.]

## 5. Conclusion

[80 words. Summarise the key takeaways and suggest directions for future work.]

## References

[List every source cited in the paper, numbered to match inline citations. Format:
[1] Author/Title. Source. Year if known. URL if available.
[2] ...]

Rules:
- Use formal academic prose throughout. No bullet points except in the references.
- Every claim that comes from a source MUST have an inline [N] citation.
- The reference numbers in the body MUST match the numbered list at the end.
- Do NOT use markdown bold (**) for section headings — the ## markers define them.
- Do NOT add any text after the References section.
- Use only plain ASCII dashes (-) not em-dashes or special characters."""


def _mock_paper(query: str) -> str:
    return f"""# {query}: A Comprehensive Analysis

**Abstract:** This paper presents a comprehensive analysis of {query} based on current research findings. The study examines key metrics, market dynamics, and strategic implications. Analysis reveals strong growth trajectories and significant investment activity in this domain. The findings suggest continued expansion driven by technological innovation and increasing market demand. This paper synthesises available evidence to provide actionable insights for researchers and practitioners.

**Keywords:** {query.lower()}, market analysis, research findings, strategic assessment, growth metrics

---

## 1. Introduction

The study of {query} has gained significant attention in recent years. This paper investigates the primary drivers, key metrics, and strategic implications of developments in this area. Understanding these dynamics is critical for stakeholders seeking to navigate a rapidly evolving landscape. This paper aims to synthesise current findings into a structured academic overview.

## 2. Background

Prior research has established foundational knowledge in this domain [1]. Earlier studies have documented growth patterns and identified key market participants [2]. The theoretical framework adopted here builds on this prior work.

## 3. Analysis & Findings

### 3.1 Key Metrics

Mock findings indicate strong momentum with 30% year-over-year growth [1]. Investment activity reached $500M in the most recent funding cycle, reflecting sustained confidence from institutional investors [2].

### 3.2 Strategic Implications

The data suggests a consolidation phase is underway. Leading entities have expanded their product portfolios while maintaining strong margins [3].

## 4. Discussion

The findings align with broader market trends documented in the literature. The growth trajectory observed is consistent with sector-wide expansion. Limitations include potential data lag and variability across sources.

## 5. Conclusion

This analysis confirms robust growth in {query}. Future research should focus on longitudinal metrics and cross-market comparisons to deepen understanding.

## References

[1] Mock Source One. Industry Report. 2024. https://example.com/1
[2] Mock Source Two. Market Analysis. 2024. https://example.com/2
[3] Mock Source Three. Research Findings. 2024. https://example.com/3"""


async def academic_writer_node(state: AgentState) -> dict:
    query             = state.get("original_query", "Research Paper")
    findings          = state.get("findings", "") or state.get("raw_research", "")
    sources           = state.get("sources", []) or []
    sub_queries       = state.get("sub_queries", [])
    revision_feedback = state.get("revision_feedback", "") or ""
    revision_count    = state.get("revision_count", 0)

    if settings.mock_mode:
        paper = _mock_paper(query)
        return {"messages": [AIMessage(content=paper)], "report_content": paper}

    findings, _ = truncate_to_budget(findings, label="academic_findings")

    # Build a numbered source list to feed the LLM so it can generate
    # correct [N] references without hallucinating URLs.
    source_block = ""
    if sources:
        lines = ["\n\nAvailable sources (use these for [N] citations):"]
        for i, s in enumerate(sources, 1):
            title = s.get("title", "Untitled")
            url   = s.get("url", "")
            lines.append(f"[{i}] {title}  {url}")
        source_block = "\n".join(lines)

    context = (
        f"Research topic: {query}\n"
        + (f"Sub-topics researched: {', '.join(sub_queries)}\n" if len(sub_queries) > 1 else "")
        + f"\nFindings:\n{findings}"
        + source_block
    )

    if revision_feedback:
        context += (
            f"\n\n--- Quality critic feedback (revision {revision_count}) ---\n"
            f"{revision_feedback}\n"
            f"--- Incorporate ALL of the above feedback points in this revision ---"
        )

    llm = get_llm(streaming=True)
    response = await llm.ainvoke([
        SystemMessage(content=ACADEMIC_SYSTEM_PROMPT),
        HumanMessage(content=context),
    ])

    paper = response.content
    logger.info(f"academic_writer_complete revision={revision_count} has_feedback={bool(revision_feedback)}")

    # Extract abstract for state (used by pdf_generator to render the abstract box).
    abstract = ""
    keywords = ""
    for line in paper.splitlines():
        if line.startswith("**Abstract:**"):
            abstract = line.replace("**Abstract:**", "").strip()
        elif line.startswith("**Keywords:**"):
            keywords = line.replace("**Keywords:**", "").strip()

    return {
        "messages":          [AIMessage(content=paper)],
        "report_content":    paper,
        "paper_abstract":    abstract,
        "paper_keywords":    keywords,
        "revision_feedback": "",   # clear after consuming
    }

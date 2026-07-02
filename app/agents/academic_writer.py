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

ACADEMIC_SYSTEM_PROMPT = """You are a senior academic research writer with expertise in producing
peer-reviewed, publication-quality papers. Convert the research findings into a COMPLETE,
LONG-FORM academic paper. This is not a summary — it is a full scholarly document.

CRITICAL LENGTH REQUIREMENT: The paper body MUST be at minimum 3,500 words (excluding the
abstract and references). Every section must be substantive. Thin, brief sections are a
failure. If you are under the word target for a section, keep writing.

Output format — follow this EXACTLY, preserving all markers:

# [Title: specific, descriptive, and informative — not a question, 10-16 words]

**Abstract:** [250-300 words. Cover: (1) the importance of the topic and why it is studied now,
(2) the research questions addressed, (3) methodology and data sources, (4) the key quantitative
findings with actual numbers and entity names, (5) the significance and implications of the
findings. Formal third person throughout. No inline citations.]

**Keywords:** [6-8 terms, comma-separated]

---

## 1. Introduction

[400-500 words. Open with a compelling statement about why this topic is significant right now.
Provide rich background context — what is happening in this domain, who the key actors are, what
trends are driving interest. State the specific research questions this paper addresses.
Briefly outline the paper's structure. End with a clear statement of the paper's objective
and scholarly contribution.]

## 2. Literature Review and Background

[500-650 words. Synthesise what is already known. Organise thematically — group related facts
and findings from different sources together. Trace how the situation evolved over time. Use
at least 6 different inline citations [N] spread across this section. Identify gaps or
tensions in the existing record that this paper addresses.]

## 3. Methodology

[200-300 words. Describe how the research was conducted: data sources (news, financial
disclosures, regulatory filings, industry databases), the time period covered, selection
criteria for sources, and limitations of the approach. Be concrete — name the types of
sources used and why they are authoritative.]

## 4. Findings and Analysis

[900-1,200 words minimum. This is the heart of the paper — analyse in depth.
Every major claim must be cited [N]. Use subsections:]

### 4.1 [First major theme — e.g. "Funding Rounds and Capital Formation"]

[300-400 words. Lead with the most important quantitative data. Provide timeline of key events,
compare across entities, explain context — why these numbers matter, what they reveal about
strategy or market position. Do not summarise — analyse.]

### 4.2 [Second major theme — e.g. "Revenue Models and Path to Profitability"]

[300-400 words. Same analytical depth. Draw comparisons. Note trajectories, not just snapshots.
Cite specific figures, dates, and named actors.]

### 4.3 [Third major theme — e.g. "Strategic Positioning and Competitive Dynamics"]

[250-350 words. A third analytical lens on the topic. Could cover regulatory context,
partnerships, technology strategy, or market implications depending on the topic.]

## 5. Discussion

[500-650 words. Interpret — do not repeat findings. What patterns emerge across sections 4.1-4.3?
What do they collectively imply? Where do the entities or cases diverge and why? Address any
surprising or counterintuitive findings. Discuss broader implications for the industry, investors,
policymakers, or future research. Acknowledge limitations of the data or analysis. This section
should demonstrate scholarly judgement, not just descriptive reporting.]

## 6. Conclusion

[250-350 words. Synthesise the key findings in 3-4 sentences. State the contribution of this
paper clearly. Discuss practical implications — what should practitioners, investors, or
policymakers take from this? Propose 2-3 specific, concrete directions for future research.
End with a forward-looking closing statement.]

## References

[Numbered reference list matching ALL inline citations used in the body. Every source cited
must appear here. Format each entry on its own line:
[1] Title. Author or Publication. Year. URL if available.
Minimum 12 references.]

RULES — READ CAREFULLY:
- MANDATORY: Each section must meet its word minimum. Count carefully. If you are short, expand.
- Formal academic prose throughout. Vary sentence structure. Never use bullet points in body sections.
- Every factual claim that comes from a source MUST have an inline [N] citation immediately after it.
- Reference numbers [N] in the body MUST exactly match the numbered list in the References section.
- Do NOT use markdown bold (**) for section headings — ## and ### markers define them.
- Do NOT add any text after the References section.
- Use only plain ASCII dashes (-), not em-dashes (—) or special Unicode characters.
- Write as if this paper will be submitted to a peer-reviewed journal. Quality and depth matter."""


def _mock_paper(query: str) -> str:
    return f"""# {query}: A Comprehensive Academic Analysis of Key Trends and Implications

**Abstract:** This paper presents a comprehensive analysis of {query} based on current research findings. The study examines key metrics, market dynamics, funding trajectories, and strategic implications through a systematic review of publicly available financial disclosures, industry reports, and news sources. Analysis reveals strong growth trajectories and significant investment activity in this domain, with total capital raised exceeding $50 billion across leading entities. The findings suggest continued expansion driven by technological innovation and increasing institutional demand. Key patterns include competitive differentiation through model capabilities, divergent revenue models, and regulatory scrutiny at scale. This paper synthesises available evidence to provide actionable insights for researchers, investors, and practitioners navigating a rapidly evolving landscape.

**Keywords:** {query.lower()}, market analysis, venture capital, research findings, strategic assessment, growth metrics, AI industry

---

## 1. Introduction

The study of {query} has emerged as one of the most consequential areas of inquiry in the contemporary technology landscape. Over the past three years, the sector has witnessed unprecedented levels of capital formation, rapid product iteration, and intensifying competitive dynamics that have drawn attention from investors, regulators, and academics alike. The convergence of large-scale compute infrastructure, foundation model development, and enterprise adoption has created a structural shift in how technology companies are funded, valued, and monetised [1].

This paper investigates the primary drivers, key financial metrics, and strategic implications of these developments. Understanding these dynamics is critical for stakeholders seeking to navigate a rapidly evolving landscape where competitive positions can shift dramatically within a single funding cycle. The research questions addressed in this paper are: (1) How have leading entities in this space structured their capital formation? (2) What revenue models have emerged and which appear most durable? (3) What are the strategic implications for market structure and competition?

This paper is structured as follows. Section 2 reviews the existing literature and background context. Section 3 describes the methodology. Section 4 presents findings across three analytical themes. Section 5 interprets these findings in a broader context. Section 6 concludes with implications and directions for future research.

## 2. Literature Review and Background

Prior research has established foundational knowledge regarding the dynamics of capital-intensive technology sectors [1]. Early studies of the internet era documented similar patterns of rapid funding escalation followed by market consolidation [2]. More recent scholarship has examined how artificial intelligence investment cycles differ from prior technology waves — notably in the concentration of capital among a small number of well-resourced incumbents and their venture-backed challengers [3].

The entities relevant to {query} have been subjects of increasing academic and journalistic scrutiny. Financial historians have noted the unusual structure of certain modern AI companies, which operate as hybrids between non-profit research institutions and commercial enterprises [4]. This structural ambiguity complicates traditional valuation frameworks and introduces novel governance risks that prior literature has not fully addressed [2].

Industry analysts have documented the rapid growth in revenue across leading players, noting compound annual growth rates that significantly outpace historical technology benchmarks [5]. At the same time, operating losses remain substantial, with burn rates that would be unsustainable for most businesses at scale. The implicit assumption underlying current valuations is that model capability improvements will eventually unlock margins comparable to software-as-a-service businesses [3].

Regulatory literature has begun to address the concentration of AI infrastructure and the potential systemic risks posed by a small number of actors controlling foundational model capabilities [6]. This work remains nascent, and there is limited empirical evidence on the long-term competitive effects of current market structure.

## 3. Methodology

This paper adopts a systematic desk research methodology, synthesising publicly available data from a range of authoritative sources. Primary data sources include regulatory filings, official press releases, audited financial disclosures where publicly available, and reports from established financial news organisations. Secondary sources include technology industry research from analysts and academic papers addressing related topics. The time period covered is primarily 2021-2025, capturing the most recent and consequential funding cycles.

Sources were selected on the basis of verifiability and authority. Unverified claims from non-authoritative outlets were excluded. Where figures varied across sources, the most recent and conservative estimate was used. Key limitations of this approach include: the reliance on self-reported figures from private companies, the lag between events and their reporting, and the difficulty of independently verifying valuation metrics for pre-IPO entities.

## 4. Findings and Analysis

### 4.1 Funding Rounds and Capital Formation

Capital formation in the domain of {query} has accelerated dramatically since 2021. Mock data indicates total investment across leading entities reached $50 billion by 2024, with the pace of individual funding rounds increasing in both frequency and scale [1]. A single entity raised $6.6 billion in a single round in late 2024, representing one of the largest private-company funding events in technology history [2]. This capital concentration reflects investor conviction in the winner-takes-most dynamics of foundation model development.

Investor composition has also evolved. Early rounds were dominated by specialist venture capital firms with deep technical expertise. More recent rounds have attracted participation from sovereign wealth funds, large corporate strategic investors, and non-traditional capital allocators, signalling a broadening of institutional interest [3]. The involvement of Microsoft, Google, and Amazon as both investors and infrastructure partners creates complex competitive relationships that differ substantially from traditional VC-backed technology companies.

Valuation multiples have expanded alongside capital raised. Price-to-revenue multiples in the sector exceed those of comparable software companies by a significant margin, reflecting expectations of long-term growth rather than current earnings. Whether these valuations are sustainable is a matter of ongoing debate among analysts [4].

### 4.2 Revenue Models and Financial Performance

Revenue models across leading entities in {query} vary significantly, creating heterogeneous competitive dynamics. The dominant model involves API-based access to large language models, priced on a per-token consumption basis [5]. This model benefits from low marginal costs at scale and high switching costs once enterprise customers have integrated a vendor's models into their workflows. However, it also creates exposure to pricing competition as model capabilities converge.

A second revenue model involves direct consumer subscription products. These have demonstrated strong adoption velocity but face significant churn risks and require ongoing investment in product development to maintain user engagement [2]. Mock revenue figures suggest annual recurring revenue for leading consumer products exceeds $1 billion, though profitability remains elusive due to the high cost of inference compute.

Enterprise contracts represent a third and increasingly important revenue stream. These tend to be higher-value, longer-term agreements with more predictable cash flow characteristics. Several entities have announced multi-year enterprise agreements valued in the hundreds of millions of dollars, suggesting a maturation of the commercial market [6].

### 4.3 Strategic Positioning and Competitive Dynamics

The competitive landscape in {query} is characterised by simultaneous collaboration and competition among leading players. Microsoft's deep investment in one major entity while operating its own competing products illustrates the unusual structure of competitive relationships in the sector [1]. Similarly, Google's investment in a competitor while developing its own foundation models creates a competitive dynamic with few precedents in technology history.

Talent competition has emerged as a critical strategic battleground. The supply of researchers capable of training frontier models is extremely limited, and compensation packages for senior AI researchers now routinely exceed those of equivalent roles at established technology companies [3]. Several high-profile researcher movements between organisations have had measurable effects on product roadmaps and capability timelines.

Regulatory pressure is increasing across multiple jurisdictions. The European AI Act, US executive orders on AI safety, and emerging regulatory frameworks in the UK and Asia-Pacific region all create compliance obligations that favour larger, better-resourced players [4]. This dynamic may accelerate consolidation in the medium term.

## 5. Discussion

The findings presented in this paper reveal a sector characterised by extraordinary capital intensity, concentrated competitive dynamics, and unresolved tensions between commercial imperatives and stated public-interest missions. Several patterns merit interpretive attention.

First, the scale of capital formation significantly exceeds what would be required merely to develop and deploy current generation models. The excess capital reflects a strategic bet on compute scaling as a durable source of competitive advantage — an assumption that is empirically contested in recent research suggesting diminishing returns to pre-training scale [5]. If this scaling thesis is incorrect, current valuations may prove difficult to sustain.

Second, the divergence between revenue growth and profitability among leading players is striking. While revenue is growing rapidly, operating losses are also expanding, driven by the cost of training and serving increasingly large models. This pattern differs from prior software companies at analogous growth stages and suggests that the path to profitability in foundation models is longer and more uncertain than investors may currently price [2].

Third, the competitive dynamics observed — particularly the co-investment relationships between ostensible competitors — suggest that the sector is evolving toward a structure more resembling a regulated oligopoly than a competitive market. The concentration of frontier model capabilities in a small number of well-capitalised entities creates barriers to entry that are difficult to overcome through conventional venture capital financing alone [6].

Limitations of this analysis include the reliance on self-reported and often unaudited financial metrics, the rapidly changing nature of the sector (conditions described here may have shifted materially since data collection), and the inherent difficulty of applying traditional financial analysis frameworks to entities with novel organisational structures.

## 6. Conclusion

This paper has analysed {query} across three analytical dimensions: capital formation, revenue model evolution, and strategic competitive dynamics. The findings confirm that this sector is experiencing capital formation at an unprecedented scale, driven by investor belief in the long-term transformative potential of foundation model AI. Revenue is growing rapidly but profitability remains distant for most leading entities.

The practical implications are significant. For investors, current valuations imply a narrow range of successful outcomes and require careful assessment of which entities have durable competitive advantages beyond current model capabilities. For policymakers, the concentration of frontier AI capabilities in a small number of private entities raises governance questions that existing regulatory frameworks are not fully equipped to address. For practitioners, the rapid pace of capability improvement and pricing competition creates both opportunities and risks for enterprise integration strategies.

Future research should focus on: (1) longitudinal tracking of revenue model evolution as the market matures; (2) empirical analysis of switching costs and competitive moat durability in foundation model markets; (3) comparative regulatory analysis across jurisdictions. Addressing these questions will significantly deepen scholarly understanding of one of the most consequential technology transitions of the current era.

## References

[1] Mock Source One. Technology Industry Report. Research Firm. 2024. https://example.com/1
[2] Mock Source Two. AI Market Analysis. Financial Times. 2024. https://example.com/2
[3] Mock Source Three. Venture Capital Trends. Crunchbase. 2024. https://example.com/3
[4] Mock Source Four. Foundation Model Economics. Stanford HAI. 2024. https://example.com/4
[5] Mock Source Five. Revenue and Growth Metrics. Bloomberg. 2024. https://example.com/5
[6] Mock Source Six. Regulatory Landscape. European Commission. 2024. https://example.com/6"""


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

    # Extract abstract and keywords for state (used by pdf_generator).
    # Abstract may span multiple lines for a 250-300 word block.
    abstract = ""
    keywords = ""
    in_abstract = False
    for line in paper.splitlines():
        if line.startswith("**Abstract:**"):
            abstract = line.replace("**Abstract:**", "").strip()
            in_abstract = True
        elif line.startswith("**Keywords:**"):
            keywords = line.replace("**Keywords:**", "").strip()
            in_abstract = False
        elif in_abstract:
            if line.startswith("**") or line.startswith("---") or line.startswith("## "):
                in_abstract = False
            elif line.strip():
                abstract += " " + line.strip()

    return {
        "messages":          [AIMessage(content=paper)],
        "report_content":    paper,
        "paper_abstract":    abstract,
        "paper_keywords":    keywords,
        "revision_feedback": "",   # clear after consuming
    }

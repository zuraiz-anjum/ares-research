# RUN.md — Setup and run in under 10 minutes

## Prerequisites

- Python 3.11+
- A Groq API key (free at console.groq.com)
- A Tavily API key (free tier at app.tavily.com)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
```

Open `.env` and fill in your two keys:
```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 and ask about a company.

Try a compound query to see decomposition fire:
```
Compare Stripe and OpenAI's recent funding
```

## Run tests

```bash
python -m pytest tests/ -v
```

29 tests, under 10 seconds, no API keys needed.

## What to expect

Simple query: Clarity checks if it's specific enough, Decomposer passes it through as a single search, Research searches Tavily, Synthesis writes the answer.

Compound query: Decomposer splits it into parallel sub-queries, Research runs them simultaneously with ThreadPoolExecutor, findings are merged before Synthesis writes a combined answer.

Vague query: Clarity pauses and asks which company, you answer, graph resumes from that exact point via Command(resume=...).

## Mock mode

To run with zero external API calls (Tavily and all LLM agents return hardcoded responses):
```
MOCK_MODE=true
```

In mock mode the full graph runs end-to-end with no network calls. Useful for CI or offline testing.

## Run with Docker

```bash
docker build -t research-assistant .
docker run -p 8000:8000 --env-file .env research-assistant
```

Open http://localhost:8000 and ask about a company.

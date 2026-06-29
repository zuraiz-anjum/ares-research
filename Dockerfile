FROM python:3.11-slim

WORKDIR /app

# fontconfig: matplotlib font rendering  |  build-essential: chromadb/flashrank native exts
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached layer — only rebuilds when requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create runtime directories and warm the matplotlib font cache
RUN mkdir -p static/charts static/reports static/shared uploads \
 && python -c "import matplotlib; matplotlib.font_manager._load_fontmanager(try_read_cache=False)" 2>/dev/null || true

# Default port — override with PORT env var at runtime
EXPOSE 8000

# Single worker keeps SQLite checkpoints safe; scale via Railway/Render replicas if needed
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM python:3.11-slim

WORKDIR /app

# Install fontconfig for matplotlib font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-create output directories and warm the matplotlib font cache
RUN mkdir -p static/charts static/reports \
 && python -c "import matplotlib; matplotlib.font_manager._load_fontmanager(try_read_cache=False)" 2>/dev/null || true

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]

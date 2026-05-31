# Optional deployment layer: containerize the Streamlit app for cloud environments.
# Local execution via `make chat` remains the primary path; this is for hosting.
#
# Build:  docker build -t leadership-agent .
# Run:    docker run -p 8501:8501 \
#           -e OPENAI_API_KEY=... -e TAVILY_API_KEY=... \
#           -v "$(pwd)/data:/app/data" \
#           leadership-agent
#
# The data/ volume holds the SQLite DB. Populate it on the host with
# `make collect URL=...`, or mount an empty volume and collect into it.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer caches across source changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only what the running app needs (see .dockerignore for exclusions).
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

# Liveness probe against Streamlit's built-in health endpoint (no curl needed).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status == 200 else 1)"

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
# ──────────────────────────────────────────────
# DishSpace AI — Production Dockerfile
# ──────────────────────────────────────────────
# Multi-stage build for small final image.
#
# Build:  docker build -t dishspace .
# Run:    docker run -p 8000:8000 --env-file .env dishspace
# ──────────────────────────────────────────────

FROM python:3.11-slim AS base

# System deps for OpenCV, Open3D
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies ──
FROM base AS deps

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .[dev]

# ── Application ──
FROM deps AS app

COPY src/ src/
COPY demo/ demo/
COPY scripts/ scripts/
COPY .env.example .env.example

# Non-root user
RUN groupadd -r dishspace && useradd -r -g dishspace dishspace
USER dishspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]

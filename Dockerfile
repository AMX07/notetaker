FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --no-dev --frozen

COPY src/ src/
COPY static/ static/

# Create non-root user and output directory
RUN useradd -m notetaker && \
    mkdir -p /app/output/jobs && \
    chown -R notetaker:notetaker /app
USER notetaker

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uv", "run", "notetaker"]

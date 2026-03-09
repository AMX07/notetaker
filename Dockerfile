FROM ubuntu:22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System deps (uses Ubuntu repos, not Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

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

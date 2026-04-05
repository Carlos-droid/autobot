# Dockerfile for Immune Swarm v1.0
FROM python:3.12-slim

WORKDIR /app

# System dependencies for Playwright and Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy configuration
COPY pyproject.toml uv.lock ./

# Copy source code (src/original_swarm as the main package)
COPY src/ /app/src/

# Install dependencies (will update lockfile if needed)
RUN uv sync

# Install Playwright browser
RUN uv run playwright install chromium

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV MODEL_NAME=qwen2.5:14b
# Set PYTHONPATH to prioritize current package folders and support absolute imports
ENV PYTHONPATH="/app/src/original_swarm:/app"

# Default CMD: Launches the OpenHands orquestrator
CMD ["uv", "run", "python", "-m", "src.original_swarm.openhands_adapter"]

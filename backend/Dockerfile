# Stage 1: Builder - install dependencies
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy project metadata. We override the uv environments setting so that
# resolution works on Linux (the lockfile was generated for win32 only).
COPY pyproject.toml uv.lock* ./

# The lockfile was pinned to win32, so we cannot use --frozen.
# Remove win32-only environment restriction and CUDA torch source so uv
# resolves cross-platform with CPU-only PyTorch.
RUN sed -i 's/environments = \["sys_platform == .win32."\]/environments = ["sys_platform == '\''win32'\''", "sys_platform == '\''linux'\''"]/' pyproject.toml && \
    sed -i '/^\[tool\.uv\.sources\]/,/^$/d' pyproject.toml && \
    sed -i '/^\[\[tool\.uv\.index\]\]/,/^$/d' pyproject.toml

# Install only dependencies (not the project itself) so we control imports
# via PYTHONPATH pointing at /app rather than site-packages.
RUN uv sync --no-dev --no-install-project --inexact

# Stage 2: Runtime - lean image
FROM python:3.12-slim

RUN groupadd --gid 1000 app && \
    useradd --uid 1000 --gid app --shell /bin/bash --create-home app

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv

# Copy application source. PYTHONPATH=/app makes ml.* and backend.* importable.
COPY backend/ ./backend/
COPY ml/__init__.py ./ml/__init__.py
COPY ml/training/ ./ml/training/
COPY ml/utils/ ./ml/utils/
COPY monitoring/ ./monitoring/

# Models are volume-mounted at runtime (see docker-compose.yml) so we do NOT
# COPY them here. This keeps the image small (~2GB vs ~2.7GB with models).

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

USER app

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

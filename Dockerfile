# Scoring service image. The model bundle is not baked in: mount a released bundle at
# /app/models (read-only), and the service verifies it against its manifest at start-up.

FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.12.16-slim-trixie AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first, so source edits do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12.16-slim-trixie
# XGBoost needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /app fraudlens
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations ./migrations
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FRAUDLENS_MODELS_DIR=/app/models \
    PROMETHEUS_MULTIPROC_DIR=/tmp/fraudlens-metrics
USER fraudlens
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2)"]
# Migrations are idempotent; running them first means a fresh database just works.
# Metric files from a previous run of this container are cleared, so /metrics counts
# only the workers that are alive now.
CMD ["sh", "-c", "mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\" && find \"$PROMETHEUS_MULTIPROC_DIR\" -type f -delete && alembic upgrade head && exec uvicorn fraudlens.service.app:create_app --factory --host 0.0.0.0 --port 8000 --no-access-log --workers ${FRAUDLENS_WORKERS:-1}"]

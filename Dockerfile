# Pinned by digest: tags move, and every image built from this file must be
# the same bytes underneath.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS base

ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# XGBoost links against the OpenMP runtime, which slim images leave out.
# Shared by both stages so it is installed once.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*


FROM base AS builder

RUN pip install --no-cache-dir uv==0.13.0
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra dashboard
COPY src ./src
# Non-editable: the runtime stage gets the package inside the venv and
# never needs the source tree, uv or pip.
RUN uv sync --frozen --extra dashboard --no-editable


FROM base AS runtime

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/state \
    && chown app:app /app/state
WORKDIR /app
# The only directory the non-root user can write to; the settings default
# (data/driftmon.db) would point into the root-owned /app.
ENV DRIFTMON_DB_PATH=/app/state/driftmon.db
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
USER app

EXPOSE 8000 8501 9100
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"]
CMD ["driftmon", "serve", "--host", "0.0.0.0", "--port", "8000"]

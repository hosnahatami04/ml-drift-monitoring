FROM python:3.11-slim

ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# XGBoost links against the OpenMP runtime, which slim images leave out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.0

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra dashboard
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --extra dashboard

EXPOSE 8000 8501 9100
CMD ["driftmon", "serve", "--host", "0.0.0.0", "--port", "8000"]

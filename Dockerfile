FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
# Model is mounted at runtime (exported from the MLflow registry by `make train`), not baked in.
ENV MODEL_DIR=/app/model PRED_DB=/logs/predictions.db MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "churn.serve:app", "--host", "0.0.0.0", "--port", "8000"]

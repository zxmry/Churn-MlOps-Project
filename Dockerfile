FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
# The current @champion, exported from the MLflow registry and committed. Promoting a model
# therefore goes through a PR and CI. Mount a different ./model over it to serve another version.
COPY model ./model
RUN useradd --create-home app && mkdir /logs && chown app /logs
USER app
ENV MODEL_DIR=/app/model PRED_DB=/logs/predictions.db MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8000
# Hosting platforms (Render, Fly, Railway) inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "exec /app/.venv/bin/uvicorn churn.serve:app --host 0.0.0.0 --port ${PORT:-8000}"]

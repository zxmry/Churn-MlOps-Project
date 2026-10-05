export MLFLOW_DISABLE_AGENT_HINT=1
RUN = uv run

.PHONY: data train export test serve replay drift drift-check retrain mlflow docker docker-run

data:        ; $(RUN) python -m churn.data
train:       ; $(RUN) python -m churn.train            # sweep, register best as @champion, export to ./model
export:      ; $(RUN) python -m churn.train export     # re-export current @champion
test:        ; $(RUN) ruff check . && $(RUN) pytest -q
serve:       ; $(RUN) uvicorn churn.serve:app --port 8000
replay:      ; $(RUN) python scripts/replay.py         # needs `make serve` (or docker-run) in another shell
drift:       ; $(RUN) python -m churn.drift
drift-check: ; $(RUN) python -m churn.drift --check     # exit 1 if retrain recommended
retrain:     ; $(RUN) python -m churn.retrain          # champion/challenger, promotes only if better
mlflow:      ; $(RUN) mlflow ui --backend-store-uri sqlite:///mlflow.db
docker:      ; docker build -t churn .
docker-run:  ; docker run --rm -p 8000:8000 churn           # serves the committed model/

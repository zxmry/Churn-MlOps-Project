"""FastAPI service: batch churn scoring + prediction logging to SQLite."""
import datetime as dt
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, create_model

from churn.features import FEATURES

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
PRED_DB = os.getenv("PRED_DB", "predictions.db")

Instance = create_model(
    "Instance",
    customer_id=(int, ...),
    **{f: (float, Field(allow_inf_nan=False)) for f in FEATURES},
)


# Shown prefilled in /docs "Try it out": a fading customer next to an active, growing one.
EXAMPLE = {"as_of": "2011-08-01", "instances": [
    {"customer_id": 1, "recency_days": 90, "n_invoices_180d": 10, "n_invoices_90d": 0,
     "spend_180d": 230.55, "spend_90d": 0, "spend_30d": 0, "avg_basket": 23.06,
     "n_products_180d": 7, "spend_trend": -5.44, "is_uk": 1},
    {"customer_id": 2, "recency_days": 10, "n_invoices_180d": 12, "n_invoices_90d": 6,
     "spend_180d": 4200, "spend_90d": 2300, "spend_30d": 900, "avg_basket": 350,
     "n_products_180d": 85, "spend_trend": 0.2, "is_uk": 1},
]}


class PredictRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [EXAMPLE]}}
    as_of: dt.date = Field(default_factory=dt.date.today)  # replay sets simulated time
    instances: list[Instance] = Field(min_length=1, max_length=10_000)


state = {}


@asynccontextmanager
async def lifespan(app):
    state["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
    state["version"] = json.loads((MODEL_DIR / "meta.json").read_text())["version"]
    yield


app = FastAPI(title="churn", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")  # hosted demos open on the interactive API docs


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state["version"]}


@app.post("/predict")
def predict(req: PredictRequest):
    df = pd.DataFrame([i.model_dump() for i in req.instances])
    df["score"] = state["model"].predict_proba(df[FEATURES])[:, 1]
    # Features stored as JSON so the log table survives feature-set changes between versions.
    log = pd.DataFrame({
        "logged_at": dt.datetime.now(dt.UTC).isoformat(), "as_of": str(req.as_of),
        "model_version": state["version"], "customer_id": df["customer_id"],
        "score": df["score"], "features": df[FEATURES].to_json(orient="records", lines=True)
        .splitlines(),
    })
    # ponytail: SQLite append per request; swap for a queue/warehouse sink at real traffic.
    with sqlite3.connect(PRED_DB) as con:
        log.to_sql("predictions", con, if_exists="append", index=False)
    return {
        "model_version": state["version"],
        "predictions": [{"customer_id": int(c), "churn_probability": round(float(s), 4)}
                        for c, s in zip(df["customer_id"], df["score"])],
    }

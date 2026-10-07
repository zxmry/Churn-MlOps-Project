"""FastAPI service: Retention Console page, batch churn scoring, prediction logging to SQLite."""
import datetime as dt
import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, create_model

from churn.features import FEATURES

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
PRED_DB = os.getenv("PRED_DB", "predictions.db")
CONSOLE = Path(__file__).with_name("console.html")

# Shown next to each field in /docs. Windows end the day before `as_of`.
FEATURE_DOCS = {
    "recency_days": "Days since the customer's last purchase.",
    "n_invoices_180d": "Orders in the last 180 days (at least 1: the population is recent buyers).",
    "n_invoices_90d": "Orders in the last 90 days.",
    "spend_180d": "Spend in GBP over the last 180 days.",
    "spend_90d": "Spend in GBP over the last 90 days.",
    "spend_30d": "Spend in GBP over the last 30 days.",
    "avg_basket": "Average order value in GBP: spend_180d / n_invoices_180d.",
    "n_products_180d": "Distinct products bought in the last 180 days.",
    "spend_trend": "log1p(spend last 90d) - log1p(spend the 90d before). Negative = slowing down.",
    "is_uk": "1 if the customer's main country is the United Kingdom, else 0.",
}
assert set(FEATURE_DOCS) == set(FEATURES)

Instance = create_model(
    "Instance",
    customer_id=(int, Field(description="Your customer identifier, echoed back.")),
    **{f: (float, Field(allow_inf_nan=False, description=FEATURE_DOCS[f])) for f in FEATURES},
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
    as_of: dt.date = Field(default_factory=dt.date.today,
                           description="Scoring date. Replay sets simulated dates.")
    instances: list[Instance] = Field(min_length=1, max_length=10_000)
    log: bool = Field(True, description="Set false for what-if simulations, so they are not "
                                        "logged as production traffic and do not skew drift monitoring.")


state = {}


@asynccontextmanager
async def lifespan(app):
    state["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
    state["meta"] = json.loads((MODEL_DIR / "meta.json").read_text())
    state["version"] = state["meta"]["version"]
    yield


app = FastAPI(
    title="Churn Risk API",
    version="1.0",
    description="Scores how likely each customer of a UK online gift wholesaler is to make **no "
                "purchase in the next 90 days**.\n\n"
                "- Open the [Retention Console](/) for the business view.\n"
                "- `POST /predict` scores up to 10,000 customers per call.\n"
                "- Source, results and model card: "
                "[GitHub](https://github.com/zxmry/Churn-MlOps-Project)",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def console():
    return CONSOLE.read_text()


@app.get("/health", summary="Liveness and serving model version")
def health():
    return {"status": "ok", "model_version": state["version"]}


@app.get("/model", summary="Serving model: version, training window and holdout metrics")
def model_info():
    return state["meta"]


@app.post("/predict", summary="Churn probability for a batch of customers")
def predict(req: PredictRequest):
    df = pd.DataFrame([i.model_dump() for i in req.instances])
    df["score"] = state["model"].predict_proba(df[FEATURES])[:, 1]
    if not req.log:
        return _response(df)
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
    return _response(df)


def _response(df: pd.DataFrame) -> dict:
    return {
        "model_version": state["version"],
        "predictions": [{"customer_id": int(c), "churn_probability": round(float(s), 4)}
                        for c, s in zip(df["customer_id"], df["score"])],
    }

"""Quality gate + API contract, on synthetic data so CI needs no dataset."""
import sqlite3

import mlflow
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.metrics import average_precision_score

from churn import serve
from churn.features import FEATURES
from churn.train import TRUSTED_TYPES, gbm


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.gamma(2.0, 30.0, size=(n, len(FEATURES))), columns=FEATURES)
    p = 1 / (1 + np.exp(-(X["recency_days"] - 60) / 20))  # churn driven by recency
    return X, pd.Series(rng.random(n) < p).astype(int)


def test_quality_gate():
    X, y = synthetic()
    model = gbm().fit(X[:2000], y[:2000])
    assert average_precision_score(y[2000:], model.predict_proba(X[2000:])[:, 1]) > 0.75


def test_predict_contract_and_logging(tmp_path, monkeypatch):
    X, y = synthetic(500)
    mlflow.sklearn.save_model(gbm().fit(X, y), tmp_path / "model",
                              skops_trusted_types=TRUSTED_TYPES)
    (tmp_path / "model" / "VERSION").write_text("7")
    monkeypatch.setattr(serve, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(serve, "PRED_DB", str(tmp_path / "pred.db"))

    row = {"customer_id": 42, **X.iloc[0].to_dict()}
    with TestClient(serve.app) as client:
        assert client.get("/health").json()["model_version"] == "7"
        r = client.post("/predict", json={"as_of": "2011-08-01", "instances": [row]})
        assert r.status_code == 200
        body = r.json()
        assert body["model_version"] == "7"
        assert 0 <= body["predictions"][0]["churn_probability"] <= 1
        bad = client.post("/predict", json={"instances": [{**row, "recency_days": "x"}]})
        assert bad.status_code == 422
        assert client.post("/predict", json={"instances": []}).status_code == 422

    with sqlite3.connect(tmp_path / "pred.db") as con:
        logged = pd.read_sql("select * from predictions", con)
    assert len(logged) == 1 and logged.loc[0, "as_of"] == "2011-08-01"

"""Drift report: PSI of logged features/scores vs training, plus delayed ground-truth PR-AUC."""
import json
import sqlite3
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd

from churn.data import CLEAN
from churn.features import FEATURES, HORIZON, SPLITS, build_dataset, build_labels
from churn.serve import MODEL_DIR, PRED_DB
from churn.train import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PSI_ALERT = 0.2
OUT = Path("reports")


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    tx = pd.read_parquet(CLEAN)
    ref = build_dataset(tx, SPLITS["train"])
    ref["score"] = mlflow.sklearn.load_model(str(MODEL_DIR)).predict_proba(ref[FEATURES])[:, 1]
    with sqlite3.connect(PRED_DB) as con:
        logs = pd.read_sql("select * from predictions where model_version = ?", con,
                           params=[(MODEL_DIR / "VERSION").read_text().strip()])
    logs = logs.join(pd.json_normalize(logs["features"].map(json.loads)))

    rows = []
    for as_of, cur in logs.groupby("as_of"):
        t = pd.Timestamp(as_of)
        row = {"as_of": as_of, "n": len(cur), "mean_score": cur["score"].mean()}
        row |= {f"psi_{c}": psi(ref[c].to_numpy(), cur[c].to_numpy()) for c in [*FEATURES, "score"]}
        if t + HORIZON <= tx["ts"].max():  # labels have matured for this month
            y = build_labels(tx, t, pd.Index(cur["customer_id"]))
            m = evaluate(y.to_numpy(), cur["score"].to_numpy())
            row |= {"actual_churn": y.mean(), "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"]}
        rows.append(row)
    report = pd.DataFrame(rows).set_index("as_of")
    report.round(4).to_csv(OUT / "drift.csv")

    psi_cols = [c for c in report if c.startswith("psi_")]
    alerts = report[psi_cols].gt(PSI_ALERT)
    for as_of, flagged in alerts.iterrows():
        names = [c[4:] for c in flagged[flagged].index]
        print(f"{as_of}: {'DRIFT ' + ', '.join(names) if names else 'ok'}")

    top = report[psi_cols].max().nlargest(5).index
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4))
    report[top].rename(columns=lambda c: c[4:]).plot(ax=a, marker="o")
    a.axhline(PSI_ALERT, ls="--", c="red", lw=1, label=f"alert ({PSI_ALERT})")
    a.set(title="PSI vs training distribution", ylabel="PSI", xlabel="")
    a.legend(fontsize=8)
    report[["mean_score", "actual_churn"]].plot(ax=b, marker="o")
    b.axhline(ref["churn"].mean(), ls=":", c="gray", lw=1, label="train churn rate")
    b.set(title="Predicted vs actual churn (labels mature after 90d)", xlabel="", ylim=(0, 1))
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "drift.png", dpi=120)
    print(report[["n", "mean_score", "actual_churn", "pr_auc", "psi_score"]].round(3))


if __name__ == "__main__":
    main()

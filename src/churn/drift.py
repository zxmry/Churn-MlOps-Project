"""Drift report: PSI of logged features/scores vs training, plus delayed ground-truth PR-AUC.

`python -m churn.drift --check` exits 1 when a retrain is recommended, so a scheduler can act on it.
"""
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd

from churn.data import CLEAN
from churn.features import FEATURES, HORIZON, build_dataset, build_labels
from churn.serve import MODEL_DIR, PRED_DB
from churn.train import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PSI_ALERT = 0.2
MAX_PR_AUC_DROP = 0.05  # retrain if matured-label PR-AUC falls this far below the holdout score
OUT = Path("reports")


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def main() -> bool:
    OUT.mkdir(exist_ok=True)
    tx = pd.read_parquet(CLEAN)
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    # Reference = the snapshots this model version was actually trained on.
    ref = build_dataset(tx, pd.date_range(meta["train_start"], meta["train_end"], freq="MS"))
    ref["score"] = mlflow.sklearn.load_model(str(MODEL_DIR)).predict_proba(ref[FEATURES])[:, 1]
    with sqlite3.connect(PRED_DB) as con:
        logs = pd.read_sql("select * from predictions where model_version = ?", con,
                           params=[meta["version"]])
    if logs.empty:
        sys.exit(f"No logged predictions for model v{meta['version']}. Run `make replay` first.")
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
    a.set(title=f"PSI vs training distribution (model v{meta['version']})", ylabel="PSI", xlabel="")
    a.legend(fontsize=8)
    report[["mean_score", "actual_churn"]].plot(ax=b, marker="o")
    b.axhline(ref["churn"].mean(), ls=":", c="gray", lw=1, label="train churn rate")
    b.set(title="Predicted vs actual churn (labels mature after 90d)", xlabel="", ylim=(0, 1))
    b.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "drift.png", dpi=120)
    print(report[["n", "mean_score", "actual_churn", "pr_auc", "psi_score"]].round(3))

    reasons = []
    if report["psi_score"].iloc[-1] > PSI_ALERT:
        reasons.append(f"score PSI {report['psi_score'].iloc[-1]:.2f} > {PSI_ALERT} in latest month")
    floor = meta["metrics"]["test_pr_auc"] - MAX_PR_AUC_DROP
    for as_of, v in report.get("pr_auc", pd.Series(dtype=float)).dropna().items():
        if v < floor:
            reasons.append(f"{as_of} PR-AUC {v:.3f} < floor {floor:.3f}")
    print("RETRAIN: " + "; ".join(reasons) if reasons else "No retrain needed.")
    return bool(reasons)


if __name__ == "__main__":
    retrain = main()
    sys.exit(1 if retrain and "--check" in sys.argv else 0)

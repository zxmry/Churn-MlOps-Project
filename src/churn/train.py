"""Train baseline + gradient boosting sweep, log to MLflow, register the best as @champion."""
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from churn.data import CLEAN, RAW, sha256
from churn.features import FEATURES, SPLITS, build_dataset

MODEL_NAME = "churn-model"
TRACKING_URI = "sqlite:///mlflow.db"
# Types we produce ourselves; skops refuses to (de)serialize anything not listed.
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                 "churn.train.signed_log"]


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def evaluate(y, p) -> dict:
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    precision_top10 = float(np.asarray(y)[top].mean())
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "precision_top10": precision_top10,
        "lift_top10": precision_top10 / float(np.mean(y)),
    }


def bootstrap_pr_auc(y, *scores, n: int = 1000, seed: int = 0) -> np.ndarray:
    """PR-AUC of each score vector on the same n resamples. Shape (n, len(scores)).

    Paired resampling, so column differences give a CI for "model A beats model B".
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    return np.array([[average_precision_score(y[i], np.asarray(p)[i]) for p in scores] for i in idx])


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1 - level) / 2 * 100
    return float(np.percentile(samples, tail)), float(np.percentile(samples, 100 - tail))


def snapshot_params(snaps) -> dict:
    return {"train_start": str(snaps[0].date()), "train_end": str(snaps[-1].date())}


def baseline():
    return make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                         LogisticRegression(max_iter=1000))


def gbm(**params):
    return HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.15,
                                          max_iter=500, random_state=0, **params)


GRID = {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31], "min_samples_leaf": [20, 100],
        "l2_regularization": [0.0, 1.0]}


def git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                       check=False)
    return r.stdout.strip() or "uncommitted"


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("churn")
    tx = pd.read_parquet(CLEAN)
    d = {k: build_dataset(tx, SPLITS[k]) for k in ("train", "val", "test")}
    X = {k: v[FEATURES] for k, v in d.items()}
    y = {k: v["churn"] for k, v in d.items()}
    tags = {"data_sha256": sha256(RAW), "git_sha": git_sha()}

    candidates = [("logreg_baseline", baseline(), {})]
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(GRID, combo))
        candidates.append(("hgb", gbm(**params), params))

    best = None
    for name, model, params in candidates:
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(tags | {"model_family": name})
            mlflow.log_params(params | snapshot_params(SPLITS["train"])
                              | {"n_train": len(X["train"]), "features": ",".join(FEATURES)})
            model.fit(X["train"], y["train"])
            val = evaluate(y["val"], model.predict_proba(X["val"])[:, 1])
            mlflow.log_metrics({f"val_{k}": v for k, v in val.items()})
            print(f"{name:16s} {params} val_pr_auc={val['pr_auc']:.4f}")
            if name == "logreg_baseline":
                base_model = model
            if best is None or val["pr_auc"] > best[2]:
                best = (run.info.run_id, model, val["pr_auc"], name)

    run_id, model, _, name = best
    with mlflow.start_run(run_id=run_id):
        p_test = model.predict_proba(X["test"])[:, 1]
        test = evaluate(y["test"], p_test)
        boot = bootstrap_pr_auc(y["test"], p_test, base_model.predict_proba(X["test"])[:, 1])
        test["pr_auc_ci_low"], test["pr_auc_ci_high"] = ci(boot[:, 0])
        test["pr_auc_vs_baseline_ci_low"], test["pr_auc_vs_baseline_ci_high"] = ci(boot[:, 0] - boot[:, 1])
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME,
                                        input_example=X["train"].head(3),
                                        skops_trusted_types=TRUSTED_TYPES)
    version = info.registered_model_version
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"champion: {name} run={run_id} v{version} test={ {k: round(v, 4) for k, v in test.items()} }")
    export()


def export(dst: str = "model") -> None:
    """Copy the @champion model out of the registry into ./model for serving/Docker."""
    mlflow.set_tracking_uri(TRACKING_URI)
    shutil.rmtree(dst, ignore_errors=True)
    mlflow.artifacts.download_artifacts(f"models:/{MODEL_NAME}@champion", dst_path=dst)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    run = client.get_run(mv.run_id)
    meta = {"version": str(mv.version), "run_id": mv.run_id,
            "train_start": run.data.params["train_start"], "train_end": run.data.params["train_end"],
            "metrics": {k: round(v, 4) for k, v in run.data.metrics.items()}}
    Path(dst, "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"exported {MODEL_NAME} v{mv.version} -> {dst}/")


if __name__ == "__main__":
    export() if sys.argv[1:] == ["export"] else main()

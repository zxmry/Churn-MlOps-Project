"""Champion/challenger retrain.

Train a challenger on every matured snapshot, score it and the current @champion on the most
recent matured months, and move the alias only if the challenger is better with confidence.
"""
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from churn.data import CLEAN, RAW, sha256
from churn.features import FEATURES, HORIZON, build_dataset, matured_snapshots
from churn.train import (
    GRID,
    MODEL_NAME,
    TRACKING_URI,
    TRUSTED_TYPES,
    baseline,
    bootstrap_pr_auc,
    ci,
    evaluate,
    export,
    gbm,
    git_sha,
    snapshot_params,
)

EVAL_MONTHS = 2  # most recent matured months, used to compare champion vs challenger
CONFIDENCE = 0.90  # promote only if the lower bound of this CI on the PR-AUC gain is > 0
BRIER_TOLERANCE = 0.005  # guardrail: calibration may not get worse than this


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("churn")
    client = MlflowClient()
    tx = pd.read_parquet(CLEAN)

    matured = matured_snapshots(tx)
    eval_snaps = matured[-EVAL_MONTHS:]
    train_snaps = matured[matured + HORIZON <= eval_snaps[0]]  # embargo before the eval window
    train, recent = build_dataset(tx, train_snaps), build_dataset(tx, eval_snaps)

    champ_mv = client.get_model_version_by_alias(MODEL_NAME, "champion")
    champ_run = client.get_run(champ_mv.run_id)
    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    family = champ_run.data.tags["model_family"]
    # Same family and hyperparameters as the champion; only the training window changes.
    params = {k: type(GRID[k][0])(champ_run.data.params[k]) for k in GRID if k in champ_run.data.params}
    challenger = baseline() if family == "logreg_baseline" else gbm(**params)

    with mlflow.start_run(run_name="retrain-challenger"):
        mlflow.set_tags({"model_family": family, "data_sha256": sha256(RAW), "git_sha": git_sha(),
                         "champion_version": champ_mv.version})
        mlflow.log_params(params | snapshot_params(train_snaps) | {
            "n_train": len(train), "features": ",".join(FEATURES),
            "eval_start": str(eval_snaps[0].date()), "eval_end": str(eval_snaps[-1].date())})
        challenger.fit(train[FEATURES], train["churn"])

        y = recent["churn"]
        p_new = challenger.predict_proba(recent[FEATURES])[:, 1]
        p_old = champion.predict_proba(recent[list(champion.feature_names_in_)])[:, 1]
        new, old = evaluate(y, p_new), evaluate(y, p_old)
        boot = bootstrap_pr_auc(y, p_new, p_old)
        gain_low, gain_high = ci(boot[:, 0] - boot[:, 1], CONFIDENCE)
        new["pr_auc_ci_low"], new["pr_auc_ci_high"] = ci(boot[:, 0])
        ranks_better = gain_low > 0
        calibration_ok = new["brier"] <= old["brier"] + BRIER_TOLERANCE
        promote = ranks_better and calibration_ok

        mlflow.log_metrics({f"test_{k}": v for k, v in new.items()}
                           | {f"champion_test_{k}": v for k, v in old.items()}
                           | {"pr_auc_gain_ci_low": gain_low, "pr_auc_gain_ci_high": gain_high})
        mlflow.set_tag("decision", "promote" if promote else "keep")
        if promote:
            info = mlflow.sklearn.log_model(challenger, name="model", registered_model_name=MODEL_NAME,
                                            input_example=train[FEATURES].head(3),
                                            skops_trusted_types=TRUSTED_TYPES)
            client.set_registered_model_alias(MODEL_NAME, "champion", info.registered_model_version)

    print(f"train {train_snaps[0].date()}..{train_snaps[-1].date()} ({len(train):,} rows), "
          f"eval {eval_snaps[0].date()}..{eval_snaps[-1].date()} ({len(recent):,} rows)")
    print(f"champion v{champ_mv.version}: PR-AUC {old['pr_auc']:.4f}  Brier {old['brier']:.4f}")
    print(f"challenger:  PR-AUC {new['pr_auc']:.4f}  Brier {new['brier']:.4f}")
    print(f"PR-AUC gain {CONFIDENCE:.0%} CI [{gain_low:+.4f}, {gain_high:+.4f}]: "
          f"{'significant' if ranks_better else 'not significant'}")
    print(f"Brier change {new['brier'] - old['brier']:+.4f} (guardrail +{BRIER_TOLERANCE}): "
          f"{'ok' if calibration_ok else 'FAILED'}")
    print("PROMOTE challenger" if promote else "KEEP champion")
    if promote:
        export()


if __name__ == "__main__":
    main()

"""Baseline models under every split scheme, to measure how much each split inflates scores.

Hyperparameters are fixed in params.yaml at this stage. Scalers live inside the
pipeline, so they are fitted on training folds only.
"""
import warnings

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import PROCESSED, ROOT, TARGET_LEVELS, TARGETS, load_params

# Each cooler level exists in only one period, so holding a period out removes the class.
NOT_APPLICABLE = {("cooler", "leave_period_out")}


def build_models(params):
    return {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logreg": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=params["logreg_c"], max_iter=3000)
        ),
        "lightgbm": lambda: lgb.LGBMClassifier(random_state=params["seed"], verbose=-1, **params["lightgbm"]),
    }


def score(y_true, y_pred, probabilities, classes, target):
    present = np.unique(y_true)
    precision = []
    for level in present:
        column = probabilities[:, list(classes).index(level)] if level in classes else np.zeros(len(y_true))
        precision.append(average_precision_score(y_true == level, column))
    rank = {level: index for index, level in enumerate(TARGET_LEVELS[target])}
    true_rank, pred_rank = [np.array([rank[v] for v in a]) for a in (y_true, y_pred)]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_pr_auc": float(np.mean(precision)),
        "ordinal_mae": float(np.abs(true_rank - pred_rank).mean()),
    }


def variant_data(features, labels, variant):
    columns = [c for c in features.columns if not (variant == "no_ps4" and c.startswith("PS4__"))]
    keep = labels["stable"].to_numpy() == 0 if variant == "stable_only" else np.ones(len(labels), dtype=bool)
    return features[columns].to_numpy(), keep


def main():
    params = load_params()["baseline"]
    features = pd.read_parquet(PROCESSED / "features.parquet")
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    folds = pd.read_parquet(PROCESSED / "folds.parquet")
    models = build_models(params)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("baseline")

    rows = []
    for variant in params["variants"]:
        X, keep = variant_data(features, labels, variant)
        names = list(models) if variant == "base" else ["lightgbm"]
        for target in TARGETS:
            y = labels[target].to_numpy()
            for scheme, scheme_folds in folds.groupby("scheme"):
                if (target, scheme) in NOT_APPLICABLE:
                    continue
                for model_name in names:
                    fold_rows = []
                    for fold, part in scheme_folds.groupby("fold"):
                        train = part.loc[part["role"] == "train", "cycle"].to_numpy()
                        test = part.loc[part["role"] == "test", "cycle"].to_numpy()
                        train, test = train[keep[train]], test[keep[test]]
                        model = models[model_name]().fit(X[train], y[train])
                        result = score(y[test], model.predict(X[test]), model.predict_proba(X[test]), model.classes_, target)
                        fold_rows.append(
                            {"variant": variant, "target": target, "scheme": scheme, "model": model_name, "fold": fold,
                             "n_train": len(train), "n_test": len(test), **result}
                        )
                    frame = pd.DataFrame(fold_rows)
                    rows.append(frame)
                    means = frame[["accuracy", "balanced_accuracy", "macro_f1", "macro_pr_auc", "ordinal_mae"]].mean()
                    with mlflow.start_run(run_name=f"{variant}/{target}/{scheme}/{model_name}"):
                        mlflow.log_params({"variant": variant, "target": target, "scheme": scheme, "model": model_name})
                        mlflow.log_metrics({k: float(v) for k, v in means.items()})
                    print(f"{variant:12s}{target:12s}{scheme:18s}{model_name:10s}"
                          f"f1={means['macro_f1']:.3f} pr={means['macro_pr_auc']:.3f}", flush=True)

    output = ROOT / "reports" / "baseline"
    output.mkdir(parents=True, exist_ok=True)
    all_folds = pd.concat(rows, ignore_index=True)
    all_folds.to_csv(output / "folds.csv", index=False)
    summary = (
        all_folds.groupby(["variant", "target", "scheme", "model"])
        .agg(
            macro_f1=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_pr_auc=("macro_pr_auc", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            accuracy=("accuracy", "mean"),
            ordinal_mae=("ordinal_mae", "mean"),
            folds=("fold", "count"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)


if __name__ == "__main__":
    main()

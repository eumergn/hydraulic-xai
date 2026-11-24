"""Compare the model families. LightGBM is tuned on inner folds that mirror the outer split."""
import json
import warnings
from collections import Counter

import mlflow
import numpy as np
import pandas as pd

from common import ROOT, load_params
from modeling import fold_indices, load_processed, make_dummy, make_lightgbm, make_logreg, make_mlp, make_shallow_tree, score, tune_lightgbm

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "macro_pr_auc", "ordinal_mae"]


def main():
    params = load_params()["models"]
    features, labels, folds = load_processed()
    X = features.to_numpy()
    seed = params["seed"]
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("models")

    builders = {
        "dummy": make_dummy,
        "shallow_tree": lambda: make_shallow_tree(params["shallow_tree_depth"], seed),
        "logreg": lambda: make_logreg(params["logreg_c"]),
        "mlp": lambda: make_mlp(params["mlp"], seed),
    }
    rows, chosen = [], []
    for target in params["targets"]:
        y = labels[target].to_numpy()
        for scheme in params["schemes"]:
            scheme_folds = sorted(folds.loc[folds["scheme"] == scheme, "fold"].unique())
            for model_name in [*builders, "lightgbm"]:
                fold_rows = []
                for fold in scheme_folds:
                    train, test = fold_indices(folds, scheme, fold)
                    if model_name == "lightgbm":
                        tuned, inner = tune_lightgbm(
                            X, y, labels, train, scheme, params["lightgbm_fixed"], params["lightgbm_grid"], target, seed
                        )
                        chosen.append({"target": target, "scheme": scheme, "fold": int(fold), **tuned, "inner_f1": max(inner)})
                        model = make_lightgbm(params["lightgbm_fixed"], tuned, seed)
                    else:
                        model = builders[model_name]()
                    model.fit(X[train], y[train])
                    result = score(y[test], model.predict(X[test]), model.predict_proba(X[test]), model.classes_, target)
                    fold_rows.append({"target": target, "scheme": scheme, "model": model_name, "fold": int(fold), **result})
                frame = pd.DataFrame(fold_rows)
                rows.append(frame)
                means = frame[METRICS].mean()
                with mlflow.start_run(run_name=f"{target}/{scheme}/{model_name}"):
                    mlflow.log_params({"target": target, "scheme": scheme, "model": model_name})
                    mlflow.log_metrics({name: float(value) for name, value in means.items()})
                print(f"{target:12s}{scheme:18s}{model_name:14s}f1={means['macro_f1']:.3f} pr={means['macro_pr_auc']:.3f}", flush=True)

    output = ROOT / "reports" / "models"
    output.mkdir(parents=True, exist_ok=True)
    all_folds = pd.concat(rows, ignore_index=True)
    all_folds.to_csv(output / "folds.csv", index=False)
    summary = all_folds.groupby(["target", "scheme", "model"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)

    selections = pd.DataFrame(chosen)
    selections.to_csv(output / "lightgbm_selection.csv", index=False)
    best = {}
    for target, part in selections[selections["scheme"] == "leave_period_out"].groupby("target"):
        pick = Counter(zip(part["num_leaves"], part["colsample_bytree"])).most_common(1)[0][0]
        best[target] = {"num_leaves": int(pick[0]), "colsample_bytree": float(pick[1])}
    (output / "best_lightgbm.json").write_text(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()

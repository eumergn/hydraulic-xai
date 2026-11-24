"""Robustness of trained models to feature noise and to the loss of sensors.

Evaluated on the leave_period_out folds. A lost sensor has all its window features
replaced by the training-fold means, as a naive imputation would do in production.
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import ROOT, SENSOR_COLUMNS, load_params
from modeling import fold_indices, load_processed, make_lightgbm, make_logreg, make_mlp

FAMILIES = {
    "pressure": [f"PS{i}" for i in range(1, 7)],
    "flow": ["FS1", "FS2"],
    "temperature": [f"TS{i}" for i in range(1, 5)],
    "motor_power": ["EPS1"],
    "vibration": ["VS1"],
    "virtual": ["CE", "CP", "SE"],
}


def sensor_columns(columns, sensors):
    wanted = {sensor for sensor in sensors}
    return [index for index, name in enumerate(columns) if name.split("__")[0] in wanted]


def main():
    all_params = load_params()
    params, robustness = all_params["models"], all_params["robustness"]
    best = json.loads((ROOT / "reports" / "models" / "best_lightgbm.json").read_text())
    features, labels, folds = load_processed()
    X = features.to_numpy()
    columns = list(features.columns)
    rng = np.random.default_rng(robustness["seed"])
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    rows = []
    for target in params["targets"]:
        y = labels[target].to_numpy()
        builders = {
            "lightgbm": lambda: make_lightgbm(params["lightgbm_fixed"], best[target], params["seed"]),
            "mlp": lambda: make_mlp(params["mlp"], params["seed"]),
            "logreg": lambda: make_logreg(params["logreg_c"]),
        }
        for fold in sorted(folds.loc[folds["scheme"] == "leave_period_out", "fold"].unique()):
            train, test = fold_indices(folds, "leave_period_out", fold)
            means, scales = X[train].mean(axis=0), X[train].std(axis=0)
            for model_name, build in builders.items():
                model = build().fit(X[train], y[train])

                def record(kind, level, inputs):
                    value = f1_score(y[test], model.predict(inputs), average="macro")
                    rows.append({"target": target, "fold": int(fold), "model": model_name, "kind": kind, "level": str(level), "macro_f1": float(value)})

                for noise in robustness["noise_levels"]:
                    record("noise", noise, X[test] + rng.normal(size=X[test].shape) * scales * noise)
                for sensor in SENSOR_COLUMNS:
                    damaged = X[test].copy()
                    damaged[:, sensor_columns(columns, [sensor])] = means[sensor_columns(columns, [sensor])]
                    record("sensor_loss", sensor, damaged)
                for family, sensors in FAMILIES.items():
                    damaged = X[test].copy()
                    index = sensor_columns(columns, sensors)
                    damaged[:, index] = means[index]
                    record("family_loss", family, damaged)
                print(f"{target} fold {fold} {model_name}", flush=True)

    output = ROOT / "reports" / "robustness"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "folds.csv", index=False)
    frame.groupby(["target", "model", "kind", "level"], sort=False)["macro_f1"].agg(["mean", "std"]).reset_index().to_csv(
        output / "summary.csv", index=False
    )


if __name__ == "__main__":
    main()

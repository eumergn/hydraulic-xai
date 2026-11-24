"""Counterfactual explanations with DiCE, and an evaluation of their quality.

A technician cannot set a sensor value directly. A counterfactual is read as
"what the measurements of this cycle would look like if the model saw the healthy state".
Only the top SHAP features vary, because DiCE samples poorly in 510 dimensions.
"""
import json
import warnings

import dice_ml
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from common import ROOT, TARGET_LEVELS, load_params
from modeling import fold_indices, load_processed, make_lightgbm

OUTPUT = ROOT / "reports" / "xai"
N_QUERIES = 12
N_COUNTERFACTUALS = 3
N_VARY = 12
HEALTHY = {"valve": 100, "accumulator": 130}


class IndexModel:
    """Exposes a fitted LightGBM model to DiCE through integer class indices and DataFrames."""

    def __init__(self, model, columns):
        self.model, self.columns = model, columns

    def predict_proba(self, frame):
        return self.model.predict_proba(frame[self.columns].to_numpy(dtype=np.float32))

    def predict(self, frame):
        return self.predict_proba(frame).argmax(axis=1)


def evaluate(counterfactual, query, training, scale, wrapper, desired, varied):
    values = counterfactual[list(training.columns)].to_numpy(dtype=np.float32)
    changes = (np.abs(values - query) > 1e-6 * (np.abs(query) + 1)).sum(axis=1)
    distance = (np.abs(values - query) / scale).sum(axis=1) / max(1, len(varied))
    finder = NearestNeighbors(n_neighbors=2).fit(training.to_numpy() / scale)
    real_gap = finder.kneighbors(training.to_numpy() / scale)[0][:, 1]
    gap = finder.kneighbors(values / scale, n_neighbors=1)[0][:, 0]
    return {
        "validity": float((wrapper.predict(counterfactual[list(training.columns)]) == desired).mean()),
        "n_changed": float(changes.mean()),
        "proximity": float(distance.mean()),
        "plausibility_ratio": float(gap.mean() / np.median(real_gap)),
    }


def run_context(name, params, fixed, best, data):
    context = next(c for c in params["contexts"] if c["name"] == name)
    target = context["target"]
    features, labels, folds = data
    columns = list(features.columns)
    train, test = fold_indices(folds, context["scheme"], context["fold"])
    y = labels[target].to_numpy()
    model = make_lightgbm(fixed, best[target], params["seed"]).fit(features.to_numpy(dtype=np.float32)[train], y[train])

    importance = pd.read_parquet(OUTPUT / "importances.parquet")
    shap_values = importance[(importance["context"] == name) & (importance["model"] == "lightgbm") & (importance["method"] == "shap")]
    varied = shap_values.sort_values("importance", ascending=False)["feature"].head(N_VARY).tolist()

    wrapper = IndexModel(model, columns)
    training = features.iloc[train]
    frame = training.copy().astype(float)
    frame["outcome"] = np.searchsorted(model.classes_, y[train])
    dice_data = dice_ml.Data(dataframe=frame, continuous_features=columns, outcome_name="outcome")
    dice_model = dice_ml.Model(model=wrapper, backend="sklearn", model_type="classifier")
    explainer = dice_ml.Dice(dice_data, dice_model, method="random")

    healthy = HEALTHY[target]
    desired = int(np.searchsorted(model.classes_, healthy))
    predicted = model.predict(features.to_numpy(dtype=np.float32)[test])
    candidates = test[(y[test] != healthy) & (predicted == y[test])]
    rng = np.random.default_rng(params["seed"])
    queries = rng.choice(candidates, size=min(N_QUERIES, len(candidates)), replace=False)
    scale = training.std().replace(0, 1).to_numpy()

    rows, examples = [], []
    for cycle in queries:
        query = features.iloc[[cycle]].astype(float)
        try:
            result = explainer.generate_counterfactuals(query, total_CFs=N_COUNTERFACTUALS, desired_class=desired, features_to_vary=varied, random_seed=params["seed"], verbose=False)
            found = result.cf_examples_list[0].final_cfs_df
        except Exception as error:
            rows.append({"context": name, "cycle": int(cycle), "found": False, "error": type(error).__name__})
            continue
        if found is None or found.empty:
            rows.append({"context": name, "cycle": int(cycle), "found": False, "error": "none"})
            continue
        rows.append({"context": name, "cycle": int(cycle), "found": True, "true_level": int(y[cycle]),
                     **evaluate(found, query.to_numpy()[0], training, scale, wrapper, desired, varied)})
        if len(examples) < 3:
            first = found.iloc[0]
            changed = [(c, float(query[c].iloc[0]), float(first[c])) for c in varied if abs(float(first[c]) - float(query[c].iloc[0])) > 1e-6]
            examples.append({"cycle": int(cycle), "true_level": int(y[cycle]), "desired_level": healthy,
                             "changes": [{"feature": c, "from": a, "to": b} for c, a, b in changed]})
        print(f"{name} cycle {cycle}: {rows[-1].get('validity')}", flush=True)
    return rows, examples


def main():
    all_params = load_params()
    params = all_params["xai"]
    fixed = all_params["models"]["lightgbm_fixed"]
    best = json.loads((ROOT / "reports" / "models" / "best_lightgbm.json").read_text())
    warnings.filterwarnings("ignore")
    data = load_processed()
    rows, examples = [], {}
    for name in ["valve_lopo", "accumulator_lopo"]:
        context_rows, context_examples = run_context(name, params, fixed, best, data)
        rows += context_rows
        examples[name] = context_examples
    pd.DataFrame(rows).to_csv(OUTPUT / "counterfactuals.csv", index=False)
    (OUTPUT / "counterfactual_examples.json").write_text(json.dumps(examples, indent=2))


if __name__ == "__main__":
    main()

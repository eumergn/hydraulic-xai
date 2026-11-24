"""Compute explanations and, above all, evaluate them.

Methods: permutation importance, tree SHAP (LightGBM), Integrated Gradients (MLP), LIME,
ALE and PDP, a shallow tree and an EBM. Evaluation: deletion curves at feature and sensor
level, retraining without top sensors, stability, agreement between methods, parameter and
label randomization (Adebayo et al., 2018) and comparison with docs/hypotheses_metier.md.
"""
import copy
import itertools
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
import torch
from captum.attr import IntegratedGradients
from interpret.glassbox import ExplainableBoostingClassifier
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from torch import nn

from common import ROOT, load_params
from modeling import fold_indices, load_processed, make_lightgbm, make_mlp, make_shallow_tree

OUTPUT = ROOT / "reports" / "xai"
MODELS = ROOT / "models"


def sensors_of(columns):
    return np.array([name.split("__")[0] for name in columns])


def sensor_scores(importance, names):
    return pd.Series(np.abs(importance)).groupby(names).sum()


def rho(first, second):
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(spearmanr(first, second).statistic)


def jaccard_top(first, second, k):
    a, b = set(np.argsort(-np.abs(first))[:k]), set(np.argsort(-np.abs(second))[:k])
    return len(a & b) / len(a | b)


def cross_entropy(probabilities, indices):
    return float(-np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-9, 1)).mean())


def permutation_importance(model, X, y, repeats, rng):
    indices = np.searchsorted(model.classes_, y)
    base = cross_entropy(model.predict_proba(X), indices)
    result = np.zeros(X.shape[1])
    for column in range(X.shape[1]):
        losses = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[:, column] = rng.permutation(shuffled[:, column])
            losses.append(cross_entropy(model.predict_proba(shuffled), indices))
        result[column] = np.mean(losses) - base
    return result


def tree_shap_local(model, X):
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    predicted = model.predict_proba(X).argmax(axis=1)
    return values[np.arange(len(X)), :, predicted]


def integrated_gradients_local(model, X, steps):
    inputs = model._scale(X)
    predicted = torch.tensor(model.predict_proba(X).argmax(axis=1))
    method = IntegratedGradients(model.net_)
    attributions = method.attribute(inputs, baselines=torch.zeros_like(inputs), target=predicted, n_steps=steps)
    return attributions.detach().numpy()


def lime_local(model, background, X, samples, seed):
    explainer = LimeTabularExplainer(
        background, mode="classification", discretize_continuous=False, feature_selection="none",
        sample_around_instance=True, random_state=seed,
    )
    output = np.zeros(X.shape)
    predicted = model.predict_proba(X).argmax(axis=1)
    for row, label in enumerate(predicted):
        explanation = explainer.explain_instance(
            X[row], model.predict_proba, labels=(int(label),), top_labels=None,
            num_features=X.shape[1], num_samples=samples,
        )
        for column, weight in explanation.local_exp[int(label)]:
            output[row, column] = weight
    return output


def deletion_curve(model, X, reference, means, ranking, counts):
    predicted = model.predict_proba(X).argmax(axis=1)
    curve = []
    for count in counts:
        damaged = X.copy()
        damaged[:, ranking[:count]] = means[ranking[:count]]
        probabilities = model.predict_proba(damaged)
        curve.append(
            {
                "count": int(count),
                "confidence": float(probabilities[np.arange(len(X)), predicted].mean()),
                "macro_f1": float(f1_score(reference, model.classes_[probabilities.argmax(axis=1)], average="macro")),
            }
        )
    return curve


def area(curve, key):
    values = [point[key] for point in curve]
    return float(np.trapezoid(values, dx=1.0 / (len(values) - 1)))


def evaluate_deletion(model, model_name, rankings, X, y, means, counts, context, rng, repeats, level, groups=None):
    rows, areas = [], []
    p = X.shape[1]

    def expand(order):
        if groups is None:
            return np.array(order)
        return np.concatenate([np.flatnonzero(groups == name) for name in order])

    def counts_for(order):
        if groups is None:
            return counts
        sizes = np.cumsum([0] + [int((groups == name).sum()) for name in order])
        return sizes.tolist()

    for method, order in rankings.items():
        curve = deletion_curve(model, X, y, means, expand(order), counts_for(order))
        rows += [{"context": context, "model": model_name, "level": level, "method": method, "step": step, **point}
                 for step, point in enumerate(curve)]
        areas.append({"context": context, "model": model_name, "level": level, "method": method,
                      "confidence_auc": area(curve, "confidence"), "f1_auc": area(curve, "macro_f1")})

    random_areas, random_curves = [], []
    population = np.arange(p) if groups is None else np.unique(groups)
    for _ in range(repeats):
        order = rng.permutation(population)
        curve = deletion_curve(model, X, y, means, expand(order), counts_for(order))
        random_curves.append(curve)
        random_areas.append({"confidence_auc": area(curve, "confidence"), "f1_auc": area(curve, "macro_f1")})
    for step in range(len(random_curves[0])):
        rows.append({"context": context, "model": model_name, "level": level, "method": "random", "step": step,
                     "count": int(np.mean([c[step]["count"] for c in random_curves])),
                     "confidence": float(np.mean([c[step]["confidence"] for c in random_curves])),
                     "macro_f1": float(np.mean([c[step]["macro_f1"] for c in random_curves]))})
    random_frame = pd.DataFrame(random_areas)
    areas.append({"context": context, "model": model_name, "level": level, "method": "random",
                  "confidence_auc": float(random_frame["confidence_auc"].mean()), "f1_auc": float(random_frame["f1_auc"].mean()),
                  "confidence_auc_std": float(random_frame["confidence_auc"].std())})
    return rows, areas


def resample_runs(train, labels, rng):
    runs = labels["run"].to_numpy()[train]
    unique = np.unique(runs)
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([train[runs == run] for run in chosen])


def cascading_randomization(model, X, steps, reference_global, seed):
    torch.manual_seed(seed)
    randomized = copy.deepcopy(model)
    linear = [layer for layer in randomized.net_.layers if isinstance(layer, nn.Linear)]
    rows, reference_local = [], integrated_gradients_local(model, X, steps)
    for depth, layer in enumerate(reversed(linear), start=1):
        layer.reset_parameters()
        local = integrated_gradients_local(randomized, X, steps)
        per_cycle = [rho(np.abs(local[i]), np.abs(reference_local[i])) for i in range(len(X))]
        rows.append({"layers_randomized": depth, "global_spearman": rho(np.abs(local).mean(axis=0), reference_global),
                     "local_spearman": float(np.nanmean(per_cycle))})
    return rows


def ale_and_pdp(model, X, column, class_index, bins=10):
    edges = np.unique(np.quantile(X[:, column], np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return []
    bucket = np.clip(np.searchsorted(edges, X[:, column], side="left") - 1, 0, len(edges) - 2)
    effects = []
    for index in range(len(edges) - 1):
        members = X[bucket == index]
        if len(members) == 0:
            effects.append(0.0)
            continue
        low, high = members.copy(), members.copy()
        low[:, column], high[:, column] = edges[index], edges[index + 1]
        effects.append(float((model.predict_proba(high)[:, class_index] - model.predict_proba(low)[:, class_index]).mean()))
    accumulated = np.cumsum(effects)
    ale = accumulated - accumulated.mean()
    rows = []
    for index in range(len(edges) - 1):
        grid_value = (edges[index] + edges[index + 1]) / 2
        modified = X.copy()
        modified[:, column] = grid_value
        rows.append({"value": float(grid_value), "ale": float(ale[index]), "pdp": float(model.predict_proba(modified)[:, class_index].mean())})
    return rows


def evaluate_domain(rankings, columns, expected, thermal, k, context, model_name):
    rows = []
    for method, order in rankings.items():
        top = [columns[i].split("__")[0] for i in order[:k]]
        share = sum(name in expected for name in top) / k
        rows.append({"context": context, "model": model_name, "method": method, "expected_share": share,
                     "thermal_share": sum(name in thermal for name in top) / k, "consistent": bool(share >= 0.5),
                     "top_features": "; ".join(columns[i] for i in order[:k])})
    return rows


def run_context(context, params, fixed, mlp_config, best, data):
    features, labels, folds = data
    name, target = context["name"], context["target"]
    seed = params["seed"]
    rng = np.random.default_rng(seed)
    X, columns = features.to_numpy(dtype=np.float32), list(features.columns)
    groups = sensors_of(columns)
    y = labels[target].to_numpy()
    train, test = fold_indices(folds, context["scheme"], context["fold"])
    subset = np.sort(rng.choice(test, size=min(params["n_test"], len(test)), replace=False))
    background = X[rng.choice(train, size=min(params["n_background"], len(train)), replace=False)]
    means = X[train].mean(axis=0)
    X_test, y_test = X[subset], y[subset]
    print(f"== {name}", flush=True)

    lightgbm_params = best[target]
    lightgbm_model = make_lightgbm(fixed, lightgbm_params, seed).fit(X[train], y[train])
    mlp_model = make_mlp(mlp_config, seed).fit(X[train], y[train])
    performance = {m: float(f1_score(y[test], model.predict(X[test]), average="macro"))
                   for m, model in (("lightgbm", lightgbm_model), ("mlp", mlp_model))}

    shallow = make_shallow_tree(4, seed).fit(X[train], y[train])
    top_columns = np.argsort(-lightgbm_model.booster_.feature_importance("gain"))[:30]
    ebm = ExplainableBoostingClassifier(interactions=0, outer_bags=2, n_jobs=4, random_state=seed, feature_names=[columns[i] for i in top_columns])
    ebm.fit(X[train][:, top_columns], y[train])
    ebm_importance = np.zeros(X.shape[1])
    ebm_importance[top_columns] = ebm.term_importances()
    performance["shallow_tree"] = float(f1_score(y[test], shallow.predict(X[test]), average="macro"))
    performance["ebm"] = float(f1_score(y[test], ebm.predict(X[test][:, top_columns]), average="macro"))
    print(f"performance {performance}", flush=True)

    outputs = {"deletion": [], "areas": [], "stability": [], "agreement": [], "randomization": [], "domain": [], "ale": [], "retrain": []}
    importances_by_model, local_examples = {}, {}
    for model_name, model in (("lightgbm", lightgbm_model), ("mlp", mlp_model)):
        attribution_name = "shap" if model_name == "lightgbm" else "integrated_gradients"
        local_fn = (lambda m, data_: tree_shap_local(m, data_)) if model_name == "lightgbm" else (lambda m, data_: integrated_gradients_local(m, data_, params["ig_steps"]))
        attribution = local_fn(model, X_test)
        global_attribution = np.abs(attribution).mean(axis=0)

        lime_rows = rng.choice(len(X_test), size=min(params["n_lime"], len(X_test)), replace=False)
        lime_runs = [lime_local(model, background, X_test[lime_rows], params["lime_samples"], seed + repeat) for repeat in range(3)]
        lime_global = np.abs(lime_runs[0]).mean(axis=0)
        permutation = permutation_importance(model, X_test, y_test, params["n_permutation_repeats"], rng)
        importances = {"permutation": permutation, attribution_name: global_attribution, "lime": lime_global}
        if model_name == "lightgbm":
            importances |= {"shallow_tree": shallow.feature_importances_, "ebm": ebm_importance}
        importances_by_model[model_name] = importances

        own = ["permutation", attribution_name, "lime"]
        rankings = {method: np.argsort(-importances[method]) for method in own}
        rows, areas = evaluate_deletion(model, model_name, rankings, X_test, y_test, means, params["deletion_feature_counts"], name, rng, params["n_random_orders"], "feature")
        outputs["deletion"] += rows
        outputs["areas"] += areas
        sensor_rankings = {method: sensor_scores(importances[method], groups).sort_values(ascending=False).index.to_numpy() for method in own}
        rows, areas = evaluate_deletion(model, model_name, sensor_rankings, X_test, y_test, means, None, name, rng, params["n_random_orders"], "sensor", groups=groups)
        outputs["deletion"] += rows
        outputs["areas"] += areas

        for first, second in itertools.combinations(importances, 2):
            outputs["agreement"].append({
                "context": name, "model": model_name, "first": first, "second": second,
                "feature_spearman": rho(np.abs(importances[first]), np.abs(importances[second])),
                "top_k_jaccard": jaccard_top(importances[first], importances[second], params["top_k"]),
                "sensor_spearman": rho(sensor_scores(importances[first], groups).to_numpy(), sensor_scores(importances[second], groups).to_numpy()),
            })

        all_rankings = {method: np.argsort(-np.abs(values)) for method, values in importances.items()}
        outputs["domain"] += evaluate_domain(all_rankings, columns, params["expected_sensors"].get(target, []), params["thermal_sensors"], params["top_k"], name, model_name)

        repeats = params["n_bootstrap_lightgbm"] if model_name == "lightgbm" else params["n_bootstrap_mlp"]
        for repeat in range(repeats):
            rows_boot = resample_runs(train, labels, rng)
            boot_model = (make_lightgbm(fixed, lightgbm_params, seed + repeat + 1) if model_name == "lightgbm" else make_mlp(mlp_config, seed + repeat + 1)).fit(X[rows_boot], y[rows_boot])
            boot_global = np.abs(local_fn(boot_model, X_test)).mean(axis=0)
            outputs["stability"].append({"context": name, "model": model_name, "method": attribution_name, "test": "bootstrap_retrain", "repeat": repeat,
                                         "spearman": rho(boot_global, global_attribution), "top_k_jaccard": jaccard_top(boot_global, global_attribution, params["top_k"])})
        scales = X[train].std(axis=0)
        probe = X_test[: min(50, len(X_test))]
        noisy = probe + rng.normal(size=probe.shape) * scales * 0.01
        clean, perturbed = local_fn(model, probe), local_fn(model, noisy)
        outputs["stability"].append({"context": name, "model": model_name, "method": attribution_name, "test": "input_noise_1pct", "repeat": 0,
                                     "spearman": float(np.nanmean([rho(np.abs(clean[i]), np.abs(perturbed[i])) for i in range(len(probe))])),
                                     "top_k_jaccard": float(np.mean([jaccard_top(clean[i], perturbed[i], params["top_k"]) for i in range(len(probe))]))})
        for first, second in itertools.combinations(range(3), 2):
            per_cycle = [rho(np.abs(lime_runs[first][i]), np.abs(lime_runs[second][i])) for i in range(len(lime_rows))]
            top = [jaccard_top(lime_runs[first][i], lime_runs[second][i], params["top_k"]) for i in range(len(lime_rows))]
            outputs["stability"].append({"context": name, "model": model_name, "method": "lime", "test": "seed", "repeat": first + second,
                                         "spearman": float(np.nanmean(per_cycle)), "top_k_jaccard": float(np.mean(top))})

        if model_name == "lightgbm":
            probe_lime = X_test[:10]
            first_run = lime_local(model, background, probe_lime, params["lime_samples_high"], seed)
            second_run = lime_local(model, background, probe_lime, params["lime_samples_high"], seed + 1)
            outputs["stability"].append({
                "context": name, "model": model_name, "method": "lime", "test": f"seed_{params['lime_samples_high']}_samples", "repeat": 0,
                "spearman": float(np.nanmean([rho(np.abs(first_run[i]), np.abs(second_run[i])) for i in range(len(probe_lime))])),
                "top_k_jaccard": float(np.mean([jaccard_top(first_run[i], second_run[i], params["top_k"]) for i in range(len(probe_lime))])),
            })

        if model_name == "mlp":
            for row in cascading_randomization(model, X_test, params["ig_steps"], global_attribution, seed):
                outputs["randomization"].append({"context": name, "model": model_name, "test": "cascading_parameters", **row})
        permuted = rng.permutation(y[train])
        random_model = (make_lightgbm(fixed, lightgbm_params, seed) if model_name == "lightgbm" else make_mlp(mlp_config, seed)).fit(X[train], permuted)
        random_global = np.abs(local_fn(random_model, X_test)).mean(axis=0)
        outputs["randomization"].append({"context": name, "model": model_name, "test": "label_permutation", "layers_randomized": None,
                                         "global_spearman": rho(random_global, global_attribution), "local_spearman": None,
                                         "macro_f1": float(f1_score(y_test, random_model.predict(X_test), average="macro"))})
        print(f"{name} {model_name} explained", flush=True)

        if model_name == "lightgbm":
            shap_order = sensor_rankings["shap"]
            removal_sets = {"shap": [shap_order[:k] for k in params["retrain_sensor_counts"]]}
            for order in range(3):
                shuffled = rng.permutation(np.unique(groups))
                removal_sets[f"random_{order}"] = [shuffled[:k] for k in params["retrain_sensor_counts"]]
            for method, removal_list in removal_sets.items():
                for removed in removal_list:
                    keep = ~np.isin(groups, removed)
                    refit = make_lightgbm(fixed, lightgbm_params, seed).fit(X[train][:, keep], y[train])
                    outputs["retrain"].append({"context": name, "method": method, "sensors_removed": len(removed),
                                               "removed": ";".join(removed), "macro_f1": float(f1_score(y[test], refit.predict(X[test][:, keep]), average="macro"))})
            for column in np.argsort(-global_attribution)[:3]:
                class_index = int(np.argmax(np.bincount(model.predict_proba(X_test).argmax(axis=1))))
                for point in ale_and_pdp(model, X_test, column, class_index):
                    outputs["ale"].append({"context": name, "feature": columns[column], "class": int(model.classes_[class_index]), **point})
            examples = rng.choice(len(X_test), size=3, replace=False)
            shap_local, lime_for_examples = attribution[examples], lime_local(model, background, X_test[examples], params["lime_samples"], seed)
            local_examples = [
                {"cycle": int(subset[index]), "true": int(y_test[index]), "predicted": int(model.classes_[model.predict_proba(X_test[[index]]).argmax()]),
                 "shap_top5": [columns[i] for i in np.argsort(-np.abs(shap_local[position]))[:5]],
                 "lime_top5": [columns[i] for i in np.argsort(-np.abs(lime_for_examples[position]))[:5]],
                 "top5_jaccard": jaccard_top(shap_local[position], lime_for_examples[position], 5)}
                for position, index in enumerate(examples)
            ]
            MODELS.mkdir(exist_ok=True)
            joblib.dump({"model": model, "columns": columns, "test_cycles": subset, "target": target, "context": name, "macro_f1": performance["lightgbm"]}, MODELS / f"{name}.joblib")

    return outputs, performance, importances_by_model, local_examples, columns


def main():
    all_params = load_params()
    params = all_params["xai"]
    fixed, mlp_config = all_params["models"]["lightgbm_fixed"], all_params["models"]["mlp"]
    best = json.loads((ROOT / "reports" / "models" / "best_lightgbm.json").read_text())
    data = load_processed()
    warnings.filterwarnings("ignore")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tables, performance, importances, locals_ = {}, {}, {}, {}
    for context in params["contexts"]:
        outputs, perf, importance, local, columns = run_context(context, params, fixed, mlp_config, best, data)
        for key, rows in outputs.items():
            tables.setdefault(key, []).extend(rows)
        performance[context["name"]], importances[context["name"]], locals_[context["name"]] = perf, importance, local

    for key, rows in tables.items():
        pd.DataFrame(rows).to_csv(OUTPUT / f"{key}.csv", index=False)
    long_rows = [
        {"context": context, "model": model, "method": method, "feature": columns[i], "importance": float(values[i])}
        for context, models in importances.items() for model, methods in models.items() for method, values in methods.items() for i in range(len(columns))
    ]
    pd.DataFrame(long_rows).to_parquet(OUTPUT / "importances.parquet")

    reference = importances["accumulator_lopo"]["lightgbm"]["shap"]
    other = importances["accumulator_random"]["lightgbm"]["shap"]
    thermal = set(params["thermal_sensors"])
    groups = sensors_of(columns)
    cross = {
        "feature_spearman": rho(reference, other),
        "top_k_jaccard": jaccard_top(reference, other, params["top_k"]),
        "sensor_spearman": rho(sensor_scores(reference, groups).to_numpy(), sensor_scores(other, groups).to_numpy()),
        "thermal_share_top_k_lopo": float(np.mean([groups[i] in thermal for i in np.argsort(-reference)[: params["top_k"]]])),
        "thermal_share_top_k_random": float(np.mean([groups[i] in thermal for i in np.argsort(-other)[: params["top_k"]]])),
    }
    (OUTPUT / "summary.json").write_text(json.dumps({"performance": performance, "accumulator_random_vs_lopo": cross, "local_examples": locals_}, indent=2))


if __name__ == "__main__":
    main()

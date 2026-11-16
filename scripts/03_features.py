"""Turn the multi-rate sensor matrices into one table of per-cycle window statistics.

Every sensor keeps its native sampling rate and is summarised over the same time
windows, which aligns 100, 10 and 1 Hz signals without resampling. No statistic is
fitted on the dataset, so this stage cannot leak between train and test.
"""
import numpy as np
import pandas as pd

from common import CYCLE_SECONDS, PROCESSED, SENSOR_COLUMNS, TARGETS, data_dir, load_params, load_profile, load_sensor


def window_statistics(values):
    n_points = values.shape[1]
    time = np.arange(n_points, dtype=np.float32)
    centred_time = time - time.mean()
    centred_values = values - values.mean(axis=1, keepdims=True)
    slope = (centred_values * centred_time).sum(axis=1) / (centred_time**2).sum() if n_points > 1 else 0.0
    return {
        "mean": values.mean(axis=1),
        "std": values.std(axis=1),
        "min": values.min(axis=1),
        "max": values.max(axis=1),
        "slope": slope * n_points,
    }


def sensor_features(name, values, n_windows):
    columns = {}
    edges = np.linspace(0, values.shape[1], n_windows + 1).astype(int)
    seconds = CYCLE_SECONDS / n_windows
    for index in range(n_windows):
        window = values[:, edges[index] : edges[index + 1]]
        label = f"{int(index * seconds)}-{int((index + 1) * seconds)}s"
        for stat, column in window_statistics(window).items():
            columns[f"{name}__{label}__{stat}"] = column.astype(np.float32)
    return columns


def build_labels(profile):
    labels = profile.copy()
    labels.index.name = "cycle"
    state = labels[TARGETS].astype(str).agg("-".join, axis=1)
    labels["run"] = (state != state.shift()).cumsum() - 1
    labels["accumulator_block"] = (labels["accumulator"] != labels["accumulator"].shift()).cumsum() - 1
    labels["cooler_period"] = (labels["cooler"] != labels["cooler"].shift()).cumsum() - 1
    return labels


def main():
    params = load_params()
    directory = data_dir(params)
    n_windows = params["features"]["n_windows"]
    PROCESSED.mkdir(parents=True, exist_ok=True)

    columns = {}
    for name in SENSOR_COLUMNS:
        columns.update(sensor_features(name, load_sensor(directory, name), n_windows))
        print(f"features: {name}", flush=True)
    features = pd.DataFrame(columns)
    features.index.name = "cycle"
    if features.isna().any().any():
        raise ValueError("features contain missing values")

    features.to_parquet(PROCESSED / "features.parquet")
    build_labels(load_profile(directory)).to_parquet(PROCESSED / "labels.parquet")
    print(f"features shape: {features.shape}")


if __name__ == "__main__":
    main()

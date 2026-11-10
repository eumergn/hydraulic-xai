"""Raw data audit for the UCI hydraulic systems dataset.

Usage: python -I scripts/00_audit_raw.py <data_dir> <output_dir>

Reads the raw matrices one at a time and keeps only per-cycle summaries, so
the 100 Hz files never sit in memory together.
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = {
    **{f"PS{i}": 6000 for i in range(1, 7)},
    "EPS1": 6000,
    "FS1": 600,
    "FS2": 600,
    **{f"TS{i}": 60 for i in range(1, 5)},
    "VS1": 60,
    "CE": 60,
    "CP": 60,
    "SE": 60,
}
N_CYCLES = 2205
TARGETS = ["cooler", "valve", "pump", "accumulator"]
EXPECTED_COUNTS = {
    "cooler": {3: 732, 20: 732, 100: 741},
    "valve": {100: 1125, 90: 360, 80: 360, 73: 360},
    "pump": {0: 1221, 1: 492, 2: 492},
    "accumulator": {130: 599, 115: 399, 100: 399, 90: 808},
    "stable": {0: 1449, 1: 756},
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sensors(data_dir):
    rows, cycle_means, hashes = [], {}, {}
    for name, n_columns in EXPECTED_COLUMNS.items():
        path = data_dir / f"{name}.txt"
        hashes[f"{name}.txt"] = sha256(path)
        frame = pd.read_csv(path, sep="\t", header=None, dtype=np.float32)
        values = frame.to_numpy()
        cycle_means[name] = values.mean(axis=1)
        rows.append(
            {
                "sensor": name,
                "rows": values.shape[0],
                "columns": values.shape[1],
                "shape_ok": values.shape == (N_CYCLES, n_columns),
                "nan": int(np.isnan(values).sum()),
                "inf": int(np.isinf(values).sum()),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
                "mean": float(np.nanmean(values)),
                "share_zero": float((values == 0).mean()),
                "constant_cycles": int((values.std(axis=1) == 0).sum()),
                "duplicate_cycles": int(frame.duplicated().sum()),
            }
        )
        print(f"read {name}: {values.shape}", flush=True)
        del frame, values
    return pd.DataFrame(rows), pd.DataFrame(cycle_means), hashes


def contiguous_runs(profile):
    state = profile[TARGETS].astype(str).agg("-".join, axis=1)
    run_id = (state != state.shift()).cumsum() - 1
    runs = (
        profile.assign(state=state, run=run_id, cycle=profile.index)
        .groupby("run")
        .agg(
            state=("state", "first"),
            start=("cycle", "min"),
            end=("cycle", "max"),
            length=("cycle", "size"),
            unstable=("stable", "sum"),
        )
    )
    return run_id, runs


def chronological_coverage(profile, train_share):
    cut = int(len(profile) * train_share)
    train, test = profile.iloc[:cut], profile.iloc[cut:]
    out = {"cut_cycle": cut}
    for target in TARGETS:
        out[target] = {
            "train_classes": sorted(int(v) for v in train[target].unique()),
            "test_classes": sorted(int(v) for v in test[target].unique()),
            "test_only": sorted(int(v) for v in set(test[target]) - set(train[target])),
        }
    combos_train = set(map(tuple, train[TARGETS].to_numpy()))
    combos_test = set(map(tuple, test[TARGETS].to_numpy()))
    out["test_combinations"] = len(combos_test)
    out["test_combinations_unseen_in_train"] = len(combos_test - combos_train)
    return out


def plot_timeline(profile, cycle_means, output_dir):
    panels = TARGETS + ["stable"]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(12, 9), sharex=True)
    for axis, target in zip(axes, panels):
        axis.step(profile.index, profile[target], where="post", linewidth=0.9)
        axis.set_ylabel(target)
        axis.set_yticks(sorted(profile[target].unique()))
    axes[-1].plot(profile.index, cycle_means["TS1"], linewidth=0.9)
    axes[-1].set_ylabel("TS1 mean (C)")
    axes[-1].set_xlabel("Cycle number")
    fig.suptitle("Component states and oil temperature over the recording")
    fig.tight_layout()
    fig.savefig(output_dir / "timeline.png", dpi=130)
    plt.close(fig)


def main():
    data_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors, cycle_means, hashes = audit_sensors(data_dir)
    hashes["profile.txt"] = sha256(data_dir / "profile.txt")
    profile = pd.read_csv(
        data_dir / "profile.txt", sep="\t", header=None, names=TARGETS + ["stable"]
    )

    class_counts = {
        column: {int(k): int(v) for k, v in profile[column].value_counts().items()}
        for column in profile.columns
    }
    run_id, runs = contiguous_runs(profile)
    combos = profile[TARGETS].value_counts()
    runs_per_state = runs.groupby("state").size()

    correlation = cycle_means.corr()
    pairs = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1)).stack()
    strong_pairs = pairs[pairs.abs() >= 0.95].sort_values(key=np.abs, ascending=False)

    unstable_positions = np.flatnonzero(profile["stable"].to_numpy() == 1)
    summary = {
        "profile_shape": list(profile.shape),
        "profile_nan": int(profile.isna().sum().sum()),
        "total_sensor_columns": int(sensors["columns"].sum()),
        "all_sensor_shapes_ok": bool(sensors["shape_ok"].all()),
        "class_counts_match_documentation": class_counts == EXPECTED_COUNTS,
        "class_counts": class_counts,
        "n_runs": int(len(runs)),
        "run_length": runs["length"].describe().to_dict(),
        "n_distinct_combinations": int(len(combos)),
        "possible_combinations": 3 * 4 * 3 * 4,
        "cycles_per_combination": combos.describe().to_dict(),
        "runs_per_combination": runs_per_state.describe().to_dict(),
        "combinations_seen_in_one_run": int((runs_per_state == 1).sum()),
        "unstable_first_cycles": unstable_positions[:10].tolist(),
        "unstable_last_cycles": unstable_positions[-10:].tolist(),
        "unstable_runs_share": float((runs["unstable"] > 0).mean()),
        "first_transitions_by_target": {
            target: (np.flatnonzero(profile[target].diff().fillna(0).to_numpy() != 0)[:12]).tolist()
            for target in TARGETS
        },
        "n_transitions_by_target": {
            target: int((profile[target].diff().fillna(0) != 0).sum()) for target in TARGETS
        },
        "chronological_60_40": chronological_coverage(profile, 0.6),
        "chronological_80_20": chronological_coverage(profile, 0.8),
        "strong_sensor_pairs_abs_ge_0_95": {
            f"{a}~{b}": round(float(v), 4) for (a, b), v in strong_pairs.items()
        },
    }

    sensors.to_csv(output_dir / "sensor_summary.csv", index=False)
    runs.to_csv(output_dir / "runs.csv")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "sha256.json").write_text(json.dumps(hashes, indent=2))
    plot_timeline(profile, cycle_means, output_dir)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(sensors.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(runs.head(25).to_string())


if __name__ == "__main__":
    main()

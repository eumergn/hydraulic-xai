"""Follow-up audit: recording blocks, PS4 zero cycles, stable flag, startup zeros.

Usage: python -I scripts/01_audit_structure.py <data_dir> <output_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["cooler", "valve", "pump", "accumulator"]


def load(data_dir, name):
    return pd.read_csv(data_dir / f"{name}.txt", sep="\t", header=None, dtype=np.float32).to_numpy()


def blocks(series):
    change = (series != series.shift()).cumsum() - 1
    frame = pd.DataFrame({"value": series, "block": change, "cycle": series.index})
    return (
        frame.groupby("block")
        .agg(value=("value", "first"), start=("cycle", "min"), end=("cycle", "max"), length=("cycle", "size"))
        .reset_index(drop=True)
    )


def leading_zero_columns(matrix):
    share = (matrix == 0).mean(axis=0)
    columns = np.flatnonzero(share > 0.99)
    return {
        "columns_almost_always_zero": int(len(columns)),
        "first": int(columns[0]) if len(columns) else None,
        "last": int(columns[-1]) if len(columns) else None,
        "columns": columns[:20].tolist(),
    }


def main():
    data_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    profile = pd.read_csv(data_dir / "profile.txt", sep="\t", header=None, names=TARGETS + ["stable"])

    ps4 = load(data_dir, "PS4")
    all_zero = (ps4 == 0).all(axis=1)
    nonzero_blocks = blocks(pd.Series(all_zero.astype(int)))
    by_target = {
        target: {
            int(level): float(all_zero[profile[target].to_numpy() == level].mean())
            for level in sorted(profile[target].unique())
        }
        for target in TARGETS + ["stable"]
    }

    accumulator = blocks(profile["accumulator"])
    accumulator["cooler_values"] = [
        sorted(int(v) for v in profile["cooler"].iloc[row.start : row.end + 1].unique())
        for row in accumulator.itertuples()
    ]
    cooler = blocks(profile["cooler"])

    unstable = profile["stable"].to_numpy() == 1
    state = profile[TARGETS].astype(str).agg("-".join, axis=1)
    state_change = np.array(state != state.shift())
    state_change[0] = True
    position = np.zeros(len(profile), dtype=int)
    for index in range(1, len(profile)):
        position[index] = 0 if state_change[index] else position[index - 1] + 1

    report = {
        "ps4_all_zero_cycles": int(all_zero.sum()),
        "ps4_nonzero_share_by_block": nonzero_blocks.assign(value=lambda d: d["value"].astype(int)).head(30).to_dict("records"),
        "ps4_n_blocks": int(len(nonzero_blocks)),
        "ps4_all_zero_share_by_level": by_target,
        "cooler_blocks": cooler.to_dict("records"),
        "accumulator_blocks": accumulator.to_dict("records"),
        "unstable_share_by_position_in_run": {
            int(k): round(float(v), 3)
            for k, v in pd.Series(unstable).groupby(np.minimum(position, 12)).mean().items()
        },
        "unstable_outside_first_210_cycles": int(unstable[210:].sum()),
        "unstable_by_cooler": {
            int(k): int(v) for k, v in profile.groupby("cooler")["stable"].sum().items()
        },
        "startup_zero_columns": {},
    }
    for name in ["PS2", "PS3", "FS1", "SE"]:
        report["startup_zero_columns"][name] = leading_zero_columns(load(data_dir, name))

    (Path(output_dir) / "structure.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()

"""Build the validation folds. See docs/protocole_decoupage.md for the reasoning.

random              shuffled cycles, kept only to measure score inflation
group_run           whole contiguous runs held out
block_latin         whole accumulator blocks held out, mixed across cooler periods
leave_period_out    train on two cooler periods, test on the third
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import PROCESSED, ROOT, TARGET_LEVELS, TARGETS, load_params


def purge(train, test, gap):
    if gap == 0:
        return train
    test = np.sort(test)
    positions = np.searchsorted(test, train)
    left = test[np.clip(positions - 1, 0, len(test) - 1)]
    right = test[np.clip(positions, 0, len(test) - 1)]
    distance = np.minimum(np.abs(train - left), np.abs(train - right))
    return train[distance > gap]


def random_folds(labels, n_splits, seed):
    state = labels[TARGETS].astype(str).agg("-".join, axis=1)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(labels, state)


def group_run_folds(labels, n_splits, seed):
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(labels, groups=labels["run"])


def block_latin_folds(labels):
    blocks = labels.groupby("accumulator_block")["accumulator"].first()
    repetition = blocks.groupby(blocks).cumcount()
    levels = TARGET_LEVELS["accumulator"]
    for fold in range(3):
        held_out = [
            block
            for block, level in blocks.items()
            if repetition[block] == (fold + levels.index(level)) % 3
        ]
        mask = labels["accumulator_block"].isin(held_out).to_numpy()
        yield np.flatnonzero(~mask), np.flatnonzero(mask)


def leave_period_folds(labels):
    for period in sorted(labels["cooler_period"].unique()):
        mask = (labels["cooler_period"] == period).to_numpy()
        yield np.flatnonzero(~mask), np.flatnonzero(mask)


def assert_clean(scheme, train, test, labels):
    if set(train) & set(test):
        raise AssertionError(f"{scheme}: cycle in both train and test")
    group = {"group_run": "run", "block_latin": "accumulator_block", "leave_period_out": "cooler_period"}.get(scheme)
    if group and set(labels[group].iloc[train]) & set(labels[group].iloc[test]):
        raise AssertionError(f"{scheme}: group shared between train and test")


def main():
    params = load_params()["splits"]
    labels = pd.read_parquet(PROCESSED / "labels.parquet")
    schemes = {
        "random": (random_folds(labels, params["n_splits"], params["seed"]), 0),
        "group_run": (group_run_folds(labels, params["n_splits"], params["seed"]), 0),
        "block_latin": (block_latin_folds(labels), params["purge_cycles"]),
        "leave_period_out": (leave_period_folds(labels), params["purge_cycles"]),
    }
    rows, coverage = [], []
    for scheme, (folds, gap) in schemes.items():
        for fold, (train, test) in enumerate(folds):
            train = purge(train, test, gap)
            assert_clean(scheme, train, test, labels)
            rows.append(pd.DataFrame({"scheme": scheme, "fold": fold, "cycle": train, "role": "train"}))
            rows.append(pd.DataFrame({"scheme": scheme, "fold": fold, "cycle": test, "role": "test"}))
            for target in TARGETS:
                train_levels, test_levels = set(labels[target].iloc[train]), set(labels[target].iloc[test])
                coverage.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "target": target,
                        "n_train": len(train),
                        "n_test": len(test),
                        "test_levels_missing_from_train": sorted(int(v) for v in test_levels - train_levels),
                        "levels_absent_from_test": sorted(int(v) for v in train_levels - test_levels),
                    }
                )
    pd.concat(rows, ignore_index=True).to_parquet(PROCESSED / "folds.parquet")
    output = ROOT / "reports" / "splits_coverage.csv"
    pd.DataFrame(coverage).to_csv(output, index=False)
    print(pd.DataFrame(coverage).groupby("scheme")[["n_train", "n_test"]].agg(["min", "max"]))


if __name__ == "__main__":
    main()

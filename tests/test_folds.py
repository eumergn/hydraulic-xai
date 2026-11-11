"""Leakage checks on the generated folds. Run after the `splits` stage."""
from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "folds.parquet").exists(), reason="run `dvc repro splits` first to generate the folds"
)
GROUP_COLUMN = {"group_run": "run", "block_latin": "accumulator_block", "leave_period_out": "cooler_period"}
PURGED = {"block_latin": 10, "leave_period_out": 10}


@pytest.fixture(scope="module")
def folds():
    return pd.read_parquet(PROCESSED / "folds.parquet")


@pytest.fixture(scope="module")
def labels():
    return pd.read_parquet(PROCESSED / "labels.parquet")


def parts(folds, scheme):
    for fold, part in folds[folds["scheme"] == scheme].groupby("fold"):
        yield fold, part[part["role"] == "train"]["cycle"], part[part["role"] == "test"]["cycle"]


@pytest.mark.parametrize("scheme", ["random", "group_run", "block_latin", "leave_period_out"])
def test_no_cycle_in_both_roles(folds, scheme):
    for _, train, test in parts(folds, scheme):
        assert not set(train) & set(test)


@pytest.mark.parametrize("scheme", ["random", "group_run"])
def test_every_cycle_tested_once(folds, labels, scheme):
    tested = pd.concat([test for _, _, test in parts(folds, scheme)])
    assert sorted(tested) == list(labels.index)


@pytest.mark.parametrize("scheme", GROUP_COLUMN)
def test_groups_are_disjoint(folds, labels, scheme):
    column = GROUP_COLUMN[scheme]
    for _, train, test in parts(folds, scheme):
        assert not set(labels.loc[train, column]) & set(labels.loc[test, column])


@pytest.mark.parametrize("scheme", PURGED)
def test_purge_gap(folds, scheme):
    for _, train, test in parts(folds, scheme):
        ordered = sorted(test)
        assert min(min(abs(c - t) for t in (ordered[0], ordered[-1])) for c in train if c < ordered[0] or c > ordered[-1]) > PURGED[scheme]


def test_leave_period_keeps_all_valve_pump_accumulator_levels(folds, labels):
    for _, train, test in parts(folds, "leave_period_out"):
        for target in ["valve", "pump", "accumulator"]:
            assert set(labels.loc[test, target]) <= set(labels.loc[train, target])

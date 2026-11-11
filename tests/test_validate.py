"""Tests for scripts/02_validate.py."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("validate", ROOT / "scripts" / "02_validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)

N = validate.N_CYCLES


def good_profile():
    rng = np.random.default_rng(0)
    columns = {target: rng.choice(levels, size=N) for target, levels in validate.TARGET_LEVELS.items()}
    columns["stable"] = rng.choice([0, 1], size=N)
    return pd.DataFrame(columns)


def good_summary(low=1.0, high=2.0):
    return pd.DataFrame(
        {
            "nan_count": np.zeros(N, dtype=int),
            "inf_count": np.zeros(N, dtype=int),
            "minimum": np.full(N, low),
            "maximum": np.full(N, high),
        }
    )


@pytest.mark.parametrize("sensor, expected", [("PS3", (0, 250)), ("EPS1", (0, 4000)), ("SE", (0, 105))])
def test_bounds_for_matches_sensor_prefix(sensor, expected):
    assert validate.bounds_for(sensor) == expected


def test_bounds_for_unknown_sensor_raises():
    with pytest.raises(KeyError):
        validate.bounds_for("XX1")


def test_every_sensor_has_bounds():
    for sensor in validate.SENSOR_COLUMNS:
        validate.bounds_for(sensor)


def test_profile_schema_accepts_valid_profile():
    validate.profile_schema().validate(good_profile())


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.assign(valve=p["valve"].replace(73, 75)),
        lambda p: p.assign(stable=2),
        lambda p: p.assign(extra=0),
        lambda p: p.iloc[:-1],
    ],
    ids=["unknown_level", "bad_stable_flag", "extra_column", "missing_cycle"],
)
def test_profile_schema_rejects_invalid_profile(change):
    with pytest.raises(pandera.errors.SchemaErrors):
        validate.profile_schema().validate(change(good_profile()), lazy=True)


def test_sensor_schema_accepts_values_within_bounds():
    validate.sensor_schema("TS1").validate(good_summary(20.0, 60.0))


@pytest.mark.parametrize(
    "summary",
    [
        good_summary().assign(nan_count=1),
        good_summary().assign(inf_count=1),
        good_summary(low=-1.0),
        good_summary(high=101.0),
    ],
    ids=["nan", "inf", "below_bound", "above_bound"],
)
def test_sensor_schema_rejects_invalid_values(summary):
    with pytest.raises(pandera.errors.SchemaErrors):
        validate.sensor_schema("TS1").validate(summary, lazy=True)


@pytest.mark.skipif(not (ROOT / "condition+monitoring+of+hydraulic+systems" / "profile.txt").exists(), reason="raw dataset not present")
def test_real_profile_passes_schema():
    profile = validate.load_profile(validate.data_dir(validate.load_params()))
    validate.profile_schema().validate(profile, lazy=True)

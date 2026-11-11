"""Tests for scripts/00_audit_raw.py and scripts/01_audit_structure.py."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "condition+monitoring+of+hydraulic+systems"
TARGETS = ["cooler", "valve", "pump", "accumulator"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_raw = load_script("00_audit_raw.py")
audit_structure = load_script("01_audit_structure.py")


def tiny_profile():
    rows = [
        (3, 100, 0, 130, 1),
        (3, 100, 0, 130, 1),
        (3, 73, 0, 130, 0),
        (3, 73, 0, 130, 0),
        (20, 73, 2, 115, 0),
        (20, 100, 2, 115, 0),
    ]
    return pd.DataFrame(rows, columns=TARGETS + ["stable"])


def write_tiny_dataset(directory, n_cycles=6):
    rng = np.random.default_rng(0)
    for name, n_columns in audit_raw.EXPECTED_COLUMNS.items():
        values = rng.uniform(1, 10, size=(n_cycles, min(n_columns, 12)))
        if name == "PS4":
            values[:3] = 0
        if name == "SE":
            values[:, 1:4] = 0
        np.savetxt(directory / f"{name}.txt", values, delimiter="\t", fmt="%.3f")
    tiny_profile().to_csv(directory / "profile.txt", sep="\t", header=False, index=False)


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "file.txt"
    path.write_bytes(b"hydraulic" * 1000)
    assert audit_raw.sha256(path) == hashlib.sha256(b"hydraulic" * 1000).hexdigest()


def test_contiguous_runs_splits_on_any_state_change():
    run_id, runs = audit_raw.contiguous_runs(tiny_profile())
    assert run_id.tolist() == [0, 0, 1, 1, 2, 3]
    assert runs["length"].tolist() == [2, 2, 1, 1]
    assert runs["unstable"].tolist() == [2, 0, 0, 0]


def test_chronological_coverage_flags_classes_only_in_test():
    coverage = audit_raw.chronological_coverage(tiny_profile(), 0.5)
    assert coverage["cut_cycle"] == 3
    assert coverage["cooler"]["test_only"] == [20]
    assert coverage["accumulator"]["test_only"] == [115]


def test_blocks_groups_consecutive_values():
    result = audit_structure.blocks(pd.Series([5, 5, 7, 7, 7, 5]))
    assert result["value"].tolist() == [5, 7, 5]
    assert result["length"].tolist() == [2, 3, 1]
    assert result["start"].tolist() == [0, 2, 5]


def test_leading_zero_columns_finds_always_zero_columns():
    matrix = np.ones((200, 6))
    matrix[:, 2:4] = 0
    result = audit_structure.leading_zero_columns(matrix)
    assert result["columns_almost_always_zero"] == 2
    assert (result["first"], result["last"]) == (2, 3)


def test_audit_scripts_run_end_to_end_on_tiny_data(tmp_path, monkeypatch):
    data, output = tmp_path / "data", tmp_path / "out"
    data.mkdir()
    write_tiny_dataset(data)
    monkeypatch.setattr(sys, "argv", ["00_audit_raw.py", str(data), str(output)])
    audit_raw.main()
    for name in ["summary.json", "sensor_summary.csv", "runs.csv", "sha256.json", "timeline.png"]:
        assert (output / name).exists()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["n_runs"] == 4
    assert summary["profile_nan"] == 0

    monkeypatch.setattr(sys, "argv", ["01_audit_structure.py", str(data), str(output)])
    audit_structure.main()
    structure = json.loads((output / "structure.json").read_text())
    assert structure["ps4_all_zero_cycles"] == 3
    assert structure["startup_zero_columns"]["SE"]["columns"] == [1, 2, 3]


@pytest.mark.skipif(not (DATA / "profile.txt").exists(), reason="raw dataset not present")
def test_real_profile_structure_matches_audit():
    profile = pd.read_csv(DATA / "profile.txt", sep="\t", header=None, names=TARGETS + ["stable"])
    _, runs = audit_raw.contiguous_runs(profile)
    assert len(profile) == 2205
    assert len(runs) == 194
    assert len(profile[TARGETS].drop_duplicates()) == 144
    cooler = audit_structure.blocks(profile["cooler"])
    assert cooler["start"].tolist() == [0, 732, 1464]
    assert len(audit_structure.blocks(profile["accumulator"])) == 12

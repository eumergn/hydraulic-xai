"""Shared constants and loaders for the pipeline stages."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["cooler", "valve", "pump", "accumulator"]
TARGET_LEVELS = {
    "cooler": [3, 20, 100],
    "valve": [73, 80, 90, 100],
    "pump": [2, 1, 0],
    "accumulator": [90, 100, 115, 130],
}
SENSOR_COLUMNS = {
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
CYCLE_SECONDS = 60
PROCESSED = ROOT / "data" / "processed"


def load_params():
    return yaml.safe_load((ROOT / "params.yaml").read_text())


def data_dir(params):
    return ROOT / params["data_dir"]


def load_sensor(directory, name):
    frame = pd.read_csv(directory / f"{name}.txt", sep="\t", header=None, dtype=np.float32)
    return frame.to_numpy()


def load_profile(directory):
    return pd.read_csv(
        directory / "profile.txt", sep="\t", header=None, names=TARGETS + ["stable"]
    )

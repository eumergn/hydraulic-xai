"""Validity checks on the raw data, using pandera. Fails the pipeline on any error."""
import json

import numpy as np
import pandas as pd
import pandera.pandas as pa

from common import N_CYCLES, ROOT, SENSOR_COLUMNS, TARGET_LEVELS, data_dir, load_params, load_profile, load_sensor

# Plausibility bounds for the physical quantities, deliberately wider than the observed ranges.
BOUNDS = {
    "PS": (0, 250),
    "EPS": (0, 4000),
    "FS": (0, 30),
    "TS": (0, 100),
    "VS": (0, 10),
    "CE": (0, 100),
    "CP": (0, 10),
    "SE": (0, 105),
}


def bounds_for(sensor):
    for prefix, limits in BOUNDS.items():
        if sensor.startswith(prefix):
            return limits
    raise KeyError(sensor)


def profile_schema():
    columns = {
        target: pa.Column(int, pa.Check.isin(levels), nullable=False)
        for target, levels in TARGET_LEVELS.items()
    }
    columns["stable"] = pa.Column(int, pa.Check.isin([0, 1]), nullable=False)
    return pa.DataFrameSchema(columns, strict=True, checks=pa.Check(lambda d: len(d) == N_CYCLES))


def sensor_schema(sensor):
    low, high = bounds_for(sensor)
    return pa.DataFrameSchema(
        {
            "nan_count": pa.Column(int, pa.Check.eq(0)),
            "inf_count": pa.Column(int, pa.Check.eq(0)),
            "minimum": pa.Column(float, pa.Check.ge(low)),
            "maximum": pa.Column(float, pa.Check.le(high)),
        },
        checks=pa.Check(lambda d: len(d) == N_CYCLES),
    )


def main():
    params = load_params()
    directory = data_dir(params)
    profile_schema().validate(load_profile(directory), lazy=True)
    results = {}
    for sensor, n_columns in SENSOR_COLUMNS.items():
        values = load_sensor(directory, sensor)
        if values.shape != (N_CYCLES, n_columns):
            raise ValueError(f"{sensor}: unexpected shape {values.shape}")
        summary = pd.DataFrame(
            {
                "nan_count": np.isnan(values).sum(axis=1).astype(int),
                "inf_count": np.isinf(values).sum(axis=1).astype(int),
                "minimum": np.nanmin(values, axis=1).astype(float),
                "maximum": np.nanmax(values, axis=1).astype(float),
            }
        )
        sensor_schema(sensor).validate(summary, lazy=True)
        results[sensor] = {
            "shape": list(values.shape),
            "observed_min": float(summary["minimum"].min()),
            "observed_max": float(summary["maximum"].max()),
            "bounds": list(bounds_for(sensor)),
        }
        print(f"valid: {sensor}", flush=True)
        del values
    output = ROOT / "reports" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "passed", "sensors": results}, indent=2))


if __name__ == "__main__":
    main()

"""Core data preparation and analysis for the 6-DoF Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation


REQUIRED_FIELDS = ("time", "x", "y", "z", "qx", "qy", "qz", "qw")
SIGNAL_COLUMNS = ("speed", "acceleration", "angular_speed")
FEATURE_COLUMNS = (
    "duration_s",
    "distance",
    "displacement",
    "mean_speed",
    "speed_std",
    "max_speed",
    "mean_acceleration",
    "mean_angular_speed",
    "path_efficiency",
    "spatial_entropy_bits",
    "spatial_coverage",
)


@dataclass(frozen=True)
class PreparationReport:
    input_rows: int
    valid_rows: int
    dropped_rows: int
    quaternion_repairs: int
    timestamp_mode: str


def infer_column_mapping(columns: Iterable[str]) -> dict[str, str | None]:
    """Infer common 6-DoF column names without assuming a specific dataset."""

    columns = list(columns)
    normalized = {
        str(column).strip().lower().replace(" ", "").replace("_", ""): column
        for column in columns
    }
    aliases = {
        "time": ("timestamp", "time", "t", "frame", "frameno", "frameid"),
        "x": ("x", "posx", "positionx", "tx"),
        "y": ("y", "posy", "positiony", "ty"),
        "z": ("z", "posz", "positionz", "tz"),
        "qx": ("qx", "rotx", "rotationx", "quatx"),
        "qy": ("qy", "roty", "rotationy", "quaty"),
        "qz": ("qz", "rotz", "rotationz", "quatz"),
        "qw": ("qw", "rotw", "rotationw", "quatw"),
        "participant": (
            "participant",
            "participantid",
            "subject",
            "subjectid",
            "user",
            "userid",
            "sequence",
        ),
        "condition": ("condition", "experimentcondition", "group", "task", "scene"),
    }
    mapping: dict[str, str | None] = {}
    for field, candidates in aliases.items():
        mapping[field] = next(
            (normalized[name] for name in candidates if name in normalized), None
        )
    return mapping


def infer_timestamp_mode(values: pd.Series, column_name: str = "") -> str:
    """Infer seconds, Unix units, .NET ticks, or a frame index."""

    name = column_name.strip().lower()
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "seconds"
    if "frame" in name:
        return "frame"

    magnitude = float(numeric.abs().median())
    positive_deltas = numeric.sort_values().diff()
    positive_deltas = positive_deltas[positive_deltas > 0]
    median_delta = float(positive_deltas.median()) if not positive_deltas.empty else 0.0

    if magnitude > 5e16 and 1e4 <= median_delta <= 1e7:
        return "dotnet_ticks"
    if magnitude > 1e17:
        return "nanoseconds"
    if magnitude > 1e14:
        return "microseconds"
    if magnitude > 1e11:
        return "milliseconds"
    if magnitude > 1e8:
        return "seconds"
    if median_delta >= 0.9 and np.allclose(numeric.head(100) % 1, 0):
        return "frame"
    return "seconds"


def _time_to_seconds(values: pd.Series, mode: str, frame_rate: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    scales = {
        "seconds": 1.0,
        "milliseconds": 1e-3,
        "microseconds": 1e-6,
        "nanoseconds": 1e-9,
        "dotnet_ticks": 1e-7,
        "frame": 1.0 / frame_rate,
    }
    if mode not in scales:
        raise ValueError(f"Unsupported timestamp mode: {mode}")
    return numeric * scales[mode]


def _apply_coordinate_convention(
    frame: pd.DataFrame, convention: str
) -> pd.DataFrame:
    if convention == "unity_to_right_handed":
        frame["z"] = -frame["z"]
    elif convention == "y_up_to_z_up":
        original_y = frame["y"].copy()
        frame["y"] = frame["z"]
        frame["z"] = -original_y
    elif convention != "native":
        raise ValueError(f"Unsupported coordinate convention: {convention}")
    return frame


def _transform_quaternions(quaternion: np.ndarray, convention: str) -> np.ndarray:
    """Express orientations in the same transformed basis as positions."""

    if convention == "native":
        return quaternion
    if convention == "unity_to_right_handed":
        basis = np.diag([1.0, 1.0, -1.0])
    elif convention == "y_up_to_z_up":
        basis = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    else:
        raise ValueError(f"Unsupported coordinate convention: {convention}")

    rotation_matrices = Rotation.from_quat(quaternion).as_matrix()
    transformed = basis @ rotation_matrices @ basis.T
    return Rotation.from_matrix(transformed).as_quat()


def prepare_data(
    raw: pd.DataFrame,
    mapping: dict[str, str | None],
    *,
    timestamp_mode: str = "auto",
    frame_rate: float = 60.0,
    coordinate_convention: str = "native",
    smoothing_window: int = 5,
) -> tuple[pd.DataFrame, PreparationReport]:
    """Map, validate, normalize, and derive motion signals from a raw table."""

    missing = [field for field in REQUIRED_FIELDS if not mapping.get(field)]
    if missing:
        raise ValueError("Missing required mappings: " + ", ".join(missing))
    if frame_rate <= 0:
        raise ValueError("Frame rate must be positive")

    input_rows = len(raw)
    output = pd.DataFrame(index=raw.index)
    for field in REQUIRED_FIELDS:
        output[field] = pd.to_numeric(raw[mapping[field]], errors="coerce")

    participant_column = mapping.get("participant")
    condition_column = mapping.get("condition")
    if participant_column:
        output["participant"] = raw[participant_column].astype("string").fillna("Unnamed sample")
    else:
        output["participant"] = "Sample 1"
    if condition_column:
        output["condition"] = raw[condition_column].astype("string").fillna("Unspecified")
    else:
        output["condition"] = "Unspecified"

    output = output.dropna(subset=list(REQUIRED_FIELDS)).copy()
    resolved_mode = (
        infer_timestamp_mode(output["time"], str(mapping["time"]))
        if timestamp_mode == "auto"
        else timestamp_mode
    )
    output = output.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["time", "x", "y", "z", "qx", "qy", "qz", "qw"]
    )

    # Some frame-index datasets concatenate multiple trials and restart Frame
    # at 1 without an explicit trial ID. Split those runs before sorting so a
    # reset can never create a false trajectory jump.
    if resolved_mode == "frame":
        resets = output.groupby("participant", sort=False)["time"].diff().le(0)
        if resets.any():
            segments = resets.groupby(output["participant"], sort=False).cumsum().add(1)
            output["participant"] = (
                output["participant"].astype(str)
                + " · Segment "
                + segments.astype(int).astype(str).str.zfill(2)
            )

    output = _apply_coordinate_convention(output, coordinate_convention)
    output = output.sort_values(["participant", "time"], kind="stable").reset_index(drop=True)
    # Convert relative values rather than epoch-scale integers. This preserves
    # sub-millisecond precision for .NET ticks and Unix nanoseconds.
    output["time_s"] = output.groupby("participant", sort=False)["time"].transform(
        lambda values: _time_to_seconds(values - values.iloc[0], resolved_mode, frame_rate)
    )

    # Pandas 3 may expose a read-only NumPy view; normalization is intentionally
    # performed on an owned copy.
    quaternion = output[["qx", "qy", "qz", "qw"]].to_numpy(float, copy=True)
    norms = np.linalg.norm(quaternion, axis=1)
    valid_quaternion = norms > 1e-8
    quaternion_repairs = int(np.count_nonzero(np.abs(norms - 1.0) > 1e-3))
    quaternion[valid_quaternion] /= norms[valid_quaternion, None]
    quaternion[valid_quaternion] = _transform_quaternions(
        quaternion[valid_quaternion], coordinate_convention
    )
    quaternion[~valid_quaternion] = np.nan
    output[["qx", "qy", "qz", "qw"]] = quaternion

    output["elapsed_s"] = output.groupby("participant", sort=False)["time_s"].transform(
        lambda values: values - values.iloc[0]
    )
    group = output.groupby("participant", sort=False, group_keys=False)
    output["dt"] = group["time_s"].diff()
    output.loc[output["dt"] <= 0, "dt"] = np.nan

    differences = group[["x", "y", "z"]].diff()
    output["step_distance"] = np.sqrt((differences**2).sum(axis=1))
    output["distance"] = output["step_distance"].fillna(0).groupby(output["participant"]).cumsum()
    output["speed_raw"] = output["step_distance"] / output["dt"]

    window = max(int(smoothing_window), 1)
    output["speed"] = group["speed_raw"].transform(
        lambda values: values.rolling(window, center=True, min_periods=1).median()
    )
    output["acceleration_raw"] = group["speed"].diff() / output["dt"]
    output["acceleration"] = group["acceleration_raw"].transform(
        lambda values: values.rolling(window, center=True, min_periods=1).median()
    )

    previous_quaternion = group[["qx", "qy", "qz", "qw"]].shift().to_numpy(float)
    dots = np.abs(np.sum(quaternion * previous_quaternion, axis=1))
    angles = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
    output["angular_speed_raw"] = angles / output["dt"].to_numpy(float)
    output["angular_speed"] = group["angular_speed_raw"].transform(
        lambda values: values.rolling(window, center=True, min_periods=1).median()
    )

    for column in SIGNAL_COLUMNS:
        output.loc[~np.isfinite(output[column]), column] = np.nan

    report = PreparationReport(
        input_rows=input_rows,
        valid_rows=len(output),
        dropped_rows=input_rows - len(output),
        quaternion_repairs=quaternion_repairs,
        timestamp_mode=resolved_mode,
    )
    return output, report


def spatial_metrics(
    positions: np.ndarray | pd.DataFrame, bins: int = 12
) -> tuple[float, float, int, int]:
    """Return Shannon entropy (bits), coverage, occupied cells, total cells."""

    points = np.asarray(positions, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return np.nan, np.nan, 0, 0
    varying = np.ptp(points, axis=0) > 1e-12
    if not varying.any():
        return 0.0, 1.0, 1, 1
    histogram, _ = np.histogramdd(points[:, varying], bins=bins)
    counts = histogram.ravel()
    counts = counts[counts > 0]
    probabilities = counts / counts.sum()
    entropy = float(stats.entropy(probabilities, base=2))
    total_cells = int(bins ** int(varying.sum()))
    occupied_cells = int(len(counts))
    return entropy, occupied_cells / total_cells, occupied_cells, total_cells


def participant_features(data: pd.DataFrame, bins: int = 12) -> pd.DataFrame:
    """Aggregate independent trajectory-level behavior features."""

    records: list[dict[str, float | str]] = []
    for participant, sample in data.groupby("participant", sort=True):
        sample = sample.sort_values("time_s")
        position = sample[["x", "y", "z"]].to_numpy(float)
        entropy, coverage, _, _ = spatial_metrics(position, bins)
        duration = float(sample["elapsed_s"].max())
        distance = float(sample["step_distance"].sum(skipna=True))
        displacement = float(np.linalg.norm(position[-1] - position[0]))
        records.append(
            {
                "participant": str(participant),
                "condition": str(sample["condition"].mode().iat[0]),
                "n_points": len(sample),
                "duration_s": duration,
                "distance": distance,
                "displacement": displacement,
                "mean_speed": float(sample["speed"].mean()),
                "speed_std": float(sample["speed"].std()),
                "max_speed": float(sample["speed"].max()),
                "mean_acceleration": float(sample["acceleration"].abs().mean()),
                "mean_angular_speed": float(sample["angular_speed"].mean()),
                "path_efficiency": displacement / distance if distance > 0 else np.nan,
                "spatial_entropy_bits": entropy,
                "spatial_coverage": coverage,
            }
        )
    return pd.DataFrame.from_records(records)


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    columns = ["speed", "acceleration", "angular_speed"]
    summary = data[columns].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.rename(
        columns={
            "count": "valid_points",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "25%": "P25",
            "50%": "median",
            "75%": "P75",
            "max": "max",
        }
    )
    return summary


def condition_tests(features: pd.DataFrame) -> pd.DataFrame:
    """Run non-parametric tests only with >=2 independent samples per condition."""

    conditions = [
        (name, group) for name, group in features.groupby("condition") if len(group) >= 2
    ]
    if len(conditions) < 2:
        return pd.DataFrame()

    results = []
    for feature in ("distance", "mean_speed", "mean_angular_speed", "spatial_entropy_bits"):
        groups = [group[feature].dropna().to_numpy() for _, group in conditions]
        if any(len(values) < 2 for values in groups):
            continue
        if len(groups) == 2:
            statistic, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
            test_name = "Mann–Whitney U"
        else:
            statistic, p_value = stats.kruskal(*groups)
            test_name = "Kruskal–Wallis"
        results.append(
            {
                "feature": feature,
                "test": test_name,
                "groups": len(groups),
                "statistic": statistic,
                "p_value": p_value,
                "note": "exploratory_uncorrected",
            }
        )
    return pd.DataFrame(results)


def evenly_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame.copy()
    indices = np.linspace(0, len(frame) - 1, limit, dtype=int)
    return frame.iloc[np.unique(indices)].copy()

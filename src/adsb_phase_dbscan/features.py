from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from .airport_context import infer_airport_context_from_points, smooth_airport_contexts
from .cleaning import filter_valid_adsb_points, normalize_adsb_dataframe
from .config import phase_tolerances
from .geometry import bearing_deg, circular_mean_deg, haversine_nm, heading_change_deg
from .phase_classifier import classify_segments
from .track_builder import build_tracks, make_segments


AVIATION_FEATURE_COLUMNS = [
    "start_altitude_agl_ft",
    "end_altitude_agl_ft",
    "mean_altitude_agl_ft",
    "delta_altitude_ft",
    "mean_ground_speed_kt",
    "min_ground_speed_kt",
    "max_ground_speed_kt",
    "delta_ground_speed_kt",
    "mean_vertical_rate_fpm",
    "max_abs_vertical_rate_fpm",
    "heading_change_deg",
    "turn_rate_deg_per_s",
    "sinuosity",
    "segment_duration_s",
]

REQUIRED_AVIATION_FEATURE_COLUMNS = [
    "start_altitude_agl_ft",
    "end_altitude_agl_ft",
    "mean_altitude_agl_ft",
    "delta_altitude_ft",
    "mean_ground_speed_kt",
    "segment_duration_s",
]

LEGACY_METADATA_COLUMNS = [
    "mean_altitude_ft",
    "sin_heading",
    "cos_heading",
]

VISUAL_FEATURE_EXCLUDE_COLUMNS = {
    "phase_confidence",
    "context_anomaly",
    "behavior_scored",
    "behavior_anomaly",
    "is_anomaly",
}

VISUAL_FEATURE_EXCLUDE_PREFIXES = (
    "dbscan_",
    "global_",
    "model_",
    "visual_",
)

VISUAL_FEATURE_EXCLUDE_FRAGMENTS = (
    "reason",
    "candidate",
    "cluster",
    "score",
)

CIRCULAR_VISUAL_FEATURE_COLUMNS = {
    "mean_heading_deg",
}


def _circular_delta_deg(values, center: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return ((arr - float(center) + 180.0) % 360.0) - 180.0


def _nanmean(values: Iterable[float], default: float = float("nan")) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return default
    return float(np.nanmean(arr))


def _nanmin(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmin(arr))


def _nanmax(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmax(arr))


def compute_segment_features(segments_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows = []
    for _, segment in segments_df.iterrows():
        points = pd.DataFrame(segment["points"]).sort_values("timestamp")
        if points.empty:
            continue
        lats = points["lat"].astype(float).to_numpy()
        lons = points["lon"].astype(float).to_numpy()
        alt = pd.to_numeric(points["altitude_ft"], errors="coerce").to_numpy()
        gs = pd.to_numeric(points["gs"], errors="coerce").to_numpy()
        vr = pd.to_numeric(points["vertical_rate_fpm"], errors="coerce").to_numpy()
        tracks = pd.to_numeric(points["track"], errors="coerce").to_numpy()
        step_dist = [haversine_nm(lats[i - 1], lons[i - 1], lats[i], lons[i]) for i in range(1, len(points))]
        path_distance = float(np.sum(step_dist)) if step_dist else 0.0
        net_displacement = haversine_nm(lats[0], lons[0], lats[-1], lons[-1]) if len(points) > 1 else 0.0
        mean_heading = circular_mean_deg(tracks)
        if math.isnan(mean_heading) and len(points) > 1:
            mean_heading = bearing_deg(lats[0], lons[0], lats[-1], lons[-1])
        duration = (points["timestamp"].iloc[-1] - points["timestamp"].iloc[0]).total_seconds()
        start_altitude = float(alt[0]) if len(alt) and not np.isnan(alt[0]) else float("nan")
        end_altitude = float(alt[-1]) if len(alt) and not np.isnan(alt[-1]) else float("nan")
        mean_altitude = _nanmean(alt)
        delta_altitude = end_altitude - start_altitude if not (np.isnan(start_altitude) or np.isnan(end_altitude)) else float("nan")
        min_ground_speed = _nanmin(gs)
        max_ground_speed = _nanmax(gs)
        start_ground_speed = float(gs[0]) if len(gs) and not np.isnan(gs[0]) else float("nan")
        end_ground_speed = float(gs[-1]) if len(gs) and not np.isnan(gs[-1]) else float("nan")
        delta_ground_speed = (
            end_ground_speed - start_ground_speed
            if not (np.isnan(start_ground_speed) or np.isnan(end_ground_speed))
            else float("nan")
        )
        max_abs_vertical_rate = _nanmax(np.abs(vr))
        heading_change = heading_change_deg(tracks)
        turn_rate = heading_change / duration if duration and duration > 0 else float("nan")
        row = {
            "segment_id": segment["segment_id"],
            "hex": segment.get("hex", ""),
            "flight": segment.get("flight", ""),
            "flight_id": segment.get("flight_id", ""),
            "start_time": points["timestamp"].iloc[0],
            "end_time": points["timestamp"].iloc[-1],
            "duration_seconds": float(duration),
            "start_lat": float(lats[0]),
            "start_lon": float(lons[0]),
            "end_lat": float(lats[-1]),
            "end_lon": float(lons[-1]),
            "mean_lat": float(np.mean(lats)),
            "mean_lon": float(np.mean(lons)),
            "start_altitude_ft": start_altitude,
            "end_altitude_ft": end_altitude,
            "mean_altitude_ft": mean_altitude,
            "delta_altitude_ft": delta_altitude,
            "mean_ground_speed_kt": _nanmean(gs),
            "min_ground_speed_kt": min_ground_speed,
            "max_ground_speed_kt": max_ground_speed,
            "delta_ground_speed_kt": delta_ground_speed,
            "speed_change_kt": delta_ground_speed,
            "mean_vertical_rate_fpm": _nanmean(vr),
            "max_abs_vertical_rate_fpm": max_abs_vertical_rate,
            "mean_heading_deg": float(mean_heading) if not math.isnan(mean_heading) else 0.0,
            "sin_heading": math.sin(math.radians(mean_heading)) if not math.isnan(mean_heading) else 0.0,
            "cos_heading": math.cos(math.radians(mean_heading)) if not math.isnan(mean_heading) else 1.0,
            "heading_change_deg": heading_change,
            "turn_rate_deg_per_s": turn_rate,
            "net_displacement_nm": float(net_displacement),
            "path_distance_nm": float(path_distance),
            "sinuosity": float(path_distance / max(net_displacement, 0.05)),
            "segment_duration_s": float(duration),
            "altitude_known": bool(not np.isnan(mean_altitude)),
            "speed_known": bool(not np.isnan(_nanmean(gs))),
            "vertical_rate_known": bool(not np.isnan(_nanmean(vr))),
        }
        context = infer_airport_context_from_points(lats, lons, row, config)
        row.update(context)
        elevation = float(row.get("airport_elevation_ft") or 0.0)
        row["start_altitude_agl_ft"] = start_altitude - elevation if not np.isnan(start_altitude) else float("nan")
        row["end_altitude_agl_ft"] = end_altitude - elevation if not np.isnan(end_altitude) else float("nan")
        row["mean_altitude_agl_ft"] = mean_altitude - elevation if not np.isnan(mean_altitude) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns() -> list[str]:
    return AVIATION_FEATURE_COLUMNS.copy()


def required_feature_columns() -> list[str]:
    return REQUIRED_AVIATION_FEATURE_COLUMNS.copy()


def rows_with_required_features(segments_df: pd.DataFrame, columns: list[str] | None = None) -> pd.Series:
    columns = columns or required_feature_columns()
    missing_columns = [col for col in columns if col not in segments_df.columns]
    if missing_columns:
        return pd.Series(False, index=segments_df.index)
    required = segments_df[columns].replace([np.inf, -np.inf], np.nan)
    return required.notna().all(axis=1)


def compute_imputation_values(segments_df: pd.DataFrame, columns: list[str] | None = None) -> dict[str, float]:
    columns = columns or feature_columns()
    missing_columns = [col for col in columns if col not in segments_df.columns]
    if missing_columns:
        raise ValueError(f"Missing DBSCAN feature columns: {missing_columns}")
    values: dict[str, float] = {}
    for col in columns:
        series = pd.to_numeric(segments_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = series.median(skipna=True)
        if pd.isna(median):
            raise ValueError(f"Cannot impute DBSCAN feature {col!r}: no finite training values")
        values[col] = float(median)
    return values


def _feature_scales(phase: str, config: dict) -> dict[str, float]:
    tol = phase_tolerances(config, phase)
    return {
        "start_altitude_agl_ft": float(tol["altitude_ft"]),
        "end_altitude_agl_ft": float(tol["altitude_ft"]),
        "mean_altitude_agl_ft": float(tol["altitude_ft"]),
        "delta_altitude_ft": float(tol["delta_altitude_ft"]),
        "mean_ground_speed_kt": float(tol["speed_kt"]),
        "min_ground_speed_kt": float(tol["speed_kt"]),
        "max_ground_speed_kt": float(tol["speed_kt"]),
        "delta_ground_speed_kt": float(tol["speed_kt"]),
        "mean_vertical_rate_fpm": float(tol["vertical_rate_fpm"]),
        "max_abs_vertical_rate_fpm": float(tol["vertical_rate_fpm"]),
        "heading_change_deg": float(tol["heading_deg"]),
        "turn_rate_deg_per_s": 3.0,
        "sinuosity": 3.0,
        "segment_duration_s": float(config["data"].get("segment_seconds", 60)),
    }


def normalize_features_by_phase(
    segments_df: pd.DataFrame,
    phase: str,
    config: dict,
    columns: list[str] | None = None,
    imputation_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    columns = columns or feature_columns()
    missing_columns = [col for col in columns if col not in segments_df.columns]
    if missing_columns:
        raise ValueError(f"Missing DBSCAN feature columns: {missing_columns}")
    imputation_values = imputation_values or compute_imputation_values(segments_df, columns)
    missing_imputations = [col for col in columns if col not in imputation_values]
    if missing_imputations:
        raise ValueError(f"Missing DBSCAN imputation values: {missing_imputations}")
    scales = _feature_scales(phase, config)
    out = pd.DataFrame(index=segments_df.index)
    for col in columns:
        scale = scales.get(col)
        if scale is None or scale <= 0:
            raise ValueError(f"No positive normalization scale configured for feature {col!r}")
        raw = pd.to_numeric(segments_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        raw = raw.fillna(float(imputation_values[col]))
        out[col] = raw / scale
    if not np.isfinite(out.to_numpy(dtype=float)).all():
        raise ValueError("DBSCAN feature matrix contains non-finite values after imputation")
    return out


def build_feature_matrix(
    segments_df: pd.DataFrame,
    phase: str,
    config: dict,
    columns: list[str] | None = None,
    imputation_values: dict[str, float] | None = None,
) -> np.ndarray:
    columns = columns or feature_columns()
    normalized = normalize_features_by_phase(segments_df, phase, config, columns, imputation_values)
    return normalized[columns].to_numpy(dtype=float)


def visual_feature_columns(segments_df: pd.DataFrame) -> list[str]:
    """Return numeric segment features for exploratory visual clustering.

    This intentionally excludes identifiers, timestamps, labels, reasons, and
    model outputs. The remaining numeric/bool columns include aviation behavior
    plus spatial context such as lat/lon and distance-to-airport.
    """

    columns = []
    for column in segments_df.columns:
        lowered = str(column).lower()
        if column in VISUAL_FEATURE_EXCLUDE_COLUMNS:
            continue
        if lowered in {"flight", "phase"} or lowered.endswith("_time") or lowered.endswith("_id") or lowered.endswith("_name"):
            continue
        if any(lowered.startswith(prefix) for prefix in VISUAL_FEATURE_EXCLUDE_PREFIXES):
            continue
        if any(fragment in lowered for fragment in VISUAL_FEATURE_EXCLUDE_FRAGMENTS):
            continue
        series = segments_df[column]
        if not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)):
            continue
        numeric = pd.to_numeric(series, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        if numeric.notna().sum() == 0 or numeric.nunique(dropna=True) <= 1:
            continue
        columns.append(str(column))
    return columns


def compute_visual_feature_stats(segments_df: pd.DataFrame, columns: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    imputation_values: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in columns:
        series = pd.to_numeric(segments_df[column], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        if column in CIRCULAR_VISUAL_FEATURE_COLUMNS:
            finite = series.dropna()
            center = circular_mean_deg(finite.tolist())
            if math.isnan(center):
                raise ValueError(f"Cannot impute visual feature {column!r}: no finite training values")
            deltas = pd.Series(np.abs(_circular_delta_deg(finite.to_numpy(dtype=float), center)))
            q25 = deltas.quantile(0.25)
            q75 = deltas.quantile(0.75)
            iqr = q75 - q25 if not (pd.isna(q25) or pd.isna(q75)) else np.nan
            std = deltas.std(skipna=True)
            scale = iqr if pd.notna(iqr) and float(iqr) > 0 else std
            imputation_values[column] = float(center)
        else:
            median = series.median(skipna=True)
            if pd.isna(median):
                raise ValueError(f"Cannot impute visual feature {column!r}: no finite training values")
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25 if not (pd.isna(q25) or pd.isna(q75)) else np.nan
            std = series.std(skipna=True)
            scale = iqr if pd.notna(iqr) and float(iqr) > 0 else std
            imputation_values[column] = float(median)
        if pd.isna(scale) or float(scale) <= 0:
            scale = 1.0
        scales[column] = float(scale)
    return imputation_values, scales


def build_visual_feature_matrix(
    segments_df: pd.DataFrame,
    columns: list[str],
    imputation_values: dict[str, float],
    scales: dict[str, float],
) -> np.ndarray:
    out = pd.DataFrame(index=segments_df.index)
    for column in columns:
        if column in segments_df.columns:
            raw = pd.to_numeric(segments_df[column], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            raw = pd.Series(np.nan, index=segments_df.index)
        median = float(imputation_values[column])
        scale = max(float(scales.get(column, 1.0)), 1e-9)
        values = raw.fillna(median)
        if column in CIRCULAR_VISUAL_FEATURE_COLUMNS:
            out[column] = _circular_delta_deg(values.to_numpy(dtype=float), median) / scale
        else:
            out[column] = (values - median) / scale
    matrix = out[columns].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Visual feature matrix contains non-finite values after imputation")
    return matrix


def records_to_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    normalized = normalize_adsb_dataframe(df, config)
    valid = filter_valid_adsb_points(normalized, config)
    tracks = build_tracks(valid, config)
    segments = make_segments(tracks, config)
    if segments.empty:
        return pd.DataFrame()
    features = compute_segment_features(segments, config)
    if features.empty:
        return features
    return classify_segments(smooth_airport_contexts(features, config), config)

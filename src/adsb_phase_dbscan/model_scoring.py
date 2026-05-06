from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_matrix

MODEL_SEVERITY_BANDS = (
    ("NORMAL", 0.70),
    ("EDGE", 1.00),
    ("UNUSUAL", 1.35),
)

CALIBRATED_SEVERITY_BANDS = (
    ("NORMAL", 75.0),
    ("EDGE", 90.0),
    ("UNUSUAL", 97.5),
)

DEFAULT_PHASE_ASSIGNMENT = {
    "max_best_score": 1.35,
    "min_score_margin_ratio": 0.90,
}


def model_severity_from_score(score: float | None, *, scored: bool = True, no_core: bool = False) -> str:
    if no_core:
        return "NO_CORE"
    if not scored or score is None:
        return "NO_MODEL"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "NO_MODEL"
    if not math.isfinite(value):
        return "NO_MODEL"
    for name, upper in MODEL_SEVERITY_BANDS:
        if value < upper:
            return name
    return "OUTLIER"


def calibrated_percentile_from_distance(distance: float | None, calibration: dict[str, Any] | None) -> float | None:
    if not calibration:
        return None
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    cached = calibration.get("_reference_array")
    if cached is None:
        reference = calibration.get("reference_distances") or []
        if not reference:
            return None
        values = np.asarray(reference, dtype=float)
        cached = np.sort(values[np.isfinite(values)])
        calibration["_reference_array"] = cached
    values = np.asarray(cached, dtype=float)
    if len(values) == 0:
        return None
    return float(np.searchsorted(values, value, side="right") / len(values) * 100.0)


def model_severity_from_percentile(
    percentile: float | None,
    fallback_score: float | None = None,
    *,
    scored: bool = True,
    no_core: bool = False,
) -> str:
    if no_core:
        return "NO_CORE"
    if not scored:
        return "NO_MODEL"
    try:
        value = float(percentile)
    except (TypeError, ValueError):
        return model_severity_from_score(fallback_score, scored=scored, no_core=no_core)
    if not math.isfinite(value):
        return model_severity_from_score(fallback_score, scored=scored, no_core=no_core)
    for name, upper in CALIBRATED_SEVERITY_BANDS:
        if value < upper:
            return name
    return "OUTLIER"


def calibrated_model_severity(label: int | None, percentile: float | None, score: float | None) -> str:
    try:
        cluster = int(label)
    except (TypeError, ValueError):
        cluster = -1
    if cluster == -1:
        return model_severity_from_percentile(percentile, score)
    return model_severity_from_score(score)


def calibrated_reason(phase: str, percentile: float | None, score: float | None = None) -> str:
    try:
        value = float(percentile)
    except (TypeError, ValueError):
        return "outside_phase_dbscan_core"
    if not math.isfinite(value):
        return "outside_phase_dbscan_core"
    score_text = ""
    try:
        score_value = float(score)
        if math.isfinite(score_value):
            score_text = f"; eps score {score_value:.2f}"
    except (TypeError, ValueError):
        pass
    return f"outside learned {phase} pattern; farther than {value:.1f}% of training inliers{score_text}"


def _finite(row: pd.Series, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _finite_value(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def model_phase_is_plausible(row: pd.Series, phase: str) -> bool:
    """Reject phase/model matches that contradict basic flight state.

    DBSCAN distance alone can always pick a nearest model, even for ground
    traffic or cruise overflights when only arrival/departure models exist.
    These gates keep "best fit" model-first behavior from assigning obviously
    inapplicable operational phases.
    """

    speed = _finite(row, "mean_ground_speed_kt")
    delta_alt = _finite(row, "delta_altitude_ft")
    vertical_rate = _finite(row, "mean_vertical_rate_fpm")
    mean_alt_agl = _finite(row, "mean_altitude_agl_ft")
    end_alt_agl = _finite(row, "end_altitude_agl_ft")
    start_alt_agl = _finite(row, "start_altitude_agl_ft")
    delta_distance = _finite(row, "delta_distance_airport_nm")
    min_distance = _finite(row, "min_distance_airport_nm", float("inf"))

    runway_threshold_distance = _finite(row, "distance_to_runway_threshold_nm", float("inf"))
    runway_aligned = bool(row.get("runway_aligned", False)) and bool(row.get("moving_toward_runway_threshold", False))

    if phase == "FINAL_APPROACH":
        return (
            speed >= 60
            and (delta_distance < -0.2 or (runway_aligned and runway_threshold_distance <= 25))
            and min_distance <= 25
            and end_alt_agl <= 5000
            and (delta_alt < -200 or vertical_rate < -150 or (runway_aligned and runway_threshold_distance <= 25))
        )
    if phase == "INTERMEDIATE_APPROACH":
        return (
            speed >= 80
            and (delta_distance < -0.2 or (runway_aligned and runway_threshold_distance <= 80))
            and min_distance <= 80
            and end_alt_agl <= 14000
            and (delta_alt < -300 or vertical_rate < -150 or (runway_aligned and runway_threshold_distance <= 80))
        )
    if phase == "INITIAL_APPROACH":
        return (
            speed >= 160
            and (delta_distance < -0.2 or runway_aligned)
            and mean_alt_agl >= 5000
            and (delta_alt < -300 or vertical_rate < -150 or runway_aligned)
        )
    if phase == "DEPARTURE":
        return (
            speed >= 70
            and delta_distance > 0.2
            and start_alt_agl <= 12000
            and (delta_alt > 300 or vertical_rate > 150 or end_alt_agl > 1500)
        )
    if phase == "ENROUTE_CLIMB":
        return (
            speed >= 180
            and delta_alt > 300
            and vertical_rate > 150
            and 5000 <= mean_alt_agl <= 35000
        )
    return True


def phase_model_candidates_by_index(
    features: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    config: dict,
) -> dict[Any, list[dict[str, Any]]]:
    candidates_by_index: dict[Any, list[dict[str, Any]]] = {idx: [] for idx in features.index}
    if features.empty:
        return candidates_by_index

    for phase, artifact in sorted(models.items()):
        artifact_columns = artifact.get("feature_columns")
        imputation_values = artifact.get("imputation_values")
        model = artifact.get("model")
        if not artifact_columns or not imputation_values or model is None:
            continue

        if len(getattr(model, "components_", [])) == 0:
            for idx in features.index:
                candidates_by_index[idx].append(
                    {
                        "phase": phase,
                        "cluster": "NO_CORE_MODEL",
                        "score": None,
                        "distance": None,
                        "severity": "NO_CORE",
                        "scored": False,
                    }
                )
            continue

        X = build_feature_matrix(features, phase, artifact.get("config", config), artifact_columns, imputation_values)
        labels = model.predict_from_core_samples(X)
        distances = model.nearest_core_distance(X)
        scores = distances / max(float(model.eps), 1e-9)
        calibration = artifact.get("distance_calibration", {})
        for idx, label, distance, score in zip(features.index, labels, distances, scores):
            score_value = float(score)
            percentile = calibrated_percentile_from_distance(float(distance), calibration)
            plausible = model_phase_is_plausible(features.loc[idx], phase)
            candidates_by_index[idx].append(
                {
                    "phase": phase,
                    "cluster": int(label),
                    "score": score_value,
                    "distance": float(distance),
                    "calibrated_score": percentile,
                    "phase_distance_percentile": percentile,
                    "severity": calibrated_model_severity(int(label), percentile, score_value),
                    "scored": True,
                    "plausible": plausible,
                }
            )

    for candidates in candidates_by_index.values():
        candidates.sort(key=lambda item: float("inf") if item["score"] is None else float(item["score"]))
    return candidates_by_index


def phase_model_candidates(features: pd.DataFrame, models: dict[str, dict[str, Any]], config: dict) -> list[dict[str, Any]]:
    if features.empty:
        return []
    candidates_by_index = phase_model_candidates_by_index(features, models, config)
    return candidates_by_index.get(features.index[0], [])


def best_phase_model_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in candidates
            if item.get("scored")
            and item.get("score") is not None
            and item.get("plausible", True)
        ),
        None,
    )


def select_phase_model_candidate(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Select a phase from all scored DBSCAN phase candidates.

    The selector is deliberately small:
    - only candidates passing physical plausibility gates can win
    - the best normalized DBSCAN distance must be inside the configured limit
    - if another plausible model is too close, the phase remains UNKNOWN
    """

    assignment_cfg = {
        **DEFAULT_PHASE_ASSIGNMENT,
        **config.get("phase_assignment", {}),
    }
    max_best_score = _finite_value(assignment_cfg.get("max_best_score"), DEFAULT_PHASE_ASSIGNMENT["max_best_score"])
    min_score_margin_ratio = _finite_value(
        assignment_cfg.get("min_score_margin_ratio"),
        DEFAULT_PHASE_ASSIGNMENT["min_score_margin_ratio"],
    )
    plausible = [
        item
        for item in candidates
        if item.get("scored")
        and item.get("score") is not None
        and item.get("plausible", True)
        and math.isfinite(_finite_value(item.get("score"), math.inf))
    ]
    plausible.sort(key=lambda item: float(item["score"]))
    if not plausible:
        return {
            "phase": "UNKNOWN",
            "confidence": 0.0,
            "reason": "model_score_rejected:no_plausible_model",
            "candidate": None,
            "second_candidate": None,
        }

    best = plausible[0]
    best_score = float(best["score"])
    second = plausible[1] if len(plausible) > 1 else None
    if best_score > max_best_score:
        return {
            "phase": "UNKNOWN",
            "confidence": 0.0,
            "reason": "model_score_rejected:best_score_above_limit",
            "candidate": best,
            "second_candidate": second,
        }

    if second is not None:
        second_score = float(second["score"])
        if best_score >= second_score * min_score_margin_ratio:
            return {
                "phase": "UNKNOWN",
                "confidence": 0.0,
                "reason": "model_score_rejected:ambiguous_best_fit",
                "candidate": best,
                "second_candidate": second,
            }

    score_confidence = 1.0 - (best_score / max(max_best_score, 1e-9)) * 0.5
    return {
        "phase": best["phase"],
        "confidence": max(0.5, min(1.0, score_confidence)),
        "reason": "model_score_selected",
        "candidate": best,
        "second_candidate": second,
    }

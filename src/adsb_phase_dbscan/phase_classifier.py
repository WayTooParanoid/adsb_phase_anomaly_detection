from __future__ import annotations

from typing import Any

import pandas as pd

from .airport_context import context_anomaly_for_phase

ARRIVAL_PHASES = {"INITIAL_APPROACH", "INTERMEDIATE_APPROACH", "FINAL_APPROACH"}
DEPARTURE_PHASES = {"DEPARTURE", "ENROUTE_CLIMB"}
DEFAULT_RELEVANT_PHASES = ["INITIAL_APPROACH", "INTERMEDIATE_APPROACH", "FINAL_APPROACH", "DEPARTURE", "ENROUTE_CLIMB"]
PHASES = [*DEFAULT_RELEVANT_PHASES, "UNKNOWN"]

DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "common": {"min_primary_context_confidence": 0.55},
    "arrival_path": {"min_closure_nm": 0.5},
    "initial_approach": {
        "min_context_confidence": 0.8,
        "min_mean_altitude_agl_ft": 5000,
        "max_mean_altitude_agl_ft": 32000,
        "min_speed_kt": 160,
        "max_speed_kt": 390,
    },
    "intermediate_approach": {
        "min_context_confidence": 0.55,
        "max_distance_airport_nm": 70,
        "max_end_altitude_agl_ft": 15000,
        "max_mean_altitude_agl_ft": 18000,
        "min_speed_kt": 80,
        "max_speed_kt": 390,
    },
    "final_approach": {
        "min_context_confidence": 0.75,
        "max_distance_airport_nm": 20,
        "max_end_altitude_agl_ft": 2000,
        "max_mean_altitude_agl_ft": 3000,
        "min_speed_kt": 60,
        "max_speed_kt": 220,
    },
    "departure": {
        "min_delta_altitude_ft": 300,
        "min_vertical_rate_fpm": 150,
        "max_start_altitude_agl_ft": 10000,
        "min_speed_kt": 80,
    },
    "enroute_climb": {
        "min_delta_altitude_ft": 500,
        "min_vertical_rate_fpm": 300,
        "min_speed_kt": 180,
        "min_mean_altitude_agl_ft": 8000,
        "max_mean_altitude_agl_ft": 32000,
    },
}

LEGACY_THRESHOLD_ALIASES = {
    "outer_arrival": "initial_approach",
    "terminal_arrival": "intermediate_approach",
    "focused_terminal_arrival": "intermediate_approach",
    "final_arrival": "final_approach",
}


def _finite(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if pd.notna(out) else default


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = {key: dict(value) if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in (updates or {}).items():
        target = LEGACY_THRESHOLD_ALIASES.get(key, key)
        if target not in out:
            continue
        if isinstance(value, dict) and isinstance(out.get(target), dict):
            out[target].update(value)
        else:
            out[target] = value
    return out


def phase_thresholds(config: dict) -> dict[str, dict[str, float]]:
    return _deep_merge(DEFAULT_THRESHOLDS, config.get("phase_classifier", {}).get("thresholds", {}))


def relevant_phases(config: dict) -> list[str]:
    configured = config.get("phase_classifier", {}).get("relevant_phases")
    phases = configured or DEFAULT_RELEVANT_PHASES
    return [str(phase) for phase in phases if str(phase) in DEFAULT_RELEVANT_PHASES]


def _phase_candidates(scores: dict[str, int], lengths: dict[str, int], allowed: set[str]) -> list[dict[str, object]]:
    items = []
    for phase, score in scores.items():
        if phase not in allowed:
            continue
        length = max(int(lengths.get(phase, 1)), 1)
        items.append({"phase": phase, "confidence": min(1.0, float(score) / length), "evidence": f"{score}/{length}"})
    items.sort(key=lambda item: (float(item["confidence"]), str(item["phase"])), reverse=True)
    return items


def _primary_context(row: dict[str, Any], config: dict[str, Any]) -> bool:
    primary_id = str(config.get("airport", {}).get("id") or "").upper()
    airport_id = str(row.get("airport_id") or "").upper()
    return bool(airport_id) and (not primary_id or airport_id == primary_id)


def classify_segment(row: pd.Series | dict, config: dict) -> tuple[str, float, str, bool, str | None, list[dict[str, object]]]:
    r = row if isinstance(row, dict) else row.to_dict()
    thresholds = phase_thresholds(config)
    allowed = set(relevant_phases(config))

    context_type = str(r.get("airport_context_type", "none"))
    context_confidence = _finite(r, "airport_context_confidence")
    has_context = bool(r.get("airport_context_available")) and bool(r.get("airport_id"))
    primary_context = _primary_context(r, config)

    common = thresholds["common"]
    arrival_path = thresholds["arrival_path"]
    initial = thresholds["initial_approach"]
    intermediate = thresholds["intermediate_approach"]
    final = thresholds["final_approach"]
    departure = thresholds["departure"]
    climb = thresholds["enroute_climb"]

    distance_nm = _finite(r, "min_distance_airport_nm", 999999)
    threshold_distance_nm = min(distance_nm, _finite(r, "distance_to_runway_threshold_nm", 999999))
    start_alt_agl = _finite(r, "start_altitude_agl_ft", 999999)
    end_alt_agl = _finite(r, "end_altitude_agl_ft", 999999)
    mean_alt_agl = _finite(r, "mean_altitude_agl_ft", 999999)
    speed_kt = _finite(r, "mean_ground_speed_kt")
    min_speed_kt = _finite(r, "min_ground_speed_kt", speed_kt)
    delta_alt_ft = _finite(r, "delta_altitude_ft")
    vertical_rate_fpm = _finite(r, "mean_vertical_rate_fpm")
    delta_distance_nm = _finite(r, "delta_distance_airport_nm")

    runway_arrival = bool(r.get("runway_aligned")) and bool(r.get("moving_toward_runway_threshold"))
    closing = delta_distance_nm <= -float(arrival_path["min_closure_nm"]) or runway_arrival
    opening = delta_distance_nm >= float(arrival_path["min_closure_nm"])
    descending = delta_alt_ft <= -300 or vertical_rate_fpm <= -150
    climbing = delta_alt_ft >= float(departure["min_delta_altitude_ft"]) or vertical_rate_fpm >= float(departure["min_vertical_rate_fpm"])
    arrival_operation = (
        has_context
        and primary_context
        and context_type in {"arrival", "terminal"}
        and context_confidence >= float(common["min_primary_context_confidence"])
        and closing
    )

    final_gates = [
        arrival_operation,
        threshold_distance_nm <= float(final["max_distance_airport_nm"]),
        end_alt_agl <= float(final["max_end_altitude_agl_ft"]),
        mean_alt_agl <= float(final["max_mean_altitude_agl_ft"]),
        float(final["min_speed_kt"]) <= speed_kt <= float(final["max_speed_kt"]),
        min_speed_kt >= float(final["min_speed_kt"]),
    ]
    intermediate_gates = [
        arrival_operation,
        descending or runway_arrival,
        distance_nm <= float(intermediate["max_distance_airport_nm"]),
        end_alt_agl <= float(intermediate["max_end_altitude_agl_ft"]),
        mean_alt_agl <= float(intermediate["max_mean_altitude_agl_ft"]),
        float(intermediate["min_speed_kt"]) <= speed_kt <= float(intermediate["max_speed_kt"]),
    ]
    initial_gates = [
        has_context,
        primary_context,
        context_type == "arrival",
        context_confidence >= float(initial["min_context_confidence"]),
        closing,
        descending,
        float(initial["min_mean_altitude_agl_ft"]) <= mean_alt_agl <= float(initial["max_mean_altitude_agl_ft"]),
        float(initial["min_speed_kt"]) <= speed_kt <= float(initial["max_speed_kt"]),
    ]
    departure_gates = [
        has_context,
        primary_context,
        context_type in {"departure", "terminal"},
        context_confidence >= float(common["min_primary_context_confidence"]),
        opening,
        climbing,
        start_alt_agl <= float(departure["max_start_altitude_agl_ft"]),
        speed_kt >= float(departure["min_speed_kt"]),
    ]
    climb_gates = [
        has_context,
        primary_context,
        context_type in {"departure", "terminal"},
        opening,
        delta_alt_ft >= float(climb["min_delta_altitude_ft"]),
        vertical_rate_fpm >= float(climb["min_vertical_rate_fpm"]),
        speed_kt >= float(climb["min_speed_kt"]),
        float(climb["min_mean_altitude_agl_ft"]) <= mean_alt_agl <= float(climb["max_mean_altitude_agl_ft"]),
    ]

    gates = {
        "INITIAL_APPROACH": initial_gates,
        "INTERMEDIATE_APPROACH": intermediate_gates,
        "FINAL_APPROACH": final_gates,
        "DEPARTURE": departure_gates,
        "ENROUTE_CLIMB": climb_gates,
    }
    scores = {phase: sum(values) for phase, values in gates.items()}
    lengths = {phase: len(values) for phase, values in gates.items()}
    candidates = _phase_candidates(scores, lengths, allowed)

    if "FINAL_APPROACH" in allowed and all(final_gates):
        phase, confidence = "FINAL_APPROACH", 0.95
    elif "INTERMEDIATE_APPROACH" in allowed and all(intermediate_gates[:1]) and sum(intermediate_gates) >= len(intermediate_gates) - 1:
        phase, confidence = "INTERMEDIATE_APPROACH", 0.85
    elif "INITIAL_APPROACH" in allowed and all(initial_gates):
        phase, confidence = "INITIAL_APPROACH", 0.8
    elif "DEPARTURE" in allowed and all(departure_gates):
        phase, confidence = "DEPARTURE", 0.9
    elif "ENROUTE_CLIMB" in allowed and all(climb_gates):
        phase, confidence = "ENROUTE_CLIMB", 0.85
    else:
        phase, confidence = "UNKNOWN", 0.0

    context_anomaly, context_reason = context_anomaly_for_phase(r, phase, config)
    prefix = "focused_phase_rules" if phase != "UNKNOWN" else "outside_focused_phase_rules"
    return phase, confidence, f"{prefix}:{scores}", context_anomaly, context_reason, candidates


def classify_segments(segments_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    labels, confidences, reasons, context_anomalies, context_reasons, candidates = [], [], [], [], [], []
    for _, row in segments_df.iterrows():
        label, confidence, reason, context_anomaly, context_reason, phase_candidates = classify_segment(row, config)
        labels.append(label)
        confidences.append(confidence)
        reasons.append(reason)
        context_anomalies.append(context_anomaly)
        context_reasons.append(context_reason)
        candidates.append(phase_candidates)
    out = segments_df.copy()
    out["phase"] = labels
    out["phase_confidence"] = confidences
    out["phase_reason"] = reasons
    out["phase_rule_candidates"] = candidates
    out["context_anomaly"] = context_anomalies
    out["context_reason"] = context_reasons
    return out

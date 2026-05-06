from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .geometry import angular_difference_deg, haversine_nm, local_xy_m

ARRIVAL_PHASES = {
    "INITIAL_APPROACH",
    "INTERMEDIATE_APPROACH",
    "FINAL_APPROACH",
}


def airports_from_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    airports = config.get("airports") or [config["airport"]]
    scope = str(config.get("context", {}).get("airport_scope", "all")).lower()
    if scope in {"primary", "primary_only", "atl_only"}:
        primary_id = str(config.get("airport", {}).get("id") or "").upper()
        primary = next(
            (
                airport
                for airport in airports
                if bool(airport.get("primary")) or str(airport.get("id") or "").upper() == primary_id
            ),
            config["airport"],
        )
        airports = [primary]
    out = []
    for airport in airports:
        item = dict(airport)
        item.setdefault("id", str(item.get("name", "AIRPORT")).upper())
        item.setdefault("name", item["id"])
        item.setdefault("arrival_radius_nm", 45)
        item.setdefault("departure_radius_nm", 25)
        item.setdefault("ground_radius_nm", 5)
        item.setdefault("terminal_radius_nm", 60)
        item.setdefault("elevation_ft", 0)
        out.append(item)
    return out


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _confidence(parts: list[bool], floor: float = 0.0) -> float:
    if not parts:
        return floor
    return max(floor, sum(bool(part) for part in parts) / len(parts))


def _runway_anchor(runway: dict[str, Any], airport: dict[str, Any]) -> tuple[float, float]:
    lat = runway.get("threshold_lat", runway.get("lat", airport["lat"]))
    lon = runway.get("threshold_lon", runway.get("lon", airport["lon"]))
    return float(lat), float(lon)


def _runway_approaches(airport: dict[str, Any]) -> list[dict[str, Any]]:
    approaches = []
    for runway in airport.get("runways") or []:
        try:
            bearing = float(runway.get("approach_bearing_deg", runway.get("bearing_deg")))
        except (TypeError, ValueError):
            continue
        anchor_lat, anchor_lon = _runway_anchor(runway, airport)
        runway_id = str(runway.get("id") or runway.get("name") or "runway")
        approaches.append(
            {
                "id": runway_id,
                "bearing_deg": bearing % 360.0,
                "lat": anchor_lat,
                "lon": anchor_lon,
            }
        )
        if "approach_bearing_deg" not in runway and bool(runway.get("bidirectional", True)):
            approaches.append(
                {
                    "id": f"{runway_id}_reciprocal",
                    "bearing_deg": (bearing + 180.0) % 360.0,
                    "lat": anchor_lat,
                    "lon": anchor_lon,
                }
            )
    return approaches


def _runway_alignment_fields(
    lats: np.ndarray,
    lons: np.ndarray,
    feature_row: dict[str, Any],
    airport: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    approaches = _runway_approaches(airport)
    if not approaches or lats.size == 0 or lons.size == 0:
        return {
            "nearest_runway_id": "",
            "runway_bearing_error_deg": math.nan,
            "runway_lateral_offset_nm": math.nan,
            "distance_to_runway_threshold_nm": math.nan,
            "moving_toward_runway_threshold": False,
            "runway_aligned": False,
        }

    context_cfg = config.get("context", {})
    max_bearing_error = _finite(context_cfg.get("runway_max_bearing_error_deg"), 18.0)
    max_lateral_offset = _finite(context_cfg.get("runway_max_lateral_offset_nm"), 2.5)
    max_threshold_distance = _finite(context_cfg.get("runway_max_threshold_distance_nm"), 20.0)
    min_closure = _finite(context_cfg.get("runway_min_closure_nm"), 0.25)
    heading = _finite(feature_row.get("mean_heading_deg"), math.nan)
    end_lat, end_lon = float(lats[-1]), float(lons[-1])
    start_lat, start_lon = float(lats[0]), float(lons[0])

    best: dict[str, Any] | None = None
    best_rank: tuple[float, float, float] | None = None
    for approach in approaches:
        alat, alon = float(approach["lat"]), float(approach["lon"])
        bearing = float(approach["bearing_deg"])
        bearing_error = angular_difference_deg(heading, bearing) if math.isfinite(heading) else math.nan
        x_m, y_m = local_xy_m(end_lat, end_lon, alat, alon)
        theta = math.radians(bearing)
        lateral_m = x_m * math.cos(theta) - y_m * math.sin(theta)
        lateral_offset_nm = abs(lateral_m) / 1852.0
        start_distance = haversine_nm(start_lat, start_lon, alat, alon)
        end_distance = haversine_nm(end_lat, end_lon, alat, alon)
        closure = start_distance - end_distance
        moving_toward = closure >= min_closure
        aligned = (
            math.isfinite(bearing_error)
            and bearing_error <= max_bearing_error
            and lateral_offset_nm <= max_lateral_offset
            and end_distance <= max_threshold_distance
            and moving_toward
        )
        rank = (
            0.0 if aligned else 1.0,
            bearing_error if math.isfinite(bearing_error) else 999.0,
            lateral_offset_nm,
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = {
                "nearest_runway_id": approach["id"],
                "runway_bearing_error_deg": float(bearing_error) if math.isfinite(bearing_error) else math.nan,
                "runway_lateral_offset_nm": float(lateral_offset_nm),
                "distance_to_runway_threshold_nm": float(end_distance),
                "moving_toward_runway_threshold": bool(moving_toward),
                "runway_aligned": bool(aligned),
            }
    assert best is not None
    return best


def _candidate_airports_for_segment(lats: np.ndarray, lons: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
    airports = airports_from_config(config)
    if len(airports) <= 20 or lats.size == 0 or lons.size == 0:
        return airports
    center_lat = float(np.nanmean(lats))
    center_lon = float(np.nanmean(lons))
    if not (math.isfinite(center_lat) and math.isfinite(center_lon)):
        return airports
    segment_span_nm = max(haversine_nm(center_lat, center_lon, float(lat), float(lon)) for lat, lon in zip(lats, lons))
    candidates = []
    for airport in airports:
        max_context_radius = max(
            _finite(airport.get("arrival_radius_nm"), 45),
            _finite(airport.get("departure_radius_nm"), 25),
            _finite(airport.get("ground_radius_nm"), 5),
            _finite(airport.get("terminal_radius_nm"), 60),
        )
        distance_to_segment_center = haversine_nm(center_lat, center_lon, float(airport["lat"]), float(airport["lon"]))
        if distance_to_segment_center <= segment_span_nm + max_context_radius + 10:
            candidates.append(airport)
    return candidates


def _focused_primary_mode(config: dict[str, Any]) -> bool:
    focus = str(config.get("phase_classifier", {}).get("focus", "")).lower()
    return focus in {"primary_arrival_departure", "atl_arrival_departure"}


def _primary_airport_id(config: dict[str, Any]) -> str:
    return str(config.get("airport", {}).get("id") or "").upper()


def infer_airport_context_from_points(
    lats: np.ndarray,
    lons: np.ndarray,
    feature_row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    best_rank = 0.0
    best_kind = "none"
    primary_id = _primary_airport_id(config)
    focused_primary = _focused_primary_mode(config)
    primary_best: dict[str, Any] | None = None
    primary_best_score = 0.0
    primary_best_rank = 0.0
    primary_best_kind = "none"

    for airport in _candidate_airports_for_segment(lats, lons, config):
        alat, alon = float(airport["lat"]), float(airport["lon"])
        dists = np.array([haversine_nm(lat, lon, alat, alon) for lat, lon in zip(lats, lons)])
        if dists.size == 0:
            continue

        start_distance = float(dists[0])
        end_distance = float(dists[-1])
        min_distance = float(np.nanmin(dists))
        max_distance = float(np.nanmax(dists))
        delta_distance = end_distance - start_distance
        elev = _finite(airport.get("elevation_ft"))
        start_alt_agl = _finite(feature_row.get("start_altitude_ft"), math.nan) - elev
        end_alt_agl = _finite(feature_row.get("end_altitude_ft"), math.nan) - elev
        mean_alt_agl = _finite(feature_row.get("mean_altitude_ft"), math.nan) - elev
        delta_alt = _finite(feature_row.get("delta_altitude_ft"))
        vr = _finite(feature_row.get("mean_vertical_rate_fpm"))
        speed = _finite(feature_row.get("mean_ground_speed_kt"))

        arrival_radius = _finite(airport.get("arrival_radius_nm"), 45)
        departure_radius = _finite(airport.get("departure_radius_nm"), 25)
        ground_radius = _finite(airport.get("ground_radius_nm"), 5)
        terminal_radius = _finite(airport.get("terminal_radius_nm"), 60)

        ground_parts = [
            mean_alt_agl < 1000,
            end_alt_agl < 1000,
            speed < 60,
            min_distance <= ground_radius,
            abs(delta_alt) < 500,
        ]
        arrival_parts = [
            delta_alt < -1000,
            vr < -300,
            end_distance < start_distance,
            end_distance <= arrival_radius,
            end_alt_agl < 12000,
        ]
        short_final_parts = [
            end_distance < start_distance,
            end_distance <= min(arrival_radius, terminal_radius),
            end_alt_agl < 2000,
            mean_alt_agl < 3000,
            70 <= speed <= 220,
        ]
        departure_parts = [
            delta_alt > 1000,
            vr > 300,
            start_distance <= departure_radius,
            end_distance > start_distance,
            start_alt_agl < 10000,
        ]
        terminal_parts = [
            min_distance <= terminal_radius,
            mean_alt_agl < 18000,
            speed > 60,
        ]

        ground_score = _confidence(ground_parts) if min_distance <= ground_radius else 0.0
        arrival_score = 0.0
        arrival_evidence = arrival_parts
        if end_distance < start_distance and end_distance <= arrival_radius:
            arrival_score = _confidence(arrival_parts)
        if all(short_final_parts):
            arrival_score = max(arrival_score, 0.9)
            arrival_evidence = short_final_parts
        departure_score = (
            _confidence(departure_parts)
            if start_distance <= departure_radius and end_distance > start_distance
            else 0.0
        )
        terminal_score = _confidence(terminal_parts, floor=0.15) if min_distance <= terminal_radius else 0.0
        terminal_score = min(terminal_score, 0.6)
        candidates = [
            ("ground", ground_score, ground_parts, min_distance, ground_radius),
            ("arrival", arrival_score, arrival_evidence, end_distance, arrival_radius),
            ("departure", departure_score, departure_parts, start_distance, departure_radius),
        ]
        if not all(short_final_parts):
            candidates.append(("terminal", terminal_score, terminal_parts, min_distance, terminal_radius))
        kind, score, parts, proximity_distance, proximity_radius = max(candidates, key=lambda item: item[1])
        proximity_bonus = max(0.0, 1.0 - proximity_distance / max(proximity_radius, 0.1)) * 0.05
        rank = score + proximity_bonus
        if rank > best_rank:
            best_score = score
            best_rank = rank
            best_kind = kind
            best = {
                "airport_id": airport["id"],
                "airport_name": airport.get("name", airport["id"]),
                "airport_elevation_ft": elev,
                "airport_context_available": True,
                "airport_context_type": kind,
                "airport_context_confidence": float(score),
                "start_distance_airport_nm": start_distance,
                "end_distance_airport_nm": end_distance,
                "min_distance_airport_nm": min_distance,
                "max_distance_airport_nm": max_distance,
                "delta_distance_airport_nm": delta_distance,
                **_runway_alignment_fields(lats, lons, feature_row, airport, config),
                "context_reason": f"{kind}_evidence:{sum(parts)}/{len(parts)}",
            }
        if focused_primary and str(airport["id"]).upper() == primary_id and rank > primary_best_rank:
            primary_best_score = score
            primary_best_rank = rank
            primary_best_kind = kind
            primary_best = {
                "airport_id": airport["id"],
                "airport_name": airport.get("name", airport["id"]),
                "airport_elevation_ft": elev,
                "airport_context_available": True,
                "airport_context_type": kind,
                "airport_context_confidence": float(score),
                "start_distance_airport_nm": start_distance,
                "end_distance_airport_nm": end_distance,
                "min_distance_airport_nm": min_distance,
                "max_distance_airport_nm": max_distance,
                "delta_distance_airport_nm": delta_distance,
                **_runway_alignment_fields(lats, lons, feature_row, airport, config),
                "context_reason": f"primary_{kind}_evidence:{sum(parts)}/{len(parts)}",
            }

    if (
        focused_primary
        and primary_best is not None
        and primary_best_score >= 0.55
        and primary_best_kind in {"arrival", "departure", "terminal"}
        and (
            _finite(feature_row.get("mean_altitude_ft"), math.nan) - _finite(primary_best.get("airport_elevation_ft"))
            >= 3000
            or _finite(primary_best.get("min_distance_airport_nm"), math.inf) <= 15
        )
    ):
        best = primary_best
        best_score = primary_best_score
        best_kind = primary_best_kind

    if best is None or best_score < 0.4 or best_kind == "none":
        return {
            "airport_id": "",
            "airport_name": "",
            "airport_elevation_ft": 0.0,
            "airport_context_available": False,
            "airport_context_type": "none",
            "airport_context_confidence": 0.0,
            "start_distance_airport_nm": math.nan,
            "end_distance_airport_nm": math.nan,
            "min_distance_airport_nm": math.nan,
            "max_distance_airport_nm": math.nan,
            "delta_distance_airport_nm": math.nan,
            "nearest_runway_id": "",
            "runway_bearing_error_deg": math.nan,
            "runway_lateral_offset_nm": math.nan,
            "distance_to_runway_threshold_nm": math.nan,
            "moving_toward_runway_threshold": False,
            "runway_aligned": False,
            "context_reason": "no_plausible_airport_context",
        }
    return best


def context_anomaly_for_phase(row: dict[str, Any], phase: str, config: dict[str, Any]) -> tuple[bool, str | None]:
    confidence = _finite(row.get("airport_context_confidence"))
    context_type = str(row.get("airport_context_type", "none"))
    has_context = bool(row.get("airport_context_available")) and bool(row.get("airport_id"))
    min_confidence = _finite(config.get("context", {}).get("min_confidence", 0.55), 0.55)

    if phase in {*ARRIVAL_PHASES, "DEPARTURE"}:
        if not has_context:
            return True, f"{phase.lower()}_without_airport_context"
        expected = "arrival" if phase in ARRIVAL_PHASES else "departure"
        if context_type != expected and confidence < 0.8:
            return True, f"{phase.lower()}_weak_{context_type}_context"
        if confidence < min_confidence:
            return True, f"{phase.lower()}_low_context_confidence"

    if phase in ARRIVAL_PHASES and _finite(row.get("delta_distance_airport_nm")) > 1:
        return True, "arrival_moving_away_from_airport"
    if phase == "DEPARTURE" and _finite(row.get("delta_distance_airport_nm")) < -1:
        return True, "departure_moving_toward_airport"

    low_alt = _finite(row.get("mean_altitude_agl_ft"), _finite(row.get("mean_altitude_ft"))) < _finite(
        config.get("context", {}).get("low_slow_far_altitude_ft", 1500),
        1500,
    )
    slow = _finite(row.get("mean_ground_speed_kt")) < _finite(config.get("context", {}).get("low_slow_far_speed_kt", 80), 80)
    if phase != "WAITING" and low_alt and slow and not has_context:
        return True, "low_slow_far_from_known_airport"

    return False, None


def _airport_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(airport["id"]): airport for airport in airports_from_config(config)}


def _point_distance_nm(row: pd.Series, lat_col: str, lon_col: str, airport: dict[str, Any]) -> float:
    lat = _finite(row.get(lat_col), math.nan)
    lon = _finite(row.get(lon_col), math.nan)
    if math.isnan(lat) or math.isnan(lon):
        return math.nan
    return haversine_nm(lat, lon, float(airport["lat"]), float(airport["lon"]))


def recompute_airport_relative_fields(row: pd.Series, airport: dict[str, Any] | None) -> dict[str, Any]:
    if airport is None:
        elevation = 0.0
        start_distance = math.nan
        end_distance = math.nan
        mean_distance = math.nan
        min_distance = math.nan
        max_distance = math.nan
        delta_distance = math.nan
        airport_fields = {
            "airport_id": "",
            "airport_name": "",
            "airport_elevation_ft": elevation,
            "airport_context_available": False,
            "nearest_runway_id": "",
            "runway_bearing_error_deg": math.nan,
            "runway_lateral_offset_nm": math.nan,
            "distance_to_runway_threshold_nm": math.nan,
            "moving_toward_runway_threshold": False,
            "runway_aligned": False,
        }
    else:
        elevation = _finite(airport.get("elevation_ft"))
        start_distance = _point_distance_nm(row, "start_lat", "start_lon", airport)
        end_distance = _point_distance_nm(row, "end_lat", "end_lon", airport)
        mean_distance = _point_distance_nm(row, "mean_lat", "mean_lon", airport)
        finite_distances = [value for value in [start_distance, end_distance, mean_distance] if math.isfinite(value)]
        min_distance = min(finite_distances) if finite_distances else math.nan
        max_distance = max(finite_distances) if finite_distances else math.nan
        delta_distance = end_distance - start_distance if math.isfinite(start_distance) and math.isfinite(end_distance) else math.nan
        airport_fields = {
            "airport_id": airport["id"],
            "airport_name": airport.get("name", airport["id"]),
            "airport_elevation_ft": elevation,
            "airport_context_available": True,
        }
        lats = np.array([_finite(row.get("start_lat"), math.nan), _finite(row.get("end_lat"), math.nan)], dtype=float)
        lons = np.array([_finite(row.get("start_lon"), math.nan), _finite(row.get("end_lon"), math.nan)], dtype=float)
        if np.isfinite(lats).all() and np.isfinite(lons).all():
            airport_fields.update(_runway_alignment_fields(lats, lons, row.to_dict(), airport, {"context": {}}))
        else:
            airport_fields.update(
                {
                    "nearest_runway_id": "",
                    "runway_bearing_error_deg": math.nan,
                    "runway_lateral_offset_nm": math.nan,
                    "distance_to_runway_threshold_nm": math.nan,
                    "moving_toward_runway_threshold": False,
                    "runway_aligned": False,
                }
            )

    start_altitude = _finite(row.get("start_altitude_ft"), math.nan)
    end_altitude = _finite(row.get("end_altitude_ft"), math.nan)
    mean_altitude = _finite(row.get("mean_altitude_ft"), math.nan)
    return {
        **airport_fields,
        "start_distance_airport_nm": start_distance,
        "end_distance_airport_nm": end_distance,
        "min_distance_airport_nm": min_distance,
        "max_distance_airport_nm": max_distance,
        "delta_distance_airport_nm": delta_distance,
        "start_altitude_agl_ft": start_altitude - elevation if math.isfinite(start_altitude) else math.nan,
        "end_altitude_agl_ft": end_altitude - elevation if math.isfinite(end_altitude) else math.nan,
        "mean_altitude_agl_ft": mean_altitude - elevation if math.isfinite(mean_altitude) else math.nan,
    }


def smooth_airport_contexts(features: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if features.empty or "airport_id" not in features.columns or "flight_id" not in features.columns:
        return features
    out = features.copy()
    raw_columns = [
        "airport_id",
        "airport_name",
        "airport_elevation_ft",
        "airport_context_type",
        "airport_context_confidence",
        "context_reason",
        "start_distance_airport_nm",
        "end_distance_airport_nm",
        "min_distance_airport_nm",
        "max_distance_airport_nm",
        "delta_distance_airport_nm",
        "nearest_runway_id",
        "runway_bearing_error_deg",
        "runway_lateral_offset_nm",
        "distance_to_runway_threshold_nm",
        "moving_toward_runway_threshold",
        "runway_aligned",
    ]
    for col in [
        *raw_columns,
    ]:
        if col in out.columns and f"raw_{col}" not in out.columns:
            out[f"raw_{col}"] = out[col]

    airports_by_id = _airport_lookup(config)
    min_conf = _finite(config.get("context", {}).get("min_confidence", 0.55), 0.55)
    for _, group in out.groupby("flight_id", sort=False):
        plausible = group[
            group["airport_context_type"].isin(["arrival", "departure", "ground"])
            & group["airport_id"].astype(bool)
            & (pd.to_numeric(group["airport_context_confidence"], errors="coerce") >= min_conf)
        ]
        if plausible.empty:
            continue
        for context_type, type_group in plausible.groupby("airport_context_type", sort=False):
            scores = type_group.groupby("airport_id")["airport_context_confidence"].sum().sort_values(ascending=False)
            if scores.empty:
                continue
            dominant_airport = scores.index[0]
            if (type_group["airport_id"] == dominant_airport).sum() < 2:
                continue
            dominant_row = type_group[type_group["airport_id"] == dominant_airport].sort_values(
                "airport_context_confidence", ascending=False
            ).iloc[0]
            for idx, row in type_group.iterrows():
                if row["airport_id"] == dominant_airport:
                    continue
                confidence = _finite(row.get("airport_context_confidence"))
                if confidence >= 0.95:
                    continue
                if _finite(dominant_row.get("airport_context_confidence")) + 0.1 < confidence:
                    continue
                out.loc[idx, "airport_id"] = dominant_airport
                out.loc[idx, "airport_name"] = dominant_row.get("airport_name", dominant_airport)
                out.loc[idx, "airport_elevation_ft"] = dominant_row.get("airport_elevation_ft", 0.0)
                out.loc[idx, "airport_context_available"] = True
                out.loc[idx, "airport_context_confidence"] = min(0.95, max(confidence, min_conf))
                out.loc[idx, "context_reason"] = f"smoothed_{context_type}_context_from_track"
    raw_airport_ids = out["raw_airport_id"].astype(str) if "raw_airport_id" in out.columns else pd.Series("", index=out.index)
    changed_airport = out["airport_id"].astype(str) != raw_airport_ids
    for idx, row in out.loc[changed_airport].iterrows():
        airport_id = str(row.get("airport_id") or "")
        airport = airports_by_id.get(airport_id)
        if not airport_id:
            airport = None
        recomputed = recompute_airport_relative_fields(row, airport)
        for col, value in recomputed.items():
            out.loc[idx, col] = value
    return out

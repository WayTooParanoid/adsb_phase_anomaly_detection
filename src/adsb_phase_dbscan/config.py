from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_AIRPORTS: list[dict[str, Any]] = [
    {
        "id": "JFK",
        "name": "John F. Kennedy International Airport",
        "lat": 40.6413,
        "lon": -73.7781,
        "elevation_ft": 13,
        "arrival_radius_nm": 45,
        "departure_radius_nm": 25,
        "ground_radius_nm": 5,
        "terminal_radius_nm": 60,
    },
    {
        "id": "LGA",
        "name": "LaGuardia Airport",
        "lat": 40.7769,
        "lon": -73.8740,
        "elevation_ft": 21,
        "arrival_radius_nm": 35,
        "departure_radius_nm": 20,
        "ground_radius_nm": 5,
        "terminal_radius_nm": 45,
    },
    {
        "id": "EWR",
        "name": "Newark Liberty International Airport",
        "lat": 40.6895,
        "lon": -74.1745,
        "elevation_ft": 18,
        "arrival_radius_nm": 40,
        "departure_radius_nm": 25,
        "ground_radius_nm": 5,
        "terminal_radius_nm": 50,
    },
]


DEFAULT_CONFIG: dict[str, Any] = {
    "airport": {"id": "JFK", "name": "JFK", "lat": 40.6413, "lon": -73.7781, "elevation_ft": 13},
    "airports": DEFAULT_AIRPORTS,
    "data": {
        "max_time_gap_minutes": 15,
        "segment_seconds": 60,
        "segment_step_seconds": 30,
        "min_points_per_segment": 5,
        "stale_seen_pos_seconds": 30,
    },
    "dbscan": {
        "default_eps": 1.0,
        "train_unknown_phase": False,
        "eps_selection": {
            "enabled": False,
            "method": "kth_neighbor_percentile",
            "percentile": 90,
            "min_eps": 0.25,
            "max_eps": 2.5,
        },
        "min_unique_tracks_by_phase": {},
        "feature_weights_by_phase": {},
        "min_samples_by_phase": {
            "INITIAL_APPROACH": 10,
            "INTERMEDIATE_APPROACH": 20,
            "FINAL_APPROACH": 20,
            "DEPARTURE": 20,
            "ENROUTE_CLIMB": 10,
        },
        "tolerances_by_phase": {
            "INITIAL_APPROACH": {
                "xy_m": 50000,
                "altitude_ft": 3000,
                "speed_kt": 100,
                "vertical_rate_fpm": 1500,
                "heading_deg": 30,
                "delta_altitude_ft": 1500,
                "delta_distance_nm": 15,
            },
            "INTERMEDIATE_APPROACH": {
                "xy_m": 20000,
                "altitude_ft": 2000,
                "speed_kt": 80,
                "vertical_rate_fpm": 1500,
                "heading_deg": 35,
                "delta_altitude_ft": 1500,
                "delta_distance_nm": 15,
            },
            "FINAL_APPROACH": {
                "xy_m": 10000,
                "altitude_ft": 1000,
                "speed_kt": 50,
                "vertical_rate_fpm": 1000,
                "heading_deg": 20,
                "delta_altitude_ft": 800,
                "delta_distance_nm": 8,
            },
            "DEPARTURE": {
                "xy_m": 20000,
                "altitude_ft": 2500,
                "speed_kt": 90,
                "vertical_rate_fpm": 1800,
                "heading_deg": 35,
                "delta_altitude_ft": 1800,
                "delta_distance_nm": 15,
            },
            "ENROUTE_CLIMB": {
                "xy_m": 50000,
                "altitude_ft": 3000,
                "speed_kt": 100,
                "vertical_rate_fpm": 1500,
                "heading_deg": 30,
                "delta_altitude_ft": 1500,
                "delta_distance_nm": 15,
            },
        },
    },
    "live": {
        "api_url_template": "https://api.adsb.lol/v2/lat/{lat}/lon/{lon}/dist/{radius_nm}",
        "radius_nm": 50,
        "airport_marker_top_k": 12,
        "poll_seconds": 2,
        "track_history_points": 80,
        "stale_aircraft_seconds": 120,
        "alert_window_segments": 10,
        "alert_min_segments": 5,
        "alert_anomaly_ratio": 0.3,
        "user_agent": "adsb-phase-dbscan-class-project/0.1",
    },
    "context": {
        "min_confidence": 0.55,
        "arrival_confidence": 0.8,
        "departure_confidence": 0.8,
        "ground_confidence": 0.9,
        "low_slow_far_altitude_ft": 1500,
        "low_slow_far_speed_kt": 80,
        "runway_max_bearing_error_deg": 18,
        "runway_max_lateral_offset_nm": 2.5,
        "runway_max_threshold_distance_nm": 20,
        "runway_min_closure_nm": 0.25,
    },
    "phase_classifier": {
        "focus": "primary_arrival_departure",
        "min_phase_confidence": 0.55,
        "relevant_phases": [
            "INITIAL_APPROACH",
            "INTERMEDIATE_APPROACH",
            "FINAL_APPROACH",
            "DEPARTURE",
            "ENROUTE_CLIMB",
        ],
    },
    "phase_assignment": {
        "max_best_score": 1.35,
        "min_score_margin_ratio": 0.90,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    user_config: dict[str, Any] = {}
    if path:
        with Path(path).open("r", encoding="utf-8") as fh:
            user_config = yaml.safe_load(fh) or {}
        deep_update(config, user_config)
    if path and "airport" in user_config and "airports" not in user_config:
        config["airports"] = [_airport_to_context_item(config["airport"])]
    config = normalize_airports_config(config)
    return config


def _airport_to_context_item(airport: dict[str, Any]) -> dict[str, Any]:
    item = dict(airport)
    item.setdefault("id", str(item.get("name", "AIRPORT")).upper())
    item.setdefault("name", item["id"])
    item.setdefault("arrival_radius_nm", 45)
    item.setdefault("departure_radius_nm", 25)
    item.setdefault("ground_radius_nm", 5)
    item.setdefault("terminal_radius_nm", 60)
    item.setdefault("elevation_ft", 0)
    return item


def normalize_airports_config(config: dict[str, Any]) -> dict[str, Any]:
    airports = config.get("airports")
    if airports:
        config["airports"] = [_airport_to_context_item(item) for item in airports]
    else:
        config["airports"] = [_airport_to_context_item(config["airport"])]
    primary = next((item for item in config["airports"] if bool(item.get("primary"))), config["airports"][0])
    primary = dict(primary)
    config["airport"] = {
        "id": primary.get("id"),
        "name": primary.get("name", primary.get("id")),
        "lat": primary["lat"],
        "lon": primary["lon"],
        "elevation_ft": primary.get("elevation_ft", 0),
    }
    return config


def phase_tolerances(config: dict[str, Any], phase: str) -> dict[str, float]:
    tolerances = config["dbscan"]["tolerances_by_phase"]
    if phase in tolerances:
        return tolerances[phase]
    raise KeyError(f"No DBSCAN tolerances configured for phase {phase!r}")

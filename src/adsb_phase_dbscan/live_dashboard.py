from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import gzip
import hashlib
import json
import logging
import math
import pickle
import socket
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .cleaning import clean_altitude, clean_hex, clean_numeric, clean_text, filter_valid_adsb_points, normalize_adsb_dataframe, parse_timestamp
from .config import load_config
from .geometry import haversine_nm
from .io import _open_text_maybe_gzip, _trace_row_to_record, load_adsb_file
from .live_adsb_lol import _current_segment, fetch_aircraft, score_live_segments
from .model_store import load_models

LOGGER = logging.getLogger(__name__)
REPLAY_TIMEZONE = ZoneInfo("America/New_York")
REPLAY_RUNTIME_CACHE_VERSION = 2
REPLAY_TRACK_CACHE_VERSION = 1
DASHBOARD_TEMPLATE_PATH = Path(__file__).with_name("dashboard_assets") / "dashboard.html"

PHASE_COLORS = {
    "INITIAL_APPROACH": "#2563eb",
    "INTERMEDIATE_APPROACH": "#0891b2",
    "FINAL_APPROACH": "#7c3aed",
    "OUTER_ARRIVAL": "#2563eb",
    "TERMINAL_ARRIVAL": "#0891b2",
    "FINAL_ARRIVAL": "#7c3aed",
    "DEPARTURE": "#16a34a",
    "ENROUTE_CLIMB": "#ea580c",
    "ENROUTE_DESCENT": "#0f766e",
    "CRUISE_OVERFLIGHT": "#3730a3",
    "LOW_LEVEL_TRANSIT": "#84cc16",
    "HOLDING_LOCAL": "#db2777",
    "GROUND_LOCAL": "#92400e",
    "UNKNOWN": "#475569",
    "WAITING": "#94a3b8",
    "UNKNOWN_MODEL": "#334155",
    "ANOMALY_EDGE": "#facc15",
    "ANOMALY": "#ef4444",
    "ANOMALY_HIGH": "#7f1d1d",
}

DEFAULT_PHASE_ORDER = [
    "INITIAL_APPROACH",
    "INTERMEDIATE_APPROACH",
    "FINAL_APPROACH",
    "OUTER_ARRIVAL",
    "TERMINAL_ARRIVAL",
    "FINAL_ARRIVAL",
    "DEPARTURE",
    "ENROUTE_CLIMB",
    "ENROUTE_DESCENT",
    "CRUISE_OVERFLIGHT",
    "LOW_LEVEL_TRANSIT",
    "HOLDING_LOCAL",
    "GROUND_LOCAL",
]

REASON_LABELS = {
    "arrival_moving_away_from_airport": "arrival is moving away from the selected airport",
    "departure_moving_toward_airport": "departure is moving back toward the selected airport",
    "low_slow_far_from_known_airport": "aircraft is low and slow without a plausible nearby airport context",
    "model_has_no_core_samples": "the phase model has no core samples, so this phase cannot be compared safely",
    "no_model_for_phase": "there is no trained DBSCAN model for this phase",
    "waiting_for_segment": "waiting for enough recent track history to build a segment",
}

FEATURE_LABELS = {
    "start_altitude_agl_ft": "start altitude",
    "end_altitude_agl_ft": "end altitude",
    "mean_altitude_agl_ft": "mean altitude",
    "delta_altitude_ft": "altitude change",
    "mean_ground_speed_kt": "average speed",
    "min_ground_speed_kt": "minimum speed",
    "max_ground_speed_kt": "maximum speed",
    "delta_ground_speed_kt": "speed change",
    "mean_vertical_rate_fpm": "vertical rate",
    "max_abs_vertical_rate_fpm": "peak vertical rate",
    "heading_change_deg": "heading change",
    "turn_rate_deg_per_s": "turn rate",
    "sinuosity": "path curvature",
    "segment_duration_s": "segment duration",
}


def _dashboard_template() -> str:
    return DASHBOARD_TEMPLATE_PATH.read_text(encoding="utf-8")


def readable_reason(reason: Any) -> str:
    text = str(reason or "").strip()
    if not text:
        return ""
    if text in REASON_LABELS:
        return REASON_LABELS[text]
    if text.startswith("weak_or_ambiguous_evidence:"):
        return "phase evidence is weak or ambiguous"
    if text.startswith("evidence:"):
        return "behavior fell outside the learned DBSCAN core for this phase"
    if "_without_airport_context" in text:
        phase = text.split("_without_airport_context", 1)[0].replace("_", " ")
        return f"{phase} phase has no plausible airport context"
    if "_low_context_confidence" in text:
        phase = text.split("_low_context_confidence", 1)[0].replace("_", " ")
        return f"{phase} phase has low airport-context confidence"
    if "_weak_" in text and text.endswith("_context"):
        return text.replace("_", " ")
    return text.replace("_", " ")


def readable_feature_name(feature: Any) -> str:
    text = str(feature or "").strip()
    return FEATURE_LABELS.get(text, text.replace("_", " "))


def feature_contribution_text(contributions: Any, *, min_delta: float = 0.6) -> str:
    if not isinstance(contributions, list) or not contributions:
        return ""
    parts = []
    for item in contributions[:3]:
        if not isinstance(item, dict):
            continue
        feature = readable_feature_name(item.get("feature"))
        try:
            delta = abs(float(item.get("normalized_delta", 0.0)))
        except (TypeError, ValueError):
            delta = 0.0
        if delta < min_delta:
            continue
        if feature:
            direction = "higher" if float(item.get("normalized_delta", 0.0) or 0.0) > 0 else "lower"
            parts.append(f"{feature} {direction} by {delta:.1f}x tolerance")
    return ", ".join(parts)


def _score_text(score: Any) -> str:
    try:
        return f"{float(score):.2f}"
    except (TypeError, ValueError):
        return "NA"


def anomaly_source_and_explanation(row: dict[str, Any] | None, *, phase: str, cluster: Any, score: Any, is_anomaly: bool) -> tuple[str, str]:
    if row is None:
        return "Waiting", readable_reason("waiting_for_segment")
    if bool(row.get("context_anomaly", False)):
        return "Context", readable_reason(row.get("context_reason") or row.get("reason"))
    if bool(row.get("behavior_anomaly", False)):
        features = feature_contribution_text(row.get("dbscan_feature_contributions"))
        if str(cluster) == "NO_CORE_MODEL":
            return "Model", readable_reason("model_has_no_core_samples")
        try:
            percentile = float(row.get("phase_distance_percentile"))
        except (TypeError, ValueError):
            percentile = float("nan")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = float("nan")
        if not features and math.isfinite(numeric_score) and numeric_score < 1.5:
            if math.isfinite(percentile):
                return "Behavior", f"Borderline {phase} model outlier; farther than {percentile:.1f}% of training inliers."
            return "Behavior", f"Borderline {phase} model outlier (score {_score_text(score)}); the combined feature pattern is just outside the learned core."
        feature_text = f" Main differences: {features}." if features else ""
        if math.isfinite(percentile):
            return "Behavior", f"Outside learned {phase} pattern; farther than {percentile:.1f}% of training inliers (eps score {_score_text(score)}).{feature_text}"
        return "Behavior", f"Outside learned {phase} pattern (score {_score_text(score)}; normal core boundary is 1.00).{feature_text}"
    if not bool(row.get("behavior_scored", False)):
        return "Unscored", readable_reason(row.get("behavior_reason") or row.get("reason"))
    if is_anomaly:
        return "Anomaly", readable_reason(row.get("reason"))
    return "Normal", f"fits learned {phase} behavior"


def model_display_phase(row: dict[str, Any] | None, *, improvement_ratio: float = 0.85) -> tuple[str, str, bool]:
    if row is None:
        return "WAITING", "WAITING", False
    rule_phase = str(row.get("phase") or "UNKNOWN")
    return rule_phase, rule_phase, False


def nearby_airports(config: dict[str, Any], lat: float, lon: float, radius_nm: float, top_k: int) -> list[dict[str, Any]]:
    airports = []
    primary_id = str(config.get("airport", {}).get("id") or "").upper()
    for airport in config.get("airports") or [config.get("airport", {})]:
        if airport.get("lat") is None or airport.get("lon") is None:
            continue
        try:
            airport_lat = float(airport["lat"])
            airport_lon = float(airport["lon"])
        except (TypeError, ValueError):
            continue
        distance_nm = haversine_nm(lat, lon, airport_lat, airport_lon)
        if distance_nm > radius_nm:
            continue
        airport_id = str(airport.get("id") or airport.get("name") or "airport")
        is_primary = bool(airport.get("primary") or airport_id.upper() == primary_id)
        airports.append(
            {
                "id": airport_id,
                "name": airport.get("name") or airport_id,
                "lat": airport_lat,
                "lon": airport_lon,
                "primary": is_primary,
                "distance_nm": float(distance_nm),
                "arrival_radius_nm": float(airport.get("arrival_radius_nm") or 0),
                "departure_radius_nm": float(airport.get("departure_radius_nm") or 0),
                "terminal_radius_nm": float(airport.get("terminal_radius_nm") or 0),
            }
        )
    airports.sort(key=lambda item: (not item["primary"], item["distance_nm"], item["id"]))
    return airports[: max(0, int(top_k))]


def _json_safe(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is None:
                value = value.tz_localize("UTC")
            return value.tz_convert("UTC").floor("us").isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


class LiveDashboardState:
    def __init__(
        self,
        config: dict,
        models: dict,
        lat: float,
        lon: float,
        radius_nm: float,
        fetcher: Callable[[dict, float, float, float], list[dict[str, Any]]] = fetch_aircraft,
    ):
        self.config = config
        self.models = models
        self.lat = lat
        self.lon = lon
        self.radius_nm = radius_nm
        self.fetcher = fetcher
        self.histories = defaultdict(lambda: deque(maxlen=int(config["live"]["track_history_points"])))
        self.anomaly_windows = defaultdict(lambda: deque(maxlen=int(config["live"]["alert_window_segments"])))
        self.last_phase: dict[str, str] = {}
        self.last_scored_segment_id: dict[str, str] = {}
        self.scored_segment_cache: dict[str, dict[str, Any]] = {}
        self.latest: dict[str, dict[str, Any]] = {}
        self.last_error: str | None = None
        self.last_poll_time: str | None = None
        self.poll_count = 0
        self.poll_wake_event = threading.Event()
        self.lock = threading.Lock()

    def _apply_records(self, records: list[dict[str, Any]], now: datetime, *, mode: str = "live") -> None:
        updates = {}
        pending_records = []
        pending_segments = []
        cached_scores = {}
        for record in records:
            hex_id = record.get("hex")
            if not hex_id:
                continue
            self.histories[hex_id].append(record)
            segment = _current_segment(self.histories[hex_id], self.config)
            key = len(pending_records)
            if not segment.empty:
                segment_id = str(segment.iloc[-1].get("segment_id", ""))
                if segment_id and segment_id in self.scored_segment_cache:
                    cached_scores[key] = self.scored_segment_cache[segment_id]
                else:
                    pending_segments.append((key, segment))
            point = {
                "lat": record.get("lat"),
                "lon": record.get("lon"),
                "timestamp": record.get("timestamp"),
                "altitude_ft": record.get("altitude_ft"),
                "gs": record.get("gs"),
            }
            pending_records.append((key, hex_id, record, point))

        scored_by_key = score_live_segments(pending_segments, self.models, self.config) if pending_segments else {}
        for key, hex_id, record, point in pending_records:
            scored = cached_scores.get(key) or scored_by_key.get(key)
            if scored:
                scored_segment_id = str(scored.get("segment_id", ""))
                if scored_segment_id:
                    if len(self.scored_segment_cache) > 50000:
                        self.scored_segment_cache.clear()
                    self.scored_segment_cache[scored_segment_id] = scored
            if scored is None:
                phase = "WAITING"
                confidence = 0.0
                cluster = None
                anomaly_score = None
                is_anomaly = False
                reason = "waiting_for_segment"
            else:
                phase = scored.get("phase", "UNKNOWN")
                confidence = scored.get("phase_confidence", 0.0)
                cluster = scored.get("dbscan_cluster")
                anomaly_score = scored.get("anomaly_score")
                is_anomaly = bool(scored.get("is_anomaly", False))
                reason = scored.get("reason") or scored.get("context_reason") or scored.get("behavior_reason") or scored.get("phase_reason", "")
            if self.last_phase.get(hex_id) != phase:
                self.anomaly_windows[hex_id].clear()
                self.last_phase[hex_id] = phase
            segment_id = str(scored.get("segment_id", "")) if scored else ""
            is_new_segment = bool(segment_id) and self.last_scored_segment_id.get(hex_id) != segment_id
            if is_new_segment:
                self.anomaly_windows[hex_id].append(is_anomaly)
                self.last_scored_segment_id[hex_id] = segment_id
            anomaly_count, anomaly_window_count, anomaly_ratio, alert = self._derived_alert_fields(hex_id)
            anomaly_source, anomaly_explanation = anomaly_source_and_explanation(
                scored,
                phase=phase,
                cluster=cluster,
                score=anomaly_score,
                is_anomaly=is_anomaly,
            )
            display_phase, rule_phase, _ = model_display_phase(scored)
            trail = [
                {
                    "lat": p.get("lat"),
                    "lon": p.get("lon"),
                    "timestamp": p.get("timestamp"),
                    "altitude_ft": p.get("altitude_ft"),
                    "gs": p.get("gs"),
                }
                for p in self.histories[hex_id]
                if p.get("lat") is not None and p.get("lon") is not None
            ]
            updates[hex_id] = {
                "hex": hex_id,
                "flight": record.get("flight", ""),
                "lat": point["lat"],
                "lon": point["lon"],
                "altitude_ft": point["altitude_ft"],
                "ground_speed_kt": point["gs"],
                "track": record.get("track"),
                "phase": phase,
                "display_phase": display_phase,
                "rule_phase": rule_phase,
                "airport_id": scored.get("airport_id") if scored else "",
                "airport_context_confidence": scored.get("airport_context_confidence") if scored else None,
                "context_anomaly": bool(scored.get("context_anomaly", False)) if scored else False,
                "context_reason": scored.get("context_reason") if scored else None,
                "behavior_scored": bool(scored.get("behavior_scored", False)) if scored else False,
                "behavior_anomaly": bool(scored.get("behavior_anomaly", False)) if scored else False,
                "behavior_reason": scored.get("behavior_reason") if scored else None,
                "phase_confidence": confidence,
                "cluster": cluster,
                "dbscan_distance": scored.get("dbscan_distance") if scored else None,
                "calibrated_anomaly_score": scored.get("calibrated_anomaly_score") if scored else None,
                "phase_distance_percentile": scored.get("phase_distance_percentile") if scored else None,
                "dbscan_feature_contributions": scored.get("dbscan_feature_contributions") if scored else [],
                "model_severity": scored.get("model_severity") if scored else "NO_MODEL",
                "phase_candidates": scored.get("phase_candidates") if scored else [],
                "model_scored_phase": scored.get("model_scored_phase") if scored else None,
                "model_best_phase": scored.get("model_best_phase") if scored else None,
                "model_best_score": scored.get("model_best_score") if scored else None,
                "model_best_calibrated_score": scored.get("model_best_calibrated_score") if scored else None,
                "model_best_cluster": scored.get("model_best_cluster") if scored else None,
                "model_best_severity": scored.get("model_best_severity") if scored else "NO_MODEL",
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "anomaly_ratio": anomaly_ratio,
                "anomaly_window_anomalies": anomaly_count,
                "anomaly_window_segments": anomaly_window_count,
                "alert": alert,
                "reason": reason,
                "anomaly_source": anomaly_source,
                "anomaly_explanation": anomaly_explanation,
                "trail": trail,
                "updated_at": now.isoformat(),
                "mode": mode,
            }
        with self.lock:
            self.latest.update(updates)
            self._prune_stale_locked(now)
            self.last_error = None
            self.last_poll_time = now.isoformat()
            self.poll_count += 1

    def poll_seconds(self) -> float:
        with self.lock:
            return float(self.config["live"]["poll_seconds"])

    def set_poll_seconds(self, seconds: float) -> float:
        seconds = max(1.0, min(float(seconds), 60.0))
        with self.lock:
            self.config["live"]["poll_seconds"] = seconds
        self.poll_wake_event.set()
        return seconds

    def set_radius_nm(self, radius_nm: float) -> float:
        radius_nm = max(5.0, min(float(radius_nm), 250.0))
        with self.lock:
            if math.isclose(float(self.radius_nm), radius_nm, rel_tol=0.0, abs_tol=0.01):
                return self.radius_nm
            self.radius_nm = radius_nm
            self.config["live"]["radius_nm"] = radius_nm
            self.histories.clear()
            self.anomaly_windows.clear()
            self.last_phase.clear()
            self.last_scored_segment_id.clear()
            self.scored_segment_cache.clear()
            self.latest.clear()
            self.poll_count = 0
            self.last_error = None
        self.poll_wake_event.set()
        return radius_nm

    def airport_marker_top_k(self) -> int:
        with self.lock:
            return int(self.config["live"].get("airport_marker_top_k", 12))

    def set_airport_marker_top_k(self, top_k: int) -> int:
        top_k = max(0, min(int(top_k), 100))
        with self.lock:
            self.config["live"]["airport_marker_top_k"] = top_k
        return top_k

    def _prune_stale_locked(self, now: datetime) -> None:
        stale_seconds = float(self.config["live"].get("stale_aircraft_seconds", 120))
        stale_hexes = []
        for hex_id, row in self.latest.items():
            updated_at = row.get("updated_at")
            if not updated_at:
                stale_hexes.append(hex_id)
                continue
            try:
                updated = datetime.fromisoformat(str(updated_at))
            except ValueError:
                stale_hexes.append(hex_id)
                continue
            if (now - updated).total_seconds() > stale_seconds:
                stale_hexes.append(hex_id)
        for hex_id in stale_hexes:
            self.latest.pop(hex_id, None)
            self.histories.pop(hex_id, None)
            self.anomaly_windows.pop(hex_id, None)
            self.last_phase.pop(hex_id, None)
            self.last_scored_segment_id.pop(hex_id, None)

    def set_alert_settings(self, min_segments: int | None = None, anomaly_ratio: float | None = None) -> dict[str, Any]:
        changed = False
        with self.lock:
            if min_segments is not None:
                value = max(1, min(int(min_segments), int(self.config["live"]["alert_window_segments"])))
                changed = changed or value != int(self.config["live"].get("alert_min_segments", 5))
                self.config["live"]["alert_min_segments"] = value
            if anomaly_ratio is not None:
                value = max(0.0, min(float(anomaly_ratio), 1.0))
                changed = changed or value != float(self.config["live"]["alert_anomaly_ratio"])
                self.config["live"]["alert_anomaly_ratio"] = value
            if changed:
                self.anomaly_windows.clear()
        return {
            "alert_min_segments": self.config["live"].get("alert_min_segments", 5),
            "alert_anomaly_ratio": self.config["live"]["alert_anomaly_ratio"],
        }

    def _derived_alert_fields(self, hex_id: str) -> tuple[int, int, float, bool]:
        window = self.anomaly_windows.get(hex_id)
        anomaly_count = int(sum(window)) if window else 0
        window_count = len(window) if window else 0
        anomaly_ratio = anomaly_count / window_count if window_count else 0.0
        enough_alert_history = window_count >= int(self.config["live"].get("alert_min_segments", 5))
        alert = enough_alert_history and anomaly_ratio >= float(self.config["live"]["alert_anomaly_ratio"])
        return anomaly_count, window_count, anomaly_ratio, alert

    def _with_derived_alert_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        result = dict(row)
        hex_id = clean_hex(result.get("hex"))
        anomaly_count, window_count, anomaly_ratio, alert = self._derived_alert_fields(hex_id) if hex_id else (0, 0, 0.0, False)
        result["anomaly_ratio"] = anomaly_ratio
        result["anomaly_window_anomalies"] = anomaly_count
        result["anomaly_window_segments"] = window_count
        result["alert"] = alert
        return result

    def poll_once(self) -> None:
        try:
            records = self.fetcher(self.config, self.lat, self.lon, self.radius_nm)
        except requests.RequestException as exc:
            with self.lock:
                self.last_error = str(exc)
                self.last_poll_time = datetime.now(timezone.utc).isoformat()
            LOGGER.warning("ADSB.lol request failed: %s", exc)
            return

        self._apply_records(records, datetime.now(timezone.utc), mode="live")

    def snapshot(
        self,
        *,
        selected_hex: str | None = None,
        all_trails: bool = False,
        compact_trail_points: int = 8,
    ) -> dict[str, Any]:
        with self.lock:
            aircraft = []
            selected_hex = clean_hex(selected_hex)
            compact_trail_points = max(0, int(compact_trail_points))
            derived_rows = [self._with_derived_alert_fields(row) for row in self.latest.values()]
            for row in sorted(derived_rows, key=lambda item: (not item.get("alert", False), item.get("hex", ""))):
                trail = list(row.get("trail") or [])
                if clean_hex(row.get("hex")) != selected_hex:
                    trail = trail[-compact_trail_points:] if compact_trail_points else []
                row["trail"] = trail
                aircraft.append(row)
            return _json_safe(
                {
                    "center": {"lat": self.lat, "lon": self.lon},
                    "radius_nm": self.radius_nm,
                    "airport_marker_top_k": int(self.config["live"].get("airport_marker_top_k", 12)),
                    "airports": nearby_airports(
                        self.config,
                        self.lat,
                        self.lon,
                        float(self.radius_nm),
                        int(self.config["live"].get("airport_marker_top_k", 12)),
                    ),
                    "poll_seconds": self.config["live"]["poll_seconds"],
                    "alert_min_segments": self.config["live"].get("alert_min_segments", 5),
                    "alert_anomaly_ratio": self.config["live"]["alert_anomaly_ratio"],
                    "alert_window_segments": self.config["live"]["alert_window_segments"],
                    "last_poll_time": self.last_poll_time,
                    "last_error": self.last_error,
                    "poll_count": self.poll_count,
                    "models_loaded": sorted(self.models.keys()),
                    "phase_colors": PHASE_COLORS,
                    "aircraft": aircraft,
                }
            )


def run_poller(state: LiveDashboardState, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        started = time.monotonic()
        state.poll_once()
        interval = state.poll_seconds()
        deadline = started + interval
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            woke_for_setting = state.poll_wake_event.wait(min(remaining, 0.5))
            if woke_for_setting:
                state.poll_wake_event.clear()
                break


def _jsonl_gzip_records(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl_records(path))


def _iter_jsonl_records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _write_jsonl_gzip(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for record in records:
            serializable = dict(record)
            serializable["points"] = [
                {key: value for key, value in point.items() if key != "_timestamp"}
                for point in record.get("points", [])
            ]
            fh.write(json.dumps(serializable, separators=(",", ":"), allow_nan=False))
            fh.write("\n")


def replay_cache_manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".manifest.json")


def _replay_source_files(source_path: Path) -> list[Path]:
    if source_path.is_file():
        return [source_path]
    if not source_path.is_dir():
        return []
    return sorted({*source_path.rglob("trace_full_*.json"), *source_path.rglob("trace_full_*.json.gz")})


def _replay_source_metadata(source_path: Path) -> dict[str, Any]:
    files = _replay_source_files(source_path)
    if not files and source_path.exists():
        files = [source_path]
    file_rows = []
    total_size = 0
    max_mtime_ns = 0
    for file_path in files:
        try:
            stat = file_path.stat()
        except OSError:
            continue
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
        total_size += size
        max_mtime_ns = max(max_mtime_ns, mtime_ns)
        try:
            relative = str(file_path.relative_to(source_path)) if source_path.is_dir() else file_path.name
        except ValueError:
            relative = file_path.name
        file_rows.append((relative, size, mtime_ns))
    fingerprint_payload = json.dumps(file_rows, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(source_path.resolve()),
        "file_count": len(file_rows),
        "total_size": total_size,
        "max_mtime_ns": max_mtime_ns,
        "fingerprint": hashlib.sha1(fingerprint_payload.encode("utf-8")).hexdigest(),
    }


def _replay_cache_metadata(source_path: Path, config: dict, radius_nm: float) -> dict[str, Any]:
    airport = config.get("airport", {})
    return {
        "version": REPLAY_TRACK_CACHE_VERSION,
        "source": _replay_source_metadata(source_path),
        "radius_nm": float(radius_nm),
        "airport": {
            "id": airport.get("id"),
            "name": airport.get("name"),
            "lat": airport.get("lat"),
            "lon": airport.get("lon"),
        },
    }


def _read_replay_track_cache(cache_path: Path, source_path: Path, config: dict, radius_nm: float) -> list[dict[str, Any]] | None:
    manifest_path = replay_cache_manifest_path(cache_path)
    if not cache_path.exists() or not manifest_path.exists():
        return None
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata != _replay_cache_metadata(source_path, config, radius_nm):
        return None
    try:
        tracks = _jsonl_gzip_records(cache_path)
    except (OSError, json.JSONDecodeError):
        return None
    return _restore_track_epochs(tracks)


def _write_replay_track_cache(cache_path: Path, source_path: Path, config: dict, radius_nm: float, tracks: list[dict[str, Any]]) -> None:
    _write_jsonl_gzip(cache_path, tracks)
    replay_cache_manifest_path(cache_path).write_text(
        json.dumps(_replay_cache_metadata(source_path, config, radius_nm), indent=2, default=str),
        encoding="utf-8",
    )


def replay_runtime_cache_path(source_path: Path, cache_dir: str | Path) -> Path:
    resolved = str(Path(source_path).resolve())
    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]
    token = f"{Path(source_path).name}_{digest}_runtime_tracks"
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in token).strip("_")
    return Path(cache_dir) / "runtime" / f"{safe}.pkl"


def _runtime_source_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size) if path.is_file() else None,
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _restore_track_epochs(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for track in tracks:
        epochs = []
        for point in track.get("points", []):
            epoch = float(point["epoch_s"])
            epochs.append(epoch)
            point["_timestamp"] = datetime.fromtimestamp(epoch, timezone.utc)
        track["_epochs"] = epochs
    return tracks


def _track_point_count(tracks: list[dict[str, Any]]) -> int:
    return sum(len(track.get("points") or []) for track in tracks if isinstance(track, dict))


def load_replay_runtime_cache(source_path: Path, cache_dir: str | Path, *, allow_empty: bool = False) -> list[dict[str, Any]] | None:
    cache_path = replay_runtime_cache_path(source_path, cache_dir)
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as fh:
            payload = pickle.load(fh)
    except (OSError, pickle.PickleError, EOFError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != REPLAY_RUNTIME_CACHE_VERSION:
        return None
    try:
        if payload.get("source") != _runtime_source_metadata(source_path):
            return None
    except OSError:
        return None
    tracks = payload.get("tracks")
    if not isinstance(tracks, list):
        return None
    if not allow_empty and _track_point_count(tracks) == 0:
        return None
    return _restore_track_epochs(tracks)


def write_replay_runtime_cache(source_path: Path, cache_dir: str | Path, tracks: list[dict[str, Any]]) -> Path:
    cache_path = replay_runtime_cache_path(source_path, cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": REPLAY_RUNTIME_CACHE_VERSION,
        "source": _runtime_source_metadata(source_path),
        "track_count": len(tracks),
        "point_count": _track_point_count(tracks),
        "tracks": tracks,
    }
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with open(temp_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(cache_path)
    return cache_path


def _is_jsonl_path(path: Path) -> bool:
    return path.name.endswith((".jsonl", ".jsonl.gz", ".ndjson", ".ndjson.gz"))


def _read_first_jsonl_record(path: Path) -> dict[str, Any] | None:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    parsed = json.loads(line)
                    return parsed if isinstance(parsed, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _utc_iso_from_epoch(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat(timespec="milliseconds")


def _replay_point_from_raw(record: dict[str, Any]) -> dict[str, Any] | None:
    timestamp = parse_timestamp(record.get("timestamp"))
    if pd.isna(timestamp):
        return None
    timestamp = pd.Timestamp(timestamp).round("ms")
    hex_id = clean_hex(record.get("hex"))
    lat = clean_numeric(record.get("lat"))
    lon = clean_numeric(record.get("lon"))
    if not hex_id or not math.isfinite(lat) or not math.isfinite(lon):
        return None
    altitude_ft = clean_altitude(record.get("alt_baro", record.get("altitude_ft")))
    gs = clean_numeric(record.get("gs"))
    track = clean_numeric(record.get("track"))
    vertical_rate_fpm = clean_numeric(record.get("baro_rate", record.get("vertical_rate_fpm")))
    return {
        "epoch_s": float(timestamp.timestamp()),
        "timestamp": timestamp.isoformat(),
        "hex": hex_id,
        "flight": clean_text(record.get("flight")),
        "lat": lat,
        "lon": lon,
        "altitude_ft": _finite_or_none(altitude_ft),
        "gs": _finite_or_none(gs),
        "track": _finite_or_none(track),
        "vertical_rate_fpm": _finite_or_none(vertical_rate_fpm),
        "type": clean_text(record.get("type")),
        "r": clean_text(record.get("r")),
        "t": clean_text(record.get("t")),
    }


def _track_from_points(points: list[dict[str, Any]], fallback_key: str) -> dict[str, Any] | None:
    if not points:
        return None
    points = sorted(points, key=lambda item: float(item["epoch_s"]))
    first = points[0]
    hex_id = first.get("hex") or fallback_key
    track = {
        "hex": hex_id,
        "flight": first.get("flight", ""),
        "type": first.get("type", ""),
        "r": first.get("r", ""),
        "t": first.get("t", ""),
        "start_epoch_s": float(points[0]["epoch_s"]),
        "end_epoch_s": float(points[-1]["epoch_s"]),
        "points": points,
        "_epochs": [float(point["epoch_s"]) for point in points],
    }
    return _restore_track_epochs([track])[0]


def _track_from_trace_file(file_path: Path, center: tuple[float, float], radius_nm: float) -> dict[str, Any] | None:
    try:
        with _open_text_maybe_gzip(file_path) as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    trace = payload.get("trace", []) if isinstance(payload, dict) else []
    points = []
    for row in trace:
        if not isinstance(row, list):
            continue
        raw_record = _trace_row_to_record(row, payload, center, radius_nm)
        if raw_record is None:
            continue
        point = _replay_point_from_raw(raw_record)
        if point is not None:
            points.append(point)
    fallback_key = clean_hex(payload.get("icao")) if isinstance(payload, dict) else file_path.stem
    return _track_from_points(points, fallback_key or file_path.stem)


def _day_name_from_processed_path(path: Path) -> str:
    stem = path.name
    for suffix in (".jsonl.gz", ".jsonl", ".ndjson.gz", ".ndjson"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem.replace("_", " ")


def _is_processed_track_record(record: dict[str, Any]) -> bool:
    return isinstance(record, dict) and isinstance(record.get("points"), list) and (
        "icao" in record or "hex" in record or "start_epoch_s" in record
    )


def _point_from_processed_track(record: dict[str, Any], point: dict[str, Any]) -> dict[str, Any] | None:
    epoch = clean_numeric(point.get("epoch_s"))
    lat = clean_numeric(point.get("lat"))
    lon = clean_numeric(point.get("lon"))
    hex_id = clean_hex(record.get("icao") or record.get("hex"))
    if not hex_id or not math.isfinite(epoch) or not math.isfinite(lat) or not math.isfinite(lon):
        return None
    return {
        "epoch_s": float(epoch),
        "timestamp": _utc_iso_from_epoch(epoch),
        "hex": hex_id,
        "flight": clean_text(record.get("callsign") or record.get("registration") or record.get("icao")),
        "lat": lat,
        "lon": lon,
        "altitude_ft": _finite_or_none(clean_altitude(point.get("baro_alt_ft", point.get("altitude_ft")))),
        "gs": _finite_or_none(clean_numeric(point.get("groundspeed_kt", point.get("gs")))),
        "track": _finite_or_none(clean_numeric(point.get("track_deg", point.get("track")))),
        "vertical_rate_fpm": _finite_or_none(clean_numeric(point.get("vertical_rate_fpm"))),
        "type": clean_text(point.get("source_type")),
        "r": clean_text(record.get("registration")),
        "t": clean_text(record.get("type_code")),
    }


def _track_from_processed_record(record: dict[str, Any]) -> dict[str, Any] | None:
    points = []
    for point in record.get("points") or []:
        if not isinstance(point, dict):
            continue
        replay_point = _point_from_processed_track(record, point)
        if replay_point is not None:
            points.append(replay_point)
    track = _track_from_points(points, clean_hex(record.get("icao") or record.get("hex")) or clean_text(record.get("trace_file")))
    if track is None:
        return None
    track["flight"] = clean_text(record.get("callsign") or record.get("registration") or track.get("flight"))
    track["type"] = clean_text(record.get("dominant_source_type") or track.get("type"))
    track["r"] = clean_text(record.get("registration") or track.get("r"))
    track["t"] = clean_text(record.get("type_code") or track.get("t"))
    track["day"] = clean_text(record.get("day"))
    track["trace_file"] = clean_text(record.get("trace_file"))
    return track


def _load_processed_replay_tracks(
    path: Path,
    *,
    day_filter: str | None = None,
    max_tracks: int | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    tracks = []
    day_key = day_filter.casefold() if day_filter else None
    for record in _iter_jsonl_records(path):
        if cancel is not None and cancel():
            return []
        if not _is_processed_track_record(record):
            continue
        record_day = clean_text(record.get("day"))
        if day_key and record_day and record_day.casefold() != day_key:
            continue
        track = _track_from_processed_record(record)
        if track is None:
            continue
        tracks.append(track)
        if max_tracks is not None and len(tracks) >= max_tracks:
            break
        if progress is not None and len(tracks) % 25 == 0:
            progress(list(tracks))
    tracks.sort(key=lambda item: (float(item.get("start_epoch_s") or 0.0), item.get("hex") or ""))
    return tracks


def _tracks_from_dataframe(records: pd.DataFrame, max_tracks: int | None = None) -> list[dict[str, Any]]:
    if records.empty or "hex" not in records.columns:
        return []
    tracks = []
    normalized = records.sort_values(["hex", "timestamp"]).reset_index(drop=True)
    for hex_id, group in normalized.groupby("hex", sort=False):
        points = []
        for row in group.to_dict("records"):
            timestamp = parse_timestamp(row.get("timestamp"))
            if pd.isna(timestamp):
                continue
            timestamp = pd.Timestamp(timestamp).round("ms")
            lat = clean_numeric(row.get("lat"))
            lon = clean_numeric(row.get("lon"))
            if not math.isfinite(lat) or not math.isfinite(lon):
                continue
            point = {
                "epoch_s": float(timestamp.timestamp()),
                "timestamp": timestamp.isoformat(),
                "hex": clean_hex(row.get("hex")),
                "flight": clean_text(row.get("flight")),
                "lat": lat,
                "lon": lon,
                "altitude_ft": _finite_or_none(clean_altitude(row.get("altitude_ft"))),
                "gs": _finite_or_none(clean_numeric(row.get("gs"))),
                "track": _finite_or_none(clean_numeric(row.get("track"))),
                "vertical_rate_fpm": _finite_or_none(clean_numeric(row.get("vertical_rate_fpm"))),
                "type": clean_text(row.get("type")),
                "r": clean_text(row.get("r")),
                "t": clean_text(row.get("t")),
            }
            points.append(point)
        track = _track_from_points(points, str(hex_id))
        if track is not None:
            tracks.append(track)
            if max_tracks is not None and len(tracks) >= max_tracks:
                break
    return tracks


def _dataframe_from_tracks(tracks: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for track in tracks:
        for point in track.get("points", []):
            row = {
                "timestamp": point.get("timestamp"),
                "hex": point.get("hex") or track.get("hex"),
                "flight": point.get("flight") or track.get("flight", ""),
                "lat": point.get("lat"),
                "lon": point.get("lon"),
                "altitude_ft": point.get("altitude_ft"),
                "gs": point.get("gs"),
                "track": point.get("track"),
                "vertical_rate_fpm": point.get("vertical_rate_fpm"),
                "seen": None,
                "seen_pos": 0,
                "type": point.get("type") or track.get("type", ""),
                "r": point.get("r") or track.get("r", ""),
                "t": point.get("t") or track.get("t", ""),
            }
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_replay_tracks(
    path: str | Path,
    config: dict,
    radius_nm: float,
    limit_files: int | None = None,
    max_tracks: int | None = None,
    cache_dir: str | Path | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[list[dict[str, Any]]], None] | None = None,
    day_filter: str | None = None,
    workers: int = 1,
    use_processes: bool = False,
) -> list[dict[str, Any]]:
    path = Path(path)
    center = (float(config["airport"]["lat"]), float(config["airport"]["lon"]))
    use_cache = path.is_dir() and limit_files is None and max_tracks is None and cache_dir is not None
    cache_path = replay_cache_path(path, config, radius_nm, cache_dir) if use_cache else None
    if cache_path is not None:
        cached_tracks = _read_replay_track_cache(cache_path, path, config, radius_nm)
        if cached_tracks is not None:
            return cached_tracks
    if not path.is_dir():
        if cache_dir is not None and limit_files is None and max_tracks is None:
            runtime_tracks = load_replay_runtime_cache(path, cache_dir)
            if runtime_tracks is not None:
                return runtime_tracks
        if _is_jsonl_path(path):
            first_record = _read_first_jsonl_record(path)
            if first_record is not None and _is_processed_track_record(first_record):
                return _load_processed_replay_tracks(
                    path,
                    day_filter=day_filter,
                    max_tracks=max_tracks,
                    cancel=cancel,
                    progress=progress,
                )
        raw = load_adsb_file(path, config=config, radius_nm=radius_nm, limit_files=limit_files)
        normalized = normalize_adsb_dataframe(raw, config)
        normalized = filter_valid_adsb_points(normalized, config)
        return _tracks_from_dataframe(normalized, max_tracks=max_tracks)

    files = sorted({*path.rglob("trace_full_*.json"), *path.rglob("trace_full_*.json.gz")})
    if limit_files is not None and limit_files < len(files):
        if limit_files <= 1:
            files = files[:limit_files]
        else:
            step = (len(files) - 1) / (limit_files - 1)
            indices = sorted({round(i * step) for i in range(limit_files)})
            files = [files[i] for i in indices]

    tracks = []
    workers = max(1, int(workers or 1))
    if workers == 1:
        for file_index, file_path in enumerate(files, start=1):
            if cancel is not None and cancel():
                return []
            if file_index % 5000 == 0:
                LOGGER.info("Replay cache scan %s/%s files, retained %s tracks", file_index, len(files), len(tracks))
            track = _track_from_trace_file(file_path, center, radius_nm)
            if track is not None:
                tracks.append(track)
                if max_tracks is not None and len(tracks) >= max_tracks:
                    break
            if progress is not None and file_index % 500 == 0 and tracks:
                progress(list(tracks))
    else:
        next_index = 0
        completed = 0
        pending: set[concurrent.futures.Future] = set()
        executor_class = concurrent.futures.ProcessPoolExecutor if use_processes else concurrent.futures.ThreadPoolExecutor
        with executor_class(max_workers=workers) as executor:
            while next_index < len(files) and len(pending) < workers * 4:
                pending.add(executor.submit(_track_from_trace_file, files[next_index], center, radius_nm))
                next_index += 1
            while pending:
                if cancel is not None and cancel():
                    return []
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    completed += 1
                    track = future.result()
                    if track is not None:
                        tracks.append(track)
                    if max_tracks is not None and len(tracks) >= max_tracks:
                        return tracks
                    while next_index < len(files) and len(pending) < workers * 4:
                        pending.add(executor.submit(_track_from_trace_file, files[next_index], center, radius_nm))
                        next_index += 1
                    if completed % 5000 == 0:
                        LOGGER.info("Replay cache scan %s/%s files, retained %s tracks", completed, len(files), len(tracks))
                    if progress is not None and completed % 500 == 0 and tracks:
                        progress(list(tracks))
    tracks.sort(key=lambda item: (float(item.get("start_epoch_s") or 0.0), item.get("hex") or ""))
    if cancel is not None and cancel():
        return []
    if cache_path is not None:
        _write_replay_track_cache(cache_path, path, config, radius_nm, tracks)
    return tracks


def load_replay_records(
    path: str | Path,
    config: dict,
    radius_nm: float,
    limit_files: int | None = None,
    max_records: int | None = None,
    cache_dir: str | Path | None = None,
    day_filter: str | None = None,
) -> pd.DataFrame:
    tracks = load_replay_tracks(
        path,
        config,
        radius_nm,
        limit_files=limit_files,
        max_tracks=max_records,
        cache_dir=cache_dir,
        day_filter=day_filter,
    )
    return _dataframe_from_tracks(tracks)


def replay_cache_path(source_path: Path, config: dict, radius_nm: float, cache_dir: str | Path) -> Path:
    airport = str(config.get("airport", {}).get("ident") or config.get("airport", {}).get("name") or "airport")
    token = f"{source_path.name}_{airport}_{radius_nm:.1f}nm_tracks"
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in token).strip("_")
    return Path(cache_dir) / f"{safe}.jsonl.gz"


def replay_day_options(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    root = Path(path)
    if not root.exists():
        return []
    if root.is_dir() and not any(root.glob("*.jsonl*")) and (root / "extracted_days").is_dir():
        root = root / "extracted_days"
    if root.is_file() and _is_jsonl_path(root):
        counts: dict[str, int] = defaultdict(int)
        for record in _jsonl_gzip_records(root):
            if _is_processed_track_record(record):
                day = clean_text(record.get("day")) or _day_name_from_processed_path(root)
                counts[day] += 1
        if counts:
            return [
                {"day": day, "path": str(root), "files": count, "records": count, "source": "processed"}
                for day, count in sorted(counts.items())
            ]
    if root.is_file():
        return [{"day": root.stem, "path": str(root), "files": 1}]
    if root.is_dir():
        jsonl_files = sorted(
            file for file in root.iterdir() if file.is_file() and _is_jsonl_path(file)
        )
        if jsonl_files:
            return [
                {
                    "day": _day_name_from_processed_path(file),
                    "path": str(file),
                    "files": 1,
                    "records": None,
                    "source": "processed",
                }
                for file in jsonl_files
            ]
    day_dirs = sorted([item for item in root.iterdir() if item.is_dir()])
    if not day_dirs:
        return [{"day": root.name, "path": str(root), "files": len(list(root.rglob("trace_full_*.json*")))}]
    return [
        {
            "day": item.name,
            "path": str(item),
            "files": len(list(item.rglob("trace_full_*.json*"))),
        }
        for item in day_dirs
    ]


class ReplayDashboardState(LiveDashboardState):
    def __init__(
        self,
        config: dict,
        models: dict,
        lat: float,
        lon: float,
        radius_nm: float,
        replay_records: pd.DataFrame | None = None,
        step_seconds: float = 15.0,
        replay_input: str | Path | None = None,
        replay_limit_files: int | None = None,
        replay_max_records: int | None = None,
        replay_cache_dir: str | Path | None = "artifacts/replay_cache",
        replay_preload: bool = False,
    ):
        super().__init__(config, models, lat, lon, radius_nm, fetcher=lambda *_: [])
        self.replay_input = Path(replay_input) if replay_input else None
        self.replay_limit_files = replay_limit_files
        self.replay_max_records = replay_max_records
        self.replay_cache_dir = Path(replay_cache_dir) if replay_cache_dir else None
        self.selected_day: str | None = None
        self.loading_day: str | None = None
        self.loading_started: datetime | None = None
        self.loading_error: str | None = None
        self._load_thread: threading.Thread | None = None
        self._load_generation = 0
        self._preloaded_tracks: dict[str, list[dict[str, Any]]] = {}
        self._preload_loading: str | None = None
        self._preload_done = False
        self._preload_error: str | None = None
        self._preload_thread: threading.Thread | None = None
        self.pending_replay_time: str | None = None
        self._day_options = replay_day_options(self.replay_input)
        initial_records = replay_records if replay_records is not None else pd.DataFrame()
        self.tracks = _tracks_from_dataframe(initial_records)
        self.records = (
            initial_records.sort_values("timestamp").reset_index(drop=True).copy()
            if "timestamp" in initial_records.columns
            else pd.DataFrame()
        )
        self.step_seconds = max(1.0, float(step_seconds))
        self.replay_speed = 1.0
        self.cursor = 0
        self.replay_time: pd.Timestamp | None = None
        self.replay_start: pd.Timestamp | None = None
        self.replay_end: pd.Timestamp | None = None
        self.activity_start: pd.Timestamp | None = None
        self.activity_end: pd.Timestamp | None = None
        if self.tracks:
            self._set_bounds_from_tracks()
            self.replay_time = self.activity_start or self.replay_start
        elif not self.records.empty:
            self.replay_start = pd.Timestamp(self.records["timestamp"].iloc[0])
            self.replay_end = pd.Timestamp(self.records["timestamp"].iloc[-1])
            self.activity_start = self.replay_start
            self.activity_end = self.replay_end
            self.replay_time = self.replay_start
        if replay_preload and self.replay_input is not None and self._day_options:
            self.start_preload()

    def available(self) -> bool:
        return (bool(self.tracks) or not self.records.empty) and self.replay_time is not None

    def day_options(self) -> list[dict[str, Any]]:
        return self._day_options

    def _day_bounds_from_name(self, day: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        if not day:
            return None, None
        try:
            local_start = datetime.strptime(day, "%b %d %Y").replace(tzinfo=REPLAY_TIMEZONE)
        except ValueError:
            return None, None
        start = pd.Timestamp(local_start).tz_convert("UTC")
        return start, start + pd.Timedelta(days=1)

    def _set_bounds_from_tracks(self) -> None:
        starts = [float(track["start_epoch_s"]) for track in self.tracks if track.get("start_epoch_s") is not None]
        ends = [float(track["end_epoch_s"]) for track in self.tracks if track.get("end_epoch_s") is not None]
        if not starts:
            self.replay_start = None
            self.replay_end = None
            self.activity_start = None
            self.activity_end = None
            return
        self.activity_start = pd.to_datetime(min(starts), unit="s", utc=True)
        self.activity_end = pd.to_datetime(max(ends), unit="s", utc=True)
        day_start, day_end = self._day_bounds_from_name(self.selected_day)
        self.replay_start = day_start or self.activity_start
        self.replay_end = day_end or self.activity_end

    def start_preload(self) -> None:
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return
        thread = threading.Thread(target=self._preload_worker, daemon=True)
        self._preload_thread = thread
        thread.start()

    def _preload_worker(self) -> None:
        try:
            for option in self.day_options():
                day = str(option["day"])
                with self.lock:
                    if day in self._preloaded_tracks:
                        continue
                    self._preload_loading = day
                tracks = load_replay_tracks(
                    option["path"],
                    self.config,
                    self.radius_nm,
                    limit_files=None,
                    max_tracks=self.replay_max_records,
                    cache_dir=self.replay_cache_dir,
                    day_filter=option.get("day"),
                )
                with self.lock:
                    self._preloaded_tracks[day] = tracks
                    self._preload_loading = None
            with self.lock:
                self._preload_done = True
                self._preload_error = None
        except Exception as exc:
            LOGGER.exception("Failed to preload replay data")
            with self.lock:
                self._preload_loading = None
                self._preload_error = str(exc)

    def _activate_tracks(self, day: str, tracks: list[dict[str, Any]]) -> None:
        self._load_generation += 1
        self.selected_day = day
        self.tracks = tracks
        self.records = pd.DataFrame()
        self.cursor = 0
        self.replay_time = None
        self.replay_start = None
        self.replay_end = None
        self.activity_start = None
        self.activity_end = None
        self.loading_day = None
        self.loading_started = None
        self.loading_error = None
        self.reset_replay()
        self._set_bounds_from_tracks()
        selected = self._parse_replay_time(self.pending_replay_time) if self.pending_replay_time else None
        self.replay_time = selected or self.activity_start or self.replay_start

    def load_day(self, day: str | None) -> None:
        if self.replay_input is None:
            return
        options = self.day_options()
        if not options:
            return
        selected = day or self.selected_day or options[0]["day"]
        selected_option = next((item for item in options if item["day"] == selected), options[0])
        if self.selected_day == selected_option["day"] and not self.records.empty:
            return
        if self.selected_day == selected_option["day"] and self.tracks:
            return
        with self.lock:
            preloaded = self._preloaded_tracks.get(selected_option["day"])
        if preloaded is not None:
            self._activate_tracks(selected_option["day"], preloaded)
            return
        if self.loading_day == selected_option["day"] and self._load_thread is not None and self._load_thread.is_alive():
            return
        limit_files = self.replay_limit_files
        if limit_files is not None and limit_files <= 0:
            limit_files = None
        self._load_generation += 1
        generation = self._load_generation
        self.selected_day = selected_option["day"]
        self.tracks = []
        self.records = pd.DataFrame()
        self.cursor = 0
        self.replay_time = None
        self.replay_start = None
        self.replay_end = None
        self.activity_start = None
        self.activity_end = None
        self.loading_day = selected_option["day"]
        self.loading_started = datetime.now(timezone.utc)
        self.loading_error = None
        self.reset_replay()
        thread = threading.Thread(
            target=self._load_day_worker,
            args=(selected_option, limit_files, generation),
            daemon=True,
        )
        self._load_thread = thread
        thread.start()

    def _publish_tracks(self, tracks: list[dict[str, Any]], *, final: bool = False) -> None:
        if not tracks:
            return
        with self.lock:
            self.tracks = tracks
            self.records = pd.DataFrame()
            self._set_bounds_from_tracks()
            if self.replay_time is None:
                selected = self._parse_replay_time(self.pending_replay_time) if self.pending_replay_time else None
                self.replay_time = selected or self.activity_start or self.replay_start
            if final:
                self.loading_day = None
                self.loading_error = None

    def _load_day_worker(self, selected_option: dict[str, Any], limit_files: int | None, generation: int) -> None:
        try:
            def publish_partial(partial_tracks: list[dict[str, Any]]) -> None:
                if generation == self._load_generation:
                    self._publish_tracks(partial_tracks)

            tracks = load_replay_tracks(
                selected_option["path"],
                self.config,
                self.radius_nm,
                limit_files=limit_files,
                max_tracks=self.replay_max_records,
                cache_dir=self.replay_cache_dir,
                cancel=lambda: generation != self._load_generation,
                progress=publish_partial,
                day_filter=selected_option.get("day"),
            )
            if generation != self._load_generation:
                return
            if not tracks:
                with self.lock:
                    self.loading_day = None
                    self.loading_error = None
                return
            self._publish_tracks(tracks, final=True)
        except Exception as exc:
            LOGGER.exception("Failed to load replay day %s", selected_option.get("day"))
            if generation != self._load_generation:
                return
            with self.lock:
                self.loading_day = None
                self.loading_error = str(exc)
                self.last_error = str(exc)

    def _parse_replay_time(self, value: str | None) -> pd.Timestamp | None:
        if not value:
            return None
        if "T" not in value and self.selected_day:
            try:
                hours, minutes = [int(part) for part in value.split(":")[:2]]
                local_day = datetime.strptime(self.selected_day, "%b %d %Y")
                local_time = local_day.replace(hour=hours, minute=minutes, tzinfo=REPLAY_TIMEZONE)
                return pd.Timestamp(local_time).tz_convert("UTC")
            except Exception:
                return None
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(parsed):
            return None
        return pd.Timestamp(parsed)

    def reset_replay(self, replay_time: str | None = None) -> None:
        selected_time = self._parse_replay_time(replay_time) or self.replay_start
        if replay_time is None and self.activity_start is not None:
            selected_time = self.activity_start
        if selected_time is not None and self.replay_start is not None and self.replay_end is not None:
            selected_time = max(self.replay_start, min(selected_time, self.replay_end))
        with self.lock:
            self.histories.clear()
            self.anomaly_windows.clear()
            self.last_phase.clear()
            self.last_scored_segment_id.clear()
            self.latest.clear()
            self.last_error = None
            self.poll_count = 0
        self.replay_time = selected_time
        if self.tracks:
            self.cursor = 0
        elif replay_time and selected_time is not None and not self.records.empty:
            history_seconds = float(self.config["data"].get("segment_seconds", 60)) * 3
            warmup_time = selected_time - pd.Timedelta(seconds=history_seconds)
            timestamps = pd.to_datetime(self.records["timestamp"], utc=True)
            self.cursor = int(timestamps.searchsorted(warmup_time, side="left"))
        else:
            self.cursor = 0

    def _active_replay_records(self, current_time: pd.Timestamp) -> list[dict[str, Any]]:
        current_epoch = float(current_time.timestamp())
        history_seconds = max(
            float(self.config["data"].get("segment_seconds", 60)) * 3,
            float(self.config["live"].get("stale_aircraft_seconds", 120)),
        )
        stale_seconds = float(self.config["live"].get("stale_aircraft_seconds", 120))
        records = []
        for track in self.tracks:
            epochs = track.get("_epochs") or []
            if not epochs:
                continue
            point_index = bisect.bisect_right(epochs, current_epoch) - 1
            if point_index < 0:
                continue
            if current_epoch - float(epochs[point_index]) > stale_seconds:
                continue
            start_index = bisect.bisect_left(epochs, current_epoch - history_seconds)
            history_points = []
            for point in track.get("points", [])[start_index:point_index]:
                history_points.append(self._record_from_replay_point(track, point))
            hex_id = track.get("hex")
            if hex_id:
                self.histories[hex_id] = deque(history_points, maxlen=int(self.config["live"]["track_history_points"]))
            records.append(self._record_from_replay_point(track, track["points"][point_index]))
        return records

    def _record_from_replay_point(self, track: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp": point.get("_timestamp") or pd.to_datetime(point.get("timestamp"), utc=True, errors="coerce"),
            "hex": point.get("hex") or track.get("hex"),
            "flight": point.get("flight") or track.get("flight", ""),
            "lat": point.get("lat"),
            "lon": point.get("lon"),
            "altitude_ft": point.get("altitude_ft"),
            "gs": point.get("gs"),
            "track": point.get("track"),
            "vertical_rate_fpm": point.get("vertical_rate_fpm"),
            "seen": None,
            "seen_pos": 0,
            "type": point.get("type") or track.get("type", ""),
            "r": point.get("r") or track.get("r", ""),
            "t": point.get("t") or track.get("t", ""),
        }

    def poll_replay(
        self,
        *,
        step_seconds: float | None = None,
        replay_speed: float | None = None,
        reset: bool = False,
        replay_time: str | None = None,
        replay_day: str | None = None,
    ) -> None:
        if step_seconds is not None:
            self.step_seconds = max(1.0, min(float(step_seconds), 3600.0))
        if replay_speed is not None:
            self.replay_speed = max(1.0, min(float(replay_speed), 200.0))
        if replay_time:
            self.pending_replay_time = replay_time
        if self.replay_input is not None:
            self.load_day(replay_day)
        if self.loading_day and not self.tracks:
            with self.lock:
                self.last_error = f"Loading replay day {self.loading_day}"
                self.poll_count += 1
            return
        jumped = bool(reset or replay_time)
        if jumped:
            self.reset_replay(replay_time)
        if not self.available():
            with self.lock:
                self.last_error = "No replay records loaded"
                self.poll_count += 1
            return
        assert self.replay_time is not None
        assert self.replay_start is not None
        assert self.replay_end is not None
        if jumped or (self.cursor == 0 and self.poll_count == 0):
            current_time = self.replay_time
        else:
            current_time = pd.Timestamp(self.replay_time) + pd.Timedelta(seconds=self.step_seconds * self.replay_speed)
        if current_time > self.replay_end:
            self.reset_replay()
            current_time = self.replay_start
        current_time = pd.Timestamp(current_time).floor("us")
        self.replay_time = current_time
        if self.tracks:
            records = self._active_replay_records(current_time)
            self.cursor = sum(
                max(0, bisect.bisect_right(track.get("_epochs") or [], float(current_time.timestamp())))
                for track in self.tracks
            )
        else:
            records = []
            while self.cursor < len(self.records):
                row = self.records.iloc[self.cursor]
                if pd.Timestamp(row["timestamp"]) > current_time:
                    break
                records.append(row.to_dict())
                self.cursor += 1
        now = current_time.to_pydatetime()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        self._apply_records(records, now, mode="replay")

    def snapshot(
        self,
        *,
        selected_hex: str | None = None,
        all_trails: bool = False,
        compact_trail_points: int = 8,
    ) -> dict[str, Any]:
        data = super().snapshot(
            selected_hex=selected_hex,
            all_trails=all_trails,
            compact_trail_points=compact_trail_points,
        )
        data.update(
            {
                "mode": "replay",
                "replay_available": self.available(),
                "replay_day": self.selected_day,
                "replay_days": self.day_options(),
                "replay_loading": self.loading_day is not None,
                "replay_loading_day": self.loading_day,
                "replay_loading_started": _json_safe(self.loading_started),
                "replay_loading_error": self.loading_error,
                "replay_preload_loading": self._preload_loading,
                "replay_preload_done": self._preload_done,
                "replay_preload_error": self._preload_error,
                "replay_preloaded_days": sorted(self._preloaded_tracks.keys()),
                "replay_time": _json_safe(self.replay_time),
                "replay_start": _json_safe(self.replay_start),
                "replay_end": _json_safe(self.replay_end),
                "replay_activity_start": _json_safe(self.activity_start),
                "replay_activity_end": _json_safe(self.activity_end),
                "replay_cursor": int(self.cursor),
                "replay_total_records": int(len(self.tracks) if self.tracks else len(self.records)),
                "replay_total_points": int(sum(len(track.get("points", [])) for track in self.tracks)) if self.tracks else int(len(self.records)),
                "replay_step_seconds": float(self.step_seconds),
                "replay_speed": float(self.replay_speed),
            }
        )
        return data


def dashboard_html(config: dict, lat: float, lon: float) -> str:
    title = f"ADS-B Phase DBSCAN Live Dashboard - {config['airport']['name']}"
    configured_phases = config.get("dbscan", {}).get("training_phases") or DEFAULT_PHASE_ORDER
    active_phases = []
    for phase in configured_phases:
        phase = str(phase)
        if phase in PHASE_COLORS and phase not in active_phases:
            active_phases.append(phase)
    for phase in ("UNKNOWN", "WAITING"):
        if phase not in active_phases:
            active_phases.append(phase)
    phase_order_json = json.dumps(active_phases, separators=(",", ":"))
    return _dashboard_template().format(
        title=title,
        lat=lat,
        lon=lon,
        phase_order_json=phase_order_json,
    )


class DashboardHandler(BaseHTTPRequestHandler):
    state: LiveDashboardState
    replay_state: ReplayDashboardState | None = None
    html: bytes

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", self.html)
            return
        if path == "/api/state":
            query = parse_qs(parsed.query)
            body = json.dumps(
                self.state.snapshot(
                    selected_hex=query.get("selected_hex", [None])[0],
                    all_trails=query.get("all_trails", ["0"])[0] == "1",
                ),
                allow_nan=False,
            ).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return
        if path == "/api/replay/state":
            if self.replay_state is None:
                body = json.dumps({"error": "Replay is not configured"}, allow_nan=False).encode("utf-8")
                self._send(404, "application/json; charset=utf-8", body)
                return
            query = parse_qs(parsed.query)
            reset = query.get("reset", ["0"])[0] == "1"
            try:
                step_seconds = self._query_float(query, "step_seconds", self.replay_state.step_seconds, minimum=1.0)
                replay_speed = self._query_float(query, "replay_speed", self.replay_state.replay_speed, minimum=0.0)
            except ValueError as exc:
                self._send_json_error(400, str(exc))
                return
            replay_time = query.get("replay_time", [None])[0]
            replay_day = query.get("replay_day", [None])[0] or None
            try:
                self.replay_state.poll_replay(
                    step_seconds=step_seconds,
                    replay_speed=replay_speed,
                    reset=reset,
                    replay_time=replay_time,
                    replay_day=replay_day,
                )
            except ValueError as exc:
                self._send_json_error(400, str(exc))
                return
            body = json.dumps(
                self.replay_state.snapshot(
                    selected_hex=query.get("selected_hex", [None])[0],
                    all_trails=query.get("all_trails", ["0"])[0] == "1",
                ),
                allow_nan=False,
            ).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return
        if path == "/api/replay/options":
            payload = {"available": False}
            if self.replay_state is not None:
                payload = {
                    "available": bool(self.replay_state.day_options()),
                    "start": _json_safe(self.replay_state.replay_start),
                    "end": _json_safe(self.replay_state.replay_end),
                    "activity_start": _json_safe(self.replay_state.activity_start),
                    "activity_end": _json_safe(self.replay_state.activity_end),
                    "selected_day": self.replay_state.selected_day,
                    "days": self.replay_state.day_options(),
                    "loading": self.replay_state.loading_day is not None,
                    "loading_day": self.replay_state.loading_day,
                    "loading_error": self.replay_state.loading_error,
                    "records": int(len(self.replay_state.tracks) if self.replay_state.tracks else len(self.replay_state.records)),
                    "step_seconds": self.replay_state.step_seconds,
                }
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", body)
            return
        self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/settings":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            self._send_json_error(400, "Content-Length must be an integer")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json_error(400, "Request body must be valid JSON")
            return
        if not isinstance(payload, dict):
            self._send_json_error(400, "Request body must be a JSON object")
            return
        try:
            if "poll_seconds" in payload:
                self.state.set_poll_seconds(self._payload_float(payload, "poll_seconds", minimum=0.2))
            if "radius_nm" in payload:
                radius_nm = self._payload_float(payload, "radius_nm", minimum=5.0, maximum=250.0)
                self.state.set_radius_nm(radius_nm)
                if self.replay_state is not None:
                    self.replay_state.set_radius_nm(radius_nm)
            if "airport_marker_top_k" in payload:
                top_k = self._payload_int(payload, "airport_marker_top_k", minimum=0, maximum=100)
                self.state.set_airport_marker_top_k(top_k)
                if self.replay_state is not None:
                    self.replay_state.set_airport_marker_top_k(top_k)
        except ValueError as exc:
            self._send_json_error(400, str(exc))
            return
        alert_min_segments = payload.get("alert_min_segments")
        alert_anomaly_ratio = payload.get("alert_anomaly_ratio")
        if alert_min_segments is not None or alert_anomaly_ratio is not None:
            try:
                min_segments = self._payload_int(payload, "alert_min_segments", minimum=1) if alert_min_segments is not None else None
                anomaly_ratio = self._payload_float(payload, "alert_anomaly_ratio", minimum=0.0, maximum=1.0) if alert_anomaly_ratio is not None else None
            except ValueError as exc:
                self._send_json_error(400, str(exc))
                return
            self.state.set_alert_settings(min_segments=min_segments, anomaly_ratio=anomaly_ratio)
            if self.replay_state is not None:
                self.replay_state.set_alert_settings(min_segments=min_segments, anomaly_ratio=anomaly_ratio)
        body = json.dumps(self.state.snapshot(), allow_nan=False).encode("utf-8")
        self._send(200, "application/json; charset=utf-8", body)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.debug(format, *args)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}, allow_nan=False).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body)

    @staticmethod
    def _coerce_float(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if minimum is not None and number < minimum:
            raise ValueError(f"{name} must be at least {minimum:g}")
        if maximum is not None and number > maximum:
            raise ValueError(f"{name} must be at most {maximum:g}")
        return number

    @classmethod
    def _query_float(cls, query: dict[str, list[str]], name: str, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
        value = query.get(name, [default])[0]
        return cls._coerce_float(value, name, minimum=minimum, maximum=maximum)

    @classmethod
    def _payload_float(cls, payload: dict[str, Any], name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
        return cls._coerce_float(payload.get(name), name, minimum=minimum, maximum=maximum)

    @classmethod
    def _payload_int(cls, payload: dict[str, Any], name: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
        number = cls._coerce_float(payload.get(name), name, minimum=minimum, maximum=maximum)
        if not number.is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(number)


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def choose_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 20):
        if _port_available(host, port):
            return port
    raise RuntimeError(f"No available port found from {preferred_port} to {preferred_port + 19}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve a live ADS-B phase DBSCAN dashboard")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--config", default=None)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--radius-nm", type=float)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--replay-input", default=None, help="Optional ADS-B file or readsb trace directory for replay mode")
    parser.add_argument("--replay-limit-files", type=int, default=0, help="Limit trace files loaded per replay day; 0 means full day")
    parser.add_argument("--replay-max-records", type=int, default=0, help="Limit replay records kept in memory; 0 means no limit")
    parser.add_argument("--replay-step-seconds", type=float, default=15.0, help="Simulated seconds advanced per replay tick")
    parser.add_argument("--replay-cache-dir", default="artifacts/replay_cache", help="Directory for compact per-day replay caches")
    parser.add_argument("--replay-preload", action="store_true", help="Preload replay day caches into memory for instant day switching")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    lat = args.lat if args.lat is not None else float(config["airport"]["lat"])
    lon = args.lon if args.lon is not None else float(config["airport"]["lon"])
    radius_nm = args.radius_nm if args.radius_nm is not None else float(config["live"]["radius_nm"])
    models = load_models(Path(args.models_dir))
    state = LiveDashboardState(config, models, lat, lon, radius_nm)
    replay_state = None
    if args.replay_input:
        replay_path = Path(args.replay_input)
        if replay_path.exists():
            replay_state = ReplayDashboardState(
                config,
                models,
                lat,
                lon,
                radius_nm,
                step_seconds=args.replay_step_seconds,
                replay_input=replay_path,
                replay_limit_files=args.replay_limit_files,
                replay_max_records=args.replay_max_records if args.replay_max_records > 0 else None,
                replay_cache_dir=args.replay_cache_dir,
                replay_preload=args.replay_preload,
            )
            LOGGER.info("Replay days available: %s", len(replay_state.day_options()))
        else:
            LOGGER.warning("Replay input does not exist: %s", replay_path)
    port = choose_port(args.host, args.port)
    if port != args.port:
        LOGGER.info("Port %s is busy; using %s instead", args.port, port)
    DashboardHandler.state = state
    DashboardHandler.replay_state = replay_state
    DashboardHandler.html = dashboard_html(config, lat, lon).encode("utf-8")
    server = ThreadingHTTPServer((args.host, port), DashboardHandler)
    stop_event = threading.Event()
    poller = threading.Thread(target=run_poller, args=(state, stop_event), daemon=True)
    poller.start()
    LOGGER.info("Dashboard running at http://%s:%s", args.host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping dashboard")
    finally:
        stop_event.set()
        server.server_close()


if __name__ == "__main__":
    main()

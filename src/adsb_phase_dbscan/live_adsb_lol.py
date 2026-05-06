from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .cleaning import filter_valid_adsb_points, normalize_adsb_dataframe
from .config import load_config
from .features import compute_segment_features
from .model_store import load_models
from .score_offline import score_features
from .track_builder import make_segments

LOGGER = logging.getLogger(__name__)


def aircraft_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    aircraft = payload.get("ac", payload.get("aircraft", []))
    return aircraft if isinstance(aircraft, list) else []


def records_from_live_payload(payload: dict[str, Any], config: dict) -> list[dict[str, Any]]:
    snapshot_time = datetime.now(timezone.utc)
    records = []
    for ac in aircraft_from_payload(payload):
        if not isinstance(ac, dict):
            continue
        row = dict(ac)
        row.setdefault("timestamp", snapshot_time)
        records.append(row)
    if not records:
        return []
    df = normalize_adsb_dataframe(pd.DataFrame(records), config)
    df = filter_valid_adsb_points(df, config)
    return df.to_dict("records")


def fetch_aircraft(config: dict, lat: float, lon: float, radius_nm: float) -> list[dict[str, Any]]:
    url = config["live"]["api_url_template"].format(lat=lat, lon=lon, radius_nm=radius_nm)
    headers = {"User-Agent": config["live"]["user_agent"]}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    return records_from_live_payload(response.json(), config)


def _current_segment(history: deque, config: dict) -> pd.DataFrame:
    df = pd.DataFrame(history).sort_values("timestamp")
    if len(df) < config["data"]["min_points_per_segment"]:
        return pd.DataFrame()
    df["flight_id"] = df["hex"].iloc[0] + "-live"
    df["track_id"] = df["flight_id"]
    segments = make_segments(df, config)
    if segments.empty:
        return pd.DataFrame()
    return segments.tail(1)


def score_live_segment(segment_df: pd.DataFrame, models: dict, config: dict) -> dict[str, Any] | None:
    scored = score_live_segments([("segment", segment_df)], models, config)
    return scored.get("segment")


def score_live_segments(
    segments: list[tuple[Any, pd.DataFrame]],
    models: dict,
    config: dict,
) -> dict[Any, dict[str, Any] | None]:
    valid_segments = [(key, segment) for key, segment in segments if segment is not None and not segment.empty]
    results: dict[Any, dict[str, Any] | None] = {key: None for key, _ in segments}
    if not valid_segments:
        return results

    segment_frames = []
    segment_ids_by_key: dict[Any, str] = {}
    for key, segment in valid_segments:
        segment_id = str(segment.iloc[-1].get("segment_id", key))
        segment_ids_by_key[key] = segment_id
        segment_frames.append(segment)

    combined_segments = pd.concat(segment_frames, ignore_index=True)
    features = compute_segment_features(combined_segments, config)
    if features.empty:
        return results
    scored_features = score_features(features, models, config, assign_phase_from_models=True)
    scored_by_segment_id = {str(row.get("segment_id", "")): row.to_dict() for _, row in scored_features.iterrows()}
    for key, segment_id in segment_ids_by_key.items():
        results[key] = scored_by_segment_id.get(segment_id)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Poll ADSB.lol and score live aircraft against saved models")
    parser.add_argument("--models-dir", default="artifacts/models")
    parser.add_argument("--config", default=None)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--radius-nm", type=float)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    lat = args.lat if args.lat is not None else float(config["airport"]["lat"])
    lon = args.lon if args.lon is not None else float(config["airport"]["lon"])
    radius_nm = args.radius_nm if args.radius_nm is not None else float(config["live"]["radius_nm"])
    models = load_models(Path(args.models_dir))
    histories = defaultdict(lambda: deque(maxlen=int(config["live"]["track_history_points"])))
    anomaly_windows = defaultdict(lambda: deque(maxlen=int(config["live"]["alert_window_segments"])))
    last_phase: dict[str, str] = {}
    last_scored_segment_id: dict[str, str] = {}
    last_seen: dict[str, datetime] = {}
    print("time | hex | flight | phase | confidence | cluster | anomaly_score | alert")
    while True:
        try:
            records = fetch_aircraft(config, lat, lon, radius_nm)
        except requests.RequestException as exc:
            LOGGER.warning("ADSB.lol request failed: %s", exc)
            time.sleep(float(config["live"]["poll_seconds"]))
            continue
        now = datetime.now(timezone.utc)
        pending_records = []
        pending_segments = []
        for record in records:
            hex_id = record.get("hex")
            if not hex_id:
                continue
            last_seen[hex_id] = now
            histories[hex_id].append(record)
            segment = _current_segment(histories[hex_id], config)
            key = len(pending_records)
            pending_records.append((key, hex_id))
            if not segment.empty:
                pending_segments.append((key, segment))
        scored_by_key = score_live_segments(pending_segments, models, config) if pending_segments else {}
        for key, hex_id in pending_records:
            scored = scored_by_key.get(key)
            if scored is None:
                continue
            phase = scored["phase"]
            if last_phase.get(hex_id) != phase:
                anomaly_windows[hex_id].clear()
                last_phase[hex_id] = phase
            segment_id = str(scored.get("segment_id", ""))
            is_new_segment = bool(segment_id) and last_scored_segment_id.get(hex_id) != segment_id
            if is_new_segment:
                anomaly_windows[hex_id].append(bool(scored["is_anomaly"]))
                last_scored_segment_id[hex_id] = segment_id
            ratio = sum(anomaly_windows[hex_id]) / max(len(anomaly_windows[hex_id]), 1)
            enough_alert_history = len(anomaly_windows[hex_id]) >= int(config["live"].get("alert_min_segments", 5))
            alert = enough_alert_history and ratio >= float(config["live"]["alert_anomaly_ratio"])
            print(
                f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
                f"{hex_id} | {scored.get('flight','')} | {scored['phase']} | "
                f"{scored['phase_confidence']:.2f} | {scored['dbscan_cluster']} | "
                f"{scored['anomaly_score'] if scored['anomaly_score'] is not None else 'NA'} | {alert}",
                flush=True,
            )
        stale_cutoff_seconds = float(config["live"].get("stale_aircraft_seconds", 120))
        for hex_id, seen_at in list(last_seen.items()):
            if (now - seen_at).total_seconds() > stale_cutoff_seconds:
                last_seen.pop(hex_id, None)
                histories.pop(hex_id, None)
                anomaly_windows.pop(hex_id, None)
                last_phase.pop(hex_id, None)
                last_scored_segment_id.pop(hex_id, None)
        time.sleep(float(config["live"]["poll_seconds"]))


if __name__ == "__main__":
    main()

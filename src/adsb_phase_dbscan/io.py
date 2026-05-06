from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .geometry import haversine_nm


def _open_text_maybe_gzip(path: Path):
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _trace_row_to_record(row: list[Any], metadata: dict[str, Any], center: tuple[float, float] | None, radius_nm: float | None) -> dict[str, Any] | None:
    if len(row) < 4:
        return None
    lat = row[1]
    lon = row[2]
    if lat is None or lon is None:
        return None
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None
    if center is not None and radius_nm is not None:
        if haversine_nm(lat_f, lon_f, center[0], center[1]) > radius_nm:
            return None
    extra = row[8] if len(row) > 8 and isinstance(row[8], dict) else {}
    timestamp = float(metadata.get("timestamp", 0.0)) + float(row[0] or 0.0)
    return {
        "timestamp": timestamp,
        "hex": metadata.get("icao", ""),
        "flight": extra.get("flight", ""),
        "lat": lat_f,
        "lon": lon_f,
        "alt_baro": row[3] if len(row) > 3 else None,
        "gs": row[4] if len(row) > 4 else None,
        "track": extra.get("track", row[5] if len(row) > 5 else None),
        "baro_rate": row[7] if len(row) > 7 else None,
        "seen_pos": 0,
        "type": row[9] if len(row) > 9 else extra.get("type"),
        "r": metadata.get("r", ""),
        "t": metadata.get("t", ""),
    }


def load_readsb_trace_file(path: str | Path, center: tuple[float, float] | None = None, radius_nm: float | None = None) -> pd.DataFrame:
    path = Path(path)
    with _open_text_maybe_gzip(path) as fh:
        payload = json.load(fh)
    trace = payload.get("trace", []) if isinstance(payload, dict) else []
    records = []
    for row in trace:
        if not isinstance(row, list):
            continue
        record = _trace_row_to_record(row, payload, center, radius_nm)
        if record is not None:
            records.append(record)
    return pd.DataFrame(records)


def load_readsb_trace_directory(
    path: str | Path,
    center: tuple[float, float] | None = None,
    radius_nm: float | None = None,
    limit_files: int | None = None,
) -> pd.DataFrame:
    path = Path(path)
    frames = []
    files = sorted({*path.rglob("trace_full_*.json"), *path.rglob("trace_full_*.json.gz")})
    if limit_files is not None:
        if limit_files < len(files):
            if limit_files <= 1:
                files = files[:limit_files]
            else:
                step = (len(files) - 1) / (limit_files - 1)
                indices = sorted({round(i * step) for i in range(limit_files)})
                files = [files[i] for i in indices]
    for idx, file_path in enumerate(files, start=1):
        try:
            frame = load_readsb_trace_file(file_path, center=center, radius_nm=radius_nm)
        except (OSError, json.JSONDecodeError):
            continue
        if not frame.empty:
            frames.append(frame)
        if idx % 1000 == 0:
            print(f"Loaded {idx}/{len(files)} trace files; retained {sum(len(f) for f in frames)} points")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _processed_track_point_to_record(record: dict[str, Any], point: dict[str, Any], center: tuple[float, float] | None, radius_nm: float | None) -> dict[str, Any] | None:
    lat = point.get("lat")
    lon = point.get("lon")
    if lat is None or lon is None:
        return None
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return None
    if center is not None and radius_nm is not None:
        if haversine_nm(lat_f, lon_f, center[0], center[1]) > radius_nm:
            return None
    return {
        "timestamp": point.get("epoch_s") or record.get("start_epoch_s"),
        "hex": record.get("icao", ""),
        "flight": record.get("flight", ""),
        "lat": lat_f,
        "lon": lon_f,
        "alt_baro": point.get("baro_alt_ft"),
        "gs": point.get("groundspeed_kt"),
        "track": point.get("track_deg"),
        "baro_rate": point.get("vertical_rate_fpm"),
        "seen_pos": 0,
        "type": record.get("type_desc", ""),
        "r": record.get("registration", ""),
        "t": record.get("type_code", ""),
    }


def load_processed_track_file(path: str | Path, center: tuple[float, float] | None = None, radius_nm: float | None = None) -> pd.DataFrame:
    path = Path(path)
    records = []
    with _open_text_maybe_gzip(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            points = record.get("points", []) if isinstance(record, dict) else []
            for point in points:
                if not isinstance(point, dict):
                    continue
                out = _processed_track_point_to_record(record, point, center, radius_nm)
                if out is not None:
                    records.append(out)
    return pd.DataFrame(records)


def iter_processed_track_records(path: str | Path):
    path = Path(path)
    with _open_text_maybe_gzip(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def processed_track_record_to_adsb_frame(
    record: dict[str, Any],
    center: tuple[float, float] | None = None,
    radius_nm: float | None = None,
) -> pd.DataFrame:
    points = record.get("points", []) if isinstance(record, dict) else []
    rows = []
    for point in points:
        if not isinstance(point, dict):
            continue
        out = _processed_track_point_to_record(record, point, center, radius_nm)
        if out is not None:
            rows.append(out)
    return pd.DataFrame(rows)


def _processed_track_files(path: Path) -> list[Path]:
    candidates = sorted({*path.glob("*.jsonl"), *path.glob("*.jsonl.gz")})
    extracted = path / "extracted_days"
    if extracted.is_dir():
        candidates.extend(sorted({*extracted.glob("*.jsonl"), *extracted.glob("*.jsonl.gz")}))
    return sorted(set(candidates))


def load_processed_track_directory(
    path: str | Path,
    center: tuple[float, float] | None = None,
    radius_nm: float | None = None,
    limit_files: int | None = None,
) -> pd.DataFrame:
    path = Path(path)
    frames = []
    files = _processed_track_files(path)
    if limit_files is not None:
        files = files[: max(0, int(limit_files))]
    for idx, file_path in enumerate(files, start=1):
        try:
            frame = load_processed_track_file(file_path, center=center, radius_nm=radius_nm)
        except (OSError, json.JSONDecodeError):
            continue
        if not frame.empty:
            frames.append(frame)
        print(f"Loaded processed day {idx}/{len(files)}; retained {sum(len(f) for f in frames)} points")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_adsb_file(path: str | Path, config: dict[str, Any] | None = None, radius_nm: float | None = None, limit_files: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        center = None
        if config is not None:
            center = (float(config["airport"]["lat"]), float(config["airport"]["lon"]))
        processed_files = _processed_track_files(path)
        if processed_files:
            return load_processed_track_directory(path, center=center, radius_nm=radius_nm, limit_files=limit_files)
        return load_readsb_trace_directory(path, center=center, radius_nm=radius_nm, limit_files=limit_files)
    suffix = path.suffix.lower()
    suffixes = [item.lower() for item in path.suffixes]
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"} or tuple(suffixes[-2:]) in {(".jsonl", ".gz"), (".ndjson", ".gz")}:
        return pd.read_json(path, lines=True, compression="infer")
    if suffix == ".json" or suffixes[-2:] == [".json", ".gz"]:
        with _open_text_maybe_gzip(path) as fh:
            payload = json.load(fh)
        if isinstance(payload, dict) and "trace" in payload:
            center = None
            if config is not None:
                center = (float(config["airport"]["lat"]), float(config["airport"]["lon"]))
            return load_readsb_trace_file(path, center=center, radius_nm=radius_nm)
        if isinstance(payload, dict):
            payload = payload.get("ac") or payload.get("aircraft") or payload.get("records") or [payload]
        return pd.DataFrame(payload)
    raise ValueError(f"Unsupported input format: {path}")


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str, allow_nan=False)

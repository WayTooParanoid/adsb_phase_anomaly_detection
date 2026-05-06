from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .airport_context import airports_from_config
from .config import load_config
from .features import records_to_features
from .io import _processed_track_files, iter_processed_track_records, processed_track_record_to_adsb_frame, save_json


def _day_name(path: Path) -> str:
    stem = path.name
    for suffix in (".jsonl.gz", ".jsonl", ".gz"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.replace("_", " ")


def _metadata_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".manifest.json")


def _write_cache(features: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    serializable = features.drop(columns=["points"], errors="ignore")
    if output.suffix.lower() in {".parquet", ".pq"}:
        serializable.to_parquet(output, index=False)
        return
    compression = "gzip" if output.suffix.lower() == ".gz" or output.name.endswith(".csv.gz") else None
    serializable.to_csv(output, index=False, compression=compression)


def _phase_counts(features: pd.DataFrame) -> dict[str, int]:
    phases = features.get("phase", pd.Series(dtype=object))
    return {str(key): int(value) for key, value in phases.value_counts().items()}


def _progress_counter(label: str, current: int, points: int, started: float) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    rate = current / elapsed if elapsed > 0 else 0.0
    print(f"\r{label} {current} tracks loaded | {points} points | {rate:.1f} tracks/s", end="", flush=True)


def _processed_record_flight_id(record: dict[str, Any], day_file: Path, index: int) -> str:
    hex_id = str(record.get("icao") or "unknown")
    start = record.get("start_epoch_s", "")
    end = record.get("end_epoch_s", "")
    return f"{_day_name(day_file)}-{hex_id}-{start}-{end}-{index}"


def build_features_from_processed_day(
    day_file: str | Path,
    config: dict[str, Any],
    radius_nm: float | None = None,
    *,
    progress: bool = True,
) -> pd.DataFrame:
    day_file = Path(day_file)
    center = (float(config["airport"]["lat"]), float(config["airport"]["lon"]))
    frames: list[pd.DataFrame] = []
    started = time.monotonic()
    label = _day_name(day_file)
    track_count = 0

    for track_count, record in enumerate(iter_processed_track_records(day_file), start=1):
        raw = processed_track_record_to_adsb_frame(record, center=center, radius_nm=radius_nm)
        if not raw.empty:
            raw["flight_id"] = _processed_record_flight_id(record, day_file, track_count)
            raw["track_id"] = raw["flight_id"]
            frames.append(raw)
        if progress and track_count % 1000 == 0:
            _progress_counter(f"{label} load", track_count, sum(len(frame) for frame in frames), started)

    if progress:
        _progress_counter(f"{label} load", track_count, sum(len(frame) for frame in frames), started)
        print()
    if not frames:
        return pd.DataFrame()

    features = records_to_features(pd.concat(frames, ignore_index=True), config)
    if features.empty:
        return features
    features["source_day"] = label
    features["source_file"] = day_file.name
    return features


def build_training_cache(
    input_path: str | Path,
    output: str | Path,
    config: dict[str, Any],
    *,
    radius_nm: float | None = None,
    limit_files: int | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output = Path(output)
    files = _processed_track_files(input_path)
    if limit_files is not None:
        files = files[: max(0, int(limit_files))]
    if not files:
        raise ValueError(f"No processed *.jsonl or *.jsonl.gz files found in {input_path}")

    day_summaries: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for index, day_file in enumerate(files, start=1):
        started = time.monotonic()
        print(f"Building {_day_name(day_file)} ({index}/{len(files)}) from {day_file.name}", flush=True)
        features = build_features_from_processed_day(day_file, config, radius_nm=radius_nm)
        frames.append(features)
        day_summaries.append(
            {
                "day": _day_name(day_file),
                "source_file": day_file.name,
                "segments": int(len(features)),
                "phase_counts": _phase_counts(features),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        )

    final = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _write_cache(final, output)
    metadata = {
        "input_path": str(input_path.resolve()),
        "output": str(output.resolve()),
        "radius_nm": radius_nm,
        "airport_count": int(len(airports_from_config(config))),
        "files": [str(path.resolve()) for path in files],
        "days": day_summaries,
        "n_segments": int(len(final)),
        "phase_counts": _phase_counts(final),
    }
    save_json(metadata, _metadata_path(output))
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DBSCAN-ready segment feature cache from processed ADS-B track days")
    parser.add_argument("--input", default="../processed/extracted_days")
    parser.add_argument("--output", default="artifacts/training_cache/atl_segments.csv.gz")
    parser.add_argument("--config", default="config.real.yaml")
    parser.add_argument("--input-radius-nm", type=float, default=120.0)
    parser.add_argument("--limit-files", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metadata = build_training_cache(
        args.input,
        args.output,
        load_config(args.config),
        radius_nm=args.input_radius_nm,
        limit_files=args.limit_files,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

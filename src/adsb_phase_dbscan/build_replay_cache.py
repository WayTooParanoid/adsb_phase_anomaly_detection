from __future__ import annotations

import argparse
import concurrent.futures
import logging
from pathlib import Path
from typing import Any

from .config import load_config
from .live_dashboard import load_replay_runtime_cache, load_replay_tracks, replay_cache_path, replay_day_options, replay_runtime_cache_path, write_replay_runtime_cache

LOGGER = logging.getLogger(__name__)


def selected_replay_days(data_dir: str | Path, requested_days: list[str] | None = None) -> list[dict[str, Any]]:
    options = replay_day_options(data_dir)
    if not requested_days:
        return options
    requested = {day.casefold() for day in requested_days}
    return [option for option in options if str(option["day"]).casefold() in requested]


def build_cache_for_day(
    day_option: dict[str, Any],
    config: dict,
    radius_nm: float,
    cache_dir: str | Path,
    *,
    force: bool = False,
    limit_files: int | None = None,
    max_tracks: int | None = None,
    workers: int = 1,
    use_processes: bool = False,
    runtime: bool = False,
) -> dict[str, Any]:
    day_path = Path(day_option["path"])
    if runtime:
        cache_path = replay_runtime_cache_path(day_path, cache_dir)
        if cache_path.exists() and not force:
            tracks = load_replay_runtime_cache(day_path, cache_dir)
            if tracks is not None:
                return {
                    "day": day_option["day"],
                    "status": "skipped",
                    "cache": str(cache_path),
                    "tracks": len(tracks),
                    "points": sum(len(track.get("points", [])) for track in tracks),
                }
        if cache_path.exists() and force:
            cache_path.unlink()
        tracks = load_replay_tracks(
            day_path,
            config,
            radius_nm,
            limit_files=limit_files,
            max_tracks=max_tracks,
            cache_dir=None,
            workers=workers,
            use_processes=use_processes,
            day_filter=day_option.get("day"),
        )
        write_replay_runtime_cache(day_path, cache_dir, tracks)
        return {
            "day": day_option["day"],
            "status": "built",
            "cache": str(cache_path),
            "tracks": len(tracks),
            "points": sum(len(track.get("points", [])) for track in tracks),
        }

    cache_path = replay_cache_path(day_path, config, radius_nm, cache_dir)
    if cache_path.exists() and not force:
        tracks = load_replay_tracks(day_path, config, radius_nm, cache_dir=cache_dir)
        return {
            "day": day_option["day"],
            "status": "skipped",
            "cache": str(cache_path),
            "tracks": len(tracks),
            "points": sum(len(track.get("points", [])) for track in tracks),
        }
    if cache_path.exists() and force:
        cache_path.unlink()
    tracks = load_replay_tracks(
        day_path,
        config,
        radius_nm,
        limit_files=limit_files,
        max_tracks=max_tracks,
        cache_dir=cache_dir,
        workers=workers,
        use_processes=use_processes,
    )
    return {
        "day": day_option["day"],
        "status": "built",
        "cache": str(cache_path),
        "tracks": len(tracks),
        "points": sum(len(track.get("points", [])) for track in tracks),
    }


def build_replay_cache(
    data_dir: str | Path,
    config: dict,
    radius_nm: float,
    cache_dir: str | Path,
    *,
    days: list[str] | None = None,
    force: bool = False,
    limit_files: int | None = None,
    max_tracks: int | None = None,
    workers: int = 1,
    day_workers: int = 1,
    use_processes: bool = False,
    runtime: bool = False,
) -> list[dict[str, Any]]:
    selected = selected_replay_days(data_dir, days)
    if not selected:
        return []

    def build_one(index: int, day_option: dict[str, Any]) -> dict[str, Any]:
        LOGGER.info(
            "Building replay cache %s/%s: %s (%s raw files)",
            index,
            len(selected),
            day_option["day"],
            day_option.get("files", "unknown"),
        )
        return build_cache_for_day(
            day_option,
            config,
            radius_nm,
            cache_dir,
            force=force,
            limit_files=limit_files,
            max_tracks=max_tracks,
            workers=workers,
            use_processes=use_processes,
            runtime=runtime,
        )

    day_workers = max(1, int(day_workers or 1))
    results = []
    if day_workers == 1 or len(selected) == 1:
        for index, day_option in enumerate(selected, start=1):
            result = build_one(index, day_option)
            LOGGER.info(
                "%s %s: %s tracks, %s points -> %s",
                result["status"].capitalize(),
                result["day"],
                result["tracks"],
                result["points"],
                result["cache"],
            )
            results.append(result)
        return results

    max_workers = min(day_workers, len(selected))
    LOGGER.info("Building up to %s day caches concurrently", max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_day = {
            executor.submit(build_one, index, day_option): day_option
            for index, day_option in enumerate(selected, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_day):
            result = future.result()
            LOGGER.info(
                "%s %s: %s tracks, %s points -> %s",
                result["status"].capitalize(),
                result["day"],
                result["tracks"],
                result["points"],
                result["cache"],
            )
            results.append(result)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build preprocessed replay caches from raw ADS-B trace day folders")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default="../Data")
    parser.add_argument("--out", default="artifacts/replay_cache")
    parser.add_argument("--radius-nm", type=float, default=None)
    parser.add_argument("--day", action="append", default=None, help="Day folder to build, e.g. 'Dec 26 2025'. Repeat for multiple days.")
    parser.add_argument("--force", action="store_true", help="Rebuild caches even when the output file already exists")
    parser.add_argument("--limit-files", type=int, default=0, help="Debug limit for raw trace files per day; 0 means all files")
    parser.add_argument("--max-tracks", type=int, default=0, help="Debug limit for tracks per day; 0 means all tracks")
    parser.add_argument("--workers", type=int, default=8, help="Parallel raw trace file readers")
    parser.add_argument("--day-workers", type=int, default=1, help="Number of day folders to build concurrently")
    parser.add_argument("--processes", action="store_true", help="Use process workers instead of threads")
    parser.add_argument("--runtime", action="store_true", help="Build dashboard-native binary caches for faster replay loading")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    radius_nm = args.radius_nm if args.radius_nm is not None else float(config["live"]["radius_nm"])
    limit_files = args.limit_files if args.limit_files > 0 else None
    max_tracks = args.max_tracks if args.max_tracks > 0 else None
    results = build_replay_cache(
        args.data_dir,
        config,
        radius_nm,
        args.out,
        days=args.day,
        force=args.force,
        limit_files=limit_files,
        max_tracks=max_tracks,
        workers=args.workers,
        day_workers=args.day_workers,
        use_processes=args.processes,
        runtime=args.runtime,
    )
    total_tracks = sum(result["tracks"] for result in results)
    total_points = sum(result["points"] for result in results)
    LOGGER.info("Replay cache complete: %s days, %s tracks, %s points", len(results), total_tracks, total_points)


if __name__ == "__main__":
    main()

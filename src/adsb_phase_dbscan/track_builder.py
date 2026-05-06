from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def split_tracks_by_time_gap(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df.assign(track_id=pd.Series(dtype=str), flight_id=pd.Series(dtype=str))
    max_gap = pd.Timedelta(minutes=config["data"]["max_time_gap_minutes"])
    rows = []
    for hex_id, group in df.sort_values(["hex", "timestamp"]).groupby("hex", sort=False):
        group = group.copy()
        gaps = group["timestamp"].diff().fillna(pd.Timedelta(seconds=0))
        flight_num = (gaps > max_gap).cumsum()
        group["flight_id"] = [f"{hex_id}-{int(n)}" for n in flight_num]
        group["track_id"] = group["flight_id"]
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_tracks(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if not df.empty and "flight_id" in df.columns and df["flight_id"].astype(str).str.len().gt(0).all():
        out = df.copy()
        if "track_id" not in out.columns:
            out["track_id"] = out["flight_id"].astype(str)
        elif not out["track_id"].astype(str).str.len().gt(0).all():
            out["track_id"] = out["flight_id"].astype(str)
        return out
    return split_tracks_by_time_gap(df, config)


def _segment_id(flight_id: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    digest = hashlib.sha1(f"{flight_id}-{start}-{end}".encode("utf-8")).hexdigest()[:12]
    return f"{flight_id}-{digest}"


def make_segments(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if "flight_id" not in df.columns:
        df = build_tracks(df, config)
    segment_seconds = int(config["data"]["segment_seconds"])
    step_seconds = int(config["data"]["segment_step_seconds"])
    min_points = int(config["data"]["min_points_per_segment"])
    rows = []
    for flight_id, group in df.sort_values("timestamp").groupby("flight_id", sort=False):
        group = group.reset_index(drop=True)
        if len(group) < min_points:
            continue
        times = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        start_time = pd.Timestamp(times[0])
        last_time = pd.Timestamp(times[-1])
        window_ns = np.timedelta64(segment_seconds, "s")
        step_ns = np.timedelta64(step_seconds, "s")
        cursor = times[0]
        final_cursor = times[-1] - window_ns + np.timedelta64(1000, "ns")
        while cursor <= final_cursor:
            end_time = cursor + window_ns
            start_idx = int(np.searchsorted(times, cursor, side="left"))
            end_idx = int(np.searchsorted(times, end_time, side="right"))
            if end_idx - start_idx >= min_points:
                segment = group.iloc[start_idx:end_idx]
                segment_start = pd.Timestamp(segment["timestamp"].iloc[0])
                segment_end = pd.Timestamp(segment["timestamp"].iloc[-1])
                cursor_ts = pd.Timestamp(cursor)
                end_ts = pd.Timestamp(end_time)
                records = segment.to_dict("records")
                rows.append(
                    {
                        "segment_id": _segment_id(str(flight_id), cursor_ts, end_ts),
                        "hex": segment["hex"].iloc[0],
                        "flight": segment["flight"].dropna().iloc[0] if segment["flight"].notna().any() else "",
                        "flight_id": flight_id,
                        "start_time": segment_start,
                        "end_time": segment_end,
                        "points": records,
                        "n_points": len(segment),
                    }
                )
            cursor = cursor + step_ns
    return pd.DataFrame(rows)

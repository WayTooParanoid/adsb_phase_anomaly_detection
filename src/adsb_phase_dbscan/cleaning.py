from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "timestamp",
    "hex",
    "flight",
    "lat",
    "lon",
    "altitude_ft",
    "gs",
    "track",
    "vertical_rate_fpm",
    "seen",
    "seen_pos",
    "type",
    "r",
    "t",
    "flight_id",
    "track_id",
]


def clean_numeric(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def clean_altitude(value: Any) -> float:
    if isinstance(value, str) and value.strip().lower() == "ground":
        return 0.0
    return clean_numeric(value)


def _first_present(df: pd.DataFrame, names: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for name in names:
        if name in df.columns:
            out = out.where(out.notna(), df[name])
    return out


def clean_hex(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    return "" if text in {"", "nan", "none", "null"} else text


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def parse_timestamp(value: Any) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    if isinstance(value, np.datetime64):
        return pd.to_datetime(value, utc=True, errors="coerce")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return pd.NaT
        numeric = pd.to_numeric(text, errors="coerce")
        if pd.notna(numeric):
            return parse_timestamp(float(numeric))
        return pd.to_datetime(text, errors="coerce", utc=True)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        numeric = float(value)
        if not np.isfinite(numeric):
            return pd.NaT
        magnitude = abs(numeric)
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        else:
            unit = "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(value, errors="coerce", utc=True)


def normalize_adsb_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = _first_present(df, ["timestamp", "now", "time"]).map(parse_timestamp)

    for col in ["hex", "flight", "type", "r", "t"]:
        out[col] = df[col] if col in df.columns else ""
    out["hex"] = out["hex"].map(clean_hex)
    for col in ["flight", "type", "r", "t"]:
        out[col] = out[col].map(clean_text)
    for col in ["flight_id", "track_id"]:
        out[col] = df[col].map(clean_text) if col in df.columns else ""

    out["lat"] = _first_present(df, ["lat", "latitude"]).map(clean_numeric)
    out["lon"] = _first_present(df, ["lon", "longitude"]).map(clean_numeric)
    out["altitude_ft"] = _first_present(df, ["alt_baro", "alt_geom", "altitude_ft", "altitude"]).map(clean_altitude)
    out["gs"] = _first_present(df, ["gs", "ground_speed", "speed"]).map(clean_numeric)
    out["track"] = _first_present(df, ["track", "heading"]).map(clean_numeric)
    out["vertical_rate_fpm"] = _first_present(df, ["baro_rate", "geom_rate", "vertical_rate_fpm"]).map(clean_numeric)
    out["seen"] = _first_present(df, ["seen"]).map(clean_numeric)
    out["seen_pos"] = _first_present(df, ["seen_pos"]).map(clean_numeric)

    return out[CANONICAL_COLUMNS]


def filter_valid_adsb_points(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    max_seen_pos = config["data"]["stale_seen_pos_seconds"]
    valid = (
        df["lat"].between(-90, 90)
        & df["lon"].between(-180, 180)
        & df["hex"].astype(bool)
        & df["timestamp"].notna()
        & (df["seen_pos"].isna() | (df["seen_pos"] <= max_seen_pos))
        & (df["gs"].isna() | df["gs"].between(0, 900))
        & (df["altitude_ft"].isna() | df["altitude_ft"].between(0, 70000))
    )
    return df.loc[valid].sort_values(["hex", "timestamp"]).reset_index(drop=True)

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EARTH_RADIUS_NM = 3440.065
EARTH_RADIUS_M = 6371000.0


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_NM * math.asin(math.sqrt(min(1.0, a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def local_xy_m(lat: float, lon: float, center_lat: float, center_lon: float) -> tuple[float, float]:
    x = math.radians(lon - center_lon) * EARTH_RADIUS_M * math.cos(math.radians(center_lat))
    y = math.radians(lat - center_lat) * EARTH_RADIUS_M
    return x, y


def angular_difference_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circular_mean_deg(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return float("nan")
    radians = np.deg2rad(vals)
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    if abs(sin_mean) < 1e-12 and abs(cos_mean) < 1e-12:
        return float("nan")
    return (math.degrees(math.atan2(sin_mean, cos_mean)) + 360.0) % 360.0


def heading_change_deg(sequence: Sequence[float]) -> float:
    vals = [v for v in sequence if v is not None and not np.isnan(v)]
    if len(vals) < 2:
        return 0.0
    return float(sum(angular_difference_deg(a, b) for a, b in zip(vals[:-1], vals[1:])))

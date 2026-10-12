"""Travel-time catchments (isochrones).

With a free OpenRouteService key we get a *real* walkable area that follows the
street network. Without a key (or if the API fails) we fall back to a clearly
labelled approximate circle so the app never breaks.
"""
from __future__ import annotations

import numpy as np
import requests

from .config import ALL_CATEGORIES
from .scoring import LocationData, to_latlon, to_xy

ORS_URL = "https://api.openrouteservice.org/v2/isochrones/{profile}"

PROFILES = {
    "Walking": "foot-walking",
    "Cycling": "cycling-regular",
    "Driving": "driving-car",
}
# Rough average city speeds (km/h) and a "streets aren't straight" factor,
# used only for the fallback circle.
_APPROX = {"foot-walking": (4.5, 0.75), "cycling-regular": (13.0, 0.75), "driving-car": (18.0, 0.7)}


class IsochroneError(RuntimeError):
    pass


def approx_isochrone(lat: float, lon: float, minutes: int,
                     profile: str = "foot-walking", n: int = 48) -> list[list[float]]:
    """Fallback: a circle whose radius is typical travel distance for that time."""
    speed, circuity = _APPROX.get(profile, _APPROX["foot-walking"])
    radius_m = speed * 1000 / 60 * minutes * circuity
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    la, lo = to_latlon(radius_m * np.cos(theta), radius_m * np.sin(theta), (lat, lon))
    ring = [[float(a), float(b)] for a, b in zip(lo, la)]  # [lon, lat]
    return ring + [ring[0]]


def fetch_ors(lat: float, lon: float, minutes: int, api_key: str,
              profile: str = "foot-walking") -> list[list[float]]:
    """Ask OpenRouteService for the area reachable within `minutes`."""
    try:
        resp = requests.post(
            ORS_URL.format(profile=profile),
            json={"locations": [[lon, lat]], "range": [int(minutes) * 60], "range_type": "time"},
            headers={
                "Authorization": api_key,
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/geo+json",
            },
            timeout=20,
        )
    except requests.RequestException as err:
        raise IsochroneError(f"could not reach OpenRouteService ({err})") from err
    if resp.status_code in (401, 403):
        raise IsochroneError("OpenRouteService rejected the API key")
    if resp.status_code == 429:
        raise IsochroneError("OpenRouteService rate limit reached, try again shortly")
    if resp.status_code >= 400:
        raise IsochroneError(f"OpenRouteService error {resp.status_code}")
    try:
        geom = resp.json()["features"][0]["geometry"]
        coords = geom["coordinates"]
        ring = coords[0] if geom["type"] == "Polygon" else coords[0][0]
        return [[float(x), float(y)] for x, y in ring]
    except (KeyError, IndexError, TypeError, ValueError) as err:
        raise IsochroneError("unexpected response from OpenRouteService") from err


def points_in_polygon(points_xy: np.ndarray, poly_xy: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon test (polygon in the same units)."""
    n = len(points_xy)
    if n == 0 or len(poly_xy) < 3:
        return np.zeros(n, dtype=bool)
    x, y = points_xy[:, 0], points_xy[:, 1]
    inside = np.zeros(n, dtype=bool)
    j = len(poly_xy) - 1
    for i in range(len(poly_xy)):
        xi, yi = poly_xy[i]
        xj, yj = poly_xy[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_at)
        j = i
    return inside


def polygon_area_km2(poly_xy: np.ndarray) -> float:
    x, y = poly_xy[:, 0], poly_xy[:, 1]
    return abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 1e6


def isochrone_report(data: LocationData, lat: float, lon: float, minutes: int,
                     api_key: str | None = None, mode: str = "Walking") -> dict:
    """Polygon + how many places of each kind lie inside it."""
    profile = PROFILES.get(mode, "foot-walking")
    ring, source, note = None, "approx", ""
    if api_key:
        try:
            ring = fetch_ors(lat, lon, minutes, api_key, profile)
            source = "ors"
        except IsochroneError as err:
            note = f"{err}; showing an approximate circle instead."
    if ring is None:
        ring = approx_isochrone(lat, lon, minutes, profile)
        if not api_key:
            note = "No OpenRouteService key, showing an approximate circle."

    poly_xy = to_xy([p[1] for p in ring], [p[0] for p in ring], data.center)
    counts = {
        cat: int(points_in_polygon(data.poi_xy[cat], poly_xy).sum()) if cat in data.poi_xy else 0
        for cat in ALL_CATEGORIES
    }
    return {
        "ring": ring, "source": source, "note": note, "counts": counts,
        "area_km2": round(polygon_area_km2(poly_xy), 2),
        "minutes": int(minutes), "mode": mode,
    }

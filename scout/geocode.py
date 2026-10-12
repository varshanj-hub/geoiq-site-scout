"""Free place search with OpenStreetMap Nominatim (no key needed).

Nominatim's usage policy: max 1 request/second and a descriptive User-Agent.
"""
from __future__ import annotations

import time

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "geoiq-site-scout/1.0 (learning project)"
_last_call = 0.0


class GeocodeError(RuntimeError):
    pass


# left,top,right,bottom of Greater Bengaluru, used to keep searches inside the city.
BENGALURU_VIEWBOX = "77.35,13.25,77.85,12.70"


def geocode(query: str, limit: int = 5, viewbox: str | None = None) -> list[dict]:
    """Return up to `limit` matches as [{'name', 'lat', 'lon'}].

    With `viewbox` ("left,top,right,bottom") results are restricted to that box.
    """
    global _last_call
    q = (query or "").strip()
    if len(q) < 3:
        return []
    wait = 1.1 - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    try:
        params = {"q": q, "format": "jsonv2", "limit": limit}
        if viewbox:
            params.update({"viewbox": viewbox, "bounded": 1, "countrycodes": "in"})
        resp = requests.get(
            NOMINATIM_URL,
            params=params,
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        _last_call = time.time()
        resp.raise_for_status()
        rows = resp.json()
    except Exception as err:
        raise GeocodeError(f"Place search failed: {err}") from err

    results = []
    for r in rows:
        try:
            name = str(r.get("display_name", q))
            results.append({
                "name": name if len(name) <= 90 else name[:87] + "...",
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return results

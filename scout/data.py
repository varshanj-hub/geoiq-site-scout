"""Fetching and caching points of interest (POIs) from OpenStreetMap.

Everything is free: the public Overpass API needs no key. If it is unreachable
(or you are offline) the app falls back to a built-in synthetic sample so it
always runs.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import ALL_CATEGORIES

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
USER_AGENT = "geoiq-site-scout/1.0 (learning project)"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

POI_COLUMNS = ["lat", "lon", "category", "name"]


# --------------------------------------------------------------------------
# Tag classification
# --------------------------------------------------------------------------
def classify(tags: dict) -> list[str]:
    """Map OpenStreetMap tags to our categories (an element can have several)."""
    cats: list[str] = []
    amenity = tags.get("amenity")
    shop = tags.get("shop")
    building = tags.get("building")

    if amenity == "cafe":
        cats.append("cafe")
    if amenity == "pharmacy" or shop == "chemist":
        cats.append("pharmacy")
    if amenity in ("restaurant", "fast_food"):
        cats.append("restaurant")
    if tags.get("leisure") == "fitness_centre" or amenity == "gym":
        cats.append("gym")
    if amenity in ("college", "university"):
        cats.append("college")
    if amenity in ("clinic", "hospital", "doctors", "dentist"):
        cats.append("healthcare")
    if "office" in tags or building == "office":
        cats.append("office")
    if building in ("apartments", "residential", "dormitory"):
        cats.append("apartments")
    if shop and shop != "chemist":
        cats.append("shops")
    if tags.get("highway") == "bus_stop":
        cats.append("bus_stop")
    if tags.get("station") == "subway" or tags.get("railway") == "subway_entrance":
        cats.append("metro")
    return cats


def parse_overpass(payload: dict) -> pd.DataFrame:
    """Turn an Overpass JSON response into a tidy POI table (one row per category)."""
    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        if "lat" in el and "lon" in el:
            lat, lon = el["lat"], el["lon"]
        elif "center" in el:
            lat, lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue
        for cat in classify(tags):
            rows.append((float(lat), float(lon), cat, tags.get("name", "")))
    df = pd.DataFrame(rows, columns=POI_COLUMNS)
    return df.drop_duplicates(subset=["lat", "lon", "category"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Live download
# --------------------------------------------------------------------------
def build_query(south: float, west: float, north: float, east: float) -> str:
    bbox = f"({south},{west},{north},{east})"
    return f"""
[out:json][timeout:120];
(
  nwr["amenity"~"^(cafe|pharmacy|restaurant|fast_food|gym|college|university|clinic|hospital|doctors|dentist)$"]{bbox};
  nwr["leisure"="fitness_centre"]{bbox};
  nwr["office"]{bbox};
  nwr["building"~"^(office|apartments|residential|dormitory)$"]{bbox};
  nwr["shop"]{bbox};
  nwr["highway"="bus_stop"]{bbox};
  nwr["station"="subway"]{bbox};
  nwr["railway"="subway_entrance"]{bbox};
);
out center tags;
""".strip()


def bbox_for(center: tuple[float, float], radius_m: float, pad: float = 1.15):
    lat0, lon0 = center
    dlat = radius_m * pad / 110574.0
    dlon = radius_m * pad / (111320.0 * np.cos(np.radians(lat0)))
    return lat0 - dlat, lon0 - dlon, lat0 + dlat, lon0 + dlon


def fetch_live(center: tuple[float, float], radius_m: float) -> pd.DataFrame:
    """Download POIs around `center` from the public Overpass API."""
    query = build_query(*bbox_for(center, radius_m))
    last_err: Exception | None = None
    for url in OVERPASS_ENDPOINTS:
        try:
            resp = requests.post(
                url,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
                timeout=(10, 120),
            )
            resp.raise_for_status()
            return parse_overpass(resp.json())
        except Exception as err:  # try the next mirror
            last_err = err
    raise RuntimeError(f"All Overpass endpoints failed: {last_err}")


# --------------------------------------------------------------------------
# Synthetic sample data (offline fallback / demo)
# --------------------------------------------------------------------------
_SAMPLE_COUNTS = {
    "office": 320, "college": 14, "metro": 3, "apartments": 520, "shops": 650,
    "bus_stop": 80, "healthcare": 70, "cafe": 90, "pharmacy": 45, "gym": 30,
    "restaurant": 230,
}


def make_sample(center: tuple[float, float], radius_m: float, seed: int = 7) -> pd.DataFrame:
    """Generate a plausible clustered set of POIs so the app works offline."""
    rng = np.random.default_rng(seed)
    lat0, lon0 = center
    r = radius_m * 0.9
    hubs = rng.uniform(-0.55 * r, 0.55 * r, size=(4, 2))  # commercial hubs (x, y in metres)
    rows = []
    for cat, n in _SAMPLE_COUNTS.items():
        if cat in ("office", "shops", "restaurant", "cafe", "metro", "college"):
            which = rng.integers(0, len(hubs), size=n)
            spread = 0.22 * r if cat != "metro" else 0.1 * r
            xy = hubs[which] + rng.normal(0, spread, size=(n, 2))
        else:
            xy = rng.normal(0, 0.45 * r, size=(n, 2))
        dist = np.hypot(xy[:, 0], xy[:, 1])
        xy = xy[dist <= radius_m]
        lat = lat0 + xy[:, 1] / 110574.0
        lon = lon0 + xy[:, 0] / (111320.0 * np.cos(np.radians(lat0)))
        for la, lo in zip(lat, lon):
            rows.append((float(la), float(lo), cat, ""))
    return pd.DataFrame(rows, columns=POI_COLUMNS)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def get_pois(
    area: str,
    center: tuple[float, float],
    radius_m: float,
    use_live: bool = True,
    force_refresh: bool = False,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """Return (pois, source, message). source is 'cache', 'live' or 'sample'."""
    cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{_slug(area)}_{int(radius_m)}.csv"

    if use_live:
        if cache_file.exists() and not force_refresh:
            df = pd.read_csv(cache_file, keep_default_na=False)
            df["name"] = df["name"].astype(str)
            return df, "cache", f"Loaded {len(df):,} places from local cache."
        try:
            df = fetch_live(center, radius_m)
            if len(df) < 30:
                raise RuntimeError(f"only {len(df)} places came back")
            df.to_csv(cache_file, index=False)
            return df, "live", f"Downloaded {len(df):,} places from OpenStreetMap."
        except Exception as err:
            msg = (f"Could not download live data ({err}). "
                   "Showing built-in SAMPLE data instead. Check your internet and "
                   "press 'Re-download'.")
            return make_sample(center, radius_m), "sample", msg

    return make_sample(center, radius_m), "sample", "Using built-in SAMPLE data (offline demo)."


def validate_categories(df: pd.DataFrame) -> list[str]:
    """Categories that have zero places (useful to warn about sparse OSM coverage)."""
    present = set(df["category"].unique())
    return [c for c in ALL_CATEGORIES if c not in present]

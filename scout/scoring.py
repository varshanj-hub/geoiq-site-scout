"""Grid building, catchment counting and the site-suitability score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BUSINESS_WEIGHTS, CATEGORY_LABELS, DEFAULT_RADIUS_M, DEMAND_CATEGORIES,
)

M_PER_DEG_LAT = 110574.0
M_PER_DEG_LON_EQ = 111320.0


# --------------------------------------------------------------------------
# Geometry helpers (flat-earth approximation: accurate to <1% at city scale)
# --------------------------------------------------------------------------
def to_xy(lat, lon, center: tuple[float, float]) -> np.ndarray:
    lat0, lon0 = center
    x = (np.asarray(lon, dtype=float) - lon0) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
    y = (np.asarray(lat, dtype=float) - lat0) * M_PER_DEG_LAT
    return np.column_stack([x, y])


def to_latlon(x, y, center: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lat0, lon0 = center
    lat = lat0 + np.asarray(y, dtype=float) / M_PER_DEG_LAT
    lon = lon0 + np.asarray(x, dtype=float) / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat0)))
    return lat, lon


def build_grid(center: tuple[float, float], radius_m: float, cell_m: float) -> pd.DataFrame:
    """Square grid of candidate locations covering a circle around `center`."""
    n = int(np.ceil(radius_m / cell_m))
    offsets = np.arange(-n, n + 1) * cell_m
    xs, ys = np.meshgrid(offsets, offsets)
    xs, ys = xs.ravel(), ys.ravel()
    keep = xs ** 2 + ys ** 2 <= radius_m ** 2
    xs, ys = xs[keep], ys[keep]

    lat, lon = to_latlon(xs, ys, center)
    half = cell_m / 2.0
    corners = []
    for cx, cy in zip(xs, ys):
        cxs = [cx - half, cx + half, cx + half, cx - half]
        cys = [cy - half, cy - half, cy + half, cy + half]
        la, lo = to_latlon(cxs, cys, center)
        corners.append([[float(a), float(b)] for a, b in zip(lo, la)])  # [lon, lat]

    return pd.DataFrame({
        "cell_id": np.arange(len(xs)),
        "lat": lat, "lon": lon, "x": xs, "y": ys, "polygon": corners,
    })


def count_within(points_xy: np.ndarray, targets_xy: np.ndarray,
                 radius_m: float, chunk: int = 128) -> np.ndarray:
    """For each point, how many targets lie within `radius_m` metres."""
    n = len(points_xy)
    if n == 0 or len(targets_xy) == 0:
        return np.zeros(n, dtype=int)
    r2 = float(radius_m) ** 2
    tx, ty = targets_xy[:, 0], targets_xy[:, 1]
    out = np.zeros(n, dtype=int)
    for i in range(0, n, chunk):
        px = points_xy[i:i + chunk, 0][:, None]
        py = points_xy[i:i + chunk, 1][:, None]
        d2 = (px - tx[None, :]) ** 2 + (py - ty[None, :]) ** 2
        out[i:i + chunk] = (d2 <= r2).sum(axis=1)
    return out


# --------------------------------------------------------------------------
# Location data container
# --------------------------------------------------------------------------
class LocationData:
    """Grid of candidate cells + POIs, with cached 'how many X within R m' lookups."""

    def __init__(self, center: tuple[float, float], radius_m: float,
                 cell_m: float, pois: pd.DataFrame):
        self.center = center
        self.radius_m = radius_m
        self.cell_m = cell_m
        self.pois = pois
        self.cells = build_grid(center, radius_m, cell_m)
        self.cell_xy = self.cells[["x", "y"]].to_numpy()
        self.poi_xy: dict[str, np.ndarray] = {
            cat: to_xy(g["lat"].to_numpy(), g["lon"].to_numpy(), center)
            for cat, g in pois.groupby("category")
        }
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def count(self, category: str, radius_m: float) -> np.ndarray:
        key = (category, int(radius_m))
        if key not in self._cache:
            self._cache[key] = count_within(
                self.cell_xy, self.poi_xy.get(category, np.empty((0, 2))), radius_m
            )
        return self._cache[key]

    def poi_latlon(self, category: str) -> pd.DataFrame:
        return self.pois[self.pois["category"] == category][["lat", "lon"]]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _normalise(counts: np.ndarray) -> np.ndarray:
    """Scale counts to 0-1 using the 95th percentile so outliers don't dominate."""
    counts = counts.astype(float)
    if counts.max() <= 0:
        return np.zeros_like(counts)
    ref = np.quantile(counts, 0.95)
    if ref <= 0:
        ref = counts.max()
    return np.clip(counts / ref, 0.0, 1.0)


def score_cells(
    data: LocationData,
    business_type: str,
    comp_weight: float = 0.8,
    radius_m: float = DEFAULT_RADIUS_M,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Score every cell 0-100 for `business_type` (higher = better place to open).

    score ~ weighted demand drivers nearby  -  comp_weight * competitors nearby
    """
    weights = weights or BUSINESS_WEIGHTS[business_type]
    total_w = sum(weights.values()) or 1.0

    df = data.cells.copy()
    demand = np.zeros(len(df))
    contrib: dict[str, np.ndarray] = {}
    for cat in DEMAND_CATEGORIES:
        counts = data.count(cat, radius_m)
        df[f"n_{cat}"] = counts
        w = weights.get(cat, 0.0)
        c = w * _normalise(counts) / total_w
        contrib[cat] = c
        demand += c

    comp_counts = data.count(business_type, radius_m)
    df["competitors"] = comp_counts
    comp_norm = _normalise(comp_counts)

    raw = demand - comp_weight * comp_norm
    span = raw.max() - raw.min()
    score = np.full(len(df), 50.0) if span <= 1e-12 else 100 * (raw - raw.min()) / span

    df["demand"] = np.round(100 * demand / max(demand.max(), 1e-12), 1)
    df["score"] = np.round(score, 1)
    df["rank"] = df["score"].rank(method="first", ascending=False).astype(int)

    # Human-readable explanation of each cell's score.
    biz = CATEGORY_LABELS[business_type]
    why = []
    for i in range(len(df)):
        ranked = sorted(contrib, key=lambda k: contrib[k][i], reverse=True)
        parts = [f"{int(df[f'n_{k}'].iloc[i])} {CATEGORY_LABELS[k]}"
                 for k in ranked[:2] if contrib[k][i] > 0 and df[f"n_{k}"].iloc[i] > 0]
        drivers = ", ".join(parts) if parts else "little demand nearby"
        why.append(f"{drivers} · {int(comp_counts[i])} {biz} within {int(radius_m)} m")
    df["why"] = why
    return df.sort_values("cell_id").reset_index(drop=True)

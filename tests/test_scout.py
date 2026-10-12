"""Run with:  python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout import data as data_mod
from scout.config import AREAS, ALL_CATEGORIES, BUSINESS_WEIGHTS
from scout.nlquery import apply_query, parse_question
from scout.scoring import LocationData, build_grid, count_within, score_cells, to_latlon, to_xy
from scout.viz import build_deck, score_to_rgb

CENTER = AREAS["Indiranagar"]


@pytest.fixture(scope="module")
def loc():
    pois = data_mod.make_sample(CENTER, 2000)
    return LocationData(CENTER, 2000, 250, pois)


# ---------------- data ----------------
def test_classify_tags():
    assert data_mod.classify({"amenity": "cafe"}) == ["cafe"]
    assert "pharmacy" in data_mod.classify({"shop": "chemist"})
    assert "shops" not in data_mod.classify({"shop": "chemist"})
    assert set(data_mod.classify({"building": "office", "office": "it"})) == {"office"}
    assert data_mod.classify({"station": "subway"}) == ["metro"]
    assert data_mod.classify({"highway": "bus_stop"}) == ["bus_stop"]
    assert data_mod.classify({"amenity": "bench"}) == []


def test_parse_overpass_handles_nodes_ways_and_dupes():
    payload = {"elements": [
        {"type": "node", "lat": 12.97, "lon": 77.64, "tags": {"amenity": "cafe", "name": "A"}},
        {"type": "node", "lat": 12.97, "lon": 77.64, "tags": {"amenity": "cafe", "name": "A"}},
        {"type": "way", "center": {"lat": 12.98, "lon": 77.65}, "tags": {"building": "apartments"}},
        {"type": "way", "tags": {"amenity": "cafe"}},  # no coordinates -> skipped
        {"type": "node", "lat": 1, "lon": 1},          # no tags -> skipped
    ]}
    df = data_mod.parse_overpass(payload)
    assert len(df) == 2
    assert set(df["category"]) == {"cafe", "apartments"}


def test_fetch_live_with_mock_and_fallback(monkeypatch, tmp_path):
    payload = {"elements": [
        {"type": "node", "lat": 12.97 + i * 1e-4, "lon": 77.64, "tags": {"amenity": "cafe"}}
        for i in range(60)
    ]}

    class Resp:
        def raise_for_status(self): pass
        def json(self): return payload

    monkeypatch.setattr(data_mod.requests, "post", lambda *a, **k: Resp())
    df, source, _ = data_mod.get_pois("Test Area", CENTER, 2000, cache_dir=tmp_path)
    assert source == "live" and len(df) == 60
    df2, source2, _ = data_mod.get_pois("Test Area", CENTER, 2000, cache_dir=tmp_path)
    assert source2 == "cache" and len(df2) == 60

    def boom(*a, **k): raise ConnectionError("offline")
    monkeypatch.setattr(data_mod.requests, "post", boom)
    df3, source3, msg = data_mod.get_pois("Other Area", CENTER, 2000, cache_dir=tmp_path)
    assert source3 == "sample" and "SAMPLE" in msg and len(df3) > 100


def test_sample_has_every_category():
    df = data_mod.make_sample(CENTER, 2000)
    assert set(ALL_CATEGORIES) <= set(df["category"])


# ---------------- geometry / scoring ----------------
def test_xy_roundtrip():
    xy = to_xy([CENTER[0] + 0.001], [CENTER[1] + 0.001], CENTER)
    lat, lon = to_latlon(xy[:, 0], xy[:, 1], CENTER)
    assert lat[0] == pytest.approx(CENTER[0] + 0.001)
    assert lon[0] == pytest.approx(CENTER[1] + 0.001)


def test_grid_inside_circle_and_polygons():
    g = build_grid(CENTER, 1000, 250)
    assert (np.hypot(g["x"], g["y"]) <= 1000 + 1e-6).all()
    assert all(len(p) == 4 for p in g["polygon"])


def test_count_within_bruteforce():
    rng = np.random.default_rng(0)
    pts, tgt = rng.uniform(-1000, 1000, (50, 2)), rng.uniform(-1000, 1000, (300, 2))
    expected = np.array([(np.hypot(*(tgt - p).T) <= 400).sum() for p in pts])
    assert (count_within(pts, tgt, 400, chunk=7) == expected).all()
    assert (count_within(pts, np.empty((0, 2)), 400) == 0).all()


@pytest.mark.parametrize("biz", list(BUSINESS_WEIGHTS))
def test_scores_in_range_for_every_business(loc, biz):
    s = score_cells(loc, biz)
    assert len(s) == len(loc.cells)
    assert s["score"].between(0, 100).all()
    assert s["score"].max() == pytest.approx(100.0)
    assert s["rank"].min() == 1 and s["rank"].is_unique
    assert s["why"].str.len().gt(0).all()


def test_competition_penalty_lowers_crowded_cells(loc):
    no_pen = score_cells(loc, "cafe", comp_weight=0.0)
    pen = score_cells(loc, "cafe", comp_weight=1.5)
    crowded = loc.count("cafe", 500) >= np.quantile(loc.count("cafe", 500), 0.9)
    assert (pen.loc[crowded, "score"].mean() - no_pen.loc[crowded, "score"].mean()) < 0 or \
        pen.loc[crowded, "rank"].mean() > no_pen.loc[crowded, "rank"].mean()


def test_scoring_with_empty_categories():
    pois = data_mod.make_sample(CENTER, 1500)
    pois = pois[pois["category"].isin(["office", "cafe"])]
    d = LocationData(CENTER, 1500, 250, pois)
    s = score_cells(d, "pharmacy")  # pharmacy + its demand drivers all absent
    assert s["score"].between(0, 100).all()


# ---------------- natural-language questions ----------------
def test_parse_examples():
    q = parse_question("lots of offices nearby and no cafe within 300 m")
    assert [(f.category, f.kind, f.radius_m) for f in q.filters] == [
        ("office", "high", 500), ("cafe", "none", 300)]

    q = parse_question("at least 3 apartment buildings within 0.4 km, low competition, top 5")
    assert q.top_n == 5
    assert (q.filters[0].category, q.filters[0].kind, q.filters[0].value, q.filters[0].radius_m) == \
        ("apartments", "atleast", 3, 400)
    assert (q.filters[1].category, q.filters[1].kind) == ("competitor", "atmost")

    q = parse_question("more than 5 shops within 200 meters")
    assert (q.filters[0].kind, q.filters[0].value, q.filters[0].radius_m) == ("atleast", 6, 200)

    q = parse_question("near a metro station and no competitors")
    assert [(f.category, f.kind) for f in q.filters] == [("metro", "atleast"), ("competitor", "none")]

    q = parse_question("spots with lots of students but no gym within 1 km")
    assert [(f.category, f.kind, f.radius_m) for f in q.filters] == [
        ("college", "high", 500), ("gym", "none", 1000)]


def test_parse_garbage_and_empty():
    assert parse_question("").filters == []
    q = parse_question("asdf qwerty")
    assert q.filters == [] and q.notes
    q = parse_question("lots of unicorns nearby")
    assert q.filters == [] and q.notes


def test_apply_query_filters_correctly(loc):
    scored = score_cells(loc, "cafe")
    q = parse_question("no cafe within 300 m and at least 2 offices within 300 m, top 5")
    top, mask = apply_query(q, loc, scored, "cafe")
    assert len(top) <= 5
    assert (loc.count("cafe", 300)[mask] == 0).all()
    assert (loc.count("office", 300)[mask] >= 2).all()
    assert top["score"].is_monotonic_decreasing


def test_apply_query_no_matches(loc):
    scored = score_cells(loc, "cafe")
    q = parse_question("at least 999999 offices")
    top, mask = apply_query(q, loc, scored, "cafe")
    assert top.empty and not mask.any()


# ---------------- map ----------------
def test_colors():
    assert score_to_rgb(0)[0] > score_to_rgb(0)[1]
    assert score_to_rgb(100)[1] > score_to_rgb(100)[0]


def test_deck_serialises_to_json(loc):
    scored = score_cells(loc, "cafe")
    top = scored.sort_values("score", ascending=False).head(5)
    mask = np.zeros(len(scored), bool); mask[:10] = True
    deck = build_deck(scored, CENTER, competitors=loc.poi_latlon("cafe"),
                      highlight_mask=mask, top_cells=top)
    parsed = json.loads(deck.to_json())
    assert len(parsed["layers"]) == 4


def test_deck_with_isochrone_and_selection(loc):
    from scout.isochrone import approx_isochrone
    scored = score_cells(loc, "gym")
    ring = approx_isochrone(CENTER[0], CENTER[1], 10)
    deck = build_deck(scored, CENTER, competitors=None, iso_ring=ring, selected=CENTER,
                      top_cells=scored.head(3))
    layers = json.loads(deck.to_json())["layers"]
    assert len(layers) == 5  # cells, isochrone, selection ring, top dots, labels

"""Tests for env loading, geocoding, isochrones and Gemini helpers (all network mocked)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout import data as data_mod
from scout import geocode as geo_mod
from scout import isochrone as iso_mod
from scout import llm as llm_mod
from scout.config import AREAS
from scout.env import load_env
from scout.scoring import LocationData, to_xy

CENTER = AREAS["Indiranagar"]


class FakeResp:
    def __init__(self, payload=None, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        if self._p is None:
            raise ValueError("no json")
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def gemini_payload(text):
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


@pytest.fixture(scope="module")
def loc():
    return LocationData(CENTER, 2000, 250, data_mod.make_sample(CENTER, 2000))


# ---------------- env ----------------
def test_load_env(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# comment\nFOO_KEY="abc123"\nexport BAR_KEY=xyz\nEMPTY=\nEXISTING=new\n')
    monkeypatch.setenv("EXISTING", "keep")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.delenv("BAR_KEY", raising=False)
    loaded = load_env(f)
    assert os.environ["FOO_KEY"] == "abc123" and os.environ["BAR_KEY"] == "xyz"
    assert os.environ["EXISTING"] == "keep" and "EMPTY" not in os.environ
    assert set(loaded) == {"FOO_KEY", "BAR_KEY"}
    assert load_env(tmp_path / "missing.env") == []
    monkeypatch.delenv("FOO_KEY"); monkeypatch.delenv("BAR_KEY")


# ---------------- geocode ----------------
def test_geocode(monkeypatch):
    monkeypatch.setattr(geo_mod.time, "sleep", lambda s: None)
    rows = [{"display_name": "Indiranagar, Bengaluru", "lat": "12.97", "lon": "77.64"},
            {"display_name": "bad row"}]
    seen = {}
    def fake_get(url, params, headers, timeout):
        seen.update(params=params, headers=headers)
        return FakeResp(rows)
    monkeypatch.setattr(geo_mod.requests, "get", fake_get)
    out = geo_mod.geocode("Indiranagar Bengaluru")
    assert out == [{"name": "Indiranagar, Bengaluru", "lat": 12.97, "lon": 77.64}]
    assert "User-Agent" in seen["headers"]
    assert geo_mod.geocode("  ") == [] and geo_mod.geocode("ab") == []


def test_geocode_error(monkeypatch):
    monkeypatch.setattr(geo_mod.time, "sleep", lambda s: None)
    def boom(*a, **k): raise ConnectionError("offline")
    monkeypatch.setattr(geo_mod.requests, "get", boom)
    with pytest.raises(geo_mod.GeocodeError):
        geo_mod.geocode("Koramangala")


# ---------------- isochrone ----------------
def test_point_in_polygon_square():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    pts = np.array([[5, 5], [11, 5], [-1, 5], [5, 11], [9.9, 0.1], [0.1, 9.9]], float)
    assert iso_mod.points_in_polygon(pts, square).tolist() == [True, False, False, False, True, True]
    assert iso_mod.points_in_polygon(pts, square[:2]).sum() == 0  # degenerate polygon
    assert iso_mod.polygon_area_km2(square * 1000) == pytest.approx(100.0)


def test_point_in_polygon_matches_circle():
    rng = np.random.default_rng(1)
    pts = rng.uniform(-500, 500, (2000, 2))
    theta = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.column_stack([300 * np.cos(theta), 300 * np.sin(theta)])
    got = iso_mod.points_in_polygon(pts, circle)
    exp = np.hypot(pts[:, 0], pts[:, 1]) <= 300
    assert (got != exp).sum() <= 3  # only boundary rounding


def test_isochrone_fallback_without_key(loc):
    rep = iso_mod.isochrone_report(loc, CENTER[0], CENTER[1], 10, api_key=None, mode="Walking")
    assert rep["source"] == "approx" and "No OpenRouteService key" in rep["note"]
    assert rep["ring"][0] == rep["ring"][-1]
    assert set(rep["counts"]) >= {"office", "cafe", "metro"}
    # 4.5 km/h * 10 min * 0.75 = 562.5 m radius -> area = pi*r^2 ~ 0.99 km2
    assert rep["area_km2"] == pytest.approx(np.pi * 0.5625 ** 2, rel=0.02)
    manual = int(((to_xy(*[loc.pois[loc.pois.category == "office"][c].to_numpy() for c in ("lat", "lon")],
                         CENTER) ** 2).sum(1) <= 562.5 ** 2).sum())
    assert abs(rep["counts"]["office"] - manual) <= 3


def test_isochrone_ors_success_and_failures(loc, monkeypatch):
    lat, lon = CENTER
    ring = [[lon + dx, lat + dy] for dx, dy in
            [(-0.003, -0.003), (0.003, -0.003), (0.003, 0.003), (-0.003, 0.003), (-0.003, -0.003)]]
    seen = {}
    def ok(url, json, headers, timeout):
        seen.update(url=url, json=json, headers=headers)
        return FakeResp({"features": [{"geometry": {"type": "Polygon", "coordinates": [ring]}}]})
    monkeypatch.setattr(iso_mod.requests, "post", ok)
    rep = iso_mod.isochrone_report(loc, lat, lon, 5, api_key="SECRET", mode="Cycling")
    assert rep["source"] == "ors" and rep["note"] == ""
    assert seen["url"].endswith("cycling-regular") and seen["json"]["range"] == [300]
    assert seen["json"]["locations"] == [[lon, lat]]
    assert seen["headers"]["Authorization"] == "SECRET"

    for status, text in [(403, "rejected"), (429, "rate limit"), (500, "error 500")]:
        monkeypatch.setattr(iso_mod.requests, "post", lambda *a, s=status, **k: FakeResp({}, s))
        rep = iso_mod.isochrone_report(loc, lat, lon, 5, api_key="K")
        assert rep["source"] == "approx" and text in rep["note"]

    monkeypatch.setattr(iso_mod.requests, "post", lambda *a, **k: FakeResp({"oops": 1}))
    assert "unexpected" in iso_mod.isochrone_report(loc, lat, lon, 5, api_key="K")["note"]

    def boom(*a, **k): raise iso_mod.requests.ConnectionError("down")
    monkeypatch.setattr(iso_mod.requests, "post", boom)
    assert iso_mod.isochrone_report(loc, lat, lon, 5, api_key="K")["source"] == "approx"


# ---------------- gemini ----------------
GOOD = json.dumps({"filters": [
    {"category": "office", "kind": "high", "value": 0, "radius_m": 500},
    {"category": "cafe", "kind": "none", "value": 0, "radius_m": 300},
    {"category": "unicorn", "kind": "none", "value": 0, "radius_m": 300},      # invalid category
    {"category": "gym", "kind": "weird", "value": 0, "radius_m": 300},          # invalid kind
    {"category": "metro", "kind": "atleast", "value": 9999999, "radius_m": 99999},  # clamped
], "top_n": 500})


def test_parse_llm_validates_and_clamps(monkeypatch):
    seen = {}
    def fake(url, json, headers, timeout):
        seen.update(url=url, headers=headers, body=json)
        return FakeResp(gemini_payload("```json\n" + GOOD + "\n```"))
    monkeypatch.setattr(llm_mod.requests, "post", fake)
    q, engine = llm_mod.parse_question_smart("anything", "cafe", "MY_KEY")
    assert engine == "gemini"
    assert [(f.category, f.kind) for f in q.filters] == [
        ("office", "high"), ("cafe", "none"), ("metro", "atleast")]
    assert q.filters[2].value == 1000 and q.filters[2].radius_m == 3000 and q.top_n == 50
    assert len(q.notes) == 2
    assert "MY_KEY" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "MY_KEY"
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.parametrize("resp", [
    FakeResp({}, 429), FakeResp({"error": {"message": "API key not valid"}}, 400),
    FakeResp({}, 500), FakeResp({"candidates": []}), FakeResp(gemini_payload("not json")),
    FakeResp(gemini_payload("[1,2]")), FakeResp(gemini_payload("")),
])
def test_parse_llm_failures_fall_back_to_rules(monkeypatch, resp):
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: resp)
    q, engine = llm_mod.parse_question_smart("no cafe within 300 m", "cafe", "K")
    assert engine == "rules" and q.filters and any("Gemini unavailable" in n for n in q.notes)


def test_parse_llm_network_error_and_no_key(monkeypatch):
    def boom(*a, **k): raise llm_mod.requests.Timeout("slow")
    monkeypatch.setattr(llm_mod.requests, "post", boom)
    q, engine = llm_mod.parse_question_smart("no cafe within 300 m", "cafe", "K")
    assert engine == "rules" and q.filters
    # no key -> never touches the network
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: pytest.fail("network used"))
    q, engine = llm_mod.parse_question_smart("no cafe within 300 m", "cafe", None)
    assert engine == "rules" and q.filters


def test_gemini_empty_filters_uses_rules_if_possible(monkeypatch):
    monkeypatch.setattr(llm_mod.requests, "post",
                        lambda *a, **k: FakeResp(gemini_payload('{"filters": [], "top_n": 5}')))
    q, engine = llm_mod.parse_question_smart("no cafe within 300 m", "cafe", "K")
    assert engine == "rules" and q.filters
    q, engine = llm_mod.parse_question_smart("zzz", "cafe", "K")
    assert engine == "gemini" and q.filters == [] and q.top_n == 5


FACTS = {"business_type": "cafe", "area": "Indiranagar", "rank": 1, "n_cells": 50, "score": 91.0,
         "top_drivers": ["34 offices", "80 shops"], "competitors": 1, "catchment_m": 500,
         "isochrone": {"minutes": 10, "mode": "Walking", "area_km2": 0.7, "competitors": 2}}


def test_explain_spot(monkeypatch):
    s, engine = llm_mod.explain_spot(FACTS, None)
    assert engine == "template" and "ranks #1 of 50" in s and "34 offices" in s
    monkeypatch.setattr(llm_mod.requests, "post",
                        lambda *a, **k: FakeResp(gemini_payload("Great spot because of offices.")))
    s, engine = llm_mod.explain_spot(FACTS, "K")
    assert engine == "gemini" and s == "Great spot because of offices."
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: FakeResp({}, 429))
    s, engine = llm_mod.explain_spot(FACTS, "K")
    assert engine == "template" and "unavailable" in s


def test_thinking_config_only_for_flash(monkeypatch):
    bodies = []
    monkeypatch.setattr(llm_mod.requests, "post",
                        lambda url, json, headers, timeout: bodies.append(json) or FakeResp(gemini_payload("x")))
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    llm_mod.call_gemini("p", "s", "K")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-pro")
    llm_mod.call_gemini("p", "s", "K")
    assert "thinkingConfig" in bodies[0]["generationConfig"]
    assert "thinkingConfig" not in bodies[1]["generationConfig"]

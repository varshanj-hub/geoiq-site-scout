"""End-to-end UI tests using Streamlit's AppTest. All network calls are mocked."""
import json
import sys
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
APP = str(ROOT / "app.py")

CALLS = []


class Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def gemini_reply(text):
    return Resp({"candidates": [{"content": {"parts": [{"text": text}]}}]})


GEMINI_FILTERS = json.dumps({"filters": [
    {"category": "office", "kind": "atleast", "value": 1, "radius_m": 500}], "top_n": 4})

ORS_RING = [[77.6408 + dx, 12.9784 + dy] for dx, dy in
            [(-0.004, -0.004), (0.004, -0.004), (0.004, 0.004), (-0.004, 0.004), (-0.004, -0.004)]]


@pytest.fixture(autouse=True)
def mocked_network(monkeypatch):
    """No test touches the real internet; the dispatcher plays every external service."""
    import scout.env
    monkeypatch.setattr(scout.env, "load_env", lambda *a, **k: [])
    for var in ("GEMINI_API_KEY", "ORS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    CALLS.clear()

    def post(url, **kw):
        CALLS.append(("POST", url, kw))
        if "overpass" in url or "private.coffee" in url:
            raise requests.ConnectionError("offline")
        if "generativelanguage" in url:
            prompt = kw["json"]["contents"][0]["parts"][0]["text"]
            if kw["json"]["generationConfig"].get("responseMimeType") == "application/json":
                return gemini_reply(GEMINI_FILTERS)
            return gemini_reply("Strong office demand and modest competition make this a promising cafe site.")
        if "openrouteservice" in url:
            return Resp({"features": [{"geometry": {"type": "Polygon", "coordinates": [ORS_RING]}}]})
        raise AssertionError(f"unexpected POST {url}")

    def get(url, **kw):
        CALLS.append(("GET", url, kw))
        if "nominatim" in url:
            q = kw["params"]["q"]
            if "nowhere" in q:
                return Resp([])
            if "broken" in q:
                raise requests.ConnectionError("down")
            return Resp([{"display_name": "Phoenix Marketcity, Whitefield, Bengaluru",
                          "lat": "12.9957", "lon": "77.6966"},
                         {"display_name": "Second match", "lat": "12.99", "lon": "77.70"}])
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("scout.geocode.time.sleep", lambda s: None)


def start(use_live=False):
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    if not use_live:
        at.sidebar.checkbox(key="use_live").set_value(False).run()
        assert not at.exception, at.exception
    return at


def md(at):
    return " ".join(m.value for m in at.markdown)


def test_offline_fallback_when_live_download_fails():
    at = start(use_live=True)  # Overpass is mocked to fail -> sample data + warning
    assert any("SAMPLE" in w.value for w in at.warning)
    assert "SAMPLE data" in md(at)
    assert len(at.metric) >= 4 and len(at.dataframe) == 1
    assert "Rule-based Q&A" in md(at) and "Approx. catchment" in md(at)


def test_questions_without_keys_use_rules_and_never_call_gemini():
    at = start()
    [b for b in at.button if b.key == "example_0"][0].click().run()
    assert not at.exception, at.exception
    assert at.text_input(key="question").value.startswith("lots of offices")
    assert "rule-based parser" in md(at) and "Understood by" in md(at)
    at.text_input(key="question").set_value("blah blah").run()
    assert not at.exception and len(at.warning) >= 1
    at.text_input(key="question").set_value("at least 99999 offices").run()
    assert not at.exception and len(at.error) == 1
    assert not any("generativelanguage" in c[1] for c in CALLS)


def test_settings_changes_do_not_crash():
    at = start()
    at.sidebar.selectbox(key="business").select("pharmacy").run()
    at.sidebar.selectbox(key="area").select("Koramangala").run()
    at.sidebar.selectbox(key="iso_mode").select("Driving").run()
    at.sidebar.select_slider(key="iso_minutes").set_value(15).run()
    at.sidebar.slider(key="comp_weight").set_value(1.5).run()
    at.sidebar.checkbox(key="show_iso").set_value(False).run()
    assert not at.exception, at.exception


def test_gemini_key_parses_questions_and_writes_summary():
    at = start()
    at.sidebar.text_input(key="gem_key").set_value("FAKE-GEMINI-KEY").run()
    assert "Gemini" in md(at)
    at.text_input(key="question").set_value("somewhere with plenty of workplaces").run()
    assert not at.exception, at.exception
    assert "Understood by" in md(at) and "Gemini" in md(at)
    gem_calls = [c for c in CALLS if "generativelanguage" in c[1]]
    assert len(gem_calls) == 1
    assert "FAKE-GEMINI-KEY" not in gem_calls[0][1]
    assert gem_calls[0][2]["headers"]["x-goog-api-key"] == "FAKE-GEMINI-KEY"

    # rerunning the page (e.g. changing a slider) must not call Gemini again for the same question
    at.sidebar.slider(key="comp_weight").set_value(0.5).run()
    assert len([c for c in CALLS if "generativelanguage" in c[1]]) == 1

    [b for b in at.button if b.key == "explain"][0].click().run()
    assert not at.exception, at.exception
    assert any("promising cafe site" in i.value for i in at.info)


def test_gemini_failure_falls_back_gracefully(monkeypatch):
    def failing_post(url, **kw):
        if "generativelanguage" in url:
            return Resp({}, 429)
        if "overpass" in url or "private.coffee" in url:
            raise requests.ConnectionError("offline")
        raise AssertionError(url)
    monkeypatch.setattr(requests, "post", failing_post)
    at = start()
    at.sidebar.text_input(key="gem_key").set_value("K").run()
    at.text_input(key="question").set_value("no cafe within 300 m").run()
    assert not at.exception, at.exception
    assert any("Gemini unavailable" in w.value for w in at.warning)
    assert "rule-based parser" in md(at)
    [b for b in at.button if b.key == "explain"][0].click().run()
    assert not at.exception, at.exception
    assert any("AI summary unavailable" in i.value for i in at.info)


def test_template_summary_without_gemini():
    at = start()
    [b for b in at.button if b.key == "explain"][0].click().run()
    assert not at.exception, at.exception
    assert any("OpenStreetMap place counts" in i.value for i in at.info)


def test_ors_key_gives_real_isochrone():
    at = start()
    assert "approximate circle" in md(at)
    at.sidebar.text_input(key="ors_key").set_value("FAKE-ORS-KEY").run()
    assert not at.exception, at.exception
    assert "street-network isochrone" in md(at)
    ors = [c for c in CALLS if "openrouteservice" in c[1]]
    assert ors and ors[0][2]["headers"]["Authorization"] == "FAKE-ORS-KEY"
    n = len(ors)
    at.checkbox(key="show_comp").set_value(False).run()  # same spot selected -> memoised, no new call
    assert not at.exception, at.exception
    assert len([c for c in CALLS if "openrouteservice" in c[1]]) == n
    at.sidebar.slider(key="comp_weight").set_value(0.4).run()  # top spot moves -> a new, legitimate call
    assert not at.exception, at.exception
    assert len([c for c in CALLS if "openrouteservice" in c[1]]) == n + 1


def test_search_any_address():
    at = start()
    at.sidebar.selectbox(key="area").select("🔎 Search any address…").run()
    assert not at.exception, at.exception
    assert "Search an address" in md(at)  # friendly empty state, no crash

    at.sidebar.text_input(key="place_query").set_value("nowhere land").run()
    at.sidebar.button(key="search_place").click().run()
    assert any("No place found" in e.value for e in at.sidebar.error)

    at.sidebar.text_input(key="place_query").set_value("broken").run()
    at.sidebar.button(key="search_place").click().run()
    assert any("Place search failed" in e.value for e in at.sidebar.error)

    at.sidebar.text_input(key="place_query").set_value("Phoenix Marketcity Whitefield").run()
    at.sidebar.button(key="search_place").click().run()
    assert not at.exception, at.exception
    assert len(at.sidebar.selectbox(key="geo_pick").options) == 2
    assert "Phoenix Marketcity" in md(at)  # header shows the chosen place
    assert len(at.metric) >= 4
    at.sidebar.selectbox(key="geo_pick").select_index(1).run()
    assert not at.exception, at.exception


def test_redownload_button_then_area_change():
    at = start(use_live=True)
    at.sidebar.button(key="redownload").click().run()
    assert not at.exception, at.exception
    posts_after_click = len([c for c in CALLS if "overpass" in c[1]])
    at.sidebar.selectbox(key="area").select("HSR Layout").run()
    assert not at.exception, at.exception
    assert posts_after_click >= 1


def test_question_with_place_recentres_map():
    at = start()
    at.text_input(key="question").set_value("give me a good spot in Monitor Tech Park").run()
    assert not at.exception, at.exception
    assert "Manyata Tech Park" in md(at)
    assert not any("couldn't find any place" in w.value.lower() for w in at.warning)

"""Place resolution tests (network mocked)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scout import places
from scout.places import extract_candidates, find_known, find_place, resolve_place, strip_place


@pytest.fixture(autouse=True)
def fresh_cache():
    places.clear_cache()


def fake_geocoder(table):
    calls = []

    def geo(query, limit=5, viewbox=None):
        calls.append(query)
        for key, rows in table.items():
            if key in query.lower():
                return rows
        return []
    geo.calls = calls
    return geo


RV = [{"name": "RV College of Engineering, Mysore Road, Bengaluru", "lat": 12.9237, "lon": 77.4989}]
FAR = [{"name": "Manyata Tech Park, Hyderabad", "lat": 17.4, "lon": 78.4}]  # outside Bengaluru


def test_builtin_exact_and_fuzzy():
    h = find_known("best cafe spot near manyata tech park")
    assert h["name"] == "Manyata Tech Park" and not h["fuzzy"]
    h = find_place("give me a good spot in Monitor Tech Park")
    assert h["name"] == "Manyata Tech Park" and h["fuzzy"]
    assert find_known("good spot in halasur")["name"] == "Halasuru"
    assert find_known("gym near IISc")["name"] == "IISc Bangalore"


def test_extract_candidates():
    c = extract_candidates("give me a good spot near Jain University and no cafe within 300 m")
    assert c and "jain" in c[0]["tokens"]
    c = extract_candidates("best spot Nitte Meenakshi Institute of Technology")
    assert c and "nitte" in c[0]["tokens"]
    c = extract_candidates("good spot in Prestige Tech Park")
    assert c and "prestige" in c[0]["tokens"]


def test_generic_questions_find_no_place():
    for q in ["lots of offices nearby and no cafe within 300 m",
              "busy shopping area with few competitors, top 5",
              "close to a metro station and no competitors within 400 m",
              "somewhere with plenty of workplaces", "blah blah", "",
              "at least 3 apartment buildings within 0.4 km, low competition, top 5",
              "spots with lots of students but no gym within 1 km",
              "near a college with no cafe within 300 m", "good spot in a busy residential area",
              "at least 99999 offices", "no cafe within 300 m"]:
        assert extract_candidates(q) == [], q
        geo = fake_geocoder({"": RV})
        assert resolve_place(q, geo)[0] is None and not geo.calls, q


def test_any_college_resolves_through_search():
    geo = fake_geocoder({"rv college": RV})
    hit, note = resolve_place("good spot near RV College Mysore Road", geo)
    assert hit is not None and hit["name"] == "RV College of Engineering"
    geo = fake_geocoder({"jain": [{"name": "Jain University, Kanakapura Road, Bengaluru",
                                   "lat": 12.8, "lon": 77.55}]})
    hit, note = resolve_place("cafe spot near Jain University", geo)
    assert hit["name"] == "Jain University" and hit["source"] == "search" and note == ""
    assert strip_place("cafe spot near Jain University", hit).strip() == "cafe spot"
    h2 = find_known("good spot near Christ University and no cafe within 300 m")
    assert strip_place("good spot near Christ University and no cafe within 300 m", h2) == "good spot and no cafe within 300 m"


def test_result_outside_bengaluru_or_unrelated_is_rejected():
    hit, note = resolve_place("good spot in Zorbatech Park", fake_geocoder({"zorbatech": FAR}))
    assert hit is None and "Zorbatech" in note.title() or "zorbatech" in note
    wrong = [{"name": "Sunrise Bakery, Indiranagar, Bengaluru", "lat": 12.97, "lon": 77.64}]
    hit, note = resolve_place("good spot in Zorbatech Park", fake_geocoder({"zorbatech": wrong}))
    assert hit is None


def test_lookup_failure_never_raises_and_is_cached():
    calls = []

    def boom(q, limit=5, viewbox=None):
        calls.append(q)
        raise RuntimeError("offline")
    hit, note = resolve_place("good spot in Zorbatech Park", boom)
    assert hit is None and "unavailable" in note
    hit, note = resolve_place("good spot in Foobar Tech Park", lambda *a, **k: [])
    assert hit is None and "Couldn't find" in note
    n = len(calls)
    resolve_place("good spot in Zorbatech Park", boom)
    assert len(calls) >= n  # no crash on repeat
    geo = fake_geocoder({})
    resolve_place("good spot in Foobar Tech Park", geo); resolve_place("good spot in Foobar Tech Park", geo)
    assert len(geo.calls) <= 2  # second ask served from the miss cache

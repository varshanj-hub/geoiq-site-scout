"""Site Scout: AI-assisted retail site selection (map-first dashboard).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import os

import pandas as pd
import streamlit as st

from scout.config import (
    AREAS, BUSINESS_WEIGHTS, CATEGORY_LABELS, DEFAULT_RADIUS_M, DEMAND_CATEGORIES,
)
from scout.data import get_pois, validate_categories
from scout.env import load_env
from scout.geocode import GeocodeError, geocode
from scout.isochrone import PROFILES, isochrone_report
from scout.llm import explain_spot, parse_question_smart
from scout.nlquery import apply_query
from scout.places import resolve_place, slug, strip_place
from scout.scoring import LocationData, score_cells
from scout.viz import build_deck

load_env()
st.set_page_config(page_title="Site Scout", page_icon="📍", layout="wide")

SEARCH_OPTION = "🔎 Search any address…"
EXAMPLES = [
    "lots of offices nearby and no cafe within 300 m",
    "busy shopping area with few competitors, top 5",
    "close to a metro station and no competitors within 400 m",
]

CSS = """
<style>
#MainMenu, footer, .stDeployButton {visibility: hidden;}
.block-container {padding-top: 1.1rem; max-width: 1600px;}
.stApp {background: #f5f7fb;}
.topbar {display:flex; align-items:center; justify-content:space-between; background:#fff;
  border:1px solid #e5e9f2; border-radius:14px; padding:14px 20px; margin-bottom:14px;}
.brand {display:flex; align-items:center; gap:12px;}
.logo {background:#2563eb; color:#fff; width:38px; height:38px; border-radius:10px;
  display:flex; align-items:center; justify-content:center; font-size:20px;}
.brand h1 {font-size:1.25rem; margin:0; padding:0; line-height:1.2; color:#0f172a;}
.brand p {margin:0; font-size:.82rem; color:#64748b;}
.pill {display:inline-block; padding:3px 11px; border-radius:999px; font-size:.75rem;
  font-weight:600; margin-left:6px;}
.pill.ok {background:#dcfce7; color:#166534;}
.pill.warn {background:#fef3c7; color:#92400e;}
.pill.blue {background:#dbeafe; color:#1e40af;}
.pill.gray {background:#e2e8f0; color:#334155;}
.section-title {font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  color:#64748b; margin:2px 0 8px 0;}
div[data-testid="stMetric"] {background:#fff; border:1px solid #e5e9f2; border-radius:12px;
  padding:10px 14px;}
div[data-testid="stMetricValue"] {font-size:1.5rem; color:#0f172a;}
div[data-testid="stVerticalBlockBorderWrapper"] {background:#fff; border-radius:14px;}
.badge {display:inline-block; padding:4px 12px; border-radius:999px; color:#fff; font-weight:700;}
.hint {color:#64748b; font-size:.82rem;}
/* Readable buttons in every state (the old hover turned text white-on-white) */
.stButton > button, button[data-testid^="stBaseButton"] {background:#fff; color:#0f172a;
  border:1px solid #cbd5e1; border-radius:10px; text-align:left; justify-content:flex-start;}
.stButton > button p, button[data-testid^="stBaseButton"] p {color:#0f172a !important;}
.stButton > button:hover, .stButton > button:focus, .stButton > button:active,
button[data-testid^="stBaseButton"]:hover, button[data-testid^="stBaseButton"]:focus {
  background:#eff6ff !important; color:#1d4ed8 !important; border-color:#2563eb !important;}
.stButton > button:hover p, .stButton > button:focus p, .stButton > button:active p,
button[data-testid^="stBaseButton"]:hover p {color:#1d4ed8 !important;}
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {background:#fff; color:#0f172a;}
.stApp label, .stApp p, .stApp li {color:#0f172a;}
.kpi {background:#fff; border:1px solid #e5e9f2; border-radius:12px; padding:10px 14px;}
.kpi .v {font-size:1.35rem; font-weight:700; color:#0f172a;} .kpi .l {font-size:.75rem; color:#64748b;}
.verdict {display:inline-block; padding:5px 14px; border-radius:999px; color:#fff; font-weight:700;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading map data (the first download can take up to a minute)...")
def load_location_data(key: str, lat: float, lon: float, radius_km: float,
                       cell_m: int, use_live: bool, force: bool):
    pois, source, message = get_pois(key, (lat, lon), radius_km * 1000,
                                     use_live=use_live, force_refresh=force)
    return LocationData((lat, lon), radius_km * 1000, cell_m, pois), source, message


def memo(key, fn, cache_if=lambda value: True):
    """Remember results for this browser session (so reruns don't re-call free APIs).

    Failed/fallback results are NOT remembered, so a retry can succeed later.
    """
    store = st.session_state.setdefault("_memo", {})
    if key in store:
        return store[key]
    value = fn()
    if cache_if(value):
        store[key] = value
    return value


def key_id(secret: str) -> str:
    return hashlib.sha1(secret.encode()).hexdigest()[:8] if secret else "none"


def score_color(score: float) -> str:
    return "#16a34a" if score >= 70 else "#d97706" if score >= 40 else "#dc2626"


def top_drivers(row: pd.Series, scored: pd.DataFrame, business_type: str) -> list[str]:
    weights = BUSINESS_WEIGHTS[business_type]
    ranked = []
    for cat, w in weights.items():
        n = int(row[f"n_{cat}"])
        ref = max(float(scored[f"n_{cat}"].quantile(0.95)), 1.0)
        if n > 0:
            ranked.append((w * min(n / ref, 1.0), f"{n} {CATEGORY_LABELS[cat]}"))
    return [label for _, label in sorted(ranked, reverse=True)[:3]]


def footfall_proxy(row: pd.Series, scored: pd.DataFrame) -> int:
    """0-100 index of how busy the surroundings look (OSM place counts, NOT measured footfall)."""
    w = {"metro": 3.0, "bus_stop": 1.0, "shops": 1.0, "office": 1.5, "college": 1.5, "apartments": 0.5}
    raw = lambda r: sum(wt * float(r[f"n_{c}"]) for c, wt in w.items())
    allraw = scored.apply(raw, axis=1)
    ref = max(float(allraw.quantile(0.95)), 1.0)
    return int(round(100 * min(raw(row) / ref, 1.0)))


def level(value: float, series: pd.Series) -> str:
    lo, hi = float(series.quantile(0.33)), float(series.quantile(0.66))
    return "Low" if value <= lo else "High" if value > hi else "Medium"


# A place named in the question (e.g. "good spot in Manyata Tech Park") overrides the dropdown.
place_hit, place_note = resolve_place(st.session_state.get("question", ""))

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
header_slot = st.container()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
sb = st.sidebar
sb.markdown("### ⚙️ Settings")
area_choice = sb.selectbox("Area", list(AREAS) + [SEARCH_OPTION], key="area")

if place_hit:
    center, area_name = (place_hit["lat"], place_hit["lon"]), place_hit["name"]
    area_key = f"lm_{slug(area_name)}_{place_hit['lat']:.3f}_{place_hit['lon']:.3f}"
    sb.info(f"📍 Focused on **{area_name}** because your question mentions it"
            + (" (closest match to what you typed)" if place_hit["fuzzy"] else "")
            + ". Clear the question to use the dropdown again.")
elif area_choice == SEARCH_OPTION:
    if place_note:
        sb.warning(place_note)
    place_query = sb.text_input("Address or place", key="place_query",
                                placeholder="e.g. Phoenix Marketcity, Whitefield")
    if sb.button("Search place", key="search_place"):
        try:
            found = geocode(place_query)
            st.session_state["geo_results"] = found
            st.session_state["geo_error"] = "" if found else "No place found. Try a more specific name."
        except GeocodeError as err:
            st.session_state["geo_results"] = []
            st.session_state["geo_error"] = str(err)
    if st.session_state.get("geo_error"):
        sb.error(st.session_state["geo_error"])
    geo_results = st.session_state.get("geo_results", [])
    if not geo_results:
        header_slot.markdown(
            '<div class="topbar"><div class="brand"><div class="logo">📍</div><div>'
            '<h1>Site Scout</h1><p>Search an address in the sidebar to begin.</p></div></div></div>',
            unsafe_allow_html=True)
        st.stop()
    idx = sb.selectbox("Pick the match", list(range(len(geo_results))),
                       format_func=lambda i: geo_results[i]["name"], key="geo_pick")
    pick = geo_results[idx]
    center = (pick["lat"], pick["lon"])
    area_name = pick["name"].split(",")[0]
    area_key = f"custom_{pick['lat']:.3f}_{pick['lon']:.3f}"
else:
    if place_note:
        sb.warning(place_note)
    center, area_name, area_key = AREAS[area_choice], area_choice, area_choice

business_type = sb.selectbox("What do you want to open?", list(BUSINESS_WEIGHTS),
                             format_func=lambda b: b.capitalize(), key="business")
radius_km = sb.slider("Search radius (km)", 1.0, 3.0, 2.0, 0.5, key="radius_km")
cell_m = sb.select_slider("Grid cell size (m)", options=[100, 150, 250, 400], value=250, key="cell_m")
catchment_m = sb.slider("Catchment radius (m)", 200, 1000, DEFAULT_RADIUS_M, 100, key="catchment_m")
comp_weight = sb.slider("Competition penalty", 0.0, 1.5, 0.8, 0.1, key="comp_weight",
                        help="0 ignores competitors; higher values avoid crowded spots.")

sb.markdown("### 🚶 Travel-time catchment")
show_iso = sb.checkbox("Show for selected spot", value=True, key="show_iso")
iso_mode = sb.selectbox("Mode", list(PROFILES), key="iso_mode")
iso_minutes = sb.select_slider("Minutes", options=[5, 10, 15], value=10, key="iso_minutes")

sb.markdown("### 🔑 API keys (optional, free)")
gem_input = sb.text_input("Gemini API key", type="password", key="gem_key",
                          help="Makes the question box understand any wording and enables AI summaries.")
ors_input = sb.text_input("OpenRouteService key", type="password", key="ors_key",
                          help="Real street-network travel-time areas instead of an approximate circle.")
gemini_key = gem_input.strip() or os.environ.get("GEMINI_API_KEY", "")
ors_key = ors_input.strip() or os.environ.get("ORS_API_KEY", "")
sb.caption(("✅ Gemini on" if gemini_key else "⚪ Gemini off (rule-based questions)")
           + "  ·  " + ("✅ ORS on" if ors_key else "⚪ ORS off (approximate circle)"))

sb.markdown("### 🗄️ Data")
use_live = sb.checkbox("Use live OpenStreetMap data", value=True, key="use_live",
                       help="Untick to use built-in sample data (works offline).")
refresh_clicked = sb.button("Re-download data", key="redownload")
if refresh_clicked:
    load_location_data.clear()

# --------------------------------------------------------------------------
# Load + score
# --------------------------------------------------------------------------
data, source, message = load_location_data(
    area_key, center[0], center[1], radius_km, cell_m, use_live, refresh_clicked)
scored = score_cells(data, business_type, comp_weight=comp_weight, radius_m=catchment_m)

source_pill = {"live": ("ok", "Live OpenStreetMap data"), "cache": ("ok", "OpenStreetMap data (cached)"),
               "sample": ("warn", "SAMPLE data")}[source]
header_slot.markdown(
    f'<div class="topbar"><div class="brand"><div class="logo">📍</div><div>'
    f'<h1>Site Scout</h1><p>Where should a new {business_type} open in {area_name}?</p></div></div>'
    f'<div><span class="pill {source_pill[0]}">{source_pill[1]}</span>'
    f'<span class="pill blue">{"✨ Gemini" if gemini_key else "Rule-based Q&A"}</span>'
    f'<span class="pill gray">{"Street-network isochrones" if ors_key else "Approx. catchment"}</span></div></div>',
    unsafe_allow_html=True)
if source == "sample":
    header_slot.warning(message)
missing = validate_categories(data.pois)
if missing and source != "sample":
    header_slot.info("OpenStreetMap has no data for: "
                     + ", ".join(CATEGORY_LABELS[c] for c in missing) + ". Scores ignore these.")

left, right = st.columns([0.85, 1.65], gap="large")

# --------------------------------------------------------------------------
# Left column: question + ranked spots
# --------------------------------------------------------------------------
with left:
    with st.container(border=True):
        st.markdown('<div class="section-title">Find a spot</div>', unsafe_allow_html=True)
        if "question" not in st.session_state:
            st.session_state["question"] = ""

        def _use_example(text: str) -> None:
            st.session_state["question"] = text

        for i, ex in enumerate(EXAMPLES):
            st.button(ex, key=f"example_{i}", on_click=_use_example, args=(ex,))
        question = st.text_input(
            "Describe your ideal spot (empty = overall best)", key="question",
            placeholder="e.g. lots of offices nearby and no cafe within 300 m")

        mask = None
        parsed = None
        engine = "rules"
        if question.strip():
            q_clean = strip_place(question, place_hit)
            parsed, engine = memo(
                ("parse", q_clean.strip(), business_type, catchment_m, key_id(gemini_key)),
                lambda: parse_question_smart(q_clean, business_type, gemini_key or None, catchment_m),
                cache_if=lambda r: r[1] == "gemini" or not gemini_key,
            )
            if place_hit and not parsed.filters:
                parsed.notes = []  # only a place was named: just show the best spots there
                st.markdown(f'<span class="hint">Showing the best spots around</span> **{area_name}**',
                            unsafe_allow_html=True)
            for note in parsed.notes:
                st.warning(note)
            if parsed.filters:
                label = "✨ Gemini" if engine == "gemini" else "rule-based parser"
                st.markdown(f'<span class="hint">Understood by {label}:</span> '
                            f'**{parsed.describe(business_type)}** · top {parsed.top_n}',
                            unsafe_allow_html=True)
                top, mask = apply_query(parsed, data, scored, business_type)
                if top.empty:
                    st.error("No spots match all of those conditions. Try relaxing one.")
            else:
                top = scored.sort_values("score", ascending=False).head(parsed.top_n)
        else:
            top = scored.sort_values("score", ascending=False).head(10)

    pool = top if not top.empty else scored.sort_values("score", ascending=False).head(5)
    with st.container(border=True):
        st.markdown('<div class="section-title">Top spots</div>', unsafe_allow_html=True)
        rows = {int(r.cell_id): r for r in pool.itertuples()}
        sel_id = st.radio(
            "Select a spot", list(rows), label_visibility="collapsed",
            format_func=lambda cid: f"#{int(rows[cid].rank)}  ·  score {rows[cid].score:.0f}  ·  {rows[cid].why}",
        )
sel = scored.loc[scored["cell_id"] == sel_id].iloc[0]

# --------------------------------------------------------------------------
# Travel-time catchment for the selected spot
# --------------------------------------------------------------------------
iso = None
if show_iso:
    iso = memo(
        ("iso", area_key, round(float(sel["lat"]), 5), round(float(sel["lon"]), 5),
         iso_minutes, iso_mode, key_id(ors_key)),
        lambda: isochrone_report(data, float(sel["lat"]), float(sel["lon"]), iso_minutes,
                                 ors_key or None, iso_mode),
        cache_if=lambda r: r["source"] == "ors" or not ors_key,
    )

# --------------------------------------------------------------------------
# Right column: metrics, map, report
# --------------------------------------------------------------------------
with right:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Places loaded", f"{len(data.pois):,}")
    m2.metric("Candidate cells", f"{len(scored):,}")
    m3.metric(f"Existing {CATEGORY_LABELS[business_type]}", f"{len(data.poi_xy.get(business_type, [])):,}")
    m4.metric("Best score", f"{scored['score'].max():.0f}")

    show_comp = st.checkbox(f"Show existing {CATEGORY_LABELS[business_type]} (dark dots)", value=True,
                            key="show_comp")
    deck = build_deck(
        scored, center,
        competitors=data.poi_latlon(business_type) if show_comp else None,
        highlight_mask=mask, top_cells=pool,
        zoom=14.2 - 0.6 * (radius_km - 1.0),
        iso_ring=iso["ring"] if iso else None,
        selected=(float(sel["lat"]), float(sel["lon"])),
    )
    st.pydeck_chart(deck, height=520)
    st.caption("Green = stronger, red = weaker. Numbered dots are the top picks; the blue ring is the "
               "selected spot's travel-time catchment. Hover a cell for details.")

    tab_report, tab_table, tab_method = st.tabs(["📋 Site report", "📊 All top spots", "ℹ️ Method & limits"])

    with tab_report:
        c0, c1 = st.columns([3, 1])
        c0.markdown(f"#### Spot #{int(sel['rank'])}")
        verdict = "Strong site" if sel["score"] >= 70 else "Moderate site" if sel["score"] >= 40 else "Weak site"
        c1.markdown(f'<span class="verdict" style="background:{score_color(sel["score"])}">'
                    f'{verdict} · {sel["score"]:.0f}/100</span>', unsafe_allow_html=True)
        ff = footfall_proxy(sel, scored)
        comp_level = level(float(sel["competitors"]), scored["competitors"])
        cards = [("Footfall proxy", f"{ff}/100"), ("Demand index", f"{sel['demand']:.0f}"),
                 ("Competition", f"{comp_level} ({int(sel['competitors'])} ≤ {catchment_m} m)")]
        if iso:
            cards.append((f"Rivals in {iso['minutes']}-min {iso['mode'].lower()}",
                          str(iso["counts"].get(business_type, 0))))
        for col, (lbl, val) in zip(st.columns(len(cards)), cards):
            col.markdown(f'<div class="kpi"><div class="l">{lbl}</div><div class="v">{val}</div></div>',
                         unsafe_allow_html=True)
        st.caption("Footfall proxy is built from nearby transit, offices, shops and homes in OpenStreetMap. "
                   "It is not measured footfall.")
        st.markdown(f"**Why:** {sel['why']}")
        gmaps = f"https://www.google.com/maps?q={sel['lat']:.6f},{sel['lon']:.6f}"
        st.markdown(f"[Open this spot in Google Maps]({gmaps})")

        if iso:
            tag = "street-network isochrone" if iso["source"] == "ors" else "approximate circle"
            st.markdown(f"**Inside the {iso['minutes']}-minute {iso['mode'].lower()} catchment** "
                        f"({iso['area_km2']} km², {tag})")
            if iso["note"]:
                st.caption(iso["note"])
            cats = [c for c in DEMAND_CATEGORIES if c in BUSINESS_WEIGHTS[business_type]] + [business_type]
            chart = pd.DataFrame({"places": [iso["counts"].get(c, 0) for c in cats]},
                                 index=[CATEGORY_LABELS[c] for c in cats])
            st.bar_chart(chart, height=220)

        facts = {
            "business_type": business_type, "area": area_name, "rank": int(sel["rank"]),
            "n_cells": int(len(scored)), "score": float(sel["score"]),
            "top_drivers": top_drivers(sel, scored, business_type),
            "competitors": int(sel["competitors"]), "catchment_m": int(catchment_m),
            "isochrone": ({"minutes": iso["minutes"], "mode": iso["mode"], "area_km2": iso["area_km2"],
                           "competitors": iso["counts"].get(business_type, 0)} if iso else None),
        }
        sum_key = ("summary", area_key, business_type, int(sel_id), catchment_m,
                   round(comp_weight, 2), iso_minutes if iso else 0, iso_mode if iso else "",
                   key_id(gemini_key))
        label = "✨ Explain this spot with Gemini" if gemini_key else "📝 Summarise this spot"
        if st.button(label, key="explain"):
            st.session_state[sum_key] = explain_spot(facts, gemini_key or None)
        if sum_key in st.session_state:
            text, how = st.session_state[sum_key]
            st.info(text)
            st.caption("Written by Gemini from the numbers above." if how == "gemini"
                       else "Template summary (add a Gemini key for an AI-written one).")

    with tab_table:
        table = pool[["rank", "score", "why", "lat", "lon"]].copy()
        table["map"] = [f"https://www.google.com/maps?q={a:.6f},{b:.6f}"
                        for a, b in zip(table["lat"], table["lon"])]
        st.dataframe(
            table[["rank", "score", "why", "map"]], hide_index=True,
            column_config={
                "rank": st.column_config.NumberColumn("Rank"),
                "score": st.column_config.NumberColumn("Score", format="%.1f"),
                "why": st.column_config.TextColumn("Why", width="large"),
                "map": st.column_config.LinkColumn("Google Maps", display_text="Open"),
            },
        )

    with tab_method:
        weights = BUSINESS_WEIGHTS[business_type]
        st.markdown(
            f"For each grid cell we count places within **{catchment_m} m** and combine them:\n\n"
            "`score = Σ (weight × demand-driver count, normalised) − competition penalty × competitors`, "
            "rescaled to 0-100.\n\n"
            f"Demand weights for **{business_type}**: "
            + ", ".join(f"{CATEGORY_LABELS[c]} ×{weights[c]}" for c in DEMAND_CATEGORIES if c in weights)
            + ".\n\n"
            "**Travel-time catchment:** with an OpenRouteService key the blue area follows real streets; "
            "otherwise it is an approximate circle.\n\n"
            "**AI:** Gemini only turns your wording into filters and writes a summary from the numbers shown. "
            "It never invents data, and its output is validated before use.\n\n"
            "**Limits:** this is a transparent proxy built from OpenStreetMap place counts. It has no real "
            "footfall, rent, income or population data, and OpenStreetMap coverage varies by area. "
            "Use it to shortlist spots, not to make the final call.\n\n"
            "**Next step with real data:** swap the OSM counts for measured footfall, rentals and demographics "
            "on a 100 m grid, and train a model on stores' actual sales."
        )

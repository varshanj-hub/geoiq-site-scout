"""Map rendering with pydeck (works inside Streamlit, no API token required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk


def score_to_rgb(score: float) -> list[int]:
    """0 -> red, 50 -> yellow, 100 -> green."""
    s = float(np.clip(score, 0, 100)) / 100.0
    lo, mid, hi = np.array([215, 70, 60]), np.array([250, 205, 70]), np.array([35, 160, 90])
    rgb = lo + (mid - lo) * (s / 0.5) if s < 0.5 else mid + (hi - mid) * ((s - 0.5) / 0.5)
    return [int(v) for v in rgb]


def build_deck(
    scored: pd.DataFrame,
    center: tuple[float, float],
    competitors: pd.DataFrame | None = None,
    highlight_mask: np.ndarray | None = None,
    top_cells: pd.DataFrame | None = None,
    zoom: float = 13.5,
    iso_ring: list | None = None,
    selected: tuple[float, float] | None = None,
) -> pdk.Deck:
    df = scored.copy()
    active = highlight_mask is not None
    colors = []
    for i, s in enumerate(df["score"]):
        r, g, b = score_to_rgb(s)
        alpha = 185
        if active and not highlight_mask[i]:
            alpha = 35
        colors.append([r, g, b, alpha])
    df["color"] = colors
    df["line"] = [[255, 255, 255, 90]] * len(df)

    layers = [
        pdk.Layer(
            "PolygonLayer",
            data=df[["polygon", "color", "line", "score", "rank", "why"]].to_dict("records"),
            get_polygon="polygon",
            get_fill_color="color",
            get_line_color="line",
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    ]

    if iso_ring:
        layers.append(pdk.Layer(
            "PolygonLayer",
            data=[{"polygon": iso_ring}],
            get_polygon="polygon",
            get_fill_color=[37, 99, 235, 40],
            get_line_color=[37, 99, 235, 230],
            line_width_min_pixels=3,
            stroked=True, filled=True, pickable=False,
        ))

    if competitors is not None and len(competitors):
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=competitors[["lat", "lon"]].to_dict("records"),
            get_position="[lon, lat]",
            get_radius=22,
            get_fill_color=[40, 40, 40, 200],
            pickable=False,
        ))

    if top_cells is not None and len(top_cells):
        top = top_cells.copy()
        top["label"] = [str(int(r)) for r in top["rank"]]
        recs = top[["lat", "lon", "label", "score", "rank", "why"]].to_dict("records")
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=recs, get_position="[lon, lat]",
            get_radius=55, get_fill_color=[20, 20, 20, 235], pickable=True,
        ))
        layers.append(pdk.Layer(
            "TextLayer", data=recs, get_position="[lon, lat]", get_text="label",
            get_size=16, get_color=[255, 255, 255, 255],
            get_text_anchor="'middle'", get_alignment_baseline="'center'",
        ))

    if selected is not None:
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=[{"lat": float(selected[0]), "lon": float(selected[1])}],
            get_position="[lon, lat]", get_radius=95,
            stroked=True, filled=False,
            get_line_color=[37, 99, 235, 255], line_width_min_pixels=4,
        ))

    tooltip = {
        "html": "<b>Rank #{rank}</b> &middot; Score {score}<br/>{why}",
        "style": {"backgroundColor": "#222", "color": "white", "fontSize": "12px"},
    }
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=center[0], longitude=center[1],
                                         zoom=zoom, pitch=0),
        map_provider="carto",
        map_style=pdk.map_styles.CARTO_LIGHT,
        tooltip=tooltip,
    )

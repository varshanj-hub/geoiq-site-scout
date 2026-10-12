"""Central configuration: areas, business types, categories and scoring weights."""
from __future__ import annotations

# (latitude, longitude) of the centre of each neighbourhood.
AREAS: dict[str, tuple[float, float]] = {
    "Indiranagar": (12.9784, 77.6408),
    "Koramangala": (12.9352, 77.6245),
    "HSR Layout": (12.9116, 77.6474),
    "Jayanagar": (12.9250, 77.5938),
    "MG Road / CBD": (12.9756, 77.6068),
    "Whitefield": (12.9698, 77.7500),
}

# Things that bring people (demand drivers).
DEMAND_CATEGORIES = [
    "office", "college", "metro", "apartments", "shops", "bus_stop", "healthcare",
]

# Things you might open (and that therefore also count as competitors).
BUSINESS_CATEGORIES = ["cafe", "pharmacy", "gym", "restaurant"]

ALL_CATEGORIES = DEMAND_CATEGORIES + BUSINESS_CATEGORIES

CATEGORY_LABELS = {
    "office": "offices",
    "college": "colleges",
    "metro": "metro stations",
    "apartments": "apartment buildings",
    "shops": "shops",
    "bus_stop": "bus stops",
    "healthcare": "clinics/hospitals",
    "cafe": "cafes",
    "pharmacy": "pharmacies",
    "gym": "gyms",
    "restaurant": "restaurants",
}

# How much each demand driver matters for each business type (0 = ignore).
BUSINESS_WEIGHTS: dict[str, dict[str, float]] = {
    "cafe": {
        "office": 1.0, "college": 0.8, "metro": 0.6, "shops": 0.5,
        "apartments": 0.4, "bus_stop": 0.2,
    },
    "pharmacy": {
        "apartments": 1.0, "healthcare": 1.0, "shops": 0.5,
        "metro": 0.3, "bus_stop": 0.3,
    },
    "gym": {
        "apartments": 1.0, "office": 0.6, "college": 0.4, "shops": 0.3,
    },
    "restaurant": {
        "office": 1.0, "shops": 0.7, "metro": 0.6, "apartments": 0.5, "college": 0.5,
    },
}

DEFAULT_RADIUS_M = 500  # catchment radius used for scoring and for questions

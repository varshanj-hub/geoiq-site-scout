"""Tiny rule-based natural-language question parser (no LLM, no API key needed).

Examples it understands:
    "lots of offices nearby and no cafe within 300 m"
    "at least 3 apartment buildings within 400 m, low competition, top 5"
    "near a metro station and no gyms within 0.5 km"

`parse_question` is the single entry point. If you later want an LLM to do the
parsing, replace that one function with something that returns a ParsedQuery.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CATEGORY_LABELS, DEFAULT_RADIUS_M
from .scoring import LocationData

# word root -> category. Plural handled by regex (s / es).
SYNONYMS: dict[str, str] = {
    "office": "office", "workplace": "office", "company": "office", "companies": "office",
    "tech park": "office",
    "college": "college", "university": "college", "universities": "college",
    "campus": "college", "student": "college",
    "metro": "metro", "subway": "metro", "station": "metro",
    "apartment": "apartments", "residential": "apartments", "housing": "apartments",
    "resident": "apartments", "home": "apartments",
    "shop": "shops", "retail": "shops", "store": "shops", "market": "shops",
    "bus stop": "bus_stop", "bus": "bus_stop",
    "hospital": "healthcare", "clinic": "healthcare", "healthcare": "healthcare",
    "doctor": "healthcare",
    "cafe": "cafe", "coffee": "cafe",
    "pharmacy": "pharmacy", "pharmacies": "pharmacy", "chemist": "pharmacy",
    "gym": "gym", "fitness": "gym",
    "restaurant": "restaurant", "eatery": "restaurant", "eateries": "restaurant",
    "competitor": "competitor", "competition": "competitor", "rival": "competitor",
}
_SYN_PATTERNS = [
    (re.compile(rf"\b{re.escape(word)}(?:s|es)?\b"), cat)
    for word, cat in sorted(SYNONYMS.items(), key=lambda kv: -len(kv[0]))
]

_SPLIT = re.compile(
    r"\s*(?:,|;|&|\band\b|\bbut\b|\bwith\b|\bplus\b|\bwhere\b|\bhaving\b|\bthat have\b)\s*"
)
_RADIUS = re.compile(
    r"(\d+(?:\.\d+)?)\s*(km|kilometers?|kilometres?|meters?|metres?|m)\b"
)
_TOP = re.compile(r"\b(?:top|best|first|show|list)\s*(\d+)\b")

_NEG = re.compile(
    r"\b(no|zero|without|nothing|none|not any|free of|lack of|far from|away from|not near|not close to)\b"
)
_ATMOST = re.compile(r"(at most|atmost|fewer than|less than|under|up to|maximum|max\b|<=?)")
_ATLEAST = re.compile(r"(at least|atleast|minimum|min\b|>=|or more|\d\s*\+)")
_MORETHAN = re.compile(r"(more than|over|greater than|>)")
_FEW = re.compile(r"\b(few|low|less|little|sparse|limited|small)\b")
_HIGH = re.compile(
    r"\b(lots? of|many|plenty|high|dense|several|good number|busy|heavy|strong|large)\b"
)
_TRIGGERS = re.compile(
    r"\b(no|zero|without|few|low|lots?|many|plenty|high|dense|near|nearby|close|around|least|most|more|less|than)\b"
)


@dataclass
class Filter:
    category: str          # a category key, or "competitor"
    kind: str              # "none" | "atleast" | "atmost" | "high"
    value: int = 1
    radius_m: int = DEFAULT_RADIUS_M

    def describe(self, business_type: str) -> str:
        cat = business_type if self.category == "competitor" else self.category
        label = CATEGORY_LABELS.get(cat, cat)
        if self.category == "competitor":
            label = f"competing {label}"
        where = f"within {self.radius_m} m"
        return {
            "none": f"no {label} {where}",
            "atleast": f"at least {self.value} {label} {where}",
            "atmost": f"at most {self.value} {label} {where}",
            "high": f"a high number of {label} {where} (top quartile)",
        }[self.kind]


@dataclass
class ParsedQuery:
    filters: list[Filter] = field(default_factory=list)
    top_n: int = 10
    notes: list[str] = field(default_factory=list)

    def describe(self, business_type: str) -> str:
        if not self.filters:
            return "No conditions recognised."
        return "; ".join(f.describe(business_type) for f in self.filters)


def _find_category(text: str) -> str | None:
    for pattern, cat in _SYN_PATTERNS:
        if pattern.search(text):
            return cat
    return None


def _to_metres(value: float, unit: str) -> int:
    return int(round(value * 1000)) if unit.startswith("k") else int(round(value))


def parse_question(question: str, default_radius_m: int = DEFAULT_RADIUS_M) -> ParsedQuery:
    """Turn a free-text question into structured filters."""
    q = question.lower().strip()
    result = ParsedQuery()
    if not q:
        return result

    m = _TOP.search(q)
    if m:
        result.top_n = max(1, min(int(m.group(1)), 50))
        q = _TOP.sub(" ", q)

    for clause in _SPLIT.split(q):
        clause = clause.strip()
        if not clause:
            continue
        cat = _find_category(clause)
        if cat is None:
            if _TRIGGERS.search(clause):
                result.notes.append(f"Couldn't tell which place type you meant in: “{clause}”")
            continue

        radius = default_radius_m
        rm = _RADIUS.search(clause)
        if rm:
            radius = _to_metres(float(rm.group(1)), rm.group(2))
            clause_wo_radius = _RADIUS.sub(" ", clause)
        else:
            clause_wo_radius = clause
        nums = [int(n) for n in re.findall(r"\d+", clause_wo_radius)]

        if _NEG.search(clause_wo_radius):
            flt = Filter(cat, "none", 0, radius)
        elif _MORETHAN.search(clause_wo_radius) and nums:
            flt = Filter(cat, "atleast", nums[0] + 1, radius)
        elif _ATLEAST.search(clause_wo_radius) and nums:
            flt = Filter(cat, "atleast", nums[0], radius)
        elif _ATMOST.search(clause_wo_radius) and nums:
            flt = Filter(cat, "atmost", nums[0], radius)
        elif _FEW.search(clause_wo_radius):
            flt = Filter(cat, "atmost", 2, radius)
        elif _HIGH.search(clause_wo_radius):
            flt = Filter(cat, "high", 0, radius)
        elif nums:
            flt = Filter(cat, "atleast", nums[0], radius)
        else:
            flt = Filter(cat, "atleast", 1, radius)
        result.filters.append(flt)

    if not result.filters and not result.notes:
        result.notes.append(
            "I couldn't find any place types in that question. Try e.g. "
            "“lots of offices nearby and no cafe within 300 m”."
        )
    return result


def apply_query(query: ParsedQuery, data: LocationData, scored: pd.DataFrame,
                business_type: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (top matching cells sorted by score, boolean mask over `scored`)."""
    mask = np.ones(len(scored), dtype=bool)
    for f in query.filters:
        cat = business_type if f.category == "competitor" else f.category
        counts = data.count(cat, f.radius_m)
        if f.kind == "none":
            mask &= counts == 0
        elif f.kind == "atleast":
            mask &= counts >= f.value
        elif f.kind == "atmost":
            mask &= counts <= f.value
        elif f.kind == "high":
            if counts.max() <= 0:
                mask &= False
            else:
                mask &= counts >= max(1.0, float(np.quantile(counts, 0.75)))
    matches = scored[mask].sort_values("score", ascending=False).head(query.top_n)
    return matches, mask

"""Find the place a question is about (e.g. "good spot near RV College") and re-centre on it.

Resolution order (first hit wins):
  1. a built-in list of Bengaluru landmarks, exact match (instant, works offline)
  2. any other place named in the question, looked up with free OpenStreetMap
     Nominatim search restricted to Bengaluru (so ANY college / tech park / area works)
  3. a typo-tolerant match against the built-in list (last resort)
Nothing here ever raises: failures just mean "no place found".
"""
from __future__ import annotations

import math
import re
import time
from difflib import SequenceMatcher

from .config import AREAS
from .geocode import BENGALURU_VIEWBOX, GeocodeError, geocode
from .nlquery import SYNONYMS

BBOX = (12.70, 77.35, 13.25, 77.85)  # south, west, north, east (Greater Bengaluru)

# --------------------------------------------------------------------------
# Built-in landmarks: name -> ((lat, lon), extra aliases). Approximate centres.
# --------------------------------------------------------------------------
_EXTRA: dict[str, tuple[tuple[float, float], list[str]]] = {
    "Manyata Tech Park": ((13.0472, 77.6206), ["manyata", "manyata business park", "nagawara"]),
    "Embassy Tech Village": ((12.9366, 77.6960), ["etv", "embassy tech"]),
    "Bagmane Tech Park": ((12.9833, 77.6640), ["bagmane", "cv raman nagar"]),
    "ITPL Whitefield": ((12.9876, 77.7370), ["itpl", "international tech park"]),
    "Electronic City": ((12.8456, 77.6603), ["ecity", "e city"]),
    "Hebbal": ((13.0358, 77.5970), []),
    "Halasuru": ((12.9810, 77.6260), ["halasur", "ulsoor"]),
    "Bellandur": ((12.9260, 77.6760), []),
    "Marathahalli": ((12.9569, 77.7011), ["marathalli"]),
    "Yeshwanthpur": ((13.0285, 77.5409), ["yeshwantpur"]),
    "Malleshwaram": ((13.0031, 77.5643), ["malleswaram"]),
    "Sarjapur Road": ((12.9100, 77.6850), ["sarjapur"]),
    "JP Nagar": ((12.9063, 77.5857), ["j p nagar"]),
    "Banashankari": ((12.9255, 77.5468), []),
    "IISc Bangalore": ((13.0219, 77.5671), ["iisc", "indian institute of science"]),
    "IIM Bangalore": ((12.8958, 77.6009), ["iimb", "iim bangalore", "iim bengaluru"]),
    "Christ University": ((12.9345, 77.6064), []),
    "RV College of Engineering": ((12.9237, 77.4989), ["rvce", "rv college"]),
    "PES University": ((12.9349, 77.5345), ["pes college"]),
    "BMS College of Engineering": ((12.9416, 77.5654), ["bmsce", "bms college"]),
    "MS Ramaiah Institute of Technology": ((13.0307, 77.5649), ["msrit", "ramaiah college"]),
}
_AREA_ALIASES = {"MG Road / CBD": ["mg road", "cbd"], "HSR Layout": ["hsr"]}

LANDMARKS: dict[str, tuple[float, float]] = {}
_ALIASES: list[tuple[str, str]] = []  # (alias, canonical name), longest alias first


def _build() -> None:
    for name, ll in AREAS.items():
        LANDMARKS[name] = ll
        for a in [name.lower()] + _AREA_ALIASES.get(name, []):
            _ALIASES.append((a, name))
    for name, (ll, aliases) in _EXTRA.items():
        LANDMARKS[name] = ll
        for a in [name.lower()] + aliases:
            _ALIASES.append((a, name))
    _ALIASES.sort(key=lambda t: -len(t[0]))


_build()
_TOKEN = re.compile(r"[a-z0-9]+")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _hit(name: str, lat: float, lon: float, span: tuple[int, int], fuzzy: bool, source: str) -> dict:
    return {"name": name, "lat": float(lat), "lon": float(lon), "span": span,
            "fuzzy": fuzzy, "source": source}


# --------------------------------------------------------------------------
# 1 + 3. built-in landmarks
# --------------------------------------------------------------------------
def find_known(question: str, fuzzy: bool = False) -> dict | None:
    q = (question or "").lower()
    if len(q.strip()) < 4:
        return None
    if not fuzzy:
        for alias, name in _ALIASES:
            m = re.search(rf"\b{re.escape(alias)}\b", q)
            if m:
                lat, lon = LANDMARKS[name]
                return _hit(name, lat, lon, m.span(), False, "builtin")
        return None
    toks = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(q)]
    best, best_score = None, 0.0
    for alias, name in _ALIASES:
        k = len(alias.split())
        if k > len(toks) or len(alias) < 6:
            continue
        cutoff = 0.8 if k == 1 else 0.72
        for i in range(len(toks) - k + 1):
            window = " ".join(t[0] for t in toks[i:i + k])
            if window[0] != alias[0]:
                continue
            r = SequenceMatcher(None, window, alias).ratio()
            if r >= cutoff and r > best_score:
                best_score, best = r, (name, (toks[i][1], toks[i + k - 1][2]))
    if best:
        lat, lon = LANDMARKS[best[0]]
        return _hit(best[0], lat, lon, best[1], True, "builtin")
    return None


def find_place(question: str) -> dict | None:
    """Offline lookup only (exact, then typo-tolerant)."""
    return find_known(question) or find_known(question, fuzzy=True)


# --------------------------------------------------------------------------
# 2. pull a place phrase out of the question
# --------------------------------------------------------------------------
_STOP = set("""a an the some any good best great nice spot spots place places area areas location locations
site sites somewhere anywhere near nearby around close closest to of and or with without no not lots lot many
plenty few low high little busy dense heavy strong large small quiet crowded residential commercial popular
top give me show find want open new here there city town locality neighbourhood neighborhood zone region part
side least most more less than over under at in on for from my our i is are be have has that which where
competition competitors competitor rival rivals within km m meter meters metre metres kilometer kilometers
building buildings opposite beside behind inside next nearest towards by it its this these those other
cafe cafes pharmacy pharmacies gym gyms restaurant restaurants""".split())
_CAT_TOKENS = {t for key in SYNONYMS for t in key.split()}
_NAME_KW = {"iisc", "iimb", "iim"}            # abbreviations that are names by themselves
_KW_SINGLE = {"college", "colleges", "university", "universities", "institute", "campus"} | _NAME_KW
_KW_PAIRS = {("tech", "park"), ("it", "park"), ("business", "park"), ("software", "park"), ("techpark",)}
_PREP = re.compile(r"\b(?:in|near|around|at|inside|opposite|beside|behind|towards|next to|close to|nearest to)\s+")
_END = re.compile(r"\b(?:and|but|with|where|having|that|which|within|top|show|list|plus|so|because|for)\b"
                  r"|[,;.!?]|\b\d+(?:\.\d+)?\s*(?:km|kilometers?|kilometres?|meters?|metres?|m)\b")


def _plain(tok: str) -> str:
    return tok[:-1] if tok.endswith("s") and len(tok) > 3 else tok


def _significant(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        if t in _STOP or _plain(t) in _STOP or t in _CAT_TOKENS or _plain(t) in _CAT_TOKENS:
            continue
        if t in _KW_SINGLE and t not in _NAME_KW:
            continue
        if t.isdigit() or len(t) < 2:
            continue
        out.append(t)
    return out


def extract_candidates(question: str) -> list[dict]:
    """Possible place phrases, best first: [{'text', 'tokens', 'span'}]."""
    q = (question or "").lower()
    found: list[dict] = []

    def add(start: int, end: int) -> None:
        text = q[start:end].strip(" ,;.!?")
        toks = _TOKEN.findall(text)
        sig = _significant(toks)
        if sig and not any(c["text"] == text for c in found):
            found.append({"text": text, "tokens": sig, "span": (start, start + len(q[start:end].rstrip(" ,;.!?")))})

    for m in _PREP.finditer(q):                       # "... in <phrase>"
        start = m.end()
        stop = _END.search(q, start)
        add(start, stop.start() if stop else len(q))

    toks = [(t.group(), t.start(), t.end()) for t in _TOKEN.finditer(q)]
    for i, (w, s, e) in enumerate(toks):              # "... <name> college/university/tech park"
        kw_len = 0
        if w in _KW_SINGLE:
            kw_len = 1
        elif i + 1 < len(toks) and (w, toks[i + 1][0]) in _KW_PAIRS:
            kw_len = 2
        elif (w,) in _KW_PAIRS:
            kw_len = 1
        if not kw_len:
            continue
        left = i
        while left > 0 and i - left < 4 and toks[left - 1][0] not in _STOP \
                and toks[left - 1][0] not in _CAT_TOKENS and not toks[left - 1][0].isdigit():
            left -= 1
        right = i + kw_len - 1
        if right + 2 < len(toks) and toks[right + 1][0] == "of":      # "... college of engineering"
            right += 2
            if right + 1 < len(toks) and toks[right + 1][0] not in _STOP | _CAT_TOKENS:
                right += 1
        add(toks[left][1], toks[right][2])
    return found[:2]


# --------------------------------------------------------------------------
# look the phrase up (Nominatim, restricted to Bengaluru) with a safety check
# --------------------------------------------------------------------------
_POS: dict[str, tuple[str, float, float]] = {}
_NEG: dict[str, float] = {}
NEG_TTL_S = 120


def clear_cache() -> None:
    _POS.clear()
    _NEG.clear()


def _in_bengaluru(lat: float, lon: float) -> bool:
    return BBOX[0] <= lat <= BBOX[2] and BBOX[1] <= lon <= BBOX[3]


def _name_matches(sig: list[str], result_name: str) -> bool:
    """The search result must actually resemble what was typed (blocks random matches)."""
    name_toks = _TOKEN.findall(result_name.lower())
    matched = sum(
        1 for t in sig
        if any(t == n or (len(t) >= 4 and SequenceMatcher(None, t, n).ratio() >= 0.8) for n in name_toks)
    )
    return matched >= max(1, math.ceil(len(sig) / 2))


def _lookup(cand: dict, geocode_fn) -> tuple[dict | None, str]:
    key = cand["text"]
    if key in _POS:
        name, lat, lon = _POS[key]
        return _hit(name, lat, lon, cand["span"], False, "search"), ""
    if time.time() - _NEG.get(key, 0.0) < NEG_TTL_S:
        return None, "cached-miss"
    queries = [f"{key}, Bengaluru"]
    shorter = " ".join(t for t in key.split() if t not in _CAT_TOKENS and t not in _KW_SINGLE)
    if shorter and shorter != key:
        queries.append(f"{shorter}, Bengaluru")
    err = ""
    for q in queries:
        try:
            results = geocode_fn(q, 5, BENGALURU_VIEWBOX)
        except (GeocodeError, Exception) as e:   # never let a lookup crash the app
            err = str(e)
            break
        for r in results or []:
            try:
                lat, lon, full = float(r["lat"]), float(r["lon"]), str(r.get("name", ""))
            except (KeyError, TypeError, ValueError):
                continue
            if _in_bengaluru(lat, lon) and _name_matches(cand["tokens"], full):
                name = full.split(",")[0].strip() or key.title()
                _POS[key] = (name, lat, lon)
                return _hit(name, lat, lon, cand["span"], False, "search"), ""
    if not err:
        _NEG[key] = time.time()
    return None, err


def resolve_place(question: str, geocode_fn=None) -> tuple[dict | None, str]:
    """Return (hit, note). `note` explains a failed lookup (empty when nothing to say)."""
    try:
        hit = find_known(question)
        if hit:
            return hit, ""
        geocode_fn = geocode_fn or geocode
        note = ""
        for cand in extract_candidates(question):
            hit, err = _lookup(cand, geocode_fn)
            if hit:
                return hit, ""
            if err and err != "cached-miss":
                note = f"Place search is unavailable right now ({err[:80]}); using the selected area."
            elif not note:
                note = f"Couldn't find “{cand['text']}” in Bengaluru; using the selected area instead."
        fuzzy = find_known(question, fuzzy=True)
        return (fuzzy, "") if fuzzy else (None, note)
    except Exception as err:  # absolute safety net
        return None, f"Place lookup failed ({type(err).__name__}); using the selected area."


def strip_place(question: str, hit: dict | None) -> str:
    """Remove the place words so 'tech park' in a name is not parsed as 'offices'."""
    if not hit:
        return question
    s, e = hit["span"]
    q = question.lower()
    left = re.sub(r"\b(?:in|near|around|at|inside|opposite|beside|behind|towards|next to|close to|nearest to)\s*$",
                  "", q[:s].rstrip())
    return re.sub(r"\s+", " ", left + " " + q[e:]).strip()

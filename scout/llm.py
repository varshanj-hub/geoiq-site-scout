"""Gemini (free tier) helpers: smart question parsing and 'why this spot' summaries.

Design rules:
* The API key is sent in a header, never in the URL, and never logged.
* Everything the model returns is VALIDATED before use (it is only allowed to
  produce known categories / filter kinds / sane numbers).
* Any failure (no key, quota, bad JSON...) falls back to the rule-based parser or
  a template summary, so the app keeps working.
"""
from __future__ import annotations

import json
import os
import re

import requests

from .config import ALL_CATEGORIES, CATEGORY_LABELS, DEFAULT_RADIUS_M
from .nlquery import Filter, ParsedQuery, parse_question

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-2.5-flash"
KINDS = ("none", "atleast", "atmost", "high")


class LLMError(RuntimeError):
    pass


def get_model() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def call_gemini(prompt: str, system: str, api_key: str, json_mode: bool = False,
                temperature: float = 0.2, max_tokens: int = 700, timeout: int = 25) -> str:
    """Single Gemini call. Returns the text of the first candidate."""
    if not api_key:
        raise LLMError("no Gemini API key")
    model = get_model()
    gen_cfg: dict = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    if model.startswith("gemini-2.5-flash"):
        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}  # faster + keeps free-tier tokens for the answer
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    }
    try:
        resp = requests.post(
            GEMINI_URL.format(model=model), json=body,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as err:
        raise LLMError(f"could not reach Gemini ({type(err).__name__})") from err

    if resp.status_code == 429:
        raise LLMError("Gemini free-tier rate limit reached, try again in a minute")
    if resp.status_code in (400, 401, 403):
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = ""
        raise LLMError(f"Gemini rejected the request ({resp.status_code}). {detail[:160]}".strip())
    if resp.status_code >= 400:
        raise LLMError(f"Gemini error {resp.status_code}")
    try:
        payload = resp.json()
        parts = payload["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError, ValueError) as err:
        raise LLMError("Gemini returned no usable answer") from err
    if not text:
        raise LLMError("Gemini returned an empty answer")
    return text


# --------------------------------------------------------------------------
# Question parsing
# --------------------------------------------------------------------------
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def parsed_from_json(obj, default_radius_m: int = DEFAULT_RADIUS_M) -> ParsedQuery:
    """Validate model output and build a ParsedQuery (drops anything unexpected)."""
    if not isinstance(obj, dict):
        raise LLMError("model output was not a JSON object")
    result = ParsedQuery(top_n=_clamp(obj.get("top_n", 10), 1, 50, 10))
    allowed = set(ALL_CATEGORIES) | {"competitor"}
    for item in obj.get("filters") or []:
        if not isinstance(item, dict):
            continue
        cat, kind = item.get("category"), item.get("kind")
        if cat not in allowed or kind not in KINDS:
            result.notes.append(f"Ignored an unrecognised condition from the AI: {item!r}"[:140])
            continue
        result.filters.append(Filter(
            category=cat, kind=kind,
            value=_clamp(item.get("value", 1), 0, 1000, 1),
            radius_m=_clamp(item.get("radius_m", default_radius_m), 50, 3000, default_radius_m),
        ))
    return result


def parse_question_llm(question: str, business_type: str, api_key: str,
                       default_radius_m: int = DEFAULT_RADIUS_M) -> ParsedQuery:
    system = (
        "You convert a request about where to open a shop into JSON filters over map grid cells. "
        "Reply with JSON only, no prose."
    )
    prompt = f"""Business being opened: {business_type}
Allowed categories: {", ".join(ALL_CATEGORIES)}, competitor (= existing {business_type} places)
Allowed kinds: none (zero within radius), atleast (>= value), atmost (<= value), high (top quartile of cells)
Default radius_m: {default_radius_m}. Default top_n: 10.

Output schema:
{{"filters": [{{"category": str, "kind": str, "value": int, "radius_m": int}}], "top_n": int}}

Examples:
"lots of offices nearby and no cafe within 300 m" ->
{{"filters":[{{"category":"office","kind":"high","value":0,"radius_m":{default_radius_m}}},{{"category":"cafe","kind":"none","value":0,"radius_m":300}}],"top_n":10}}
"quiet residential spot with few rivals, show 3" ->
{{"filters":[{{"category":"apartments","kind":"high","value":0,"radius_m":{default_radius_m}}},{{"category":"competitor","kind":"atmost","value":2,"radius_m":{default_radius_m}}}],"top_n":3}}

Request: {question!r}"""
    text = call_gemini(prompt, system, api_key, json_mode=True, temperature=0.0)
    try:
        obj = json.loads(_FENCE.sub("", text.strip()))
    except json.JSONDecodeError as err:
        raise LLMError("Gemini did not return valid JSON") from err
    return parsed_from_json(obj, default_radius_m)


def parse_question_smart(question: str, business_type: str, api_key: str | None,
                         default_radius_m: int = DEFAULT_RADIUS_M) -> tuple[ParsedQuery, str]:
    """Use Gemini when a key is available, otherwise (or on failure) the rule-based parser.

    Returns (parsed_query, engine) where engine is 'gemini' or 'rules'.
    """
    if api_key:
        try:
            parsed = parse_question_llm(question, business_type, api_key, default_radius_m)
            if parsed.filters:
                return parsed, "gemini"
            fallback = parse_question(question, default_radius_m)
            if fallback.filters:
                fallback.notes.append("Gemini found no conditions, used the rule-based parser.")
                return fallback, "rules"
            return parsed, "gemini"
        except LLMError as err:
            fallback = parse_question(question, default_radius_m)
            fallback.notes.append(f"Gemini unavailable ({err}); used the rule-based parser.")
            return fallback, "rules"
    return parse_question(question, default_radius_m), "rules"


# --------------------------------------------------------------------------
# "Why this spot" summary
# --------------------------------------------------------------------------
def template_summary(facts: dict) -> str:
    """Plain, deterministic fallback summary (no LLM)."""
    biz = CATEGORY_LABELS.get(facts["business_type"], facts["business_type"])
    drivers = facts.get("top_drivers") or []
    lead = (f"This spot ranks #{facts['rank']} of {facts['n_cells']} with a score of {facts['score']:.0f}/100. "
            if facts.get("rank") else "")
    if drivers:
        lead += "Demand is driven mainly by " + ", ".join(drivers) + ". "
    lead += (f"There are {facts['competitors']} competing {biz} within {facts['catchment_m']} m"
             + (", which leaves room for a new entrant." if facts["competitors"] <= 2
                else ", so differentiation will matter."))
    iso = facts.get("isochrone")
    if iso:
        lead += (f" Within a {iso['minutes']}-minute {iso['mode'].lower()} catchment "
                 f"({iso['area_km2']} km²) there are {iso['competitors']} competitors.")
    return lead + " This is a proxy from OpenStreetMap place counts; verify footfall and rent on the ground."


def explain_spot(facts: dict, api_key: str | None) -> tuple[str, str]:
    """Return (summary, engine). Falls back to the template if Gemini fails."""
    if not api_key:
        return template_summary(facts), "template"
    system = (
        "You are a retail site-selection analyst. Write a short, plain-English recommendation "
        "using ONLY the facts provided. Never invent numbers, names or facts."
    )
    prompt = (
        "Write 3-4 sentences for someone deciding whether to open a "
        f"{facts['business_type']} here. Mention the main strengths, the competition, and one "
        "honest limitation. No headings, no bullet points.\n\nFacts (JSON):\n"
        + json.dumps(facts, ensure_ascii=False)
    )
    try:
        return call_gemini(prompt, system, api_key, temperature=0.4, max_tokens=400), "gemini"
    except LLMError as err:
        return template_summary(facts) + f"\n\n(AI summary unavailable: {err})", "template"

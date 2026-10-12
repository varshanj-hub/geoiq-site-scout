# 📍 Site Scout

An AI-assisted retail site-selection dashboard. Pick a neighbourhood (or search any
address) and a business type (cafe, pharmacy, gym, restaurant). Every ~250 m grid
cell is scored 0-100 from free OpenStreetMap data, shown on a map with a "Site report"
for the spot you select. Ask questions in plain English, e.g.
*"somewhere with plenty of workplaces and no cafe within 300 m"*.

Everything works **without any API key**. Free keys unlock the extras.

| Feature | No key | With free key |
|---|---|---|
| Place data | OpenStreetMap (Overpass), cached locally | - |
| Search any address | Nominatim (free, no key) | - |
| Ask questions | Rule-based parser | **Gemini** understands any wording |
| "Explain this spot" | Template summary | **Gemini** writes a short analyst note |
| Travel-time catchment | Approximate circle | **OpenRouteService** real street-network isochrone |

## Run it

You need Python 3.9+ (tested on 3.12).

**Mac / Linux**
```bash
cd geoiq_site_scout
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

**Windows (PowerShell)**
```powershell
cd geoiq_site_scout
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens at http://localhost:8501. The first time you pick an area it
downloads data (up to a minute), then caches it in `data/cache/`. If the download
fails, or you untick "Use live OpenStreetMap data", the app uses clearly-labelled
SAMPLE data so it always works.

## Add the free API keys (optional)

1. **Gemini**: create a key at https://aistudio.google.com/apikey
2. **OpenRouteService**: sign up at https://openrouteservice.org/dev/#/signup
3. Copy `.env.example` to `.env` and paste the keys, **or** paste them into the
   sidebar's "API keys" boxes (kept only for your session).

`.env` is git-ignored. Never commit or share your keys. Free tiers have rate limits;
results are remembered per session so the app doesn't waste your quota. If Gemini's model
name changes in future, set `GEMINI_MODEL` in `.env`.

## How it works

| File | What it does |
|---|---|
| `app.py` | Streamlit dashboard (map-first layout, site report) |
| `scout/data.py` | OpenStreetMap download, classification, caching, offline sample fallback |
| `scout/scoring.py` | Grid, catchment counting, 0-100 score |
| `scout/nlquery.py` | Rule-based question parser (always available) |
| `scout/llm.py` | Gemini question parsing + summaries; output is validated, failures fall back |
| `scout/isochrone.py` | OpenRouteService isochrones + point-in-polygon counting |
| `scout/places.py` | Finds the place named in a question (any Bengaluru college / tech park / area) and re-centres the map |
| `scout/geocode.py` | Nominatim address search (1 request/second, as their policy requires) |
| `scout/viz.py` | pydeck map layers |

**Score** = weighted nearby demand drivers (offices, colleges, metro, apartments, shops...)
minus a competition penalty, rescaled to 0-100. Weights live in `scout/config.py`.

**Limits:** a transparent proxy from place counts. No real footfall, rent, income or
population data; OpenStreetMap coverage varies by area. Use it to shortlist spots.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
All network calls are mocked, so tests need no internet or API keys.

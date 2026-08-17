# 01_BACKEND — Flask Application Layer

Owner: **Backend** team member. This folder is the web/API layer of PACE — it
serves pages, reads the ML pipeline's output files, and returns them as
rendered HTML (and two data exports). It does **not** contain any ML
training code and does **not** contain any HTML/CSS/JS.

## What's in here

```
01_BACKEND/
├── app.py       — the entire Flask application (routes, auth, data-access helpers)
├── routes/      — reserved for future route modules (currently empty, see below)
├── services/    — reserved for future service/helper modules (currently empty, see below)
└── README.md    — this file
```

### Why `routes/` and `services/` are currently empty

`app.py` is a single working Flask file (~370 lines) with every route defined
in it directly — that was the existing, tested structure before this reorg,
and it was **not** split apart during the restructure. Splitting a working
file into `routes/`+`services/` modules is a real refactor with real risk of
breaking something (blueprint registration, circular imports, etc.), and the
instruction for this reorg was to move files into clear folders, not to
rewrite working code. These two folders are provided so the backend owner
can grow into them deliberately — for example, moving `/prediction` and the
`/api/*` endpoints into `routes/api.py` and `routes/pages.py`, and moving the
CSV/JSON-reading helpers (`get_final_data`, `get_json`, etc.) into
`services/data_access.py` — without anyone else's work colliding with it.
Do this as a dedicated PR/commit, not mixed in with feature work.

## How to run it

From the **repository root**:

```bash
pip install -r requirements.txt
python 01_BACKEND/app.py
```

Then open `http://127.0.0.1:5000`. Log in with `admin` / `admin123` (demo
credentials only — see `USERS` dict in `app.py`; change via `SECRET_KEY`/a
real user table before any real deployment).

The app will run and every page will load even with **no data** — pages show
an honest "no data yet" empty state until the ML pipeline has been run (see
`02_ML_AI/README.md`). Nothing is invented to fill the gap.

## Available routes

| Route | Method | Purpose |
|---|---|---|
| `/welcome` | GET | Public landing page |
| `/login`, `/logout` | GET/POST | Auth |
| `/` | GET | Dashboard (KPIs, charts, top markets) |
| `/customers` | GET | Customer list table |
| `/segmentation` | GET | Segment profiles + metrics |
| `/prediction` | GET/POST | Manual churn-probability prediction form |
| `/reports` | GET | Business recommendations + EDA insights |
| `/analytics` | GET | Category/payment/state/satisfaction charts |
| `/model-comparison` | GET | Churn model + segmentation model comparison tables |
| `/forecasting` | GET | Customer lifecycle revenue forecast |
| `/recommendations` | GET | Per-customer product recommendations (searchable) |
| `/settings` | GET | Account/appearance info |
| `/export/excel`, `/export/pdf` | GET | Download current customer data |
| `/api/customers` | GET | JSON of all customers (used by front-end Chart.js) |

## Expected inputs/outputs

- **Reads** CSV/JSON files written by the ML pipeline from:
  - `04_DATA_TESTING/datasets/processed/` (customer tables)
  - `04_DATA_TESTING/reports/` (metrics + insights JSON)
  - `02_ML_AI/models/` (`best_churn_model.pkl`, `feature_columns.pkl`)
- **Renders** templates from `03_FRONTEND/templates/` and serves assets from
  `03_FRONTEND/static/` — both are passed explicitly to `Flask(__name__, ...)`
  in `app.py` since they no longer sit next to `app.py` on disk.
- `/prediction` POST expects form fields matching the churn model's trained
  feature columns (`joblib.load(feature_columns.pkl)`); missing fields default
  to 0, matching the model's original behaviour.

## How it communicates with ML

The backend never imports or runs any ML code directly — it only **reads
static files** the pipeline (`02_ML_AI/run_pipeline.py`) already produced
(CSV/JSON/`.pkl`). This is a deliberate offline-batch-then-serve
architecture: retrain/re-run the pipeline, restart Flask, the dashboard
picks up the new numbers. There is no live retraining triggered from the web
app.

## Which files are safe to modify

- `app.py` — yes, this is the backend owner's file. Please do not rename
  existing route functions/URLs (`03_FRONTEND` templates and any bookmarks
  reference them via `url_for(...)`) without coordinating with the frontend
  owner.
- Do not edit files under `02_ML_AI/` or `03_FRONTEND/` from here — if a
  change requires touching those, document the dependency and coordinate
  with that folder's owner instead of editing directly.

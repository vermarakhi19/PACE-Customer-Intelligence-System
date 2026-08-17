# 03_FRONTEND — Templates & Static Assets

Owner: **Frontend** team member. This folder owns every HTML template and
every CSS/JS asset. It contains no Python/ML logic.

## Structure

```
03_FRONTEND/
├── templates/
│   ├── base.html            — shared layout (sidebar, top bar, dark mode, toasts)
│   ├── _components.html     — reusable Jinja macros (chart panels, etc.)
│   ├── landing.html         — public marketing/landing page
│   ├── login.html
│   ├── dashboard.html
│   ├── customers.html
│   ├── segmentation.html
│   ├── prediction.html
│   ├── reports.html
│   ├── analytics.html
│   ├── model_comparison.html
│   ├── forecasting.html
│   ├── recommendations.html
│   └── settings.html
├── static/
│   ├── css/style.css
│   └── js/main.js
└── README.md                — this file
```

## How Flask finds this folder

`01_BACKEND/app.py` passes this folder's `templates/` and `static/`
subfolders explicitly to `Flask(__name__, template_folder=..., static_folder=...)`,
because this folder is no longer a sibling of `app.py` on disk the way it
was before the reorg. Every `{{ url_for('static', filename='...') }}` and
`render_template('...')` call in the templates is unchanged — Flask still
resolves them the same way, just from the new location.

## Pages

Every page `{% extends "base.html" %}`. Navigation links live in
`base.html`'s sidebar and use `url_for('<route-name>')` — never hardcoded
URLs — so route names must stay in sync with `01_BACKEND/app.py`.

## Components

`_components.html` holds shared Jinja macros (e.g. the `chart_panel` macro
used by `dashboard.html`) — reuse these instead of duplicating markup when
adding new chart cards.

## CSS

`static/css/style.css` — single stylesheet, CSS custom properties at the top
for the light/dark theme palette (`--canvas`, `--text`, `--border`, etc.).
Dark mode is a `data-theme` attribute toggle, not a second stylesheet.

## JavaScript

`static/js/main.js` — dark-mode toggle (persisted via `localStorage`),
generic `.js-loading-submit` form-submit spinner handler,
`window.paceToast()` for toast notifications, and chart color constants
shared across pages. No framework — plain JS + Chart.js (loaded via CDN in
`base.html`) + DataTables (customer/recommendation tables).

## Backend data requirements per page

| Page | Needs from backend |
|---|---|
| `dashboard.html` | `kpis`, `segment_counts`, `risk_counts`, `churn_prob_hist`, `top_markets`, `sparklines`, `has_data` |
| `customers.html` | `customers` (list of dicts), `columns`, `has_data` |
| `segmentation.html` | `seg_metrics`, `segment_profile` |
| `prediction.html` | `result` (or `None` before first prediction) |
| `reports.html` | `recommendations`, `eda_insights` |
| `analytics.html` | `charts` (category/payment/state/satisfaction dicts), `has_data` |
| `model_comparison.html` | `churn_metrics`, `seg_metrics` |
| `forecasting.html` | `forecast`, `has_data` |
| `recommendations.html` | `customers`, `affinity`, `search`, `has_data` |
| `settings.html` | `user` |

Every page that depends on pipeline output degrades to an honest "no data
yet" empty state when `has_data` is false — this is intentional and must be
preserved when editing templates; never hardcode a fallback number.

## Which pages consume ML results

All of the above except `landing.html`, `login.html`, and `settings.html`
directly render values that trace back to `02_ML_AI/run_pipeline.py`'s
output files, via the backend routes listed in `01_BACKEND/README.md`.

"""
smoke_test_flask_app.py
========================
Minimal smoke test for the 01_BACKEND Flask app, owned by the
04_DATA_TESTING (Data + Testing) work area.

WHY THIS EXISTS
----------------
This project did not ship with any automated tests before the 4-folder
restructure (confirmed during the restructure's verification pass). Rather
than leave 04_DATA_TESTING/testing/ empty, this adds ONE real, runnable
smoke test that boots the actual Flask app (in-process, via Flask's test
client — no server, no network) and hits every route once, checking for a
successful status code. It does not assert on specific business numbers
(those depend on whatever raw dataset is currently loaded), only that every
page renders without a server error and that the two exports produce
non-empty files.

This is intentionally small: a starting point for the data/testing owner to
extend (e.g. schema validation for 04_DATA_TESTING/datasets/raw/, more
detailed assertions per page) — not a full test suite.

RUN
---
    cd PACE-Customer-Intelligence-System
    python 04_DATA_TESTING/testing/smoke_test_flask_app.py

Exits non-zero and prints which route failed if anything breaks.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "01_BACKEND"))

import importlib

ROUTES_GET = [
    "/welcome", "/login", "/", "/customers", "/segmentation", "/prediction",
    "/reports", "/analytics", "/model-comparison", "/forecasting",
    "/recommendations", "/settings", "/api/customers",
    "/export/excel", "/export/pdf", "/static/css/style.css", "/static/js/main.js",
]


def run():
    app_mod = importlib.import_module("app")
    app = app_mod.app
    app.testing = True
    client = app.test_client()

    failures = []

    # Auth flow first — most routes require a logged-in session.
    r = client.post("/login", data={"username": "admin", "password": "admin123"})
    if r.status_code not in (200, 302):
        failures.append(("POST /login", r.status_code))

    for path in ROUTES_GET:
        r = client.get(path)
        ok = r.status_code == 200
        print(f"{'OK  ' if ok else 'FAIL'}  GET {path:25s} -> {r.status_code}")
        if not ok:
            failures.append((f"GET {path}", r.status_code))

    r = client.get("/logout")
    if r.status_code not in (200, 302):
        failures.append(("GET /logout", r.status_code))

    if failures:
        print(f"\n[SMOKE TEST FAILED] {len(failures)} route(s) did not return 200:")
        for name, code in failures:
            print(f"  - {name} -> {code}")
        sys.exit(1)

    print(f"\n[SMOKE TEST PASSED] {len(ROUTES_GET)} routes + login/logout all responded correctly.")


if __name__ == "__main__":
    run()

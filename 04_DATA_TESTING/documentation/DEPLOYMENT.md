# Deployment Guide

## Option A — College demo / local network
```bash
python 01_BACKEND/app.py
```
Flask's dev server binds to `0.0.0.0:5000`, so anyone on the same Wi-Fi
can reach it via your machine's LAN IP: `http://<your-ip>:5000`.
Fine for a viva demo, **not** for production (dev server is single-threaded
and has no HTTPS).

## Option B — Production-style deployment (Gunicorn + Nginx, Linux)

1. Install Gunicorn:
   ```bash
   pip install gunicorn
   ```
2. Run with multiple workers:
   ```bash
   cd PACE-Customer-Intelligence-System
   gunicorn --chdir 01_BACKEND --workers 4 --bind 0.0.0.0:8000 app:app
   ```
   (`app.py` computes all its paths — including the frontend's
   `03_FRONTEND/templates` and `03_FRONTEND/static` — relative to the repo
   root, so `--chdir 01_BACKEND` only affects how Gunicorn imports the
   `app` module; Flask itself still finds templates/static correctly.)
3. Put Nginx in front as a reverse proxy (handles HTTPS via Let's Encrypt,
   static file caching, and gzip):
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
       location /static/ {
           alias /path/to/PACE-Customer-Intelligence-System/03_FRONTEND/static/;
       }
   }
   ```
4. Set a real secret key via environment variable instead of the dev
   default:
   ```bash
   export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"
   ```

## Option C — Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
EXPOSE 8000
CMD ["gunicorn", "--chdir", "01_BACKEND", "--workers", "4", "--bind", "0.0.0.0:8000", "app:app"]
```
```bash
docker build -t retailpulse-india .
docker run -p 8000:8000 -v $(pwd)/04_DATA_TESTING/datasets:/app/04_DATA_TESTING/datasets retailpulse-india
```

## Option D — Cloud (Render / Railway / PythonAnywhere / AWS EC2)
Any of these work with the Gunicorn command above as the start command.
For AWS/GCP, an EC2/Compute VM with Nginx + Gunicorn (Option B) is the
most transparent path for a viva demo ("here's exactly what's running").

## Before going live
- Move the hard-coded `admin/admin123` credential to a real user table
  with hashed passwords (the `USERS` dict in `app.py` is a placeholder).
- Turn `debug=True` off in `app.py` (`app.run(debug=False, ...)`).
- Re-run `02_ML_AI/run_pipeline.py` on a schedule (cron / Airflow) if the
  underlying customer data refreshes periodically, so the dashboard
  doesn't go stale.

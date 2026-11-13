# DualGuard

Hybrid two-stage network intrusion detector on CIC-IDS2017.

- **Stage 1:** Autoencoder trained on benign flows; high reconstruction error = anomaly.
- **Stage 2:** LightGBM multiclass classifier on known attack families.
- **Verdict:** BENIGN / KNOWN_ATTACK / UNKNOWN_ANOMALY.

## Setup (macOS)

```bash
cd /Volumes/user-data/Projects/DualGuard
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

## Layout

```
config.yaml          all parameters and paths
src/dualguard/       library code (data, models, thresholds, evaluation)
api/                 FastAPI service
web/                 React dashboard
tests/               pytest
data/raw/            CIC-IDS2017 MachineLearningCVE CSVs (not in git)
models/              trained artefacts (not in git)
reports/             evaluation tables and figures
```

## Dashboard (React)

Two terminals:

```bash
# 1. API
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# 2. Dashboard
cd web
npm install        # first time only
npm run dev        # http://localhost:5173
```

## Docker (API + dashboard, one command)

```bash
python scripts/08_export_replay.py   # compact replay data for the image
python scripts/freeze_runtime.py     # pin image libraries to the training versions
docker compose up -d --build
```

- Dashboard: http://localhost:8080
- API docs:  http://localhost:8000/docs
- Stop:      `docker compose down`

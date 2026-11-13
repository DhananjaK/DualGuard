"""Step 7: DualGuard REST API (FastAPI).

Run from the project root:
    uvicorn api.main:app --reload --port 8000
Interactive docs:  http://localhost:8000/docs
"""
import io
import math
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dualguard.config import load_config
from dualguard.data import FAMILY_MAP
from dualguard.service import MODES, Detector, true_labels

MAX_JSON_FLOWS = 10_000
MAX_CSV_ROWS = 200_000
MAX_RETURNED_ROWS = 5_000


class PredictRequest(BaseModel):
    model: Literal["A", "B"] = "B"
    mode: Literal["parallel", "cascade"] | None = None
    flows: list[dict[str, float | int | str | None]] = Field(..., min_length=1, max_length=MAX_JSON_FLOWS)


def _clean(obj):
    """JSON cannot hold NaN/inf: turn them into null, recursively."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    return _clean(df.astype(object).where(df.notna(), None).to_dict(orient="records"))


def create_app(models_dir: Path | None = None, tables_dir: Path | None = None,
               processed_dir: Path | None = None) -> FastAPI:
    cfg = load_config()
    models_dir = Path(models_dir or cfg["paths"]["models_dir"])
    tables_dir = Path(tables_dir or cfg["paths"]["tables_dir"])
    processed_dir = Path(processed_dir or cfg["paths"]["processed_dir"])

    detectors: dict[str, Detector] = {
        s: Detector(models_dir / s, s) for s in ("A", "B") if (models_dir / s / "thresholds.json").exists()
    }
    replay_cache: dict[str, pd.DataFrame] = {}

    app = FastAPI(title="DualGuard API", version="1.0",
                  description="Hybrid Autoencoder + LightGBM network intrusion detector (CIC-IDS2017).")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    def get(model: str) -> Detector:
        if model not in detectors:
            raise HTTPException(404, f"model {model!r} not loaded; available: {sorted(detectors)}")
        return detectors[model]

    def run(det: Detector, raw: pd.DataFrame, mode: str | None) -> dict:
        t0 = time.perf_counter()
        try:
            df, valid = det.prepare(raw)
        except KeyError as e:
            raise HTTPException(422, str(e.args[0]))
        if not valid.any():
            raise HTTPException(422, "no valid rows: every row has a missing, non-numeric or infinite feature")
        labels = true_labels(raw)
        res = det.predict(df[valid], mode)
        if labels is not None:
            res.insert(0, "true_label", labels[valid].to_numpy())
            res.insert(1, "true_family", res["true_label"].map(FAMILY_MAP).fillna("?"))
        mode_used = mode or det.th["default_mode"]
        return {
            "model": det.scenario, "mode": mode_used,
            "n_received": len(raw), "n_scored": int(valid.sum()), "n_invalid": int((~valid).sum()),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            "verdict_counts": res["verdict"].value_counts().to_dict(),
            "family_counts": res["family"].value_counts().to_dict(),
            "truncated": len(res) > MAX_RETURNED_ROWS,
            "results": _records(res.head(MAX_RETURNED_ROWS).reset_index(names="input_row")),
        }

    @app.get("/health")
    def health():
        return {"status": "ok" if detectors else "no models", "models_loaded": sorted(detectors)}

    @app.get("/models")
    def models():
        return _clean([d.info() for d in detectors.values()])

    @app.post("/predict")
    def predict(req: PredictRequest):
        return run(get(req.model), pd.DataFrame(req.flows), req.mode)

    @app.post("/predict_csv")
    async def predict_csv(file: UploadFile = File(...), model: Literal["A", "B"] = Form("B"),
                          mode: Literal["parallel", "cascade"] | None = Form(None)):
        content = await file.read()
        try:
            raw = pd.read_csv(io.BytesIO(content), encoding_errors="replace", low_memory=False,
                              nrows=MAX_CSV_ROWS + 1)
        except Exception as e:  # noqa: BLE001 - report any parse error to the user
            raise HTTPException(400, f"could not read CSV: {e}")
        if len(raw) > MAX_CSV_ROWS:
            raise HTTPException(413, f"CSV has more than {MAX_CSV_ROWS:,} rows")
        return run(get(model), raw, mode)

    @app.get("/replay")
    def replay(model: Literal["A", "B"] = "B", start: int = Query(0, ge=0),
               limit: int = Query(200, ge=1, le=2000), source_file: str | None = None):
        """Pre-computed TEST-set scores (Step 6), in capture order, for the live-replay view."""
        if model not in replay_cache:
            compact = processed_dir / f"replay_{model}.parquet"      # small, self-contained (Docker image)
            scored = processed_dir / f"test_scored_{model}.parquet"
            if compact.exists():
                sc = pd.read_parquet(compact)
            elif scored.exists():                                    # development: join capture file names
                sc = pd.read_parquet(scored)
                files = pd.read_parquet(processed_dir / "flows.parquet", columns=["SourceFile"])["SourceFile"]
                sc["SourceFile"] = files.iloc[sc["row"]].to_numpy()
            else:
                raise HTTPException(404, "no test scores; run scripts/06_evaluate_test.py --open-test")
            replay_cache[model] = sc.sort_values("row").reset_index(drop=True)
        sc = replay_cache[model]
        if source_file:
            sc = sc[sc["SourceFile"] == source_file]
        page = sc.iloc[start:start + limit]
        return {"model": model, "total": len(sc), "start": start,
                "source_files": sorted(replay_cache[model]["SourceFile"].unique()),
                "rows": _records(page)}

    @app.get("/evidence/{model}")
    def evidence(model: Literal["A", "B"]):
        """Evaluation result tables (reports/tables) as JSON, for the Evidence page."""
        names = ["test_summary", "test_detection", "test_naming", "val_detection",
                 "ae_history", "lgbm_importance", "family_counts", "ae_val_errors"]
        out = {}
        for n in names:
            p = tables_dir / f"{n}_{model}.csv"
            if p.exists():
                if n == "test_naming":       # two header rows: (mode, output)
                    df = pd.read_csv(p, header=[0, 1], index_col=0)
                    df.columns = [f"{a}|{b}" for a, b in df.columns]
                    df = df.reset_index(names="true")
                else:
                    df = pd.read_csv(p).rename(columns={"Unnamed: 0": "name"})   # saved index -> "name"
                if n == "lgbm_importance":
                    df = df.head(15)
                out[n] = _records(df)
        if model == "B" and (tables_dir / "posthoc_B_by_file.csv").exists():
            out["posthoc_by_file"] = _records(pd.read_csv(tables_dir / "posthoc_B_by_file.csv"))
        return out

    samples_dir = Path(__file__).resolve().parents[1] / "samples"

    @app.get("/samples")
    def samples():
        return sorted(p.name for p in samples_dir.glob("*.csv"))

    @app.get("/samples/{name}")
    def sample_file(name: str):
        path = (samples_dir / name).resolve()
        if path.parent != samples_dir.resolve() or not path.exists():   # no ../ path tricks
            raise HTTPException(404, "sample not found")
        return FileResponse(path, media_type="text/csv", filename=name)

    return app


app = create_app()

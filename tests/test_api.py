"""API tests against tiny models built on synthetic data (no real training needed)."""
import io
import json

import lightgbm  # noqa: F401
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from dualguard.autoencoder import reconstruction_error, save_autoencoder, train_autoencoder
from dualguard.classifier import save_classifier, train_classifier
from dualguard.hybrid import fit_thresholds
from dualguard.preprocess import Preprocessor

FEATS = ["Flow Duration", "Total Fwd Packets", "Flow Bytes/s", "Packet Length Std"]


def _flows(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=FEATS)
    fam = rng.choice(["BENIGN", "DoS"], size=n, p=[0.85, 0.15])
    df.loc[fam == "DoS", "Flow Bytes/s"] += 6
    df["Label"] = np.where(fam == "DoS", "DoS Hulk", "BENIGN")
    df["Family"], df["Day"], df["SourceFile"] = fam, "Monday", "x.csv"
    return df


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    tr, va = _flows(2000, 0), _flows(600, 1)
    for s in ("A", "B"):
        d = root / s
        pre = Preprocessor().fit(tr); pre.save(d / "preprocessor.joblib")
        Xtr, Xva = pre.transform(tr), pre.transform(va)
        ae, _ = train_autoencoder(Xtr[tr["Family"] == "BENIGN"], Xva[va["Family"] == "BENIGN"],
                                  {"hidden": [8], "bottleneck": 2, "epochs": 5, "batch_size": 128,
                                   "lr": 0.01, "patience": 3, "threads": 1}, 0, log=lambda *_: None)
        save_autoencoder(ae, d / "autoencoder.pt")
        clf = train_classifier(Xtr, tr["Family"].to_numpy(), Xva, va["Family"].to_numpy(),
                               {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 7,
                                "min_child_samples": 5, "feature_fraction": 1.0, "bagging_fraction": 1.0,
                                "bagging_freq": 0, "early_stopping_rounds": 10,
                                "class_weight": "sqrt_balanced", "threads": 1}, 0, log=lambda *_: None)
        save_classifier(clf, d / "lightgbm.joblib")
        th = fit_thresholds(reconstruction_error(ae, Xva), clf.predict_proba(Xva), clf.classes_,
                            va["Family"].to_numpy(), 0.02, 0.5)
        th["default_mode"] = "parallel"
        (d / "thresholds.json").write_text(json.dumps(th))
    tables = tmp_path_factory.mktemp("tables")
    pd.DataFrame({"method": ["parallel"], "macro_detection": [0.9]}).to_csv(tables / "test_summary_B.csv", index=False)
    return TestClient(create_app(models_dir=root, tables_dir=tables, processed_dir=root))


def test_health_and_models(client):
    assert client.get("/health").json() == {"status": "ok", "models_loaded": ["A", "B"]}
    info = client.get("/models").json()
    assert {m["scenario"] for m in info} == {"A", "B"}
    assert info[0]["features"] == FEATS and info[0]["default_mode"] == "parallel"


def test_predict_json_returns_a_verdict_per_flow(client):
    flows = _flows(20, 5)[FEATS].to_dict(orient="records")
    r = client.post("/predict", json={"model": "B", "flows": flows}).json()
    assert r["n_scored"] == 20 and r["mode"] == "parallel"
    assert {x["verdict"] for x in r["results"]} <= {"BENIGN", "KNOWN_ATTACK", "UNKNOWN_ANOMALY"}


def test_obvious_dos_is_flagged(client):
    flow = {f: 0.0 for f in FEATS} | {"Flow Bytes/s": 8.0}
    r = client.post("/predict", json={"model": "A", "flows": [flow]}).json()
    assert r["results"][0]["verdict"] != "BENIGN"


def test_missing_feature_is_rejected(client):
    r = client.post("/predict", json={"flows": [{"Flow Duration": 1.0}]})
    assert r.status_code == 422 and "missing" in r.json()["detail"]


def test_invalid_rows_are_counted_not_crashed(client):
    flows = _flows(3, 6)[FEATS].to_dict(orient="records")
    flows[1]["Flow Bytes/s"] = "Infinity"
    r = client.post("/predict", json={"flows": flows}).json()
    assert r["n_scored"] == 2 and r["n_invalid"] == 1


def test_predict_csv_with_labels_and_cascade(client):
    df = _flows(50, 7).drop(columns=["Family", "Day", "SourceFile"])
    df.columns = [" " + c for c in df.columns]                    # raw CIC headers have leading spaces
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = client.post("/predict_csv", files={"file": ("x.csv", buf, "text/csv")},
                    data={"model": "B", "mode": "cascade"}).json()
    assert r["mode"] == "cascade" and r["n_scored"] == 50
    assert set(r["results"][0]) >= {"true_label", "true_family", "verdict", "ae_ratio"}


def test_unknown_model_and_evidence(client):
    assert client.post("/predict", json={"model": "C", "flows": [{}]}).status_code == 422
    assert client.get("/evidence/B").json()["test_summary"][0]["macro_detection"] == 0.9

import numpy as np
import pandas as pd

from dualguard.data import FAMILY_MAP, clean, normalise_labels
from dualguard.splits import check_disjoint, split_scenario_b


def test_web_attack_labels_normalised():
    raw = pd.Series(["Web Attack � XSS", " BENIGN ", "Web Attack � Sql Injection"])
    out = normalise_labels(raw)
    assert list(out) == ["Web Attack - XSS", "BENIGN", "Web Attack - Sql Injection"]
    assert all(lbl in FAMILY_MAP for lbl in out)


def _toy():
    return pd.DataFrame({
        "f1": [1.0, 1.0, np.inf, 2.0, 3.0, 3.0],
        "f2": [5.0, 5.0, 1.0, 2.0, 4.0, 4.0],
        "Label": ["BENIGN", "BENIGN", "BENIGN", "DDoS", "BENIGN", "DDoS"],
        "Day": ["Monday"] * 6,
        "SourceFile": ["x.csv"] * 6,
    })


def test_clean_rules():
    df, rep = clean(_toy())
    assert rep["rows_dropped_nan_or_inf"] == 1          # the inf row
    assert rep["rows_dropped_conflicting_labels"] == 2  # (3,4) labelled both ways
    assert rep["rows_dropped_duplicates"] == 1          # second (1,5) BENIGN
    assert len(df) == 2
    assert df["f1"].dtype == np.float32


def test_scenario_b_is_temporal_and_disjoint():
    n = 200
    df = pd.DataFrame({
        "Day": ["Monday"] * 100 + ["Thursday"] * 100,
        "Family": (["BENIGN", "DoS"] * 50) + (["BENIGN", "WebAttack"] * 50),
    })
    cfg = {"data": {"scenario_b": {"train_days": ["Monday"], "test_days": ["Thursday"],
                                   "val_fraction_of_train": 0.2}}}
    s = split_scenario_b(df, cfg, seed=0)
    check_disjoint(s)
    assert set(df["Day"].iloc[s["test"]]) == {"Thursday"}
    assert "WebAttack" not in set(df["Family"].iloc[s["train"]])
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == n

import numpy as np

from dualguard.hybrid import BENIGN, KNOWN, UNKNOWN, decide, fit_thresholds, threshold_at_budget
from dualguard.metrics import detection_table

CLASSES = np.array(["BENIGN", "DoS"])


def test_threshold_spends_exactly_the_budget():
    scores = np.arange(1000, dtype=float)              # 1000 benign flows
    t = threshold_at_budget(scores, 0.02)
    assert (scores > t).sum() == 20                    # 2% of 1000


def test_already_flagged_flows_use_up_budget_first():
    scores = np.arange(1000, dtype=float)
    pre = np.zeros(1000, bool); pre[:5] = True         # another component already alarmed on 5
    t = threshold_at_budget(scores, 0.02, already_flagged=pre)
    assert pre.sum() + (scores[~pre] > t).sum() == 20


def test_budget_already_spent_means_never_fire():
    pre = np.ones(100, bool)
    assert threshold_at_budget(np.zeros(100), 0.02, already_flagged=pre) == float("inf")


def _case():
    # flow 0: normal.  1: LightGBM sure it's DoS, AE calm.  2: AE screams, LightGBM says benign.
    err = np.array([0.1, 0.1, 9.0])
    proba = np.array([[0.99, 0.01], [0.05, 0.95], [0.9, 0.1]])
    th = {"tau": 0.5, "ae": 1.0, "ae_parallel": 1.0}
    return err, proba, th


def test_parallel_logic():
    err, proba, th = _case()
    v, fam = decide(err, proba, CLASSES, th, "parallel")
    assert list(v) == [BENIGN, KNOWN, UNKNOWN]
    assert list(fam) == [BENIGN, "DoS", "?"]


def test_cascade_gate_hides_attacks_the_ae_misses():
    err, proba, th = _case()
    v, _ = decide(err, proba, CLASSES, th, "cascade")
    assert list(v) == [BENIGN, BENIGN, UNKNOWN]        # flow 1 never reaches LightGBM


def test_combined_false_alarms_stay_within_budget():
    rng = np.random.default_rng(0)
    n = 5000
    y = np.array(["BENIGN"] * n)
    err = rng.exponential(1, n)
    p_dos = rng.beta(0.5, 8, n)                        # mostly low, a few confident "DoS"
    proba = np.c_[1 - p_dos, p_dos]
    th = fit_thresholds(err, proba, CLASSES, y, budget=0.02, tau=0.5)
    for mode in ("cascade", "parallel"):
        v, _ = decide(err, proba, CLASSES, th, mode)
        assert (v != BENIGN).mean() <= 0.02 + 1e-9


def test_detection_table_macro():
    y = np.array(["BENIGN"] * 4 + ["DoS"] * 2 + ["Bot"] * 2)
    alarm = np.array([0, 0, 0, 1, 1, 1, 1, 0], bool)
    tab = detection_table(y, {"m": alarm})
    assert tab.loc["BENIGN", "m"] == 0.25 and tab.loc["DoS", "m"] == 1.0 and tab.loc["Bot", "m"] == 0.5
    assert tab.loc["MACRO (attacks)", "m"] == 0.75

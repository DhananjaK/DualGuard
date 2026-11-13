import numpy as np

from dualguard.classifier import load_classifier, save_classifier, sqrt_balanced_weights, train_classifier

P = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 5,
     "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
     "early_stopping_rounds": 10, "class_weight": "sqrt_balanced", "threads": 1}


def _data(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.choice(["BENIGN", "DoS", "PortScan"], size=n, p=[0.8, 0.15, 0.05])
    X = rng.normal(size=(n, 5)).astype(np.float32)
    X[y == "DoS", 0] += 4          # DoS: feature 0 high
    X[y == "PortScan", 1] -= 4     # PortScan: feature 1 low
    return X, y


def test_sqrt_balanced_weights():
    y = np.array(["a"] * 100 + ["b"] * 25 + ["c"] * 1)
    w = sqrt_balanced_weights(y)
    assert w[0] == 1.0 and w[100] == 2.0 and w[-1] == 10.0   # sqrt(100/100), sqrt(100/25), sqrt(100/1)


def test_learns_separable_families_and_roundtrips(tmp_path):
    X_tr, y_tr = _data(3000, 0)
    X_va, y_va = _data(1000, 1)
    m = train_classifier(X_tr, y_tr, X_va, y_va, P, seed=0, log=lambda *_: None)
    assert (m.predict(X_va) == y_va).mean() > 0.95
    proba = m.predict_proba(X_va)
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-6)
    save_classifier(m, tmp_path / "m.joblib")
    np.testing.assert_allclose(load_classifier(tmp_path / "m.joblib").predict_proba(X_va), proba)


def test_unseen_family_is_forced_into_a_known_one():
    """The blind spot: an attack type never trained on still gets a KNOWN label."""
    X_tr, y_tr = _data(3000, 0)
    X_va, y_va = _data(1000, 1)
    m = train_classifier(X_tr, y_tr, X_va, y_va, P, seed=0, log=lambda *_: None)
    novel = np.random.default_rng(5).normal(size=(50, 5)).astype(np.float32)
    novel[:, 4] += 8               # new attack: feature 4 extreme, never seen in training
    assert set(m.predict(novel)) <= {"BENIGN", "DoS", "PortScan"}

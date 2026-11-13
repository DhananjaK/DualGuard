import numpy as np
import torch

from dualguard.autoencoder import (Autoencoder, load_autoencoder, reconstruction_error,
                                   save_autoencoder, train_autoencoder)

P = {"hidden": [8], "bottleneck": 2, "epochs": 15, "batch_size": 64, "lr": 0.01,
     "patience": 5, "threads": 1}


W = np.random.default_rng(99).normal(size=(2, 6))   # fixed mixing matrix


def _normal(n, seed):
    """'Normal traffic': 6 features that really depend on only 2 hidden factors."""
    z = np.random.default_rng(seed).normal(size=(n, 2))
    return (z @ W).astype(np.float32)


def test_shapes_mirror():
    m = Autoencoder(68, [64, 32], 16)
    assert m(torch.zeros(5, 68)).shape == (5, 68)
    assert m.encoder(torch.zeros(5, 68)).shape == (5, 16)


def test_learns_normal_and_flags_outliers():
    X_tr, X_va = _normal(2000, 0), _normal(500, 1)
    model, hist = train_autoencoder(X_tr, X_va, P, seed=0, log=lambda *_: None)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    normal_err = reconstruction_error(model, X_va)
    outliers = np.random.default_rng(2).normal(0, 3, size=(200, 6)).astype(np.float32)
    assert np.median(reconstruction_error(model, outliers)) > 5 * np.median(normal_err)


def test_same_seed_same_model():
    X_tr, X_va = _normal(500, 0), _normal(100, 1)
    a, _ = train_autoencoder(X_tr, X_va, P, seed=7, log=lambda *_: None)
    b, _ = train_autoencoder(X_tr, X_va, P, seed=7, log=lambda *_: None)
    np.testing.assert_allclose(reconstruction_error(a, X_va), reconstruction_error(b, X_va), rtol=1e-5)


def test_save_load(tmp_path):
    m = Autoencoder(6, [8], 2)
    save_autoencoder(m, tmp_path / "ae.pt")
    x = np.random.default_rng(0).normal(size=(10, 6)).astype(np.float32)
    np.testing.assert_allclose(reconstruction_error(m, x), reconstruction_error(load_autoencoder(tmp_path / "ae.pt"), x))

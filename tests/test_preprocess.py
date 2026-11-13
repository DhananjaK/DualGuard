import numpy as np
import pandas as pd
import pytest

from dualguard.preprocess import Preprocessor, signed_log1p


def _df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "Destination Port": rng.integers(0, 65535, 100).astype(float),
        "bytes": rng.exponential(1e6, 100),
        "win": np.r_[np.full(50, -1.0), rng.integers(0, 65535, 50)],
        "const": np.zeros(100),
        "Label": ["BENIGN"] * 100, "Family": ["BENIGN"] * 100,
        "Day": ["Monday"] * 100, "SourceFile": ["x.csv"] * 100,
    })


def test_signed_log1p_keeps_sign_and_order():
    x = np.array([-1.0, 0.0, 1.0, 1e9])
    y = signed_log1p(x)
    assert y[0] < 0 and y[1] == 0 and np.all(np.diff(y) > 0)
    assert y[-1] < 25            # 1e9 squashed to ~20.7


def test_fit_drops_config_and_constant_columns():
    pre = Preprocessor(["Destination Port"]).fit(_df())
    assert pre.features_ == ["bytes", "win"]
    assert pre.constant_features_ == ["const"]


def test_train_is_standardised():
    df = _df()
    X = Preprocessor(["Destination Port"]).fit(df).transform(df)
    assert X.dtype == np.float32
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-3)


def test_missing_feature_raises():
    df = _df()
    pre = Preprocessor().fit(df)
    with pytest.raises(KeyError):
        pre.transform(df.drop(columns="bytes"))


def test_save_load_roundtrip(tmp_path):
    df = _df()
    pre = Preprocessor().fit(df)
    pre.save(tmp_path / "p.joblib")
    np.testing.assert_array_equal(pre.transform(df), Preprocessor.load(tmp_path / "p.joblib").transform(df))


def test_clip_z_caps_extremes():
    df = _df()
    df.loc[0, "bytes"] = 1e15                    # absurd outlier
    unclipped = Preprocessor(["Destination Port"]).fit(df.iloc[1:]).transform(df)
    clipped = Preprocessor(["Destination Port"], clip_z=3).fit(df.iloc[1:]).transform(df)
    assert np.abs(unclipped).max() > 3
    assert np.abs(clipped).max() <= 3

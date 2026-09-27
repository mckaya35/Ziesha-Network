import numpy as np
import pandas as pd

from backtest import trade_outcomes
from export_pine import manual_proba, pine_source
from features import compute_features, feature_names
from labels import make_labels
from tests.conftest import make_ohlcv
from walkforward import fit, fold_positions, predict, signals_from, train_block


def test_fold_no_leak(cfg):
    idx = pd.date_range("2020-01-01", "2021-06-30 23:00", freq="1h", tz="UTC")
    H = cfg["label"]["H"]
    for m in ("2021-01-01", "2021-03-01"):
        ts = pd.Timestamp(m, tz="UTC")
        tr, te = fold_positions(idx, ts, ts + pd.DateOffset(months=1), cfg)
        assert idx[te[0]] == ts
        # eğitimdeki son etiketin ufku (t+H) testten en az H bar önce biter (embargo)
        assert tr[-1] + H < te[0] - H
        assert idx[tr[0]] >= ts - pd.DateOffset(months=12)


def test_train_block_uses_only_train(cfg):
    df = make_ohlcv(3000, seed=3)
    feat, lab = compute_features(df, cfg), make_labels(df, 24, 0.02, 0.01)
    NOF = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    oc = {s: trade_outcomes(df, NOF, s, cfg) for s in (1, -1)}
    pos = np.arange(200, 2000)
    a = train_block(feat, lab, oc, pos, cfg)
    df2 = df.copy(); df2.iloc[2000 + 24:] *= 1.5          # ufkun ötesindeki gelecek değişir
    feat2, lab2 = compute_features(df2, cfg), make_labels(df2, 24, 0.02, 0.01)
    oc2 = {s: trade_outcomes(df2, NOF, s, cfg) for s in (1, -1)}
    b = train_block(feat2, lab2, oc2, pos, cfg)
    for s in (1, -1):
        assert np.allclose(a[s][1].coef_, b[s][1].coef_) and a[s][2] == b[s][2]


def test_signals_conflict():
    s = signals_from(np.array([.7, .7, .2, .2]), np.array([.7, .2, .7, .2]), .6, .6)
    assert list(s) == [0, 1, -1, 0]
    assert list(signals_from(np.array([.9]), np.array([.9]), None, .6)) == [-1]


def test_manual_proba_equals_sklearn(cfg):
    """Pine'a gömülen formül (b + Σ w·(x-μ)/σ -> sigmoid) sklearn ile aynı olmalı."""
    df = make_ohlcv(1500, seed=5)
    feat, lab = compute_features(df, cfg), make_labels(df, 24, 0.02, 0.01)
    X = feat[feature_names(cfg)].iloc[200:1400]
    y = lab["label_up"].iloc[200:1400].to_numpy()
    sc, m = fit(X, y, 1.0)
    assert np.allclose(manual_proba(X.to_numpy(), sc, m), predict(sc, m, X), atol=1e-12)


def test_pine_source_shape(cfg):
    df = make_ohlcv(1500, seed=5)
    feat, lab = compute_features(df, cfg), make_labels(df, 24, 0.02, 0.01)
    X = feat[feature_names(cfg)].iloc[200:1400]
    sc, m = fit(X, lab["label_up"].iloc[200:1400].to_numpy(), 1.0)
    src = pine_source(cfg, {"BTCUSDT": {1: (sc, m, 0.55), -1: (sc, m, None)}},
                      {"generated": "x", "BTCUSDT": "a → b"}, demo=True)
    assert src.startswith("//@version=6")
    assert "ta.linreg(close, 168, 0)" in src and "ta.stdev(close, 72)" in src
    assert "barstate.isconfirmed" in src and "alertcondition" in src and "DEMO" in src
    assert "thr_BTCUSDT_dn = 1.01" in src

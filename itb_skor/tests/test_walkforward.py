import numpy as np
import pandas as pd

from export_pine import manual_proba, pine_source
from features import compute_features, feature_names
from labels import make_labels
from tests.conftest import make_ohlcv
from walkforward import (fit, fold_positions, holdout_bounds, predict, prepare, run, run_holdout,
                         signals_from, train_block)

NOF = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))


def raw(n=24 * 500, k=3):
    """k sembol, 500 gün; biri geç başlar (SUI benzeri)."""
    out = {}
    for i in range(k):
        df = make_ohlcv(n, seed=i)
        if i == k - 1:
            df = df.iloc[n // 3:]
        out[f"S{i}USDT"] = (df, NOF)
    return out


def test_fold_no_leak(cfg):
    idx = pd.date_range("2020-01-01", "2021-06-30 23:00", freq="1h", tz="UTC")
    H = cfg["label"]["H"]
    for m in ("2021-01-01", "2021-03-01"):
        ts = pd.Timestamp(m, tz="UTC")
        tr, te = fold_positions(idx, ts, ts + pd.DateOffset(months=1), cfg)
        assert idx[te[0]] == ts
        assert tr[-1] + H < te[0] - H                       # etiket ufku + embargo testten önce biter
        assert idx[tr[0]] >= ts - pd.DateOffset(months=12)


def test_first_test_auto_and_holdout(cfg):
    data = prepare(cfg, raw())
    first, wf_end, hold, end = holdout_bounds(cfg, data)
    assert first == pd.Timestamp("2021-01-01", tz="UTC")    # 2020-01-01 + 12 ay
    assert end - hold >= pd.Timedelta(days=cfg["walkforward"]["holdout_days"])
    assert wf_end + (cfg["label"]["H"] + 1) * pd.Timedelta("1h") == hold


def test_walkforward_never_touches_holdout(cfg, tmp_path):
    cfg = {**cfg, "exp_out": str(tmp_path)}
    data = prepare(cfg, raw())
    _, wf_end, hold, _ = holdout_bounds(cfg, data)
    preds, models, _ = run(cfg, data)
    assert preds["time"].max() < wf_end
    assert (pd.to_datetime(models["train_last"]) < hold).all()
    # WF işlemlerinin çıkışı da holdout'tan önce
    H = cfg["label"]["H"]
    assert preds["time"].max() + (H + 1) * pd.Timedelta("1h") <= hold
    # holdout tahminlerini değiştirmek (holdout fiyatlarını bozmak) WF sonuçlarını etkilemez
    r2 = {s: (df.copy(), f) for s, (df, f) in raw().items()}
    for s, (df, _) in r2.items():
        df.loc[df.index >= hold, ["open", "high", "low", "close"]] *= 1.7
    preds2, models2, _ = run(cfg, prepare(cfg, r2))
    pd.testing.assert_frame_equal(preds, preds2)
    hp, hm, h0, _ = run_holdout(cfg, data)
    assert hp["time"].min() >= h0 and (pd.to_datetime(hm["train_last"]) < h0).all()


def test_train_block_uses_only_train(cfg):
    data = prepare(cfg, raw(24 * 200, 2))
    pos = {s: np.arange(200, 3000) for s in data}
    a = train_block(data, pos, cfg)
    r2 = {s: (d["df"].copy(), NOF) for s, d in data.items()}
    for df, _ in r2.values():
        df.iloc[3000 + 24:] *= 1.5                      # etiket ufkunun ötesi değişir
    b = train_block(prepare(cfg, r2), pos, cfg)
    for s in (1, -1):
        assert np.allclose(a[s][1].coef_, b[s][1].coef_) and a[s][2] == b[s][2]


def test_pooled_single_model(cfg):
    data = prepare(cfg, raw(24 * 200, 2))
    blk = train_block(data, {s: np.arange(200, 3000) for s in data}, cfg)
    assert blk[1][1].coef_.shape == (1, len(feature_names(cfg)))
    assert blk[1][0].n_samples_seen_ > 3000                 # iki sembolün satırları birlikte


def test_signals_conflict():
    s = signals_from(np.array([.7, .7, .2, .2]), np.array([.7, .2, .7, .2]), .6, .6)
    assert list(s) == [0, 1, -1, 0]
    assert list(signals_from(np.array([.9]), np.array([.9]), None, .6)) == [-1]


def test_manual_proba_equals_sklearn(cfg):
    df = make_ohlcv(1500, seed=5)
    feat, lab = compute_features(df, cfg), make_labels(df, 24, 0.02, 0.01)
    X = feat[feature_names(cfg)].iloc[200:1400]
    sc, m = fit(X, lab["label_up"].iloc[200:1400].to_numpy(), 1.0)
    assert np.allclose(manual_proba(X.to_numpy(), sc, m), predict(sc, m, X), atol=1e-12)


def test_pine_source_shape(cfg):
    df = make_ohlcv(1500, seed=5)
    feat, lab = compute_features(df, cfg), make_labels(df, 24, 0.02, 0.01)
    X = feat[feature_names(cfg)].iloc[200:1400]
    sc, m = fit(X, lab["label_up"].iloc[200:1400].to_numpy(), 1.0)
    src = pine_source(cfg, {1: (sc, m, 0.55), -1: (sc, m, None)},
                      {"generated": "x", "symbols": ["BTCUSDT", "ETHUSDT"], "train": "a → b"}, demo=True)
    assert src.startswith("//@version=6")
    assert "ta.linreg(close, 168, 0)" in src and "ta.stdev(close, 72)" in src
    assert "barstate.isconfirmed" in src and "alertcondition" in src and "DEMO" in src
    assert "thr_dn = 1.01" in src and 'base == "BTC" or base == "ETH"' in src
    assert 'timeframe.period == "60"' in src

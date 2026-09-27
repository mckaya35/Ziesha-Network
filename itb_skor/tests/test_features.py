import numpy as np
import pandas as pd

from features import compute_features, feature_names, linreg_slope, sma, stdev


def ref_linreg(y, n, offset):
    """Pine ta.linreg tanımı: intercept + slope*(n-1-offset), x=0 en eski bar."""
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return intercept + slope * (n - 1 - offset)


def test_against_manual_reference(ohlcv, cfg):
    f = compute_features(ohlcv, cfg)
    c, v = ohlcv["close"].to_numpy(), ohlcv["volume"].to_numpy()
    for t in (200, 377, 599):
        for n in cfg["features"]["windows"]:
            w = c[t - n + 1: t + 1]
            assert np.isclose(f[f"sma_ratio_{n}"].iat[t], sum(w) / n / c[t] - 1, rtol=1e-10, atol=1e-12)
            mu = sum(w) / n
            assert np.isclose(f[f"vol_{n}"].iat[t], np.sqrt(sum((w - mu) ** 2) / n) / c[t], rtol=1e-9)
            lr = (ref_linreg(w, n, 0) - ref_linreg(w, n, 1)) / c[t]
            assert np.isclose(f[f"slope_{n}"].iat[t], lr, rtol=1e-7, atol=1e-12)
            assert np.isclose(f[f"ret_{n}"].iat[t], c[t] / c[t - n] - 1)
        assert np.isclose(f["ret_1"].iat[t], c[t] / c[t - 1] - 1)
        assert np.isclose(f["rvol_24"].iat[t], v[t] / v[t - 23: t + 1].mean())
        h = ohlcv.index[t].hour
        assert np.isclose(f["hour_sin"].iat[t], np.sin(2 * np.pi * h / 24))


def test_stdev_is_population():
    x = np.array([1.0, 2, 3, 4])
    assert np.isclose(stdev(x, 4)[-1], np.std(x, ddof=0))
    assert np.isclose(sma(x, 2)[-1], 3.5)


def test_linreg_slope_linear():
    x = 3.0 * np.arange(20) + 7
    assert np.allclose(linreg_slope(x, 6)[5:], 3.0)
    assert np.isnan(linreg_slope(x, 6)[:5]).all()


def test_warmup_nan(ohlcv, cfg):
    f = compute_features(ohlcv, cfg)
    assert f["sma_ratio_168"].iloc[:167].isna().all() and f["sma_ratio_168"].iloc[167:].notna().all()
    assert f["ret_168"].iloc[:168].isna().all() and f["ret_168"].iloc[168:].notna().all()
    assert list(f.columns) == feature_names(cfg)


def test_no_lookahead(ohlcv, cfg):
    """Veriyi t'de kes -> t'deki özellikler tam veriyle hesaplananla birebir aynı olmalı."""
    full = compute_features(ohlcv, cfg)
    for t in (170, 250, 400, 598):
        cut = compute_features(ohlcv.iloc[: t + 1], cfg)
        pd.testing.assert_series_equal(cut.iloc[-1], full.iloc[t], check_names=False, rtol=0, atol=0)


def test_future_change_does_not_affect_past(ohlcv, cfg):
    full = compute_features(ohlcv, cfg)
    mod = ohlcv.copy()
    mod.iloc[400:] *= 3.0
    f2 = compute_features(mod, cfg)
    pd.testing.assert_frame_equal(full.iloc[:400], f2.iloc[:400])

import numpy as np
import pandas as pd

import h3_trend as h3

CFG = {"costs": {"fee": 0.0005, "slippage": 0.0002}}
NOF = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
SC = 0.0007


def daily(o, h, l, c, start="2022-01-01"):
    idx = pd.date_range(start, periods=len(c), freq="1D", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx, dtype=float)


def flat_then(extra_o, extra_h, extra_l, extra_c, n=25, px=100.0):
    """n gün yatay (h=101, l=99, c=100, atr≈2), ardından verilen günler."""
    o = [px] * n + extra_o; h = [px + 1] * n + extra_h; l = [px - 1] * n + extra_l; c = [px] * n + extra_c
    return daily(o, h, l, c)


def test_rma_atr_match_pine_definition():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 1, 60)); h = c + 1; l = c - 1; o = np.r_[c[0], c[:-1]]
    d = daily(o, h, l, c)
    a = h3.atr(d, 5)
    pc = np.r_[np.nan, c[:-1]]
    tr = np.where(np.isnan(pc), h - l, np.maximum.reduce([h - l, abs(h - pc), abs(l - pc)]))
    ref = [np.nan] * 4 + [tr[:5].mean()]
    for x in tr[5:]:
        ref.append(ref[-1] + (x - ref[-1]) / 5)
    assert np.allclose(a, ref, equal_nan=True)


def test_channels_exclude_current_bar():
    d = flat_then([100], [150], [99], [100])
    ch = h3.channels(d)
    assert ch["hh_in"][-1] == 101                   # bugünkü 150 dahil değil
    assert np.isnan(ch["hh_in"][19]) and ch["hh_in"][20] == 101


def test_long_breakout_stop_and_costs():
    # gün25: kapanış 103 > 101 -> gün26 açılışta giriş 103; stop = 103 - 2*atr(≈2) ≈ 99
    d = flat_then([100, 103, 100], [104, 104, 100.5], [100, 102, 97], [103, 103.5, 98])
    tr, dl = h3.simulate(d, NOF, CFG, "X")
    r = tr.iloc[0]
    a = h3.atr(d)[25]
    assert r["side"] == 1 and r["entry"] == 103 and np.isclose(r["stop"], 103 - 2 * a)
    assert r["reason"] == "stop" and r["exit_day"] == d.index[27]
    px = min(100, r["stop"])                           # açılış 100 > stop ise stop fiyatı
    risk = 103 - r["stop"]
    assert np.isclose(r["R"], (px - 103) / risk - SC * (103 + px) / risk)
    assert np.isclose(dl.sum(), r["R"])               # günlük R yolu toplamı = işlem R'si


def test_gap_through_stop_fills_at_open():
    d = flat_then([100, 103, 90], [104, 104, 91], [100, 102, 89], [103, 103, 90])
    tr, _ = h3.simulate(d, NOF, CFG)
    assert tr.iloc[0]["exit"] == 90 and tr.iloc[0]["R"] < -1


def test_channel_exit_next_open():
    # long aç, sonra kapanış 10 günlük dibin altına iner ama stop'a değmez -> ertesi açılışta çıkış
    o = [100, 103] + [104] * 12 + [103, 101.5]
    h = [104, 105] + [105] * 12 + [104, 102]
    l = [100, 102.5] + [103] * 12 + [101, 101]
    c = [103, 104] + [104] * 12 + [101.2, 101.5]
    d = flat_then(o, h, l, c)
    tr, dl = h3.simulate(d, NOF, CFG)
    r = tr.iloc[0]
    assert r["reason"] == "channel" and r["exit"] == 101.5 and r["exit_day"] == d.index[25 + 15]
    assert np.isclose(dl.sum(), tr["R"].sum())


def test_no_lookahead_in_trades():
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 400)))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * 1.01; l = np.minimum(o, c) * 0.99
    d = daily(o, h, l, c)
    a, _ = h3.simulate(d, NOF, CFG)
    d2 = d.copy(); d2.iloc[300:] *= 1.8
    b, _ = h3.simulate(d2, NOF, CFG)
    cut = d.index[299]
    pd.testing.assert_frame_equal(a[a["exit_day"] < cut].reset_index(drop=True), b[b["exit_day"] < cut].reset_index(drop=True))


def test_funding_sign():
    d3 = flat_then([100, 103, 90], [104, 104, 91], [100, 102, 89], [103, 103, 90])
    fund3 = pd.Series([0.001], index=pd.DatetimeIndex([d3.index[26] + pd.Timedelta(hours=8)]))
    t3, _ = h3.simulate(d3, fund3, CFG)
    r = t3.iloc[0]
    assert np.isclose(r["fund_R"], -0.001 * 103 / (103 - r["stop"]))   # long funding öder


def test_to_daily_requires_full_days():
    idx = pd.date_range("2022-01-01", periods=12, freq="4h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
    assert len(h3.to_daily(df)) == 2
    assert len(h3.to_daily(df.drop(idx[3]))) == 1

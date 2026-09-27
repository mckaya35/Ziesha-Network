import numpy as np
import pandas as pd

from backtest import run_backtest, trade_outcomes


def bars(o, h, l, c):
    idx = pd.date_range("2021-01-01", periods=len(c), freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx, dtype=float)


def cfg(H=3):
    return {"interval": "1h", "label": {"H": H, "up": 0.02, "adverse": 0.01}, "costs": {"fee": 0.0005, "slippage": 0.0002}}


NOF = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))


def test_long_tp_entry_next_open():
    df = bars([100, 100, 101, 102], [100, 101, 102.5, 102], [100, 99.5, 100.5, 101], [100, 101, 102, 102])
    oc = trade_outcomes(df, NOF, 1, cfg(2))
    r = oc.iloc[0]
    assert r["reason"] == "tp" and r["exit_idx"] == 2
    assert np.isclose(r["gross"], 0.02)
    assert np.isclose(r["cost"], 0.0007 * (1 + 1.02))
    assert np.isclose(r["R"], (0.02 - 0.0007 * 2.02) / 0.01)


def test_same_bar_is_sl_and_gap():
    df = bars([100, 100, 97], [100, 103, 97], [100, 98, 96], [100, 100, 96])
    oc = trade_outcomes(df, NOF, 1, cfg(1))
    assert oc["reason"].iat[0] == "sl" and np.isclose(oc["gross"].iat[0], -0.01)
    gap = bars([100, 100, 97], [100, 100.5, 97.5], [100, 99.5, 96], [100, 100, 96])
    oc2 = trade_outcomes(gap, NOF, 1, cfg(2))
    assert oc2["reason"].iat[0] == "sl" and np.isclose(oc2["gross"].iat[0], 97 / 100 - 1)   # boşluk: open 97 < SL 99


def test_timeout_and_short():
    df = bars([100] * 5, [100.5] * 5, [99.5] * 5, [100, 100, 100, 100.3, 100])
    oc = trade_outcomes(df, NOF, -1, cfg(3))
    assert oc["reason"].iat[0] == "time" and oc["exit_idx"].iat[0] == 3
    assert np.isclose(oc["gross"].iat[0], -(100.3 / 100 - 1))
    assert not oc["valid"].iat[2]


def test_funding_window():
    df = bars([100] * 6, [100.5] * 6, [99.5] * 6, [100] * 6)
    ft = pd.DatetimeIndex(["2021-01-01 01:00", "2021-01-01 02:00", "2021-01-01 04:00", "2021-01-01 05:00"], tz="UTC")
    fund = pd.Series([1.0, 0.001, 0.002, 1.0], index=ft)
    oc = trade_outcomes(df, fund, 1, cfg(3))
    # sinyal bar0, giriş 01:00 (hariç), zaman aşımı bar3 kapanışı = 04:00 (dahil) -> 0.001+0.002
    assert np.isclose(oc["fund"].iat[0], -0.003)
    assert np.isclose(trade_outcomes(df, fund, -1, cfg(3))["fund"].iat[0], 0.003)


def test_single_position():
    df = bars([100] * 10, [100.5] * 10, [99.5] * 10, [100] * 10)
    oc = {s: trade_outcomes(df, NOF, s, cfg(3)) for s in (1, -1)}
    sig = np.array([1, 1, 1, -1, 1, 0, 0, 0, 0, 0])
    tr = run_backtest(oc, sig)
    assert list(tr["signal_time"].dt.hour) == [0, 3]      # bar0 -> çıkış bar3; bar3 sinyali alınır
    assert tr["entry_time"].iat[0] == df.index[1]


def test_breakdown_only_short_side():
    from report import breakdown, pick
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    tr = pd.DataFrame({"symbol": "BTCUSDT", "side": -1, "signal_time": t0, "entry_time": t0, "exit_time": t0,
                       "reason": "tp", "hold_h": 2.0, "gross": 0.02, "cost": 0.0014, "fund": 0.0, "net": 0.0186,
                       "R": 1.86}, index=[0])
    out = pick(breakdown(tr, 0.01, None), ["TOPLAM", "long", "short"])
    assert list(out.index) == ["TOPLAM", "short"]

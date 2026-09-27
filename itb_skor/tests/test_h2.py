import numpy as np
import pandas as pd

import h2_momentum as h2

CFG = {"costs": {"fee": 0.0005, "slippage": 0.0002}, "walkforward": {"holdout_days": 90}}
NOF = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))


def coin(seed, start="2021-01-01", end="2022-12-31 20:00", drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="4h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, len(idx))))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0}, index=idx)


def universe(k=12):
    return {f"C{i:02d}USDT": (coin(i, drift=(i - k / 2) * 1e-4), NOF) for i in range(k)}


def test_signal_and_returns_exact():
    raw = universe(1)
    df = raw["C00USDT"][0]
    m = pd.Timestamp("2021-03-01", tz="UTC")                          # pazartesi
    r = h2.weekly_table(raw, pd.DatetimeIndex([m])).iloc[0]
    sb = m - pd.Timedelta(hours=4)
    assert np.isclose(r["signal"], df["close"][sb] / df["close"][sb - pd.Timedelta(days=28)] - 1)
    assert np.isclose(r["ret"], df["open"][m + pd.Timedelta(days=7)] / df["open"][m] - 1)


def test_signal_no_lookahead():
    raw = universe(3)
    ms = pd.date_range("2021-03-01", "2021-06-28", freq="W-MON", tz="UTC")
    a = h2.weekly_table(raw, ms)
    for m in ms:
        mod = {s: (df.copy(), f) for s, (df, f) in raw.items()}
        for df, _ in mod.values():
            df.loc[df.index >= m, ["open", "high", "low", "close"]] *= 3     # pazartesi ve sonrası değişir
        b = h2.weekly_table(mod, pd.DatetimeIndex([m]))
        assert np.allclose(a[a["monday"] == m]["signal"].to_numpy(), b["signal"].to_numpy())


def test_min_history_and_missing_bars():
    raw = universe(2)
    raw["LATEUSDT"] = (coin(9, start="2021-02-10"), NOF)
    m = pd.Timestamp("2021-03-08", tz="UTC")                          # LATE: 26 gün < 29
    assert "LATEUSDT" not in set(h2.weekly_table(raw, pd.DatetimeIndex([m]))["symbol"])
    df = raw["C00USDT"][0].drop(m + pd.Timedelta(days=7))            # çıkış barı eksik -> o hafta dışarıda
    raw["C00USDT"] = (df, NOF)
    assert "C00USDT" not in set(h2.weekly_table(raw, pd.DatetimeIndex([m]))["symbol"])


def test_weights_top_bottom():
    tab = pd.DataFrame({"monday": pd.Timestamp("2021-03-01", tz="UTC"), "symbol": [f"S{i}" for i in range(10)],
                        "signal": np.arange(10.0), "ret": 0.0, "fund": 0.0})
    w = h2.build_weights(tab)[tab["monday"].iat[0]]
    assert {s for s, v in w.items() if v > 0} == {"S9", "S8", "S7", "S6"}
    assert {s for s, v in w.items() if v < 0} == {"S0", "S1", "S2", "S3"}
    assert all(abs(v) == 0.125 for v in w.values())


def test_costs_only_on_changes_and_funding_sign():
    m1, m2 = pd.Timestamp("2021-03-01", tz="UTC"), pd.Timestamp("2021-03-08", tz="UTC")
    W = {m1: {"A": .125, "B": -.125}, m2: {"A": .125, "C": -.125}}
    tab = pd.DataFrame({"monday": [m1, m1, m2, m2], "symbol": ["A", "B", "A", "C"],
                        "ret": [0.10, 0.10, 0.0, 0.0], "fund": [0.001, 0.001, 0.0, 0.0]})
    wk = h2.portfolio(tab, W, CFG)
    c = 0.0007
    assert np.isclose(wk["cost"].iat[0], c * 0.25)                  # açılış: A + B
    assert np.isclose(wk["cost"].iat[1], c * (0.25 + 0.25))         # B kapanır, C açılır; son hafta kapanış 0.25
    assert np.isclose(wk["gross"].iat[0], 0.0)                      # long +, short − birbirini götürür
    assert np.isclose(wk["fund"].iat[0], 0.0)                       # long öder, short alır
    wk2 = h2.portfolio(tab, W, CFG, 2.0)
    assert np.isclose(wk2["cost"].iat[0], 2 * c * 0.25)


def test_newey_west():
    x = np.random.default_rng(0).normal(0.01, 0.02, 400)
    t0 = h2.newey_west_t(x, lags=0)
    assert np.isclose(t0, x.mean() / (x.std(ddof=0) / np.sqrt(len(x))))
    assert np.isfinite(h2.newey_west_t(x))


def test_periods_exclude_holdout():
    raw = universe(12)
    A, B, hold, end = h2.periods(CFG, raw)
    assert (A["monday"] + pd.Timedelta(days=7) <= hold).all()
    assert (B["monday"] >= hold).all()
    assert A.groupby("monday").size().iloc[0] >= 10

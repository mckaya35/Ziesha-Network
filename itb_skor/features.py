"""Özellikler — hepsi ORAN; yalnızca geçmiş/şimdiki bara bakar.
Her formülün Pine v6 karşılığı yorumda. Pine ile birebir aynı olmalı (TA-Lib yok)."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _rolling(x, n, fn):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = fn(sliding_window_view(x, n))
    return out


def sma(x, n):
    # Pine: ta.sma(x, n)
    return _rolling(x, n, lambda w: w.mean(axis=1))


def stdev(x, n):
    # Pine: ta.stdev(x, n)  (biased=true -> popülasyon, ddof=0)
    return _rolling(x, n, lambda w: w.std(axis=1, ddof=0))


def linreg_slope(x, n):
    # Pine: ta.linreg(x, n, 0) - ta.linreg(x, n, 1)
    # ta.linreg = intercept + slope*(n-1-offset)  => fark = en küçük kareler eğimi.
    # Pencere içinde en eski bar i=0, en yeni i=n-1.
    i = np.arange(n, dtype=float)
    w = (i - i.mean()) / ((i - i.mean()) ** 2).sum()
    return _rolling(x, n, lambda win: win @ w)


def shift(x, k):
    # Pine: x[k]
    out = np.full(len(x), np.nan)
    out[k:] = x[:-k]
    return out


def feature_names(cfg):
    names = []
    for n in cfg["features"]["windows"]:
        names += [f"sma_ratio_{n}", f"slope_{n}", f"vol_{n}", f"ret_{n}"]
    names += ["ret_1", "rvol_24"]
    if cfg["features"].get("hour"):
        names += ["hour_sin", "hour_cos"]
    return names


def compute_features(df, cfg):
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    f = {}
    for n in cfg["features"]["windows"]:
        f[f"sma_ratio_{n}"] = sma(c, n) / c - 1               # ta.sma(close, n) / close - 1
        f[f"slope_{n}"] = linreg_slope(c, n) / c               # (ta.linreg(close,n,0) - ta.linreg(close,n,1)) / close
        f[f"vol_{n}"] = stdev(c, n) / c                        # ta.stdev(close, n) / close
        f[f"ret_{n}"] = c / shift(c, n) - 1                    # close / close[n] - 1
    f["ret_1"] = c / shift(c, 1) - 1                           # close / close[1] - 1
    f["rvol_24"] = v / sma(v, 24)                              # volume / ta.sma(volume, 24)
    if cfg["features"].get("hour"):
        h = df.index.hour.to_numpy(float)                      # hour(time, "UTC")  (bar AÇILIŞ saati)
        f["hour_sin"] = np.sin(2 * np.pi * h / 24)             # math.sin(2*math.pi*h/24)
        f["hour_cos"] = np.cos(2 * np.pi * h / 24)             # math.cos(2*math.pi*h/24)
    return pd.DataFrame(f, index=df.index)[feature_names(cfg)]

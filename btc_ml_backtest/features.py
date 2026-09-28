"""Lookahead içermeyen özellikler + hedef (sonraki saatin log getirisi)."""
import numpy as np
import pandas as pd


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build(df):
    c, lc = df["close"], np.log(df["close"])
    X = pd.DataFrame(index=df.index)
    for k in (1, 3, 6, 12, 24):
        X[f"ret_{k}"] = lc.diff(k)
    X["rsi_14"] = rsi(c)
    for n in (24, 168):
        X[f"dist_sma_{n}"] = c / c.rolling(n).mean() - 1
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(), (df.low - c.shift()).abs()], axis=1).max(axis=1)
    X["atr_ratio"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    X["std_24"] = X["ret_1"].rolling(24).std()
    v = df["volume"]
    X["vol_z"] = (v - v.rolling(168).mean()) / v.rolling(168).std()
    X = X.replace([np.inf, -np.inf], np.nan)
    y = lc.diff().shift(-1).rename("target")
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]

"""Binance USDT-M Futures herkese açık mum + funding verisi (API anahtarı yok).

Kullanım:  python data.py            # config'teki sembolleri artımlı indirir/günceller
           python data.py --synthetic  # AĞ YOK: sadece kurulum/akış denemesi için sahte veri
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from common import load_config

BASE = "https://fapi.binance.com"
MS_H = 3_600_000


def _get(path, params, tries=5):
    for i in range(tries):
        try:
            r = requests.get(BASE + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):  # rate limit
                time.sleep(30 * (i + 1))
                continue
            r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError(f"{path} {params} indirilemedi")


def fetch_klines(symbol, interval, start_ms):
    rows, cur = [], start_ms
    while True:
        batch = _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval,
                                         "startTime": cur, "limit": 1000})
        if not batch:
            break
        rows += batch
        cur = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.25)
    if not rows:
        return None
    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    # Kapanmamış (son) mumu at
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"] < now_ms]
    df.index = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index.name = "time"
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_funding(symbol, start_ms):
    rows, cur = [], start_ms
    while True:
        batch = _get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": cur, "limit": 1000})
        if not batch:
            break
        rows += batch
        cur = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.25)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # funding zamanları birkaç ms kayabiliyor -> saniyeye yuvarla
    idx = pd.to_datetime(df["fundingTime"], unit="ms", utc=True).dt.round("s")
    return pd.DataFrame({"rate": df["fundingRate"].astype(float).values}, index=pd.DatetimeIndex(idx, name="time"))


def _merge_save(old, new, path):
    df = new if old is None else pd.concat([old, new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    return df


def quality_report(df, freq="1h"):
    dup = int(df.index.duplicated().sum())
    full = pd.date_range(df.index[0], df.index[-1], freq=freq)
    missing = full.difference(df.index)
    bad = int(((df["high"] < df[["open", "close"]].max(axis=1)) |
               (df["low"] > df[["open", "close"]].min(axis=1))).sum())
    return {"rows": len(df), "first": str(df.index[0]), "last": str(df.index[-1]),
            "duplicates": dup, "missing_hours": len(missing),
            "missing_examples": [str(t) for t in missing[:5]], "bad_ohlc": bad}


def update(cfg):
    d = Path(cfg["data_dir"]); d.mkdir(parents=True, exist_ok=True)
    start_ms = int(pd.Timestamp(cfg["start"], tz="UTC").timestamp() * 1000)
    reports = {}
    for sym in cfg["symbols"]:
        kp = d / f"{sym}_{cfg['interval']}.parquet"
        old = pd.read_parquet(kp) if kp.exists() else None
        s = start_ms if old is None else int(old.index[-1].timestamp() * 1000) + MS_H
        new = fetch_klines(sym, cfg["interval"], s)
        df = old if new is None else _merge_save(old, new, kp)

        fp = d / f"{sym}_funding.parquet"
        fold = pd.read_parquet(fp) if fp.exists() else None
        s = start_ms if fold is None else int(fold.index[-1].timestamp() * 1000) + 1000
        fnew = fetch_funding(sym, s)
        if fnew is not None:
            _merge_save(fold, fnew, fp)

        reports[sym] = quality_report(df)
        print(sym, reports[sym])
    return reports


def make_synthetic(cfg, seed=0):
    """SADECE akış testi: rastgele yürüyüş (öngörülebilir sinyal YOK). Gerçek sonuç değildir."""
    d = Path(cfg["data_dir"]); d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.date_range(cfg["start"], "2024-12-31 23:00", freq="1h", tz="UTC", name="time")
    for k, sym in enumerate(cfg["symbols"]):
        n = len(idx)
        lr = rng.standard_t(4, n) * 0.006
        close = (20000 / (k + 1)) * np.exp(np.cumsum(lr))
        open_ = np.r_[close[0], close[:-1]]
        spread = np.abs(rng.normal(0, 0.004, n)) * close
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.004, n)) * close
        vol = rng.lognormal(8, 0.5, n)
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
        df.to_parquet(d / f"{sym}_{cfg['interval']}.parquet")
        fidx = idx[idx.hour % 8 == 0]
        pd.DataFrame({"rate": rng.normal(0.0001, 0.0001, len(fidx))}, index=fidx).to_parquet(d / f"{sym}_funding.parquet")
    return {s: quality_report(pd.read_parquet(d / f"{s}_{cfg['interval']}.parquet")) for s in cfg["symbols"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--config")
    a = ap.parse_args()
    c = load_config(a.config)
    print(make_synthetic(c) if a.synthetic else update(c))

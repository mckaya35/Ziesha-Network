"""Binance USD-M futures BTCUSDT 1H kline + funding indirme."""
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
START = "2019-01-01"
DATA = Path(__file__).parent / "data"
KLINE_PATH = DATA / "btcusdt_1h.parquet"
FUND_PATH = DATA / "btcusdt_funding.parquet"


def _get(path, params):
    for i in range(5):
        try:
            r = requests.get(BASE + path, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if i == 4:
                raise
            time.sleep(2 ** i)


def _now():
    return pd.Timestamp.now("UTC").floor("h").tz_localize(None)


def _ms(ts):
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def download_klines():
    rows, start, end = [], _ms(START), _ms(_now())
    while start < end:
        batch = _get("/fapi/v1/klines", {"symbol": SYMBOL, "interval": "1h", "startTime": start, "limit": 1500})
        if not batch:
            break
        rows += batch
        start = batch[-1][0] + 3_600_000
        time.sleep(0.2)
    df = pd.DataFrame([r[:6] for r in rows], columns=["time", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df = df.drop_duplicates("time").set_index("time").astype(float)
    df = df[df.index < _now()]  # yalnızca kapanmış barlar
    return fill_gaps(df)


def fill_gaps(df):
    idx = pd.date_range(df.index[0], df.index[-1], freq="h")
    df = df.reindex(idx)
    df["close"] = df["close"].ffill()
    for c in ("open", "high", "low"):
        df[c] = df[c].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df.index.name = "time"
    return df


def download_funding():
    rows, start = [], _ms(START)
    while True:
        batch = _get("/fapi/v1/fundingRate", {"symbol": SYMBOL, "startTime": start, "limit": 1000})
        if not batch:
            break
        rows += batch
        start = batch[-1]["fundingTime"] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms")
    return df.set_index("time")[["fundingRate"]].astype(float).rename(columns={"fundingRate": "rate"})


def load():
    DATA.mkdir(exist_ok=True)
    if not KLINE_PATH.exists():
        download_klines().to_parquet(KLINE_PATH)
    if not FUND_PATH.exists():
        download_funding().to_parquet(FUND_PATH)
    return pd.read_parquet(KLINE_PATH), pd.read_parquet(FUND_PATH)


if __name__ == "__main__":
    k, f = load()
    print(f"klines : {len(k)} satır, {k.index[0]} -> {k.index[-1]}, sıfır hacimli saat: {(k.volume == 0).sum()}")
    print(f"funding: {len(f)} satır, {f.index[0]} -> {f.index[-1]}")

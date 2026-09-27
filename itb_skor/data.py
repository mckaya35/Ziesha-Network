"""Veri: önce yerel klasör (CSV/Parquet, SADECE okunur), eksikler Binance USDT-M Futures API'den.

python data.py                 # tüm deneyler için tarama + yükleme raporu
python data.py --exp main
"""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from common import bar_delta, experiments, load_config

BASE = "https://fapi.binance.com"
STD = ["open", "high", "low", "close", "volume"]
TIME_COLS = ["timestamp", "time", "open_time", "opentime", "date", "datetime"]


# ---------------------------------------------------------------- yerel dosyalar
def _parse_time(s):
    if pd.api.types.is_numeric_dtype(s):
        v = float(pd.Series(s).dropna().iloc[0])
        unit = "ns" if v > 1e17 else "us" if v > 1e14 else "ms" if v > 1e11 else "s"
        return pd.to_datetime(s.astype("int64"), unit=unit, utc=True)
    return pd.to_datetime(s, utc=True)


def read_local(path):
    """Herhangi bir CSV/Parquet -> index 'timestamp' (UTC), sütunlar open high low close volume."""
    path = Path(path)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    cols = {c.lower().strip(): c for c in raw.columns}
    tcol = next((cols[c] for c in TIME_COLS if c in cols), None)
    if tcol is None and isinstance(raw.index, pd.DatetimeIndex):
        t = raw.index.tz_localize("UTC") if raw.index.tz is None else raw.index.tz_convert("UTC")
    elif tcol is None:
        raise ValueError(f"{path}: zaman sütunu yok ({list(raw.columns)})")
    else:
        t = _parse_time(raw[tcol])
    missing = [c for c in STD if c not in cols]
    if missing:
        raise ValueError(f"{path}: eksik sütun {missing}")
    df = pd.DataFrame({c: pd.to_numeric(raw[cols[c]], errors="coerce").to_numpy() for c in STD},
                      index=pd.DatetimeIndex(t, name="timestamp"))
    return df, list(raw.columns)


def detect_tf(index):
    d = pd.Series(index.sort_values()).diff().dropna()
    return d.mode().iloc[0] if len(d) else None


def tf_label(td):
    h = td / pd.Timedelta(hours=1)
    return f"{int(h)}h" if h >= 1 and h.is_integer() else f"{int(td / pd.Timedelta(minutes=1))}m"


def quality(df, tf):
    dup = int(df.index.duplicated().sum())
    d = df[~df.index.duplicated()].sort_index()
    full = pd.date_range(d.index[0], d.index[-1], freq=tf)
    miss = full.difference(d.index)
    nan = int(d[STD].isna().any(axis=1).sum())
    bad = int(((d["high"] < d[["open", "close"]].max(axis=1)) | (d["low"] > d[["open", "close"]].min(axis=1))).sum())
    return {"rows": len(df), "start": d.index[0], "end": d.index[-1], "duplicates": dup,
            "missing_bars": len(miss), "nan_rows": nan, "bad_ohlc": bad}


def _coin_of(path, coins, quote):
    toks = re.split(r"[^A-Za-z0-9]+", f"{path.parent.name}_{path.stem}".upper())
    for c in coins:
        if c in toks or c + quote in toks:
            return c
    return None


def scan_local(cfg, coins):
    """Klasörü tara: dosya, sembol, zaman dilimi, başlangıç–bitiş, sütunlar, eksik/dublike."""
    d = cfg.get("local_data_dir")
    if not d or not Path(d).exists():
        return pd.DataFrame()
    rows = []
    for p in sorted(Path(d).rglob("*")):
        if p.suffix not in (".csv", ".parquet") or not p.is_file():
            continue
        coin = _coin_of(p, coins, cfg["quote"])
        if coin is None:
            continue
        try:
            df, cols = read_local(p)
        except Exception as e:                      # okunamayan/alakasız dosya
            rows.append({"file": str(p.relative_to(d)), "coin": coin, "error": str(e)[:80]})
            continue
        tf = detect_tf(df.index)
        rows.append({"file": str(p.relative_to(d)), "path": str(p), "coin": coin, "symbol": coin + cfg["quote"],
                     "tf": tf_label(tf), "tf_td": tf, "columns": ",".join(cols), **quality(df, tf)})
    return pd.DataFrame(rows)


def resample(df, src, dst):
    """Yalnızca daha küçük zaman diliminden. Eksik alt barı olan hedef barlar atılır."""
    if not (src < dst and dst % src == pd.Timedelta(0)):
        raise ValueError(f"{src} -> {dst} yeniden örneklenemez")
    g = df.resample(dst, label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    out = out[g["close"].count() == dst // src]
    return out


# ---------------------------------------------------------------- API
def _get(path, params, tries=5):
    for i in range(tries):
        try:
            r = requests.get(BASE + path, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (418, 429):
                time.sleep(30 * (i + 1)); continue
            r.raise_for_status()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError(f"{path} {params}")


def fetch_klines(symbol, interval, start):
    rows, cur = [], int(start.timestamp() * 1000)
    while True:
        b = _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "startTime": cur, "limit": 1000})
        if not b:
            break
        rows += b; cur = b[-1][0] + 1
        if len(b) < 1000:
            break
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame(columns=STD, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["t", *STD, "close_time"]
    df = df[df["close_time"] < int(time.time() * 1000)]          # kapanmamış mumu at
    df.index = pd.DatetimeIndex(pd.to_datetime(df["t"], unit="ms", utc=True), name="timestamp")
    return df[STD].astype(float)


def fetch_funding(symbol, start):
    rows, cur = [], int(start.timestamp() * 1000)
    while True:
        b = _get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": cur, "limit": 1000})
        if not b:
            break
        rows += b; cur = b[-1]["fundingTime"] + 1
        if len(b) < 1000:
            break
        time.sleep(0.25)
    idx = pd.DatetimeIndex(pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True).round("s"), name="time")
    return pd.DataFrame({"rate": [float(r["fundingRate"]) for r in rows]}, index=idx)


def _cached_update(path, fetch, start, step):
    old = pd.read_parquet(path) if path.exists() else None
    s = start if old is None or old.empty else max(start, old.index[-1] + step)
    new = fetch(s)
    df = new if old is None else pd.concat([old, new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


# ---------------------------------------------------------------- birleştirme
def load_symbol(cfg, scan, coin, use_api=True):
    """Dönüş: (df, provenance). Yerel öncelikli; hedeften küçük TF varsa yeniden örnekle; kuyruk API'den."""
    sym, tf = coin + cfg["quote"], bar_delta(cfg)
    prov = {"symbol": sym, "local_file": None, "resampled_from": None, "api_rows": 0, "api_error": None,
            "overlap_bars": 0, "overlap_med_absdiff": np.nan}
    df = None
    cand = scan[(scan["coin"] == coin) & scan["tf_td"].notna()] if len(scan) and "tf_td" in scan else scan
    if len(cand):
        exact = cand[cand["tf_td"] == tf]
        smaller = cand[(cand["tf_td"] < tf) & cand["tf_td"].map(lambda x: tf % x == pd.Timedelta(0))]
        if len(exact):
            r = exact.sort_values("rows").iloc[-1]
            df, _ = read_local(r["path"])
        elif len(smaller):
            r = smaller.sort_values("tf_td").iloc[-1]
            df = resample(read_local(r["path"])[0].pipe(lambda x: x[~x.index.duplicated()].sort_index()), r["tf_td"], tf)
            prov["resampled_from"] = r["tf"]
        if df is not None:
            prov["local_file"] = r["file"]
            df = df[~df.index.duplicated(keep="last")].sort_index().dropna()

    if use_api and cfg.get("api_fill"):
        try:
            start = df.index[-1] - 200 * tf if df is not None else pd.Timestamp(cfg["api_start"], tz="UTC")
            api = _cached_update(Path(cfg["data_dir"]) / "api" / f"{sym}_{cfg['interval']}.parquet",
                                 lambda s: fetch_klines(sym, cfg["interval"], s), start, tf)
            if df is not None:
                common = df.index.intersection(api.index)
                if len(common):
                    prov["overlap_bars"] = len(common)
                    prov["overlap_med_absdiff"] = float((api.loc[common, "close"] / df.loc[common, "close"] - 1).abs().median())
                tail = api[api.index > df.index[-1]]
            else:
                tail = api
            prov["api_rows"] = len(tail)
            df = tail if df is None else pd.concat([df, tail])
        except Exception as e:
            prov["api_error"] = str(e)[:120]
    return df, prov


def load_funding(cfg, symbol, start, use_api=True):
    p = Path(cfg["data_dir"]) / "api" / f"{symbol}_funding.parquet"
    if use_api and cfg.get("api_fill"):
        try:
            return _cached_update(p, lambda s: fetch_funding(symbol, s), start, pd.Timedelta(seconds=1))["rate"], None
        except Exception as e:
            err = str(e)[:120]
    else:
        err = "API kapalı"
    if p.exists():
        return pd.read_parquet(p)["rate"], f"önbellek kullanıldı ({err})"
    return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC")), f"funding YOK ({err}) -> 0 kabul edildi"


def load_all(cfg, use_api=True):
    """Deneyin tüm sembolleri: {sym: (df, funding)}, tarama tablosu, kaynak tablosu. Rapor md de yazar."""
    scan = scan_local(cfg, cfg["coins"])
    out, prov = {}, []
    for coin in cfg["coins"]:
        df, pv = load_symbol(cfg, scan, coin, use_api)
        if df is None or df.empty:
            pv["status"] = "VERİ YOK — dışarıda"
            prov.append(pv); continue
        fund, ferr = load_funding(cfg, pv["symbol"], df.index[0], use_api)
        pv.update({"status": "ok", "funding": ferr or f"{len(fund)} kayıt", **{k: v for k, v in quality(df, bar_delta(cfg)).items()}})
        out[pv["symbol"]] = (df, fund)
        prov.append(pv)
    prov = pd.DataFrame(prov)
    write_data_report(cfg, scan, prov)
    return out, scan, prov


def write_data_report(cfg, scan, prov):
    L = [f"# Veri raporu — {cfg['exp']} ({cfg['interval']})\n",
         f"Yerel klasör: `{cfg.get('local_data_dir') or '-'}` (yalnızca okundu, değiştirilmedi). Pazar türü (config): **{cfg['local_market']}**\n",
         "## Taranan dosyalar\n"]
    if len(scan):
        cols = [c for c in ["file", "symbol", "tf", "start", "end", "rows", "duplicates", "missing_bars", "nan_rows", "bad_ohlc", "columns", "error"] if c in scan]
        L.append("```\n" + scan[cols].to_string(index=False) + "\n```")
    else:
        L.append("(yerel dosya bulunamadı)")
    L += ["", "## Deneyde kullanılan seri (yerel + API kuyruğu)\n", "```\n" + prov.to_string(index=False) + "\n```", "",
          "- `overlap_med_absdiff`: yerel kapanış ile Binance USDT-M Futures API kapanışı arasındaki medyan |oran farkı|. "
          "≈0 ise yerel veri futures; ~1e-4 ve üstü ise büyük olasılıkla spot (funding maliyeti yaklaşık kalır).",
          "- `resampled_from`: hedef zaman diliminden küçük dosyadan yeniden örneklendi (eksik alt barlı hedef barlar atıldı)."]
    txt = "\n".join(L)
    (Path(cfg["exp_out"]) / "data_report.md").write_text(txt)
    return txt


def make_synthetic(root, seed=0):
    """SADECE akış testi: python_port biçiminde rastgele-yürüyüş CSV'leri (öngörülebilir sinyal YOK)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    spec = {"BTC": "1h", "ETH": "1h", "SOL": "1h", "SUI": "1h", "ADA": "4h", "XRP": "4h"}
    for i, (coin, tf) in enumerate(spec.items()):
        start = "2023-05-03" if coin == "SUI" else "2021-07-30"
        idx = pd.date_range(start, "2024-12-31 23:00", freq=tf, tz="UTC")
        sc = np.sqrt(pd.Timedelta(tf) / pd.Timedelta("1h"))
        c = 100 * (i + 1) * np.exp(np.cumsum(rng.standard_t(4, len(idx)) * 0.006 * sc))
        o = np.r_[c[0], c[:-1]]
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004 * sc, len(idx))))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004 * sc, len(idx))))
        df = pd.DataFrame({"time": idx.as_unit("ms").asi8, "open": o, "high": h, "low": l, "close": c,
                           "volume": rng.lognormal(8, 0.5, len(idx))})
        if coin == "BTC":                                   # tarayıcıyı sına: 1 dublike + 3 eksik bar
            df = pd.concat([df.drop(index=[1000, 1001, 1002]), df.iloc[[500]]]).sort_values("time")
        d = root / f"uzun_{coin}_{tf}"
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / "ohlcv.csv", index=False)
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None)
    a = ap.parse_args()
    for e in [a.exp] if a.exp else experiments():
        c = load_config(e)
        _, _, prov = load_all(c)
        print(open(Path(c["exp_out"]) / "data_report.md").read())

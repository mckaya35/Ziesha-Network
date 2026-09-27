"""Ortak yardımcılar: config ve veri yükleme."""
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    with open(path or ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = str(ROOT / cfg["data_dir"])
    cfg["out_dir"] = str(ROOT / cfg["out_dir"])
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def load_klines(cfg, symbol):
    """Index = bar AÇILIŞ zamanı (UTC). Sütunlar: open high low close volume."""
    return pd.read_parquet(Path(cfg["data_dir"]) / f"{symbol}_{cfg['interval']}.parquet")


def load_funding(cfg, symbol):
    """Index = funding zamanı (UTC), sütun: rate."""
    p = Path(cfg["data_dir"]) / f"{symbol}_funding.parquet"
    if not p.exists():
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name="rate")
    return pd.read_parquet(p)["rate"]

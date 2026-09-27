"""Ortak yardımcılar: config (deney bazında birleştirilmiş)."""
import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(exp="main", path=None, overrides=None):
    with open(path or ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw.update(overrides)
    cfg = copy.deepcopy({k: v for k, v in raw.items() if k != "experiments"})
    e = raw["experiments"][exp]
    cfg.update(copy.deepcopy(e))
    cfg["exp"] = exp
    cfg["symbols"] = [c + cfg["quote"] for c in e["coins"]]
    for k in ("data_dir", "out_dir", "local_data_dir", "control_predictions"):
        if cfg.get(k):
            p = Path(cfg[k]).expanduser()
            cfg[k] = str(p if p.is_absolute() else (ROOT / p).resolve())
    cfg["exp_out"] = str(Path(cfg["out_dir"]) / exp)
    Path(cfg["exp_out"]).mkdir(parents=True, exist_ok=True)
    return cfg


def experiments(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return list(yaml.safe_load(f)["experiments"])


def bar_delta(cfg):
    return pd.Timedelta(cfg["interval"])


def config_hash(cfg):
    keys = ("interval", "symbols", "label", "features", "walkforward", "costs", "criteria")
    s = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]

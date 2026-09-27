import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def cfg():
    return {"exp": "t", "interval": "1h", "quote": "USDT", "features": {"windows": [6, 24, 72, 168], "hour": True},
            "label": {"H": 24, "up": 0.02, "adverse": 0.01},
            "walkforward": {"train_months": 12, "test_months": 1, "first_test": "2021-01", "val_frac": 0.2,
                            "C": 1.0, "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7], "min_val_trades": 10,
                            "holdout_days": 30},
            "costs": {"fee": 0.0005, "slippage": 0.0002}}


def make_ohlcv(n, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": rng.lognormal(5, 1, n)}, index=idx)


@pytest.fixture
def ohlcv():
    return make_ohlcv(600)

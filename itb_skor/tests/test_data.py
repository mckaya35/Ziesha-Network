import numpy as np
import pandas as pd
import pytest

from data import detect_tf, quality, read_local, resample, scan_local
from tests.conftest import make_ohlcv


def write(path, df, tcol="time", unit="ms"):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index(drop=True)
    out.insert(0, tcol, df.index.as_unit(unit).asi8 if unit else df.index.astype(str))
    out.to_csv(path, index=False)


def test_read_local_formats(tmp_path):
    df = make_ohlcv(50)
    write(tmp_path / "a.csv", df, "time", "ms")
    write(tmp_path / "b.csv", df.rename(columns=str.upper), "Open_Time", "s")
    write(tmp_path / "c.csv", df, "timestamp", None)
    for f in ("a.csv", "b.csv", "c.csv"):
        r, _ = read_local(tmp_path / f)
        assert list(r.columns) == ["open", "high", "low", "close", "volume"]
        assert r.index.tz is not None and r.index[0] == df.index[0]
        assert np.allclose(r["close"], df["close"])


def test_scan_detects_tf_dup_missing(tmp_path):
    df = make_ohlcv(100)
    bad = pd.concat([df.drop(df.index[[10, 11]]), df.iloc[[5]]])
    write(tmp_path / "uzun_BTC_1h" / "ohlcv.csv", bad)
    write(tmp_path / "uzun_ETH_4h" / "ohlcv.csv", df.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}))
    write(tmp_path / "uzun_FOO_1h" / "ohlcv.csv", df)                       # config'te yok -> atlanır
    s = scan_local({"local_data_dir": str(tmp_path), "quote": "USDT"}, ["BTC", "ETH"]).set_index("coin")
    assert set(s.index) == {"BTC", "ETH"}
    assert s.loc["BTC", "tf"] == "1h" and s.loc["ETH", "tf"] == "4h"
    assert s.loc["BTC", "duplicates"] == 1 and s.loc["BTC", "missing_bars"] == 2


def test_resample_only_from_smaller_and_complete(tmp_path):
    df = make_ohlcv(48)
    r = resample(df.drop(df.index[5]), pd.Timedelta("1h"), pd.Timedelta("4h"))
    assert len(r) == 11                                        # eksik alt barlı 4h bar atıldı
    g = df.iloc[8:12]
    row = r.loc[df.index[8]]
    assert row["open"] == g["open"].iat[0] and row["close"] == g["close"].iat[-1]
    assert row["high"] == g["high"].max() and np.isclose(row["volume"], g["volume"].sum())
    with pytest.raises(ValueError):
        resample(r, pd.Timedelta("4h"), pd.Timedelta("1h"))


def test_quality_counts():
    df = make_ohlcv(30)
    q = quality(pd.concat([df, df.iloc[[3]]]).drop(df.index[7]), pd.Timedelta("1h"))
    assert q["duplicates"] == 1 and q["missing_bars"] == 1
    assert detect_tf(df.index) == pd.Timedelta("1h")

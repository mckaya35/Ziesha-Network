import numpy as np
import pandas as pd

from labels import make_labels


def bars(close, high=None, low=None):
    close = np.asarray(close, float)
    idx = pd.date_range("2021-01-01", periods=len(close), freq="1h", tz="UTC")
    return pd.DataFrame({"open": close, "high": high if high is not None else close,
                         "low": low if low is not None else close, "close": close, "volume": 1.0}, index=idx)


def test_tp_before_sl():
    #          t0   t1   t2    t3    t4
    h = [100, 101, 102.1, 100, 100]
    l = [100, 99.5, 100, 98, 100]
    lab = make_labels(bars([100] * 5, h, l), H=3, up=0.02, adverse=0.01)
    assert lab["label_up"].iat[0] == 1          # 102.1 >= 102 (t2) SL 99'dan önce (t3)
    assert lab["label_dn"].iat[0] == 0          # dn TP=98 t3'te ama SL=101 t1'de önce


def test_sl_before_tp():
    h = [100, 100, 103, 100]
    l = [100, 98.9, 100, 100]
    assert make_labels(bars([100] * 4, h, l), H=3, up=0.02, adverse=0.01)["label_up"].iat[0] == 0


def test_same_bar_pessimistic():
    h = [100, 103, 100, 100]
    l = [100, 98, 100, 100]
    lab = make_labels(bars([100] * 4, h, l), H=3, up=0.02, adverse=0.01)
    assert lab["label_up"].iat[0] == 0 and lab["label_dn"].iat[0] == 0


def test_outside_horizon_and_tail_nan():
    h = [100, 100, 100, 100, 103, 100]
    lab = make_labels(bars([100] * 6, h, [100] * 6), H=3, up=0.02, adverse=0.01)
    assert lab["label_up"].iat[0] == 0           # TP 4. barda, H=3 dışında
    assert lab["label_up"].iat[1] == 1           # t1 için 3. bar = t4
    assert lab["label_up"].iloc[-3:].isna().all()


def test_short_label():
    h = [100, 100.5, 100]
    l = [100, 97.9, 100]
    lab = make_labels(bars([100] * 3, h, l), H=2, up=0.02, adverse=0.01)
    assert lab["label_dn"].iat[0] == 1


def test_label_uses_only_future_window():
    """t etiketi yalnızca t..t+H verisine bağlı olmalı."""
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    df = bars(c, c * 1.004, c * 0.996)
    a = make_labels(df, 24, 0.02, 0.01)
    df2 = df.copy(); df2.iloc[200:] *= 2
    b = make_labels(df2, 24, 0.02, 0.01)
    pd.testing.assert_frame_equal(a.iloc[: 200 - 24], b.iloc[: 200 - 24])

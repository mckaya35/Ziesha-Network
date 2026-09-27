"""highlow2 / triple-barrier etiketleri.
label_up[t]=1 <=> t+1..t+H içinde high >= close[t]*(1+up), low <= close[t]*(1-adverse) OLMADAN ÖNCE.
Aynı barda ikisi -> 0 (kötümser). Son H satır -> NaN (eğitimden çıkarılır)."""
import numpy as np
import pandas as pd


def _first_hit(hits):
    """hits: (T, H) bool -> ilk True sütunu (1..H), yoksa H+1."""
    any_ = hits.any(axis=1)
    first = hits.argmax(axis=1) + 1
    return np.where(any_, first, hits.shape[1] + 1)


def _future(x, H):
    T = len(x)
    m = np.full((T, H), np.nan)
    for k in range(1, H + 1):
        m[:T - k, k - 1] = x[k:]
    return m


def make_labels(df, H, up, adverse):
    c = df["close"].to_numpy(float)
    hi, lo = _future(df["high"].to_numpy(float), H), _future(df["low"].to_numpy(float), H)
    with np.errstate(invalid="ignore"):
        up_tp = _first_hit(hi >= (c * (1 + up))[:, None])
        up_sl = _first_hit(lo <= (c * (1 - adverse))[:, None])
        dn_tp = _first_hit(lo <= (c * (1 - up))[:, None])
        dn_sl = _first_hit(hi >= (c * (1 + adverse))[:, None])
    lab_up = ((up_tp <= H) & (up_tp < up_sl)).astype(float)
    lab_dn = ((dn_tp <= H) & (dn_tp < dn_sl)).astype(float)
    lab_up[-H:] = np.nan
    lab_dn[-H:] = np.nan
    return pd.DataFrame({"label_up": lab_up, "label_dn": lab_dn}, index=df.index)

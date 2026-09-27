"""Kaydırmalı walk-forward: 12 ay eğitim -> 1 ay test, 1 ay kaydır. TEK model, tüm sembollerin verisiyle.
Purge (eğitimin son H barı) + embargo (test öncesi H bar). Scaler/model/eşik yalnızca eğitimde.
Son `holdout_days` gün MÜHÜRLÜ: walk-forward buraya hiç dokunmaz (bkz. holdout_bounds)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backtest import run_backtest, trade_outcomes
from common import bar_delta, load_config
from data import load_all
from features import compute_features, feature_names
from labels import make_labels


def prepare(cfg, raw=None, use_api=True):
    """{sym: {"df", "fund", "feat", "lab", "oc"}}"""
    if raw is None:
        raw, _, _ = load_all(cfg, use_api)
    L = cfg["label"]
    out = {}
    for sym, (df, fund) in raw.items():
        out[sym] = {"df": df, "fund": fund, "feat": compute_features(df, cfg),
                    "lab": make_labels(df, L["H"], L["up"], L["adverse"]),
                    "oc": {s: trade_outcomes(df, fund, s, cfg) for s in (1, -1)}}
    return out


def holdout_bounds(cfg, data):
    """(ilk test ayı, walk-forward sinyal sonu, holdout başlangıcı, veri sonu)."""
    bar, H = bar_delta(cfg), cfg["label"]["H"]
    first = min(d["df"].index[0] for d in data.values())
    end = max(d["df"].index[-1] for d in data.values()) + bar
    hold = (end - pd.Timedelta(days=cfg["walkforward"]["holdout_days"])).floor("D")
    wf_end = hold - (H + 1) * bar        # WF işlemleri holdout başlamadan kapanmış olur (fiyatına bile bakmaz)
    ft = cfg["walkforward"]["first_test"]
    if ft == "auto":
        t = first + pd.DateOffset(months=cfg["walkforward"]["train_months"])
        ft = (t - pd.Timedelta(1, "ns")).normalize() + pd.offsets.MonthBegin(1)     # t'den sonraki ilk ay başı (t dahil)
    else:
        ft = pd.Timestamp(ft + "-01", tz="UTC")
    return ft, wf_end, hold, end


def fit(X, y, C):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(sc.transform(X), y)
    return sc, m


def predict(sc, m, X):
    return m.predict_proba(sc.transform(X))[:, 1]


def usable(d, pos, col):
    ok = d["feat"].iloc[pos].notna().all(axis=1).to_numpy() & d["lab"][col].iloc[pos].notna().to_numpy()
    return pos[ok]


def fold_positions(index, test_start, test_end, cfg):
    """Bir sembol için eğitim ve test satır pozisyonları (sızıntısız)."""
    H = cfg["label"]["H"]
    p0, p1 = int(index.searchsorted(test_start)), int(index.searchsorted(test_end))
    q0 = int(index.searchsorted(test_start - pd.DateOffset(months=cfg["walkforward"]["train_months"])))
    q1 = p0 - H - H           # embargo H + purge H
    return np.arange(q0, max(q0, q1)), np.arange(p0, p1)


def pooled_backtest(data, pos_by_sym, probs_by_sym, thr, side):
    """Tek yön, tek eşik, tüm semboller -> işlemlerin R dizisi."""
    Rs = []
    for s, pos in pos_by_sym.items():
        sig = np.zeros(len(data[s]["df"]), dtype=int)
        sig[pos[probs_by_sym[s] >= thr]] = side
        tr = run_backtest({side: data[s]["oc"][side]}, sig, s)
        Rs.append(tr["R"].to_numpy(float))
    return np.concatenate(Rs) if Rs else np.array([])


def select_threshold(data, pos_val, p_val, side, cfg):
    """Doğrulamada net ort. R > 0 ve ≥ min_val_trades olan eşiklerden toplam R'si en yüksek olan; yoksa None."""
    wf = cfg["walkforward"]
    best, best_tot = None, -np.inf
    for thr in wf["thresholds"]:
        R = pooled_backtest(data, pos_val, p_val, thr, side)
        if len(R) >= wf["min_val_trades"] and R.mean() > 0 and R.sum() > best_tot:
            best, best_tot = thr, R.sum()
    return best


def train_block(data, pos_train, cfg):
    """pos_train: {sym: purge edilmiş eğitim pozisyonları}. Dönüş {side: (scaler, model, eşik)}."""
    H, C, vf = cfg["label"]["H"], cfg["walkforward"]["C"], cfg["walkforward"]["val_frac"]
    names = feature_names(cfg)
    X = lambda pb: pd.concat([data[s]["feat"][names].iloc[p] for s, p in pb.items()])
    res = {}
    for side, col in ((1, "label_up"), (-1, "label_dn")):
        pos = {s: usable(data[s], p, col) for s, p in pos_train.items()}
        pos = {s: p for s, p in pos.items() if len(p)}
        times = np.concatenate([data[s]["df"].index[p].as_unit("ns").asi8 for s, p in pos.items()])
        val_start = pd.Timestamp(int(np.quantile(times, 1 - vf)), tz="UTC")
        pos_val, pos_in = {}, {}
        for s, p in pos.items():
            idx = data[s]["df"].index
            k = int(idx.searchsorted(val_start))
            pos_val[s] = p[p >= k]
            pos_in[s] = p[p < k - H]                    # iç purge
        pos_in = {s: p for s, p in pos_in.items() if len(p)}
        y = lambda pb: np.concatenate([data[s]["lab"][col].to_numpy()[p] for s, p in pb.items()])
        sc, m = fit(X(pos_in), y(pos_in), C)
        pv = {s: predict(sc, m, data[s]["feat"][names].iloc[p]) for s, p in pos_val.items() if len(p)}
        thr = select_threshold(data, {s: pos_val[s] for s in pv}, pv, side, cfg)
        sc, m = fit(X(pos), y(pos), C)                  # tam eğitimle yeniden fit
        res[side] = (sc, m, thr)
    return res


def signals_from(p_up, p_dn, thr_up, thr_dn):
    lu = p_up >= thr_up if thr_up is not None else np.zeros_like(p_up, dtype=bool)
    ld = p_dn >= thr_dn if thr_dn is not None else np.zeros_like(p_dn, dtype=bool)
    return np.where(lu & ~ld, 1, np.where(ld & ~lu, -1, 0))   # ikisi birden -> işlem yok


def predict_period(data, blk, test_start, test_end, cfg, fold):
    """Bir test dönemi için: eğitim (test_start öncesi) + tüm sembollerde tahmin."""
    names = feature_names(cfg)
    rows = []
    for s, d in data.items():
        _, te = fold_positions(d["df"].index, test_start, test_end, cfg)
        te = te[d["feat"].iloc[te].notna().all(axis=1).to_numpy()]
        if len(te) == 0:
            continue
        X = d["feat"][names].iloc[te]
        rows.append(pd.DataFrame({"symbol": s, "fold": fold, "pos": te, "time": d["df"].index[te],
                                  "p_up": predict(blk[1][0], blk[1][1], X), "p_dn": predict(blk[-1][0], blk[-1][1], X),
                                  "thr_up": blk[1][2] if blk[1][2] is not None else np.nan,
                                  "thr_dn": blk[-1][2] if blk[-1][2] is not None else np.nan}))
    return rows


def train_for(data, test_start, cfg, min_rows=1000):
    pos_tr = {}
    for s, d in data.items():
        tr, _ = fold_positions(d["df"].index, test_start, test_start, cfg)
        if len(tr):
            pos_tr[s] = tr
    if sum(len(p) for p in pos_tr.values()) < min_rows:
        return None, pos_tr
    return train_block(data, pos_tr, cfg), pos_tr


def model_row(cfg, fold, side, blk_side, pos_tr, data):
    sc, m, thr = blk_side
    names = feature_names(cfg)
    r = {"fold": fold, "side": side, "threshold": thr if thr is not None else np.nan, "intercept": m.intercept_[0],
         "train_first": min(data[s]["df"].index[p[0]] for s, p in pos_tr.items()),
         "train_last": max(data[s]["df"].index[p[-1]] for s, p in pos_tr.items()),
         "n_symbols": len(pos_tr)}
    r.update({f"coef_{n}": c for n, c in zip(names, m.coef_[0])})
    r.update({f"mean_{n}": v for n, v in zip(names, sc.mean_)})
    r.update({f"scale_{n}": v for n, v in zip(names, sc.scale_)})
    return r


def run(cfg, data=None):
    data = data or prepare(cfg)
    first, wf_end, hold, end = holdout_bounds(cfg, data)
    preds, models = [], []
    m = first
    while m < wf_end:
        m_end = min(m + pd.DateOffset(months=cfg["walkforward"]["test_months"]), wf_end)
        blk, pos_tr = train_for(data, m, cfg)
        fold = m.strftime("%Y-%m")
        if blk is not None:
            preds += predict_period(data, blk, m, m_end, cfg, fold)
            for side, name in ((1, "up"), (-1, "dn")):
                models.append(model_row(cfg, fold, name, blk[side], pos_tr, data))
            print(cfg["exp"], fold, "semboller", len(pos_tr), "thr_up", blk[1][2], "thr_dn", blk[-1][2])
        m = m + pd.DateOffset(months=cfg["walkforward"]["test_months"])
    preds, models = pd.concat(preds, ignore_index=True), pd.DataFrame(models)
    preds.to_parquet(f"{cfg['exp_out']}/folds.parquet")
    models.to_parquet(f"{cfg['exp_out']}/folds_models.parquet")
    return preds, models, data


def run_holdout(cfg, data):
    """Mühürlü holdout: holdout başlangıcından önceki 12 ayla eğitilir, holdout'a TEK kez uygulanır."""
    _, _, hold, end = holdout_bounds(cfg, data)
    blk, pos_tr = train_for(data, hold, cfg)
    preds = pd.concat(predict_period(data, blk, hold, end, cfg, "HOLDOUT"), ignore_index=True)
    models = pd.DataFrame([model_row(cfg, "HOLDOUT", n, blk[s], pos_tr, data) for s, n in ((1, "up"), (-1, "dn"))])
    return preds, models, hold, end


if __name__ == "__main__":
    import sys
    run(load_config(sys.argv[1] if len(sys.argv) > 1 else "main"))

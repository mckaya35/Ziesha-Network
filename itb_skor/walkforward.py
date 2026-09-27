"""Kaydırmalı walk-forward: 12 ay eğitim -> 1 ay test, 1 ay kaydır.
Purge (eğitimin son H barı) + embargo (test öncesi H bar). Scaler/model/eşik yalnızca eğitimde."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backtest import run_backtest, trade_outcomes
from common import load_config, load_funding, load_klines
from features import compute_features, feature_names
from labels import make_labels


def prepare(cfg, symbol):
    df = load_klines(cfg, symbol)
    fund = load_funding(cfg, symbol)
    feat = compute_features(df, cfg)
    lab = make_labels(df, cfg["label"]["H"], cfg["label"]["up"], cfg["label"]["adverse"])
    oc = {1: trade_outcomes(df, fund, 1, cfg), -1: trade_outcomes(df, fund, -1, cfg)}
    return df, feat, lab, oc


def fit(X, y, C):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(sc.transform(X), y)
    return sc, m


def predict(sc, m, X):
    return m.predict_proba(sc.transform(X))[:, 1]


def usable(feat, lab, pos, side_col):
    """pos içindeki, özellik ve etiketi tam olan satır pozisyonları."""
    ok = feat.iloc[pos].notna().all(axis=1).to_numpy() & lab[side_col].iloc[pos].notna().to_numpy()
    return pos[ok]


def select_threshold(p_val, pos_val, oc_side, side, cfg):
    """Doğrulama bölümünde net ort. R > 0 olan eşikler arasından toplam R'si en yüksek olanı seç.
    Hiçbiri pozitif değilse None (o fold'da bu yönde işlem yok)."""
    wf, T = cfg["walkforward"], len(oc_side)
    best, best_tot, table = None, -np.inf, []
    for thr in wf["thresholds"]:
        sig = np.zeros(T, dtype=int)
        sig[pos_val[p_val >= thr]] = side
        tr = run_backtest({side: oc_side, -side: oc_side}, sig)
        n = len(tr); mean = tr["R"].mean() if n else np.nan; tot = tr["R"].sum() if n else 0.0
        table.append((thr, n, mean))
        if n >= wf["min_val_trades"] and mean > 0 and tot > best_tot:
            best, best_tot = thr, tot
    return best, table


def train_block(feat, lab, oc, pos_train, cfg):
    """pos_train: purge edilmiş eğitim satır pozisyonları (zaman sıralı).
    Dönüş: {side: (scaler, model, threshold)}. Eşik yalnızca eğitim içi son %20'de seçilir."""
    H, C, vf = cfg["label"]["H"], cfg["walkforward"]["C"], cfg["walkforward"]["val_frac"]
    names = feature_names(cfg)
    res = {}
    for side, col in ((1, "label_up"), (-1, "label_dn")):
        pos = usable(feat, lab, pos_train, col)
        n_val = int(len(pos) * vf)
        pos_val = pos[-n_val:]
        pos_in = pos[: len(pos) - n_val]
        pos_in = pos_in[pos_in < pos_val[0] - H]              # iç purge: etiketi doğrulamaya taşan satırlar
        sc, m = fit(feat[names].iloc[pos_in], lab[col].iloc[pos_in].to_numpy(), C)
        thr, _ = select_threshold(predict(sc, m, feat[names].iloc[pos_val]), pos_val, oc[side], side, cfg)
        sc, m = fit(feat[names].iloc[pos], lab[col].iloc[pos].to_numpy(), C)   # tam eğitimle yeniden fit
        res[side] = (sc, m, thr)
    return res


def fold_positions(index, test_start, test_end, cfg):
    """Eğitim ve test satır pozisyonları (zaman sızıntısı yok)."""
    H = cfg["label"]["H"]
    p0 = int(index.searchsorted(test_start))
    p1 = int(index.searchsorted(test_end))
    train_start = test_start - pd.DateOffset(months=cfg["walkforward"]["train_months"])
    q0 = int(index.searchsorted(train_start))
    cut = p0 - H            # embargo: test başlangıcından önce H bar boşluk
    q1 = cut - H            # purge: eğitimin son H barı (etiketi ileri taşar) atılır
    return np.arange(q0, max(q0, q1)), np.arange(p0, p1)


def signals_from(p_up, p_dn, thr_up, thr_dn):
    lu = p_up >= thr_up if thr_up is not None else np.zeros_like(p_up, dtype=bool)
    ld = p_dn >= thr_dn if thr_dn is not None else np.zeros_like(p_dn, dtype=bool)
    return np.where(lu & ~ld, 1, np.where(ld & ~lu, -1, 0))   # ikisi birden -> işlem yok


def run(cfg):
    names = feature_names(cfg)
    preds, models = [], []
    for sym in cfg["symbols"]:
        df, feat, lab, oc = prepare(cfg, sym)
        idx = df.index
        m = pd.Timestamp(cfg["walkforward"]["first_test"] + "-01", tz="UTC")
        while m <= idx[-1]:
            m_end = m + pd.DateOffset(months=cfg["walkforward"]["test_months"])
            pos_tr, pos_te = fold_positions(idx, m, m_end, cfg)
            pos_te = pos_te[feat.iloc[pos_te].notna().all(axis=1).to_numpy()]
            if len(pos_tr) < 1000 or len(pos_te) == 0:
                m = m_end; continue
            blk = train_block(feat, lab, oc, pos_tr, cfg)
            Xte = feat[names].iloc[pos_te]
            p_up = predict(blk[1][0], blk[1][1], Xte)
            p_dn = predict(blk[-1][0], blk[-1][1], Xte)
            fold = m.strftime("%Y-%m")
            preds.append(pd.DataFrame({"symbol": sym, "fold": fold, "pos": pos_te, "time": idx[pos_te],
                                       "p_up": p_up, "p_dn": p_dn,
                                       "thr_up": blk[1][2] if blk[1][2] is not None else np.nan,
                                       "thr_dn": blk[-1][2] if blk[-1][2] is not None else np.nan,
                                       "label_up": lab["label_up"].iloc[pos_te].to_numpy(),
                                       "label_dn": lab["label_dn"].iloc[pos_te].to_numpy()}))
            for side, name in ((1, "up"), (-1, "dn")):
                sc, mdl, thr = blk[side]
                row = {"symbol": sym, "fold": fold, "side": name, "threshold": thr if thr is not None else np.nan,
                       "train_first": idx[pos_tr[0]], "train_last": idx[pos_tr[-1]], "intercept": mdl.intercept_[0]}
                row.update({f"coef_{n}": c for n, c in zip(names, mdl.coef_[0])})
                models.append(row)
            print(sym, fold, "thr_up", blk[1][2], "thr_dn", blk[-1][2])
            m = m_end
    preds, models = pd.concat(preds, ignore_index=True), pd.DataFrame(models)
    preds.to_parquet(f"{cfg['out_dir']}/folds.parquet")
    models.to_parquet(f"{cfg['out_dir']}/folds_models.parquet")
    return preds, models


if __name__ == "__main__":
    run(load_config())

"""Walk-forward XGBoost backtest + karşılaştırmalar + rapor."""
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import data
import features
import strategy as st

SEED, N_RANDOM = 42, 200
GRID = list(itertools.product([3, 5], [0.03, 0.1]))
REPORTS = Path(__file__).parent / "reports"
M = pd.DateOffset


def folds(index, train=12, val=3, test=3, step=3):
    s, end = index[0], index[-1]
    while s + M(months=train + val) <= end:
        t0, v0, e0 = s, s + M(months=train), s + M(months=train + val)
        yield t0, v0, e0, min(e0 + M(months=test), end + pd.Timedelta(hours=1))
        s += M(months=step)


def _model(depth, lr, n=200, es=True):
    return XGBRegressor(max_depth=depth, learning_rate=lr, n_estimators=n, subsample=0.8, colsample_bytree=0.8,
                        early_stopping_rounds=20 if es else None, random_state=SEED, n_jobs=-1, verbosity=0)


def fit_predict(Xtr, ytr, Xva, yva, Xte):
    best = None
    for depth, lr in GRID:
        m = _model(depth, lr).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        rmse = m.evals_result()["validation_0"]["rmse"][m.best_iteration]
        if best is None or rmse < best[0]:
            best = (rmse, depth, lr, m.best_iteration + 1)
    _, depth, lr, n = best
    m = _model(depth, lr, n, es=False).fit(pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
    return m.predict(Xte), (depth, lr, n)


def walk_forward(X, y, shuffle=False):
    rng = np.random.default_rng(SEED)
    preds, fold_id, params = [], [], []
    for k, (t0, v0, e0, e1) in enumerate(folds(X.index)):
        tr, va, te = (X.index >= t0) & (X.index < v0), (X.index >= v0) & (X.index < e0), (X.index >= e0) & (X.index < e1)
        if not te.any():
            break
        ytr, yva = y[tr].to_numpy(), y[va].to_numpy()
        if shuffle:
            ytr, yva = rng.permutation(ytr), rng.permutation(yva)
        p, prm = fit_predict(X[tr], ytr, X[va], yva, X[te])
        preds.append(pd.Series(p, index=X.index[te]))
        fold_id += [k] * te.sum()
        params.append((str(e0.date()), *prm))
    return pd.concat(preds), np.array(fold_id), params


def main():
    kl, fund = data.load()
    X, y = features.build(kl)
    pred, fid, params = walk_forward(X, y)
    pred_sh, _, _ = walk_forward(X, y, shuffle=True)
    idx = pred.index
    yt, fr = y.loc[idx].to_numpy(), st.funding_per_bar(idx, fund)
    zero = np.zeros_like(fr)

    pos = st.positions(pred.to_numpy())
    r_nf, n_tr = st.net_returns(pos, yt, zero)
    r_f, _ = st.net_returns(pos, yt, fr)
    r_bh, n_bh = st.net_returns(np.ones_like(yt), yt, fr)
    r_sh, n_sh = st.net_returns(st.positions(pred_sh.to_numpy()), yt, fr)

    rng = np.random.default_rng(SEED)
    rand = [st.net_returns(st.random_positions(len(yt), n_tr, rng), yt, fr) for _ in range(N_RANDOM)]
    rand_m = pd.DataFrame([st.metrics(r, n, fid) for r, n in rand])
    rand_m["Kârlı fold"] = rand_m["Kârlı fold"].str.split("/").str[0].astype(int)
    med = rand_m.median(numeric_only=True).to_dict()
    med["İşlem"] = int(med["İşlem"])
    med["Kârlı fold"] = f"{int(med['Kârlı fold'])}/{len(np.unique(fid))}"

    rows = {
        "Strateji (fundingsiz)": st.metrics(r_nf, n_tr, fid),
        "Strateji (fundingli)": st.metrics(r_f, n_tr, fid),
        "Al-tut": st.metrics(r_bh, n_bh, fid),
        "Etiket karıştırma": st.metrics(r_sh, n_sh, fid),
        f"Rastgele medyan (n={N_RANDOM})": med,
    }
    tab = pd.DataFrame(rows).T
    for col in ("Yıllık getiri", "Yıllık vol", "Maks. düşüş"):
        tab[col] = tab[col].map(lambda v: f"{v:.1%}")
    tab["Sharpe"] = tab["Sharpe"].map(lambda v: f"{v:.2f}")
    pct = (rand_m["Sharpe"] < rows["Strateji (fundingli)"]["Sharpe"]).mean() * 100

    print(f"Test dönemi: {idx[0]} -> {idx[-1]} | fold: {len(params)} | c={st.C}, λ={st.LAM}")
    print("Fundingsiz satır hariç tüm satırlar funding ve maliyet sonrası.\n")
    print(tab.to_string())
    print(f"\nStrateji (fundingli) Sharpe'ı rastgele dağılımda %{pct:.1f} yüzdelikte.")

    REPORTS.mkdir(exist_ok=True)
    rand_eq = np.median(np.exp(np.cumsum([r for r, _ in rand], axis=1)), axis=0)
    fig, ax = plt.subplots(figsize=(11, 5))
    for lbl, r in [("Strateji (fundingsiz)", r_nf), ("Strateji (fundingli)", r_f), ("Al-tut", r_bh),
                   ("Etiket karıştırma", r_sh)]:
        ax.plot(idx, np.exp(np.cumsum(r)), label=lbl, lw=1)
    ax.plot(idx, rand_eq, label="Rastgele medyan", lw=1, ls="--")
    ax.set_yscale("log"); ax.set_ylabel("Equity (log)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(REPORTS / "equity.png", dpi=120)
    return tab


if __name__ == "__main__":
    main()

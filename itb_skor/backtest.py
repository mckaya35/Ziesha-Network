"""Kötümser işlem simülasyonu.
Sinyal bar kapanışında -> giriş bir sonraki barın AÇILIŞI. TP=+up, SL=-adverse, yoksa H bar sonunda kapanış.
Aynı barda TP ve SL -> SL. Aynı sembolde tek pozisyon. Maliyet: komisyon+kayma (taraf başına), gerçek funding."""
import numpy as np
import pandas as pd

from labels import _first_hit, _future


def trade_outcomes(df, funding, side, cfg, cost_mult=1.0):
    """Her bar i için 'i kapanışında sinyal gelseydi' işlemin sonucu (pozisyon kısıtı yok).
    side=+1 long, -1 short. Satır başına: exit_idx, reason, gross, cost, fund, net, R."""
    H, up, adv = cfg["label"]["H"], cfg["label"]["up"], cfg["label"]["adverse"]
    side_cost = (cfg["costs"]["fee"] + cfg["costs"]["slippage"]) * cost_mult
    T = len(df)
    o = df["open"].to_numpy(float)
    entry = np.r_[o[1:], np.nan]                       # open[i+1]
    fo, fh, fl, fc = (_future(df[k].to_numpy(float), H) for k in ("open", "high", "low", "close"))
    if side == 1:
        tp, sl = entry * (1 + up), entry * (1 - adv)
        with np.errstate(invalid="ignore"):
            k_tp, k_sl = _first_hit(fh >= tp[:, None]), _first_hit(fl <= sl[:, None])
    else:
        tp, sl = entry * (1 - up), entry * (1 + adv)
        with np.errstate(invalid="ignore"):
            k_tp, k_sl = _first_hit(fl <= tp[:, None]), _first_hit(fh >= sl[:, None])
    k = np.minimum(np.minimum(k_tp, k_sl), H)
    reason = np.where(k_sl <= np.minimum(k_tp, H), "sl", np.where(k_tp <= H, "tp", "time"))
    rows = np.arange(T)
    kk = k - 1
    open_k = fo[rows, kk]
    # SL'de boşluk (gap) varsa daha kötü fiyat: long için min(open, sl), short için max(open, sl)
    sl_px = np.minimum(open_k, sl) if side == 1 else np.maximum(open_k, sl)
    exit_px = np.where(reason == "sl", sl_px, np.where(reason == "tp", tp, fc[rows, H - 1]))
    gross = side * (exit_px / entry - 1)
    cost = side_cost * (1 + exit_px / entry)

    # Funding: (giriş anı, çıkış sınırı] aralığındaki funding anları. Long öder (+rate), short alır.
    t = df.index
    one_h = pd.Timedelta(hours=1)
    t_ns = t.as_unit("ns").asi8
    entry_t = np.r_[t_ns[1:], np.iinfo(np.int64).max]
    exit_idx = rows + k
    exit_idx_c = np.minimum(exit_idx, T - 1)
    exit_bound = np.where(reason == "time", t_ns[exit_idx_c] + one_h.value, t_ns[exit_idx_c])
    fund = np.zeros(T)
    if len(funding):
        ft = funding.index.as_unit("ns").asi8
        cum = np.r_[0.0, np.cumsum(funding.to_numpy(float))]
        fund = cum[np.searchsorted(ft, exit_bound, "right")] - cum[np.searchsorted(ft, entry_t, "right")]
    fund_pnl = -side * fund * cost_mult

    net = gross - cost + fund_pnl
    valid = rows + H <= T - 1
    out = pd.DataFrame({"exit_idx": exit_idx, "reason": reason, "gross": gross, "cost": cost,
                        "fund": fund_pnl, "net": net, "R": net / adv, "hold": k}, index=t)
    out.loc[~valid, ["gross", "cost", "fund", "net", "R"]] = np.nan
    out["valid"] = valid
    return out


def run_backtest(outcomes, signal, symbol=""):
    """outcomes: {1: long_df, -1: short_df}; signal: bar başına +1/-1/0 (np.array, df ile hizalı).
    Tek pozisyon kuralı: sinyal ancak önceki işlemin çıkış barı kapandıktan sonra alınır."""
    trades, free_from = [], 0
    idx = outcomes[1].index
    for i in np.flatnonzero(signal != 0):
        if i < free_from:
            continue
        s = int(signal[i]); oc = outcomes[s]
        if not oc["valid"].iat[i]:
            continue
        r = oc.iloc[i]
        ex = int(r["exit_idx"])
        trades.append({"symbol": symbol, "side": s, "signal_time": idx[i], "entry_time": idx[i + 1],
                       "exit_time": idx[ex], "reason": r["reason"], "hold_h": int(r["hold"]),
                       "gross": r["gross"], "cost": r["cost"], "fund": r["fund"], "net": r["net"], "R": r["R"]})
        free_from = ex
    return pd.DataFrame(trades)


def metrics(tr, adv, days=None):
    """tr: işlemler. days: Sharpe için gün aralığı (DatetimeIndex, UTC tarih)."""
    if tr is None or len(tr) == 0:
        return {"trades": 0}
    R = tr["R"]
    pos, neg = R[R > 0].sum(), -R[R < 0].sum()
    eq = R.loc[tr["exit_time"].sort_values().index].cumsum()
    dd = float((eq.cummax().clip(lower=0) - eq).max())
    daily = tr.groupby(tr["exit_time"].dt.floor("D"))["R"].sum()
    if days is not None:
        daily = daily.reindex(days, fill_value=0.0)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else np.nan
    gross_R, cost_R = tr["gross"].sum() / adv, (tr["cost"].sum() - tr["fund"].sum()) / adv
    return {"trades": int(len(tr)), "win_rate": float((R > 0).mean()), "avg_R": float(R.mean()),
            "total_R": float(R.sum()), "profit_factor": float(pos / neg) if neg > 0 else np.inf,
            "max_dd_R": dd, "sharpe_daily": sharpe, "avg_hold_h": float(tr["hold_h"].mean()),
            "gross_R": float(gross_R), "cost_R": float(cost_R),
            "cost_to_gross": float(cost_R / abs(gross_R)) if gross_R != 0 else np.nan}


def random_benchmark(pools, counts, reps, seed):
    """pools: {(sym, side): R dizisi (test barlarında geçerli)}, counts: {(sym, side): n}.
    Her tekrarda aynı sayıda rastgele giriş -> ortalama R dağılımı."""
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        vals = [rng.choice(pools[k], size=n, replace=False) for k, n in counts.items() if n > 0]
        out[r] = np.concatenate(vals).mean() if vals else np.nan
    return out

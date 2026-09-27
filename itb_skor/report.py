"""Walk-forward tahminlerinden işlemleri simüle eder, kıyaslar, başarı kriterini uygular -> report.md + grafikler."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import metrics, random_benchmark, run_backtest, trade_outcomes
from common import load_config, load_funding
from features import feature_names
from walkforward import prepare, signals_from


def simulate(cfg, preds):
    adv = cfg["label"]["adverse"]
    out = {"model": [], "model_x2": [], "sma168": []}
    pools, counts, bh, days = {}, {}, {}, []
    for sym in cfg["symbols"]:
        p = preds[preds["symbol"] == sym]
        if p.empty:
            continue
        df, feat, lab, oc = prepare(cfg, sym)
        fund = load_funding(cfg, sym)
        oc2 = {s: trade_outcomes(df, fund, s, cfg, cost_mult=cfg["criteria"]["cost_mult"]) for s in (1, -1)}
        T, pos = len(df), p["pos"].to_numpy()
        sig = np.zeros(T, dtype=int)
        for _, g in p.groupby("fold"):
            thr_u = None if np.isnan(g["thr_up"].iat[0]) else g["thr_up"].iat[0]
            thr_d = None if np.isnan(g["thr_dn"].iat[0]) else g["thr_dn"].iat[0]
            sig[g["pos"].to_numpy()] = signals_from(g["p_up"].to_numpy(), g["p_dn"].to_numpy(), thr_u, thr_d)
        tr = run_backtest(oc, sig, sym)
        out["model"].append(tr)
        out["model_x2"].append(run_backtest(oc2, sig, sym))

        # Kıyas 2: close > SMA168 -> long, altında -> short (aynı TP/SL, aynı test barları)
        s168 = np.zeros(T, dtype=int)
        r = feat["sma_ratio_168"].to_numpy()[pos]           # SMA/close - 1 < 0  <=> close > SMA
        s168[pos] = np.where(r < 0, 1, np.where(r > 0, -1, 0))
        out["sma168"].append(run_backtest(oc, s168, sym))

        # Kıyas 1 havuzu: test barlarında rastgele giriş
        for s in (1, -1):
            v = oc[s]["R"].to_numpy()[pos]
            pools[(sym, s)] = v[~np.isnan(v)]
            counts[(sym, s)] = int((tr["side"] == s).sum()) if len(tr) else 0

        # Kıyas 3: al-ve-tut (ilk test barı açılış -> son test barı kapanış, 2 taraf maliyet + long funding)
        o0, c1 = df["open"].iat[pos[0]], df["close"].iat[pos[-1]]
        t0, t1 = df.index[pos[0]], df.index[pos[-1]]
        f = fund[(fund.index > t0) & (fund.index <= t1 + pd.Timedelta(hours=1))].sum()
        side_cost = cfg["costs"]["fee"] + cfg["costs"]["slippage"]
        ret = c1 / o0 - 1 - side_cost * (1 + c1 / o0) - f
        yearly = df["close"].iloc[pos].groupby(df.index[pos].year).agg(["first", "last"])
        bh[sym] = {"from": str(t0), "to": str(t1), "net_return": float(ret), "R": float(ret / adv),
                   "yearly_return": {int(y): float(v["last"] / v["first"] - 1) for y, v in yearly.iterrows()}}
        days.append(pd.date_range(t0.floor("D"), t1.floor("D"), freq="D"))
    for k in out:
        out[k] = pd.concat([x for x in out[k] if len(x)], ignore_index=True) if any(len(x) for x in out[k]) else pd.DataFrame()
    all_days = pd.DatetimeIndex(sorted(set().union(*days))) if days else None
    rnd = random_benchmark(pools, counts, cfg["benchmark"]["random_reps"], cfg["benchmark"]["seed"])
    return out, rnd, bh, all_days


def breakdown(tr, adv, days):
    rows = {"TOPLAM": metrics(tr, adv, days)}
    if len(tr) == 0:
        return pd.DataFrame(rows).T
    for s, g in tr.groupby("symbol"):
        rows[f"sembol {s}"] = metrics(g, adv)
    for y, g in tr.groupby(tr["entry_time"].dt.year):
        rows[f"yıl {y}"] = metrics(g, adv)
    for s, g in tr.groupby("side"):
        rows["long" if s == 1 else "short"] = metrics(g, adv)
    return pd.DataFrame(rows).T


def criteria(cfg, tr, tr2, sma, rnd):
    c = cfg["criteria"]
    n = len(tr)
    avg = tr["R"].mean() if n else np.nan
    p95 = float(np.nanpercentile(rnd, c["random_pct"])) if n else np.nan
    sma_avg = sma["R"].mean() if len(sma) else np.nan
    yearly = tr.groupby(tr["entry_time"].dt.year)["R"].sum() if n else pd.Series(dtype=float)
    need_years = math.ceil(c["year_frac"] * len(yearly) - 1e-9)
    avg2 = tr2["R"].mean() if len(tr2) else np.nan
    res = [
        ("İşlem sayısı ≥ %d" % c["min_trades"], n >= c["min_trades"], f"{n}"),
        ("Net ort. R > 0", bool(avg > 0), f"{avg:.4f}"),
        (f"Ort. R > rastgele girişlerin %{c['random_pct']}'i", bool(avg > p95), f"model {avg:.4f} vs p{c['random_pct']} {p95:.4f}"),
        ("SMA168 kuralından iyi (ort. R)", bool(avg > sma_avg), f"model {avg:.4f} vs SMA168 {sma_avg:.4f}"),
        (f"Pozitif yıl ≥ {need_years}/{len(yearly)}", bool((yearly > 0).sum() >= need_years and len(yearly) > 0),
         ", ".join(f"{y}: {v:+.1f}R" for y, v in yearly.items())),
        (f"Maliyet ×{c['cost_mult']:g} iken ort. R ≥ 0", bool(avg2 >= 0), f"{avg2:.4f}"),
    ]
    return res, all(r[1] for r in res)


def plots(cfg, out, rnd, models, od):
    tr, sma = out["model"], out["sma168"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, t in (("Model", tr), ("SMA168 kuralı", sma)):
        if len(t):
            s = t.sort_values("exit_time")
            ax.plot(s["exit_time"], s["R"].cumsum(), label=name)
    ax.set_title("Net sermaye eğrisi (kümülatif R, tüm semboller)"); ax.set_ylabel("R"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(od / "equity.png", dpi=110); plt.close(fig)

    if len(tr):
        mon = tr.groupby([tr["entry_time"].dt.year, tr["entry_time"].dt.month])["R"].sum().unstack().reindex(columns=range(1, 13))
        fig, ax = plt.subplots(figsize=(10, 0.5 + 0.5 * len(mon)))
        v = np.nanmax(np.abs(mon.to_numpy())) or 1
        im = ax.imshow(mon.to_numpy(), cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(12), range(1, 13)); ax.set_yticks(range(len(mon)), mon.index)
        for (i, j), x in np.ndenumerate(mon.to_numpy()):
            if not np.isnan(x):
                ax.text(j, i, f"{x:.0f}", ha="center", va="center", fontsize=8)
        ax.set_title("Aylık net R"); fig.colorbar(im); fig.tight_layout(); fig.savefig(od / "monthly_R.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rnd[~np.isnan(rnd)], bins=50, color="0.7", label="Rastgele giriş (ort. R)")
    if len(tr):
        ax.axvline(tr["R"].mean(), color="C3", lw=2, label="Model")
    if len(sma):
        ax.axvline(sma["R"].mean(), color="C0", ls="--", label="SMA168")
    ax.axvline(np.nanpercentile(rnd, cfg["criteria"]["random_pct"]), color="k", ls=":", label=f"p{cfg['criteria']['random_pct']}")
    ax.set_title("Modelin rastgele giriş dağılımındaki yeri"); ax.legend(); fig.tight_layout(); fig.savefig(od / "random_dist.png", dpi=110); plt.close(fig)

    names = feature_names(cfg)
    groups = list(models.groupby(["symbol", "side"]))
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 3 * len(groups)), squeeze=False)
    for ax, ((sym, side), g) in zip(axes[:, 0], groups):
        m = g[[f"coef_{n}" for n in names]].to_numpy().T
        v = np.nanmax(np.abs(m)) or 1
        ax.imshow(m, cmap="coolwarm", vmin=-v, vmax=v, aspect="auto")
        ax.set_yticks(range(len(names)), names, fontsize=7)
        step = max(1, len(g) // 12)
        ax.set_xticks(range(0, len(g), step), g["fold"].iloc[::step], fontsize=7, rotation=45)
        ax.set_title(f"{sym} {side}: katsayılar (fold bazında, standartlaştırılmış)")
    fig.tight_layout(); fig.savefig(od / "coef_stability.png", dpi=110); plt.close(fig)


def coef_stability(cfg, models):
    names = feature_names(cfg)
    rows = []
    for (sym, side), g in models.groupby(["symbol", "side"]):
        for n in names:
            s = np.sign(g[f"coef_{n}"])
            rows.append({"symbol": sym, "side": side, "feature": n,
                         "ayni_isaret_orani": float(max((s > 0).mean(), (s < 0).mean())),
                         "isaret_degisimi": int((s.diff().fillna(0) != 0).sum())})
    return pd.DataFrame(rows)


def fmt(df):
    """Bağımlılıksız markdown tablo."""
    def cell(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return ""
            return f"{int(x)}" if float(x).is_integer() and abs(x) >= 1 else f"{x:.3f}"
        return str(x)
    cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    L = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    L += ["| " + str(i) + " | " + " | ".join(cell(v) for v in r) + " |" for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join(L)


def main(cfg=None, banner=""):
    cfg = cfg or load_config()
    od = Path(cfg["out_dir"])
    preds = pd.read_parquet(od / "folds.parquet")
    models = pd.read_parquet(od / "folds_models.parquet")
    adv = cfg["label"]["adverse"]
    out, rnd, bh, days = simulate(cfg, preds)
    tr, tr2, sma = out["model"], out["model_x2"], out["sma168"]
    for k, t in out.items():
        t.to_csv(od / f"trades_{k}.csv", index=False)
    crit, ok = criteria(cfg, tr, tr2, sma, rnd)
    plots(cfg, out, rnd, models, od)
    stab = coef_stability(cfg, models)
    stab.to_csv(od / "coef_stability.csv", index=False)
    no_trade = models.assign(none=models["threshold"].isna()).groupby(["symbol", "side"])["none"].mean()

    L = [f"# ITB Skor — Walk-forward raporu\n", banner,
         f"**SONUÇ: {'GEÇTİ' if ok else 'GEÇMEDİ'}**\n"]
    if not ok:
        L.append("Başarısız kriter(ler): " + "; ".join(f"**{c[0]}** ({c[2]})" for c in crit if not c[1]) + "\n")
        L.append("Kural gereği parametreler test sonucuna bakılarak AYARLANMADI.\n")
    L += ["## Başarı kriterleri\n", "| Kriter | Durum | Değer |", "|---|---|---|"]
    L += [f"| {c[0]} | {'✅' if c[1] else '❌'} | {c[2]} |" for c in crit]
    L += ["", "## Kurulum", f"- Semboller: {', '.join(cfg['symbols'])}, aralık {cfg['interval']}",
          f"- Etiket/bariyer: H={cfg['label']['H']} bar, TP=+{cfg['label']['up']:.2%}, SL=-{cfg['label']['adverse']:.2%} (1R)",
          f"- Walk-forward: eğitim {cfg['walkforward']['train_months']} ay → test {cfg['walkforward']['test_months']} ay, ilk test {cfg['walkforward']['first_test']}, purge+embargo {cfg['label']['H']}+{cfg['label']['H']} bar",
          f"- Maliyet: komisyon {cfg['costs']['fee']:.3%} + kayma {cfg['costs']['slippage']:.3%} / taraf, gerçek funding",
          f"- Fold sayısı: {preds['fold'].nunique()} ay × {preds['symbol'].nunique()} sembol",
          "- Eşiği bulunamayan (doğrulamada pozitif beklenen değer yok → işlem yok) fold oranı: " +
          ", ".join(f"{s} {d}: {v:.0%}" for (s, d), v in no_trade.items()), ""]
    L += ["## Model metrikleri (net)\n", fmt(breakdown(tr, adv, days)), ""]
    L += [f"## Maliyet ×{cfg['criteria']['cost_mult']:g}\n", fmt(breakdown(tr2, adv, days).loc[["TOPLAM"]]), ""]
    L += ["## Kıyaslar\n", "### SMA168 kuralı\n", fmt(breakdown(sma, adv, days)), ""]
    if len(rnd) and not np.all(np.isnan(rnd)):
        L += ["### Rastgele girişler (aynı sembol/yön bazında işlem sayısı, %d tekrar)\n" % len(rnd),
              f"- Ort. R dağılımı: ortalama {np.nanmean(rnd):.4f}, p5 {np.nanpercentile(rnd, 5):.4f}, p50 {np.nanpercentile(rnd, 50):.4f}, p95 {np.nanpercentile(rnd, 95):.4f}",
              f"- Modelin yüzdeliği: {(rnd < tr['R'].mean()).mean():.1%}" if len(tr) else "", ""]
    L += ["### Al-ve-tut (test dönemi)\n", "| Sembol | Dönem | Net getiri | R karşılığı | Yıllık getiri |", "|---|---|---|---|---|"]
    L += [f"| {s} | {v['from'][:10]} → {v['to'][:10]} | {v['net_return']:.1%} | {v['R']:.1f} | " +
          ", ".join(f"{y}: {r:+.0%}" for y, r in v["yearly_return"].items()) + " |" for s, v in bh.items()]
    L += ["", "## Katsayı kararlılığı\n",
          "Her özellik için fold'ların ne kadarında katsayının çoğunluk işaretinde kaldığı (1.0 = hiç değişmedi):\n",
          fmt(stab.pivot_table(index="feature", columns=["symbol", "side"], values="ayni_isaret_orani")), "",
          "## Grafikler\n", "![equity](equity.png)\n", "![monthly](monthly_R.png)\n", "![random](random_dist.png)\n", "![coef](coef_stability.png)\n",
          "## Notlar",
          "- Tüm işlemler kötümser: giriş sinyalden sonraki barın açılışında, aynı barda TP+SL → SL, SL'de fiyat boşluğu varsa daha kötü fiyat, TP'de iyileştirme yok.",
          "- Maliyet ×2 senaryosunda komisyon, kayma ve funding birlikte ikiye katlanır.",
          "- Bu rapor geçmiş veriye dayanır; gelecekteki performansı garanti etmez. Yatırım tavsiyesi değildir."]
    (od / "report.md").write_text("\n".join(L))
    (od / "summary.json").write_text(json.dumps({"passed": ok, "criteria": [[c[0], bool(c[1]), c[2]] for c in crit]}, ensure_ascii=False, indent=1))
    print("\n".join(f"{'OK ' if c[1] else 'XX '} {c[0]}: {c[2]}" for c in crit))
    print("SONUÇ:", "GEÇTİ" if ok else "GEÇMEDİ", "->", od / "report.md")
    return ok


if __name__ == "__main__":
    main()

"""İşlem simülasyonu + kıyaslar + başarı kriteri + trend/kontrol analizi + mühürlü holdout -> report.md."""
import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import metrics, random_benchmark, run_backtest, trade_outcomes
from common import config_hash
from features import feature_names
from walkforward import run_holdout, signals_from

TREND_FEATS = ["sma_ratio_24", "sma_ratio_72", "sma_ratio_168", "slope_24", "slope_72", "slope_168", "ret_24", "ret_72", "ret_168"]


# ---------------------------------------------------------------- simülasyon
def model_signal(T, p):
    sig = np.zeros(T, dtype=int)
    for _, g in p.groupby("fold"):
        tu = None if np.isnan(g["thr_up"].iat[0]) else g["thr_up"].iat[0]
        td = None if np.isnan(g["thr_dn"].iat[0]) else g["thr_dn"].iat[0]
        sig[g["pos"].to_numpy()] = signals_from(g["p_up"].to_numpy(), g["p_dn"].to_numpy(), tu, td)
    return sig


def simulate(cfg, data, preds):
    adv, cm = cfg["label"]["adverse"], cfg["criteria"]["cost_mult"]
    side_cost = cfg["costs"]["fee"] + cfg["costs"]["slippage"]
    out = {"model": [], "model_x2": [], "sma168": []}
    pools, counts, bh, days = {}, {}, {}, []
    for sym, p in preds.groupby("symbol"):
        d = data[sym]; df, T, pos = d["df"], len(d["df"]), p["pos"].to_numpy()
        sig = model_signal(T, p)
        tr = run_backtest(d["oc"], sig, sym)
        out["model"].append(tr)
        oc2 = {s: trade_outcomes(df, d["fund"], s, cfg, cost_mult=cm) for s in (1, -1)}
        out["model_x2"].append(run_backtest(oc2, sig, sym))
        s168 = np.zeros(T, dtype=int)                       # close > SMA168 -> long, altında -> short
        r = d["feat"]["sma_ratio_168"].to_numpy()[pos]      # SMA/close - 1 < 0  <=> close > SMA
        s168[pos] = np.where(r < 0, 1, np.where(r > 0, -1, 0))
        out["sma168"].append(run_backtest(d["oc"], s168, sym))
        for s in (1, -1):
            v = d["oc"][s]["R"].to_numpy()[pos]
            pools[(sym, s)] = v[~np.isnan(v)]
            counts[(sym, s)] = int((tr["side"] == s).sum())
        o0, c1, t0, t1 = df["open"].iat[pos[0]], df["close"].iat[pos[-1]], df.index[pos[0]], df.index[pos[-1]]
        f = d["fund"][(d["fund"].index > t0) & (d["fund"].index <= t1)].sum()
        ret = c1 / o0 - 1 - side_cost * (1 + c1 / o0) - f
        bh[sym] = {"from": t0, "to": t1, "net_return": float(ret), "R": float(ret / adv)}
        days.append(pd.date_range(t0.floor("D"), t1.floor("D"), freq="D"))
    for k in out:
        out[k] = pd.concat(out[k], ignore_index=True)
    all_days = pd.DatetimeIndex(sorted(set().union(*days)))
    counts = {k: min(n, len(pools[k])) for k, n in counts.items()}
    rnd = random_benchmark(pools, counts, cfg["benchmark"]["random_reps"], cfg["benchmark"]["seed"])
    return out, rnd, bh, all_days


def pick(df, rows):
    """Var olan satırları seç (ör. hiç long işlem yoksa 'long' satırı olmaz)."""
    return df.loc[[r for r in rows if r in df.index]]


def breakdown(tr, adv, days):
    rows = {"TOPLAM": metrics(tr, adv, days)}
    if len(tr):
        for s, g in tr.groupby("symbol"):
            rows[s] = metrics(g, adv)
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
    need = math.ceil(Fraction(str(c["year_frac"])) * len(yearly))
    avg2 = tr2["R"].mean() if len(tr2) else np.nan
    res = [
        (f"İşlem sayısı ≥ {c['min_trades']}", n >= c["min_trades"], f"{n}"),
        ("Net ort. R > 0", bool(avg > 0), f"{avg:.4f}"),
        (f"Ort. R > rastgele girişlerin %{c['random_pct']}'i", bool(avg > p95), f"model {avg:.4f} vs p{c['random_pct']} {p95:.4f}"),
        ("SMA168 kuralından iyi (işlem başına ort. R)", bool(avg > sma_avg), f"model {avg:.4f} vs SMA168 {sma_avg:.4f}"),
        (f"Pozitif yıl ≥ {need}/{len(yearly)}", bool(len(yearly) > 0 and (yearly > 0).sum() >= need),
         ", ".join(f"{y}: {v:+.1f}R" for y, v in yearly.items())),
        (f"Maliyet ×{c['cost_mult']:g} iken ort. R ≥ 0", bool(avg2 >= 0), f"{avg2:.4f}"),
    ]
    return res, all(r[1] for r in res)


# ---------------------------------------------------------------- trend / kontrol analizi
def trend_analysis(cfg, data, preds, tr, sma):
    """Skor, basit trend etkisinden farklı bir şey mi yakalıyor?"""
    rows = []
    for sym, p in preds.groupby("symbol"):
        f = data[sym]["feat"].iloc[p["pos"].to_numpy()]
        rows.append(pd.concat([p[["symbol", "time"]].reset_index(drop=True),
                               (p["p_up"] - p["p_dn"]).rename("score").reset_index(drop=True),
                               f[TREND_FEATS].reset_index(drop=True)], axis=1))
    X = pd.concat(rows, ignore_index=True).dropna()
    res = {"spearman": {k: float(X["score"].corr(-X[k] if k.startswith("sma_ratio") else X[k], method="spearman"))
                        for k in TREND_FEATS}}
    A = np.c_[np.ones(len(X)), (X[TREND_FEATS] - X[TREND_FEATS].mean()) / X[TREND_FEATS].std()]
    beta, *_ = np.linalg.lstsq(A, X["score"].to_numpy(), rcond=None)
    resid = X["score"].to_numpy() - A @ beta
    res["r2_trend"] = float(1 - resid.var() / X["score"].var())
    if len(tr):
        al = []
        for sym, g in tr.groupby("symbol"):
            r = data[sym]["feat"]["sma_ratio_168"].reindex(g["signal_time"]).to_numpy()
            al.append(pd.Series(np.sign(-r) == g["side"].to_numpy(), index=g.index))
        aligned = pd.concat(al).reindex(tr.index)
        res["aligned_frac"] = float(aligned.mean())
        res["aligned"] = (int(aligned.sum()), float(tr.loc[aligned, "R"].mean()) if aligned.any() else np.nan)
        res["against"] = (int((~aligned).sum()), float(tr.loc[~aligned, "R"].mean()) if (~aligned).any() else np.nan)
    res["sma_avg"] = float(sma["R"].mean()) if len(sma) else np.nan
    cp = cfg.get("control_predictions")
    if cp and Path(cp).exists():
        c = pd.read_csv(cp)
        c["time"] = pd.to_datetime(c["time"], utc=True, unit="ms" if np.issubdtype(c["time"].dtype, np.number) else None)
        m = X.merge(c.rename(columns={"score": "control"}), on=["symbol", "time"])
        res["control"] = (len(m), float(m["score"].corr(m["control"], method="spearman")) if len(m) else np.nan)
    return res


def trend_section(cfg, ta):
    L = ["## Skor ile trend etkisi / önceki çalışmanın kontrol modeli\n",
         "Soru: lojistik skor (p_up − p_dn), basit trend etkisinden farklı bir şey mi yakalıyor?\n",
         "| Trend göstergesi | Spearman(skor, gösterge) |", "|---|---|"]
    L += [f"| {'-' if k.startswith('sma_ratio') else ''}{k} | {v:+.3f} |" for k, v in ta["spearman"].items()]
    L += ["", f"- Skorun trend özellikleriyle (9 adet) doğrusal açıklanan varyansı: **R² = {ta['r2_trend']:.3f}**"]
    if "aligned_frac" in ta:
        L += [f"- Model işlemlerinin SMA168 trend yönünde olan oranı: **{ta['aligned_frac']:.1%}**",
              f"- Trend yönündeki işlemler: {ta['aligned'][0]} adet, ort. R {ta['aligned'][1]:.4f}",
              f"- Trende karşı işlemler: {ta['against'][0]} adet, ort. R {ta['against'][1]:.4f}",
              f"- Karşılaştırma: saf SMA168 kuralı ort. R {ta['sma_avg']:.4f}"]
        same = ta["r2_trend"] >= 0.5 and ta["aligned_frac"] >= 0.7
        L += ["", "Otomatik yorum (önceden sabitlenmiş kural: R² ≥ 0.5 **ve** trend yönlü işlem ≥ %70 → \"büyük ölçüde aynı trend etkisi\"): **" +
              ("Skor büyük ölçüde AYNI TREND ETKİSİNİ yakalıyor." if same else "Skor yalnızca trend etkisine indirgenemiyor (kural sağlanmadı).") + "**",
              "Bu bir ilişki ölçüsüdür; nedensellik iddiası değildir."]
    if "control" in ta:
        L.append(f"- Kontrol modeli tahminleriyle Spearman korelasyon: {ta['control'][1]:+.3f} ({ta['control'][0]} ortak satır)")
    prev = Path(cfg.get("local_data_dir") or "") / "final_test_report.md"
    L += ["", f"Önceki çalışma (`{prev}`): " + ("bulundu. " if prev.exists() else "BULUNAMADI. ") +
          "Bu kod o raporu otomatik yorumlamaz; kontrol modelinin tahminleri `symbol,time,score` CSV'si olarak "
          "`config.yaml > control_predictions` alanına verilirse doğrudan korelasyon hesaplanır. Yoksa yukarıdaki trend "
          "ölçüleri, o çalışmanın 'kontrol' modelinin (trend ağırlıklı temel özellikler) sonuçlarıyla elle karşılaştırılmalıdır.", ""]
    return L


# ---------------------------------------------------------------- grafikler / tablolar
def fmt(df):
    def cell(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return ""
            return f"{int(x)}" if float(x).is_integer() and abs(x) >= 1 else f"{x:.3f}"
        return str(x)
    cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    L = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    L += ["| " + str(i) + " | " + " | ".join(cell(v) for v in r) + " |" for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join(L)


def plots(cfg, out, rnd, models, od):
    tr, sma = out["model"], out["sma168"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, t in (("Model", tr), ("SMA168 kuralı", sma)):
        if len(t):
            s = t.sort_values("exit_time"); ax.plot(s["exit_time"], s["R"].cumsum(), label=name)
    ax.set_title(f"{cfg['exp']}: net sermaye eğrisi (kümülatif R, tüm semboller)"); ax.set_ylabel("R"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(od / "equity.png", dpi=110); plt.close(fig)
    if len(tr):
        mon = tr.groupby([tr["entry_time"].dt.year, tr["entry_time"].dt.month])["R"].sum().unstack().reindex(columns=range(1, 13))
        fig, ax = plt.subplots(figsize=(10, 0.8 + 0.5 * len(mon)))
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, (side, g) in zip(axes, models.groupby("side")):
        m = g[[f"coef_{n}" for n in names]].to_numpy().T
        v = np.nanmax(np.abs(m)) or 1
        ax.imshow(m, cmap="coolwarm", vmin=-v, vmax=v, aspect="auto")
        ax.set_yticks(range(len(names)), names, fontsize=7)
        step = max(1, len(g) // 12)
        ax.set_xticks(range(0, len(g), step), g["fold"].iloc[::step], fontsize=7, rotation=45)
        ax.set_title(f"{side}: standartlaştırılmış katsayılar (fold bazında; kırmızı +, mavi −)")
    fig.tight_layout(); fig.savefig(od / "coef_stability.png", dpi=110); plt.close(fig)


def coef_stability(cfg, models):
    rows = []
    for side, g in models.groupby("side"):
        for n in feature_names(cfg):
            s = np.sign(g[f"coef_{n}"])
            rows.append({"feature": n, "side": side, "same_sign": float(max((s > 0).mean(), (s < 0).mean())),
                         "sign_flips": int((s.diff().fillna(0) != 0).sum())})
    return pd.DataFrame(rows).pivot_table(index="feature", columns="side", values=["same_sign", "sign_flips"])


# ---------------------------------------------------------------- holdout
def holdout_section(cfg, data, wf_ok):
    od = Path(cfg["exp_out"])
    log_p = od / "holdout_log.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else []
    h = config_hash(cfg)
    prior = [e for e in log if e["config_hash"] != h]
    preds, models, hold, end = run_holdout(cfg, data)
    preds.to_parquet(od / "holdout.parquet")
    out, rnd, bh, days = simulate(cfg, data, preds)
    adv = cfg["label"]["adverse"]
    tr = out["model"]
    log.append({"time": pd.Timestamp.now(tz="UTC").isoformat(), "config_hash": h, "wf_passed": wf_ok,
                "trades": len(tr), "avg_R": float(tr["R"].mean()) if len(tr) else None})
    log_p.write_text(json.dumps(log, indent=1))
    L = ["## Mühürlü holdout (tek sefer)\n",
         f"Dönem: {hold:%Y-%m-%d} → {end:%Y-%m-%d} ({cfg['walkforward']['holdout_days']} gün). Model holdout öncesi 12 ayla eğitildi; "
         "eşik yine eğitim içi doğrulamada seçildi. Walk-forward hiçbir aşamada bu döneme bakmadı.",
         f"- Bu config ({h}) ile holdout değerlendirme sayısı: {sum(e['config_hash'] == h for e in log)}"]
    if prior:
        L.append(f"- ⚠️ Holdout daha önce **farklı** config ile {len(prior)} kez görüldü → artık tam olarak mühürlü sayılmaz.")
    L += ["", fmt(pick(breakdown(tr, adv, days), ["TOPLAM", "long", "short"])), "",
          f"- SMA168 kuralı (holdout): ort. R {out['sma168']['R'].mean():.4f}, {len(out['sma168'])} işlem",
          f"- Rastgele giriş p95 (holdout): {np.nanpercentile(rnd, 95):.4f}" if len(tr) else "- Holdout'ta işlem yok",
          f"- Maliyet ×{cfg['criteria']['cost_mult']:g} (holdout): ort. R {out['model_x2']['R'].mean():.4f}" if len(tr) else "",
          "- Holdout sonucu başarı kriterini DEĞİŞTİRMEZ; yalnızca bilgi içindir. Bu sonuca bakıp parametre değiştirmek holdout'u geçersiz kılar.", ""]
    return L


# ---------------------------------------------------------------- ana
def main(cfg, data, preds, models, banner="", holdout=True):
    od = Path(cfg["exp_out"])
    adv = cfg["label"]["adverse"]
    out, rnd, bh, days = simulate(cfg, data, preds)
    tr, tr2, sma = out["model"], out["model_x2"], out["sma168"]
    for k, t in out.items():
        t.to_csv(od / f"trades_{k}.csv", index=False)
    crit, ok = criteria(cfg, tr, tr2, sma, rnd)
    plots(cfg, out, rnd, models, od)
    ta = trend_analysis(cfg, data, preds, tr, sma)
    no_trade = models.assign(none=models["threshold"].isna()).groupby("side")["none"].mean()
    nofund = [s for s, d in data.items() if len(d["fund"]) == 0]
    L = [f"# ITB Skor — Walk-forward raporu: `{cfg['exp']}` ({cfg['interval']}, {len(data)} sembol)\n", banner,
         (f"> ⚠️ Funding verisi yok: {', '.join(nofund)} → funding 0 kabul edildi, maliyet EKSİK.\n" if nofund else ""),
         (f"> ⚠️ Veri bulunamayan semboller (dışarıda): {', '.join(sorted(set(cfg['symbols']) - set(data)))}\n"
          if set(cfg["symbols"]) - set(data) else ""),
         f"**SONUÇ: {'GEÇTİ' if ok else 'GEÇMEDİ'}**\n"]
    if not ok:
        L += ["Başarısız kriter(ler): " + "; ".join(f"**{c[0]}** ({c[2]})" for c in crit if not c[1]) + "\n",
              "Kural gereği parametreler test sonucuna bakılarak AYARLANMADI.\n"]
    L += ["## Başarı kriterleri (walk-forward test dönemi, holdout hariç)\n", "| Kriter | Durum | Değer |", "|---|---|---|"]
    L += [f"| {c[0]} | {'✅' if c[1] else '❌'} | {c[2]} |" for c in crit]
    L += ["", "## Kurulum",
          f"- Semboller ({len(data)}): {', '.join(data)} — TEK model, tüm sembollerin verisiyle",
          f"- Etiket/bariyer: H={cfg['label']['H']} bar, TP=+{cfg['label']['up']:.2%}, SL=−{cfg['label']['adverse']:.2%} (1R)",
          f"- Walk-forward: {preds['fold'].min()} → {preds['fold'].max()} ({preds['fold'].nunique()} test ayı), eğitim {cfg['walkforward']['train_months']} ay, purge+embargo {cfg['label']['H']}+{cfg['label']['H']} bar",
          f"- Maliyet: komisyon {cfg['costs']['fee']:.3%} + kayma {cfg['costs']['slippage']:.3%} / taraf + funding",
          "- Eşiği bulunamayan (doğrulamada pozitif beklenen değer yok → o ay o yönde işlem yok) fold oranı: " +
          ", ".join(f"{s}: {v:.0%}" for s, v in no_trade.items()),
          f"- Veri kaynağı ve kalite: bkz. `data_report.md`", ""]
    L += ["## Model metrikleri (net; toplam, sembol, yıl, yön)\n", fmt(breakdown(tr, adv, days)), ""]
    L += [f"## Maliyet ×{cfg['criteria']['cost_mult']:g}\n", fmt(breakdown(tr2, adv, days).loc[["TOPLAM"]]), ""]
    L += ["## Kıyaslar\n", "### SMA168 kuralı (close > SMA168 → long, altında → short; aynı TP/SL)\n",
          fmt(breakdown(sma, adv, days).loc[lambda x: ~x.index.isin(list(data))]), ""]
    L += [f"### Rastgele girişler (sembol/yön bazında aynı işlem sayısı, {len(rnd)} tekrar)\n",
          f"- Ort. R dağılımı: p5 {np.nanpercentile(rnd, 5):.4f}, p50 {np.nanpercentile(rnd, 50):.4f}, p95 {np.nanpercentile(rnd, 95):.4f}",
          f"- Modelin yüzdeliği: {(rnd < tr['R'].mean()).mean():.1%}" if len(tr) else "", ""]
    L += ["### Al-ve-tut (walk-forward test dönemi, net)\n", "| Sembol | Dönem | Net getiri | R karşılığı |", "|---|---|---|---|"]
    L += [f"| {s} | {v['from']:%Y-%m-%d} → {v['to']:%Y-%m-%d} | {v['net_return']:.1%} | {v['R']:.1f} |" for s, v in bh.items()]
    L += [f"| Eşit ağırlık ort. | | {np.mean([v['net_return'] for v in bh.values()]):.1%} | {np.mean([v['R'] for v in bh.values()]):.1f} |", ""]
    L += trend_section(cfg, ta)
    L += ["## Katsayı kararlılığı\n", "`same_sign` = fold'ların çoğunluk işaretinde kalan oranı (1.0 = hiç değişmedi), `sign_flips` = ardışık fold'larda işaret değişimi sayısı.\n",
          fmt(coef_stability(cfg, models)), ""]
    if holdout:
        L += holdout_section(cfg, data, ok)
    L += ["## Grafikler\n", "![equity](equity.png)\n", "![monthly](monthly_R.png)\n", "![random](random_dist.png)\n", "![coef](coef_stability.png)\n",
          "## Notlar",
          "- Kötümser simülasyon: giriş sinyalden sonraki barın açılışında, aynı barda TP+SL → SL, SL'de boşluk varsa daha kötü fiyat, TP'de iyileştirme yok, sembol başına tek pozisyon.",
          "- Maliyet ×2 senaryosunda komisyon, kayma ve funding birlikte ikiye katlanır. Funding yoksa (bkz. data_report.md) maliyet eksik hesaplanmıştır.",
          "- Geçmiş veriye dayanır; gelecekteki performansı garanti etmez. Yatırım tavsiyesi değildir."]
    (od / "report.md").write_text("\n".join(L))
    (od / "summary.json").write_text(json.dumps({"exp": cfg["exp"], "passed": ok, "config_hash": config_hash(cfg),
                                                 "criteria": [[c[0], bool(c[1]), c[2]] for c in crit]}, ensure_ascii=False, indent=1))
    print("\n".join(f"{'OK ' if c[1] else 'XX '} {c[0]}: {c[2]}" for c in crit))
    print(cfg["exp"], "SONUÇ:", "GEÇTİ" if ok else "GEÇMEDİ", "->", od / "report.md")
    return ok

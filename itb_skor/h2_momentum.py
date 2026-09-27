"""H2 — haftalık kesitsel momentum. Tanım: HYPOTHESIS_H2.md (ön kayıt). Buradaki sabitler o dosyayla birebir aynıdır.

python h2_momentum.py              # A: tarihsel test (holdout hariç) -> out/h2/report.md
python h2_momentum.py --holdout    # B: yalnızca A GEÇTİ ise, tek sefer (holdout_log.json'a yazılır)
python h2_momentum.py --signal     # C: yalnızca A GEÇTİ ise; bu haftanın portföyünü forward/h2_signals.csv'ye ekler
python h2_momentum.py --forward    # C: kayıtlı sinyallerin gerçekleşen sonuçları
"""
import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT, load_config

# ---- Ön kayıt sabitleri (DEĞİŞTİRME; değişiklik = yeni hipotez H3) ----
LOOKBACK = pd.Timedelta(days=28)
MIN_HISTORY = pd.Timedelta(days=29)
N_SIDE = 4
WEIGHT = 0.125                     # her bacak %50 brüt, coin başına %12,5
MIN_UNIVERSE_FIRST = 10
NW_LAGS = 4
T_MIN = 2.0
YEAR_FRAC = Fraction(2, 3)
COST_MULT = 2.0
RANDOM_REPS, SEED = 1000, 42
FORWARD_START = pd.Timestamp("2026-10-05", tz="UTC")
BAR = pd.Timedelta(hours=4)


def load(cfg):
    from data import load_all
    raw, _, prov = load_all(cfg)
    return raw, prov


def weekly_table(raw, mondays):
    """Her pazartesi m ve coin için: sinyal (28g getiri), giriş/çıkış açılışı, funding toplamı. Uygun olmayan -> satır yok."""
    rows = []
    for sym, (df, fund) in raw.items():
        c, o = df["close"], df["open"]
        first = df.index[0]
        ft = fund.index
        cum = np.r_[0.0, np.cumsum(fund.to_numpy(float))]
        for m in mondays:
            sig_bar = m - BAR                       # Pazar 20:00 4h barı (kapanışı = Pazartesi 00:00)
            need = [sig_bar, sig_bar - LOOKBACK, m, m + pd.Timedelta(days=7)]
            if first > m - MIN_HISTORY or not all(t in df.index for t in need[:2]):
                continue
            exit_t = need[3]
            if m not in df.index or exit_t not in df.index:
                continue
            f = cum[ft.searchsorted(exit_t, "right")] - cum[ft.searchsorted(m, "right")]   # (giriş, çıkış]
            rows.append({"monday": m, "symbol": sym, "signal": c[sig_bar] / c[need[1]] - 1,
                         "ret": o[exit_t] / o[m] - 1, "fund": f})
    return pd.DataFrame(rows)


def mondays_between(start, end):
    return pd.date_range(start.normalize(), end, freq="W-MON")


def build_weights(tab, rng=None):
    """Pazartesi -> {sym: ağırlık}. rng verilirse rastgele 4/4 (kıyas)."""
    W = {}
    for m, g in tab.groupby("monday"):
        if len(g) < 2 * N_SIDE:
            continue
        if rng is None:
            g = g.sort_values(["signal", "symbol"], ascending=[False, True])
            longs, shorts = g["symbol"].iloc[:N_SIDE], g["symbol"].iloc[-N_SIDE:]
        else:
            pick = rng.choice(g["symbol"].to_numpy(), 2 * N_SIDE, replace=False)
            longs, shorts = pick[:N_SIDE], pick[N_SIDE:]
        W[m] = {**{s: WEIGHT for s in longs}, **{s: -WEIGHT for s in shorts}}
    return W


def portfolio(tab, W, cfg, cost_mult=1.0):
    """Haftalık net getiri. Maliyet yalnızca ağırlığı değişen coin'lerde; dönem sonunda kapanış maliyeti."""
    c = (cfg["costs"]["fee"] + cfg["costs"]["slippage"]) * cost_mult
    t = {(m, s): (r, f) for m, s, r, f in zip(tab["monday"], tab["symbol"], tab["ret"], tab["fund"])}
    out, prev, prev_m = [], {}, None
    ms = sorted(W)
    for i, m in enumerate(ms):
        w = W[m]
        if prev_m is None or m - prev_m != pd.Timedelta(days=7):
            prev = {}                                          # kesinti -> sıfırdan kur
        turnover = sum(abs(w.get(s, 0) - prev.get(s, 0)) for s in set(w) | set(prev))
        last = i == len(ms) - 1 or ms[i + 1] - m != pd.Timedelta(days=7)
        if last:
            turnover += sum(abs(v) for v in w.values())         # kapanış
        gross = sum(v * t[(m, s)][0] for s, v in w.items())
        fund = -sum(v * t[(m, s)][1] for s, v in w.items()) * cost_mult   # long öder, short alır
        out.append({"monday": m, "gross": gross, "cost": c * turnover, "fund": fund,
                    "net": gross - c * turnover + fund,
                    "long": ",".join(s for s, v in w.items() if v > 0), "short": ",".join(s for s, v in w.items() if v < 0)})
        prev, prev_m = w, m
    return pd.DataFrame(out)


def newey_west_t(x, lags=NW_LAGS):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for l in range(1, lags + 1):
        var += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return x.mean() / math.sqrt(var / n) if var > 0 else np.nan


def periods(cfg, raw):
    """A: ilk pazartesi (≥10 uygun coin) -> holdout başı; B: holdout. Holdout başı H1 ile aynı yöntemle."""
    end = max(df.index[-1] for df, _ in raw.values()) + BAR
    hold = (end - pd.Timedelta(days=cfg["walkforward"]["holdout_days"])).floor("D")
    first_data = min(df.index[0] for df, _ in raw.values())
    tab = weekly_table(raw, mondays_between(first_data + MIN_HISTORY, end))
    counts = tab.groupby("monday").size()
    first = counts[counts >= MIN_UNIVERSE_FIRST].index.min()
    A = tab[(tab["monday"] >= first) & (tab["monday"] + pd.Timedelta(days=7) <= hold)]
    B = tab[(tab["monday"] >= hold)]
    return A, B, hold, end


def evaluate(cfg, tab):
    W = build_weights(tab)
    wk = portfolio(tab, W, cfg)
    wk2 = portfolio(tab, W, cfg, COST_MULT)
    rng = np.random.default_rng(SEED)
    rnd = np.array([portfolio(tab, build_weights(tab, rng), cfg)["net"].mean() for _ in range(RANDOM_REPS)])
    return wk, wk2, rnd


def criteria(wk, wk2, rnd, loo):
    mean, t = wk["net"].mean(), newey_west_t(wk["net"])
    yearly = wk.groupby(wk["monday"].dt.year)["net"].sum()
    need = math.ceil(YEAR_FRAC * len(yearly))
    p95 = float(np.percentile(rnd, 95))
    worst = min(loo.values()) if loo else np.nan
    res = [
        (f"Haftalık net ort. > 0 ve Newey-West({NW_LAGS}) t ≥ {T_MIN}", bool(mean > 0 and t >= T_MIN), f"ort {mean:.4%}, t {t:.2f}, {len(wk)} hafta"),
        (f"Pozitif yıl ≥ {need}/{len(yearly)}", bool((yearly > 0).sum() >= need), ", ".join(f"{y}: {v:+.1%}" for y, v in yearly.items())),
        (f"Maliyet ×{COST_MULT:g} iken ort. ≥ 0", bool(wk2["net"].mean() >= 0), f"{wk2['net'].mean():.4%}"),
        ("Rastgele 4/4 portföylerin %95'inden iyi", bool(mean > p95), f"model {mean:.4%} vs p95 {p95:.4%}"),
        ("Herhangi bir coin çıkarılınca ort. > 0 (20 koşu)", bool(worst > 0),
         f"en kötü: {min(loo, key=loo.get)} çıkarılınca {worst:.4%}" if loo else "-"),
    ]
    return res, all(r[1] for r in res)


def run_A(cfg, raw):
    od = Path(cfg["out_dir"]) / "h2"; od.mkdir(parents=True, exist_ok=True)
    A, _, hold, _ = periods(cfg, raw)
    wk, wk2, rnd = evaluate(cfg, A)
    loo = {}
    for s in raw:
        sub = {k: v for k, v in raw.items() if k != s}
        As, _, _, _ = periods(cfg, sub)
        loo[s] = portfolio(As, build_weights(As), cfg)["net"].mean()
    crit, ok = criteria(wk, wk2, rnd, loo)
    wk.to_csv(od / "weekly_A.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wk["monday"], (1 + wk["net"]).cumprod(), label="H2 net")
    ax.plot(wk2["monday"], (1 + wk2["net"]).cumprod(), label=f"maliyet ×{COST_MULT:g}", ls="--")
    ax.set_title("H2 kesitsel momentum — kümülatif net (A dönemi)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(od / "equity.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rnd, bins=50, color="0.7", label="Rastgele 4/4 (haftalık ort.)")
    ax.axvline(wk["net"].mean(), color="C3", lw=2, label="H2")
    ax.axvline(np.percentile(rnd, 95), color="k", ls=":", label="p95")
    ax.legend(); fig.tight_layout(); fig.savefig(od / "random_dist.png", dpi=110); plt.close(fig)

    ann = lambda x: (1 + x.mean()) ** 52 - 1
    L = ["# H2 — Haftalık kesitsel momentum: tarihsel test (A)\n",
         "Ön kayıt: `HYPOTHESIS_H2.md`. Tanım ve kriterler sonuç görülmeden sabitlendi.\n",
         f"**SONUÇ: {'GEÇTİ' if ok else 'GEÇMEDİ'}**\n"]
    if not ok:
        L += ["Başarısız: " + "; ".join(f"**{c[0]}** ({c[2]})" for c in crit if not c[1]) + "\n",
              "Ön kayıt gereği B (holdout) ve C (ileriye dönük) yapılmaz; parametre ayarlanmaz.\n"]
    L += ["| Kriter | Durum | Değer |", "|---|---|---|"] + [f"| {c[0]} | {'✅' if c[1] else '❌'} | {c[2]} |" for c in crit]
    L += ["", "## Özet",
          f"- Dönem: {wk['monday'].min():%Y-%m-%d} → {wk['monday'].max() + pd.Timedelta(days=7):%Y-%m-%d} ({len(wk)} hafta); holdout başı {hold:%Y-%m-%d} (hariç)",
          f"- Haftalık net ort. {wk['net'].mean():.3%} (yıllık bileşik ≈ {ann(wk['net']):.1%}), std {wk['net'].std():.3%}, "
          f"kazanan hafta {(wk['net'] > 0).mean():.0%}",
          f"- Brüt ort. {wk['gross'].mean():.3%}, maliyet ort. {wk['cost'].mean():.3%}, funding ort. {wk['fund'].mean():+.3%}",
          f"- Maks. düşüş (bileşik): {((1 + wk['net']).cumprod() / (1 + wk['net']).cumprod().cummax() - 1).min():.1%}",
          f"- Rastgele 4/4: p5 {np.percentile(rnd, 5):.3%}, p50 {np.percentile(rnd, 50):.3%}, p95 {np.percentile(rnd, 95):.3%}; "
          f"modelin yüzdeliği {(rnd < wk['net'].mean()).mean():.1%}",
          "", "## Bir coin çıkarılınca (haftalık net ort.)\n", "| Çıkarılan | Ort. |", "|---|---|"]
    L += [f"| {s} | {v:.3%} |" for s, v in sorted(loo.items(), key=lambda kv: kv[1])]
    L += ["", "## Grafikler\n", "![equity](equity.png)\n", "![random](random_dist.png)\n",
          "## Uyarılar",
          "- Evren bugün hâlâ büyük olan 20 coin → hayatta kalma yanlılığı tarihsel sonucu olumlu yönde şişirebilir. Asıl kanıt C (ileriye dönük).",
          "- Rastgele 4/4 portföyler her hafta yeniden seçildiği için daha yüksek devir hızı ve maliyet taşır; bu kıyas modelin lehine hafif yanlıdır.",
          "- Yatırım tavsiyesi değildir."]
    (od / "report.md").write_text("\n".join(L))
    (od / "summary.json").write_text(json.dumps({"passed_A": ok, "criteria": [[c[0], c[1], c[2]] for c in crit]}, ensure_ascii=False, indent=1))
    print("\n".join(f"{'OK ' if c[1] else 'XX '} {c[0]}: {c[2]}" for c in crit))
    print("H2 A SONUÇ:", "GEÇTİ" if ok else "GEÇMEDİ", "->", od / "report.md")
    return ok


def passed_A(cfg):
    p = Path(cfg["out_dir"]) / "h2" / "summary.json"
    return p.exists() and json.loads(p.read_text())["passed_A"]


def run_B(cfg, raw):
    od = Path(cfg["out_dir"]) / "h2"
    log_p = od / "holdout_log.json"
    log = json.loads(log_p.read_text()) if log_p.exists() else []
    _, B, hold, end = periods(cfg, raw)
    wk = portfolio(B, build_weights(B), cfg)
    log.append({"time": pd.Timestamp.now(tz="UTC").isoformat(), "weeks": len(wk), "mean": float(wk["net"].mean())})
    log_p.write_text(json.dumps(log, indent=1))
    ok = bool(wk["net"].mean() > 0)
    txt = [f"# H2 — holdout (B): {hold:%Y-%m-%d} → {end:%Y-%m-%d}\n",
           f"Değerlendirme sayısı: {len(log)}" + (" ⚠️ birden fazla" if len(log) > 1 else ""),
           f"\n**SONUÇ: {'GEÇTİ' if ok else 'GEÇMEDİ'}** (ölçüt: ortalama > 0)\n",
           f"- {len(wk)} hafta, haftalık net ort. {wk['net'].mean():.3%}, toplam {(1 + wk['net']).prod() - 1:.1%}",
           "- 13 haftada istatistiksel güç düşüktür; bu adım kırılmayı yakalamak içindir."]
    (od / "holdout.md").write_text("\n".join(txt))
    wk.to_csv(od / "weekly_B.csv", index=False)
    print("\n".join(txt))


def run_signal(cfg, raw):
    """Son pazartesinin portföyü. Giriş: o pazartesi 00:00 açılış."""
    end = max(df.index[-1] for df, _ in raw.values()) + BAR
    m = end.normalize() - pd.Timedelta(days=end.weekday())
    if m < FORWARD_START:
        raise SystemExit(f"İleriye dönük dönem {FORWARD_START:%Y-%m-%d} pazartesi başlar.")
    rows = []
    for sym, (df, _) in raw.items():
        sb = m - BAR
        if df.index[0] <= m - MIN_HISTORY and sb in df.index and sb - LOOKBACK in df.index:
            rows.append({"symbol": sym, "signal": df["close"][sb] / df["close"][sb - LOOKBACK] - 1})
    g = pd.DataFrame(rows).sort_values(["signal", "symbol"], ascending=[False, True])
    p = ROOT / "forward" / "h2_signals.csv"; p.parent.mkdir(exist_ok=True)
    old = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["monday"])
    if str(m.date()) in old["monday"].astype(str).tolist():
        raise SystemExit(f"{m:%Y-%m-%d} sinyali zaten kayıtlı; değiştirilemez.")
    row = pd.DataFrame([{"monday": str(m.date()), "long": ",".join(g["symbol"].iloc[:N_SIDE]),
                         "short": ",".join(g["symbol"].iloc[-N_SIDE:]), "n_universe": len(g),
                         "recorded_utc": pd.Timestamp.now(tz="UTC").isoformat()}])
    pd.concat([old, row]).to_csv(p, index=False)
    print(row.to_string(index=False))
    print(f"\nŞimdi commit'leyin: git add forward/h2_signals.csv && git commit -m 'H2 sinyal {m:%Y-%m-%d}'")


def run_forward(cfg, raw):
    p = ROOT / "forward" / "h2_signals.csv"
    if not p.exists():
        raise SystemExit("Kayıtlı sinyal yok.")
    sig = pd.read_csv(p)
    mondays = pd.DatetimeIndex(pd.to_datetime(sig["monday"]).dt.tz_localize("UTC"))
    tab = weekly_table(raw, mondays)
    W = {}
    for m, r in zip(mondays, sig.itertuples()):
        W[m] = {**{s: WEIGHT for s in r.long.split(",")}, **{s: -WEIGHT for s in r.short.split(",")}}
    have = set(zip(tab["monday"], tab["symbol"]))
    W = {m: w for m, w in W.items() if all((m, s) in have for s in w)}      # henüz kapanmamış haftalar dışarıda
    if not W:
        raise SystemExit("Henüz tamamlanmış hafta yok.")
    wk, wk2 = portfolio(tab, W, cfg), portfolio(tab, W, cfg, COST_MULT)
    print(wk[["monday", "gross", "cost", "fund", "net"]].to_string(index=False))
    print(f"\n{len(wk)} hafta: ort. {wk['net'].mean():.3%}, maliyet ×2 ort. {wk2['net'].mean():.3%} "
          f"(ölçüt, 26 hafta sonunda: ort. > 0 ve ×2 ≥ 0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--holdout", action="store_true")
    g.add_argument("--signal", action="store_true")
    g.add_argument("--forward", action="store_true")
    a = ap.parse_args()
    cfg = load_config("exp4h")                      # 20 coin, 4h, aynı maliyetler
    raw, _ = load(cfg)
    if a.holdout or a.signal:
        if not passed_A(cfg):
            raise SystemExit("Ön kayıt gereği: A GEÇMEDİ (veya çalıştırılmadı) → B/C yapılmaz.")
    if a.holdout:
        run_B(cfg, raw)
    elif a.signal:
        run_signal(cfg, raw)
    elif a.forward:
        run_forward(cfg, raw)
    else:
        run_A(cfg, raw)

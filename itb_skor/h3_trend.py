"""H3 — günlük trend takibi (Turtle Sistem 1). Tanım: HYPOTHESIS_H3.md (ön kayıt). Sabitler o dosyayla birebir aynıdır.

python h3_trend.py            # A: tarihsel test -> out/h3/report.md
python h3_trend.py --forward  # C: 2026-10-05 sonrası açılan işlemler (yalnızca A GEÇTİ ise anlamlı)
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

from common import load_config
from h2_momentum import newey_west_t

# ---- Ön kayıt sabitleri (DEĞİŞTİRME) ----
ENTRY_N, EXIT_N, ATR_N, STOP_ATR = 20, 10, 20, 2.0
RISK_PER_TRADE = 0.005
MIN_TRADES = 200
NW_LAGS, T_MIN = 5, 2.0
YEAR_FRAC = Fraction(2, 3)
COST_MULT = 2.0
RANDOM_REPS, SEED = 1000, 42
FORWARD_START = pd.Timestamp("2026-10-05", tz="UTC")
DAY = pd.Timedelta(days=1)


def to_daily(df4h):
    """UTC günlük mum; yalnızca 6 tam 4h barı olan günler."""
    g = df4h.resample("1D", label="left", closed="left")
    d = g.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return d[g["close"].count() == 6]


def rma(x, n):
    """Pine ta.rma: ilk değer ilk n değerin SMA'sı, sonra alpha=1/n."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) < n:
        return out
    s = valid[0]
    out[s + n - 1] = x[s:s + n].mean()
    for i in range(s + n, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def atr(d, n=ATR_N):
    """Pine ta.atr(n) = ta.rma(ta.tr(true), n)."""
    h, l, c = d["high"].to_numpy(), d["low"].to_numpy(), d["close"].to_numpy()
    pc = np.r_[np.nan, c[:-1]]
    tr = np.where(np.isnan(pc), h - l, np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)]))
    return rma(tr, n)


def channels(d):
    """Pine: ta.highest(high, n)[1] vb. — yalnızca önceki barlar."""
    return {"hh_in": d["high"].rolling(ENTRY_N).max().shift(1).to_numpy(),
            "ll_in": d["low"].rolling(ENTRY_N).min().shift(1).to_numpy(),
            "hh_out": d["high"].rolling(EXIT_N).max().shift(1).to_numpy(),
            "ll_out": d["low"].rolling(EXIT_N).min().shift(1).to_numpy(),
            "atr": atr(d)}


def simulate(d, fund, cfg, sym="", cost_mult=1.0):
    """Tek coin, gün gün. Sıra (her gün t): açılışta bekleyen giriş/çıkış -> bar içi stop -> kapanışta sinyal.
    Dönüş: işlemler + günlük R yolu (mark-to-market; maliyet ve funding dahil)."""
    o, h, l, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    ch = channels(d)
    idx, T = d.index, len(d)
    side_cost = (cfg["costs"]["fee"] + cfg["costs"]["slippage"]) * cost_mult
    ft = fund.index
    cum = np.r_[0.0, np.cumsum(fund.to_numpy(float))]
    fsum = lambda a, b: cum[ft.searchsorted(b, "right")] - cum[ft.searchsorted(a, "right")]   # (a, b]
    trades, daily = [], np.zeros(T)
    pos, pend_entry, pend_exit = None, None, False

    def close_trade(t, px, reason, exit_time):
        s, e, risk = pos["side"], pos["entry"], pos["risk"]
        daily[t] += s * (px - pos["mark"]) / risk - side_cost * px / risk
        fund_r = -s * fsum(idx[pos["i"]], exit_time) * e / risk * cost_mult
        daily[t] += fund_r
        gross_r, cost_r = s * (px - e) / risk, side_cost * (e + px) / risk
        trades.append({"symbol": sym, "side": s, "signal_day": idx[pos["sig"]], "entry_day": idx[pos["i"]],
                       "exit_day": idx[t], "reason": reason, "days": t - pos["i"] + 1, "entry": e, "exit": px,
                       "stop": pos["stop"], "gross_R": gross_r, "cost_R": cost_r, "fund_R": fund_r,
                       "R": gross_r - cost_r + fund_r})

    for t in range(T):
        # --- açılış ---
        if pend_exit:
            close_trade(t, o[t], "channel", idx[t]); pos, pend_exit = None, False
        if pend_entry is not None:
            s, stop, sig = pend_entry
            e = o[t]
            risk = s * (e - stop)
            if risk > 0:                                   # açılış stop'un ötesindeyse işlem açılmaz
                pos = {"side": s, "entry": e, "stop": stop, "risk": risk, "i": t, "sig": sig, "mark": e}
                daily[t] -= side_cost * e / risk
            pend_entry = None
        # --- bar içi stop ---
        if pos is not None:
            s, stop = pos["side"], pos["stop"]
            if (l[t] <= stop) if s == 1 else (h[t] >= stop):
                px = min(o[t], stop) if s == 1 else max(o[t], stop)
                close_trade(t, px, "stop", idx[t] + DAY); pos = None
            else:
                daily[t] += s * (c[t] - pos["mark"]) / pos["risk"]
                pos["mark"] = c[t]
        # --- kapanış: sinyaller (ertesi açılışta uygulanır) ---
        if t + 1 >= T:
            break
        if pos is not None:
            s = pos["side"]
            pend_exit = bool(c[t] < ch["ll_out"][t]) if s == 1 else bool(c[t] > ch["hh_out"][t])
        elif not np.isnan(ch["atr"][t]) and not np.isnan(ch["hh_in"][t]):
            s = 1 if c[t] > ch["hh_in"][t] else -1 if c[t] < ch["ll_in"][t] else 0
            if s:
                pend_entry = (s, c[t] - s * STOP_ATR * ch["atr"][t], t)
    return pd.DataFrame(trades), pd.Series(daily, index=idx, name=sym)


def run_all(raw, cfg, cost_mult=1.0, exclude=None):
    trs, days = [], []
    for sym, (df4h, fund) in raw.items():
        if sym == exclude:
            continue
        d = to_daily(df4h)
        tr, dl = simulate(d, fund, cfg, sym, cost_mult)
        trs.append(tr); days.append(dl)
    tr = pd.concat([x for x in trs if len(x)], ignore_index=True)
    port = pd.concat(days, axis=1).fillna(0).sum(axis=1) * RISK_PER_TRADE      # günlük portföy getirisi
    return tr, port


def random_benchmark(raw, tr, cfg, reps=RANDOM_REPS, seed=SEED):
    """Aynı coin/yön sayısı ve aynı elde tutma süreleriyle rastgele giriş günleri; stop yok, süre dolunca açılışta çıkış."""
    rng = np.random.default_rng(seed)
    pools = {}
    side_cost = cfg["costs"]["fee"] + cfg["costs"]["slippage"]
    for sym, (df4h, _) in raw.items():
        d = to_daily(df4h)
        pools[sym] = (d["open"].to_numpy(float), channels(d)["atr"])
    out = np.empty(reps)
    groups = [(r.symbol, int(r.side), max(1, int(round(r.days)))) for r in tr.itertuples()]
    for k in range(reps):
        Rs = []
        for sym, s, n in groups:
            o, a = pools[sym]
            valid = np.flatnonzero(~np.isnan(a[:-n - 1]))
            i = rng.choice(valid)
            e, x = o[i + 1], o[min(i + 1 + n, len(o) - 1)]
            risk = STOP_ATR * a[i]
            Rs.append((s * (x - e) - side_cost * (e + x)) / risk)
        out[k] = np.mean(Rs)
    return out


def criteria(tr, tr2, port, rnd, loo):
    mean = tr["R"].mean()
    t = newey_west_t(port.to_numpy(), NW_LAGS)
    yearly = tr.groupby(tr["exit_day"].dt.year)["R"].sum()
    need = math.ceil(YEAR_FRAC * len(yearly))
    p95 = float(np.percentile(rnd, 95))
    worst = min(loo, key=loo.get)
    res = [
        (f"İşlem sayısı ≥ {MIN_TRADES}", len(tr) >= MIN_TRADES, f"{len(tr)}"),
        (f"Ort. R > 0 ve günlük portföy NW({NW_LAGS}) t ≥ {T_MIN}", bool(mean > 0 and t >= T_MIN), f"ort. R {mean:.3f}, t {t:.2f}"),
        (f"Pozitif yıl ≥ {need}/{len(yearly)}", bool((yearly > 0).sum() >= need), ", ".join(f"{y}: {v:+.1f}R" for y, v in yearly.items())),
        (f"Maliyet ×{COST_MULT:g} iken ort. R ≥ 0", bool(tr2["R"].mean() >= 0), f"{tr2['R'].mean():.3f}"),
        ("Rastgele girişlerin %95'inden iyi", bool(mean > p95), f"model {mean:.3f} vs p95 {p95:.3f}"),
        ("Herhangi bir coin çıkarılınca ort. R > 0", bool(loo[worst] > 0), f"en kötü: {worst} çıkarılınca {loo[worst]:.3f}"),
    ]
    return res, all(r[1] for r in res)


def run_A(cfg, raw):
    od = Path(cfg["out_dir"]) / "h3"; od.mkdir(parents=True, exist_ok=True)
    tr, port = run_all(raw, cfg)
    tr2, _ = run_all(raw, cfg, COST_MULT)
    rnd = random_benchmark(raw, tr, cfg)
    loo = {s: run_all(raw, cfg, exclude=s)[0]["R"].mean() for s in raw}
    crit, ok = criteria(tr, tr2, port, rnd, loo)
    tr.to_csv(od / "trades.csv", index=False)

    eq = (1 + port).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq.index, eq.values); ax.set_title("H3 — portföy (işlem başı %0,5 risk), net"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(od / "equity.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rnd, bins=50, color="0.7", label="Rastgele giriş (ort. R)")
    ax.axvline(tr["R"].mean(), color="C3", lw=2, label="H3"); ax.axvline(np.percentile(rnd, 95), color="k", ls=":", label="p95")
    ax.legend(); fig.tight_layout(); fig.savefig(od / "random_dist.png", dpi=110); plt.close(fig)

    ann_ret = (1 + port.mean()) ** 365 - 1
    dd = (eq / eq.cummax() - 1).min()
    by = lambda col: tr.groupby(col)["R"].agg(["count", "mean", "sum"]).round(3)
    L = ["# H3 — Günlük trend takibi (Turtle Sistem 1): tarihsel test (A)\n",
         "Ön kayıt: `HYPOTHESIS_H3.md`. Kurallar klasik, bu veride optimize edilmedi.\n",
         f"**SONUÇ: {'GEÇTİ' if ok else 'GEÇMEDİ'}**\n"]
    if not ok:
        L += ["Başarısız: " + "; ".join(f"**{c[0]}** ({c[2]})" for c in crit if not c[1]) + "\n",
              "Ön kayıt gereği parametre ayarlanmaz; C yapılmaz.\n"]
    L += ["| Kriter | Durum | Değer |", "|---|---|---|"] + [f"| {c[0]} | {'✅' if c[1] else '❌'} | {c[2]} |" for c in crit]
    L += ["", "## Özet",
          f"- {len(tr)} işlem, {tr['symbol'].nunique()} coin, {tr['entry_day'].min():%Y-%m-%d} → {tr['exit_day'].max():%Y-%m-%d}",
          f"- Kazanma oranı {(tr['R'] > 0).mean():.0%}, ort. R {tr['R'].mean():.3f}, medyan R {tr['R'].median():.3f}, "
          f"en iyi {tr['R'].max():.1f}R, en kötü {tr['R'].min():.1f}R",
          f"- Ort. elde tutma {tr['days'].mean():.1f} gün; çıkış nedeni: " + ", ".join(f"{k} {v:.0%}" for k, v in tr['reason'].value_counts(normalize=True).items()),
          f"- Maliyet ort. {tr['cost_R'].mean():.3f}R, funding ort. {tr['fund_R'].mean():+.3f}R",
          f"- Portföy (işlem başı %0,5 risk): yıllık ≈ {ann_ret:.1%}, maks. düşüş {dd:.1%}",
          f"- Rastgele giriş: p50 {np.percentile(rnd, 50):.3f}, p95 {np.percentile(rnd, 95):.3f}; modelin yüzdeliği {(rnd < tr['R'].mean()).mean():.1%}",
          "", "## Yön / yıl\n", "```", by("side").to_string(), "", by(tr["exit_day"].dt.year).to_string(), "```",
          "", "## Coin bazında\n", "```", by("symbol").sort_values("sum").to_string(), "```",
          "", "![equity](equity.png)\n", "![random](random_dist.png)\n",
          "## Uyarılar",
          "- Veri H1/H2'de görüldü; tarihsel test tam kör değildir. Asıl kanıt ileriye dönük testtir (C).",
          "- Evren bugün hâlâ büyük olan 20 coin → hayatta kalma yanlılığı.",
          "- Rastgele kıyasta stop yoktur (aynı süre tutulur); stop'lu stratejiye göre kıyas yaklaşıktır.",
          "- Yatırım tavsiyesi değildir."]
    (od / "report.md").write_text("\n".join(L))
    (od / "summary.json").write_text(json.dumps({"passed_A": ok, "criteria": [[c[0], bool(c[1]), c[2]] for c in crit]}, ensure_ascii=False, indent=1))
    print("\n".join(f"{'OK ' if c[1] else 'XX '} {c[0]}: {c[2]}" for c in crit))
    print("H3 A SONUÇ:", "GEÇTİ" if ok else "GEÇMEDİ", "->", od / "report.md")
    return ok


def run_forward(cfg, raw):
    tr, _ = run_all(raw, cfg)
    f = tr[tr["entry_day"] >= FORWARD_START]
    if f.empty:
        raise SystemExit(f"{FORWARD_START:%Y-%m-%d} sonrası kapanmış işlem yok.")
    tr2, _ = run_all(raw, cfg, COST_MULT)
    f2 = tr2[tr2["entry_day"] >= FORWARD_START]
    print(f[["symbol", "side", "entry_day", "exit_day", "reason", "R"]].to_string(index=False))
    print(f"\n{len(f)} işlem: ort. R {f['R'].mean():.3f}, maliyet ×2 ort. R {f2['R'].mean():.3f} (ölçüt 26 hafta sonunda: > 0 ve ≥ 0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", action="store_true")
    a = ap.parse_args()
    cfg = load_config("exp4h")
    from data import load_all
    raw, _, _ = load_all(cfg)
    run_forward(cfg, raw) if a.forward else run_A(cfg, raw)

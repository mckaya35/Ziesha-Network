"""Son 12 aya yeniden eğitilen TEK (havuz) model -> TradingView Pine v6 indikatörü.

python export_pine.py                  # main deneyi; walk-forward GEÇTİ ise üretir
python export_pine.py --exp exp4h      # ikinci deney
python export_pine.py --force          # kriter geçmese de DEMO (dosyada ve tabloda uyarı)
python export_pine.py --retrain        # önce veriyi (API kuyruğu) günceller, sonra yeniden eğitir (aylık)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, bar_delta, load_config
from features import feature_names
from walkforward import predict, prepare, train_block

NEVER = 1.01     # eşik bulunamadıysa: o yönde hiç sinyal üretme
TV_PERIOD = {"1h": "60", "2h": "120", "4h": "240", "1d": "1D"}


def pine_expr(name):
    """Özellik adı -> Pine ifadesi (features.py ile birebir)."""
    n = name.split("_")[-1]
    if name.startswith("sma_ratio_"):
        return f"ta.sma(close, {n}) / close - 1"
    if name.startswith("slope_"):
        return f"(ta.linreg(close, {n}, 0) - ta.linreg(close, {n}, 1)) / close"
    if name.startswith("vol_"):
        return f"ta.stdev(close, {n}) / close"
    if name.startswith("ret_"):
        return f"close / close[{n}] - 1"
    if name == "rvol_24":
        return "volume / ta.sma(volume, 24)"
    if name == "hour_sin":
        return 'math.sin(2 * math.pi * hour(time, "UTC") / 24)'
    if name == "hour_cos":
        return 'math.cos(2 * math.pi * hour(time, "UTC") / 24)'
    raise ValueError(name)


def linear_terms(sc, m):
    return float(m.intercept_[0]), list(zip(m.coef_[0], sc.mean_, sc.scale_))


def manual_proba(X, sc, m):
    """Pine'a gömülen formül: sigmoid(b + Σ w·(x−μ)/σ)."""
    b, terms = linear_terms(sc, m)
    z = b + sum(w * (X[:, i] - mu) / sd for i, (w, mu, sd) in enumerate(terms))
    return 1 / (1 + np.exp(-z))


def pine_source(cfg, blk, meta, demo=False):
    names, f = feature_names(cfg), (lambda x: repr(float(x)))
    coins = [s[: -len(cfg["quote"])] for s in meta["symbols"]]
    L = ["//@version=6",
         f"// ITB Skor ({cfg['exp']}) — export_pine.py tarafından otomatik üretildi ({meta['generated']}). ELLE DEĞİŞTİRME.",
         f"// Tek model, {len(coins)} sembolün verisiyle eğitildi. Yalnızca {cfg['interval']} ve bu sembollerin Binance USDT-M perpetual grafiğinde geçerli.",
         "// İndikatördür; emir göndermez. Yatırım tavsiyesi değildir."]
    if demo:
        L.append("// !!! DEMO: walk-forward başarı kriteri GEÇMEDİ. Sinyaller kanıtlanmış bir avantaj GÖSTERMEZ. !!!")
    L += [f'indicator("ITB Skor {cfg["interval"]}", shorttitle="ITB Skor", overlay=false)', "",
          "// --- Özellikler (features.py ile birebir) ---"]
    L += [f"f_{n} = {pine_expr(n)}" for n in names]
    L += ["", "sigmoid(x) => 1.0 / (1.0 + math.exp(-x))", ""]
    for side, tag in ((1, "up"), (-1, "dn")):
        sc, m, thr = blk[side]
        b, terms = linear_terms(sc, m)
        L.append(f"// {tag}: sigmoid(b + Σ w·(x−μ)/σ), eşik {thr if thr is not None else 'yok (sinyal üretmez)'}")
        L.append(f"float z_{tag} = {f(b)}")
        L += [f"z_{tag} += ({f(w)}) * (f_{n} - ({f(mu)})) / ({f(sd)})" for n, (w, mu, sd) in zip(names, terms)]
        L += [f"raw_p_{tag} = sigmoid(z_{tag})", f"thr_{tag} = {f(thr if thr is not None else NEVER)}", ""]
    L += ['base = str.replace(syminfo.ticker, "USDT.P", "")',
          "sym_ok = syminfo.prefix == \"BINANCE\" and str.endswith(syminfo.ticker, \"USDT.P\") and (" +
          " or ".join(f'base == "{c}"' for c in coins) + ")",
          f'tf_ok = timeframe.period == "{TV_PERIOD.get(cfg["interval"], cfg["interval"])}"',
          "valid = sym_ok and tf_ok",
          "p_up = valid ? raw_p_up : na",
          "p_dn = valid ? raw_p_dn : na",
          "score = p_up - p_dn", "",
          'plot(score, "Skor (p_up - p_dn)", color=color.white, linewidth=2)',
          'plot(p_up, "p_up", color=color.new(color.green, 40))',
          'plot(p_dn, "p_dn", color=color.new(color.red, 40))',
          'plot(valid and thr_up <= 1 ? thr_up : na, "Eşik up", color=color.green, style=plot.style_linebr)',
          'plot(valid and thr_dn <= 1 ? thr_dn : na, "Eşik dn", color=color.red, style=plot.style_linebr)',
          'hline(0, "0", color=color.gray, linestyle=hline.style_dotted)', "",
          "up_hit = valid and raw_p_up >= thr_up",
          "dn_hit = valid and raw_p_dn >= thr_dn",
          "long_sig = barstate.isconfirmed and up_hit and not dn_hit",
          "short_sig = barstate.isconfirmed and dn_hit and not up_hit",
          'plotshape(long_sig, "Long", shape.triangleup, location.bottom, color.green, size=size.tiny)',
          'plotshape(short_sig, "Short", shape.triangledown, location.top, color.red, size=size.tiny)',
          'alertcondition(long_sig, "ITB Long", "ITB Skor LONG adayı: {{ticker}} {{close}}")',
          'alertcondition(short_sig, "ITB Short", "ITB Skor SHORT adayı: {{ticker}} {{close}}")', "",
          "var table tb = table.new(position.top_right, 2, 5, bgcolor=color.new(color.black, 20), border_width=1)",
          "if barstate.islast",
          '    table.cell(tb, 0, 0, "Model eğitimi", text_color=color.white)',
          f'    table.cell(tb, 1, 0, "{meta["train"]}", text_color=color.white)',
          '    table.cell(tb, 0, 1, "Sembol", text_color=color.white)',
          '    table.cell(tb, 1, 1, syminfo.tickerid, text_color=sym_ok ? color.white : color.red)',
          '    table.cell(tb, 0, 2, "Zaman dilimi", text_color=color.white)',
          '    table.cell(tb, 1, 2, timeframe.period, text_color=tf_ok ? color.white : color.red)',
          '    table.cell(tb, 0, 3, "Uyarı", text_color=color.yellow)',
          f'    table.cell(tb, 1, 3, valid ? "Yalnızca {cfg["interval"]} ve eğitildiği sembollerde geçerli" : '
          f'"UYUMSUZ sembol/zaman dilimi — değerler gösterilmiyor. Geçerli: BINANCE {"/".join(coins)} USDT.P {cfg["interval"]}", '
          "text_color=valid ? color.yellow : color.red)",
          '    table.cell(tb, 0, 4, "Durum", text_color=color.white)',
          '    table.cell(tb, 1, 4, "' + ("DEMO — kriter GEÇMEDİ" if demo else "Walk-forward kriteri geçti") +
          '", text_color=' + ("color.red" if demo else "color.lime") + ")"]
    return "\n".join(L) + "\n"


def export(cfg, demo, data=None, use_api=True):
    data = data or prepare(cfg, use_api=use_api)
    names = feature_names(cfg)
    end = max(d["df"].index[-1] for d in data.values())
    start = end - pd.DateOffset(months=cfg["walkforward"]["train_months"])
    pos = {s: np.arange(int(d["df"].index.searchsorted(start)), len(d["df"])) for s, d in data.items()}
    pos = {s: p for s, p in pos.items() if len(p)}          # etiketi eksik son H satır usable() ile düşer
    blk = train_block(data, pos, cfg)
    meta = {"generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"), "symbols": list(pos),
            "train": f"{start:%Y-%m-%d} → {end:%Y-%m-%d %H:%M} UTC"}
    parity = []
    for s in pos:
        X = data[s]["feat"][names].iloc[-50:]
        p = pd.DataFrame({"symbol": s, "time_open_utc": X.index.strftime("%Y-%m-%d %H:%M"),
                          "close": data[s]["df"]["close"].iloc[-50:].to_numpy(),
                          "p_up": predict(blk[1][0], blk[1][1], X), "p_dn": predict(blk[-1][0], blk[-1][1], X)})
        p["score"] = p["p_up"] - p["p_dn"]
        parity.append(p)
    pine = ROOT / cfg["pine_file"]
    pine.write_text(pine_source(cfg, blk, meta, demo))
    par = ROOT / ("parity_check.csv" if cfg["exp"] == "main" else f"parity_check_{cfg['exp']}.csv")
    pd.concat(parity).to_csv(par, index=False, float_format="%.6f")
    print(cfg["exp"], "eğitim", meta["train"], "eşik up", blk[1][2], "dn", blk[-1][2])
    print("Yazıldı:", pine, "ve", par, "(DEMO)" if demo else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.exp)
    summ = Path(cfg["exp_out"]) / "summary.json"
    passed = summ.exists() and json.loads(summ.read_text())["passed"]
    if not passed and not a.force:
        sys.exit(f"{a.exp}: walk-forward kriteri GEÇMEDİ (veya rapor yok). Pine üretilmedi. Demo için: --force")
    export(cfg, demo=not passed, use_api=a.retrain or cfg.get("api_fill", False))


if __name__ == "__main__":
    main()

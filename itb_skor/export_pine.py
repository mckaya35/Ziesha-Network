"""Son 12 aya yeniden eğitilen model -> TradingView Pine v6 indikatörü (ITB_Skor.pine).

python export_pine.py            # report GEÇTİ ise üretir
python export_pine.py --force    # kriter geçmese de DEMO olarak üretir (dosyada uyarı yazar)
python export_pine.py --retrain  # önce veriyi günceller, sonra yeniden eğitip yeni .pine üretir (aylık)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, load_config
from features import feature_names
from walkforward import predict, prepare, train_block

NEVER = 1.01   # eşik bulunamadıysa: hiç sinyal üretme


def pine_expr(name):
    """Özellik adı -> Pine ifadesi (features.py ile birebir)."""
    if name.startswith("sma_ratio_"):
        n = name.split("_")[-1]; return f"ta.sma(close, {n}) / close - 1"
    if name.startswith("slope_"):
        n = name.split("_")[-1]; return f"(ta.linreg(close, {n}, 0) - ta.linreg(close, {n}, 1)) / close"
    if name.startswith("vol_"):
        n = name.split("_")[-1]; return f"ta.stdev(close, {n}) / close"
    if name.startswith("ret_"):
        n = name.split("_")[-1]; return f"close / close[{n}] - 1"
    if name == "rvol_24":
        return "volume / ta.sma(volume, 24)"
    if name == "hour_sin":
        return "math.sin(2 * math.pi * hour(time, \"UTC\") / 24)"
    if name == "hour_cos":
        return "math.cos(2 * math.pi * hour(time, \"UTC\") / 24)"
    raise ValueError(name)


def linear_terms(sc, m):
    """Standartlaştırılmış logit: b + sum w*(x-mu)/sd."""
    return float(m.intercept_[0]), list(zip(m.coef_[0], sc.mean_, sc.scale_))


def manual_proba(X, sc, m):
    b, terms = linear_terms(sc, m)
    z = b + sum(w * (X[:, i] - mu) / sd for i, (w, mu, sd) in enumerate(terms))
    return 1 / (1 + np.exp(-z))


def pine_source(cfg, models, meta, demo=False):
    names = feature_names(cfg)
    f = lambda x: repr(float(x))
    L = ["//@version=6",
         f"// ITB Skor — export_pine.py tarafından otomatik üretildi ({meta['generated']}). ELLE DEĞİŞTİRME.",
         "// Yalnızca 1h ve modelin eğitildiği Binance USDT-M perpetual sembolünde geçerlidir.",
         "// Bu bir indikatördür; emir göndermez. Yatırım tavsiyesi değildir."]
    if demo:
        L.append("// !!! DEMO: walk-forward başarı kriteri GEÇMEDİ. Sinyaller kanıtlanmış bir avantaj GÖSTERMEZ. !!!")
    L += ['indicator("ITB Skor", shorttitle="ITB Skor", overlay=false)', "",
          "// --- Özellikler (features.py ile birebir) ---"]
    for n in names:
        L.append(f"f_{n} = {pine_expr(n)}")
    L += ["", "sigmoid(x) => 1.0 / (1.0 + math.exp(-x))", ""]
    for sym, blk in models.items():
        for side, tag in ((1, "up"), (-1, "dn")):
            sc, m, thr = blk[side]
            b, terms = linear_terms(sc, m)
            L.append(f"// {sym} {tag}: eşik {thr if thr is not None else 'yok (sinyal üretmez)'}")
            L.append(f"float z_{sym}_{tag} = {f(b)}")
            for n, (w, mu, sd) in zip(names, terms):
                L.append(f"z_{sym}_{tag} += {f(w)} * (f_{n} - {f(mu)}) / {f(sd)}")
            L.append(f"p_{sym}_{tag} = sigmoid(z_{sym}_{tag})")
            L.append(f"thr_{sym}_{tag} = {f(thr if thr is not None else NEVER)}")
        L.append("")
    syms = list(models)
    base = 'str.replace(syminfo.ticker, ".P", "")'
    L += [f"base = {base}",
          "sel = " + " : ".join(f'base == "{s}" ? {i}' for i, s in enumerate(syms)) + " : -1",
          'sym_ok = sel >= 0 and syminfo.prefix == "BINANCE" and str.endswith(syminfo.ticker, ".P")',
          'tf_ok = timeframe.period == "60"',
          "valid = sym_ok and tf_ok", ""]
    for tag in ("up", "dn"):
        for kind in ("p", "thr"):
            L.append(f"{kind}_{tag} = " + " : ".join(f"sel == {i} ? {kind}_{s}_{tag}" for i, s in enumerate(syms)) + " : na")
    L += ["score = valid ? p_up - p_dn : na", "",
          'plot(score, "Skor (p_up - p_dn)", color=color.new(color.white, 0), linewidth=2)',
          'plot(valid ? p_up : na, "p_up", color=color.new(color.green, 40))',
          'plot(valid ? p_dn : na, "p_dn", color=color.new(color.red, 40))',
          'plot(valid and thr_up <= 1 ? thr_up : na, "Eşik up", color=color.green, style=plot.style_linebr, linewidth=1)',
          'plot(valid and thr_dn <= 1 ? thr_dn : na, "Eşik dn", color=color.red, style=plot.style_linebr, linewidth=1)',
          'hline(0, "0", color=color.gray, linestyle=hline.style_dotted)', "",
          "up_hit = valid and p_up >= thr_up",
          "dn_hit = valid and p_dn >= thr_dn",
          "long_sig = barstate.isconfirmed and up_hit and not dn_hit",
          "short_sig = barstate.isconfirmed and dn_hit and not up_hit",
          'plotshape(long_sig, "Long", shape.triangleup, location.bottom, color.green, size=size.tiny)',
          'plotshape(short_sig, "Short", shape.triangledown, location.top, color.red, size=size.tiny)',
          'alertcondition(long_sig, "ITB Long", "ITB Skor LONG adayı: {{ticker}} {{close}}")',
          'alertcondition(short_sig, "ITB Short", "ITB Skor SHORT adayı: {{ticker}} {{close}}")', "",
          "var table tb = table.new(position.top_right, 2, 5, bgcolor=color.new(color.black, 20), border_width=1)",
          "if barstate.islast",
          '    table.cell(tb, 0, 0, "Model eğitimi", text_color=color.white)',
          '    table.cell(tb, 1, 0, ' + " : ".join(f'sel == {i} ? "{meta[s]}"' for i, s in enumerate(syms)) + ' : "-"' + ', text_color=color.white)',
          '    table.cell(tb, 0, 1, "Sembol", text_color=color.white)',
          '    table.cell(tb, 1, 1, syminfo.tickerid, text_color=sym_ok ? color.white : color.red)',
          '    table.cell(tb, 0, 2, "Zaman dilimi", text_color=color.white)',
          '    table.cell(tb, 1, 2, timeframe.period, text_color=tf_ok ? color.white : color.red)',
          '    table.cell(tb, 0, 3, "Uyarı", text_color=color.yellow)',
          '    table.cell(tb, 1, 3, valid ? "Yalnızca 1h ve eğitildiği sembolde geçerli" : "UYUMSUZ: yalnızca BINANCE ' +
          "/".join(f"{s}.P" for s in syms) + ' 1h — değerler gösterilmiyor", text_color=valid ? color.yellow : color.red)',
          '    table.cell(tb, 0, 4, "Durum", text_color=color.white)',
          '    table.cell(tb, 1, 4, "' + ("DEMO — kriter GEÇMEDİ" if demo else "Walk-forward kriteri geçti") +
          '", text_color=' + ("color.red" if demo else "color.lime") + ")"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    summ = Path(cfg["out_dir"]) / "summary.json"
    passed = summ.exists() and json.loads(summ.read_text())["passed"]
    if not passed and not a.force:
        sys.exit("Walk-forward kriteri GEÇMEDİ (veya rapor yok). Pine üretilmedi. Demo için: --force")
    if a.retrain:
        import data
        data.update(cfg)
    export(cfg, demo=not passed)


def export(cfg, demo):
    names = feature_names(cfg)
    models, meta, parity = {}, {"generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")}, []
    for sym in cfg["symbols"]:
        df, feat, lab, oc = prepare(cfg, sym)
        idx = df.index
        start = idx[-1] - pd.DateOffset(months=cfg["walkforward"]["train_months"])
        pos = np.arange(int(idx.searchsorted(start)), len(idx))   # etiketi eksik son H satır usable() ile düşer
        models[sym] = blk = train_block(feat, lab, oc, pos, cfg)
        meta[sym] = f"{idx[pos[0]]:%Y-%m-%d} → {idx[-1]:%Y-%m-%d %H:%M} UTC"
        X = feat[names].iloc[-50:]
        parity.append(pd.DataFrame({"symbol": sym, "time_open_utc": X.index.strftime("%Y-%m-%d %H:%M"),
                                    "close": df["close"].iloc[-50:].to_numpy(),
                                    "p_up": predict(blk[1][0], blk[1][1], X), "p_dn": predict(blk[-1][0], blk[-1][1], X)}))
        parity[-1]["score"] = parity[-1]["p_up"] - parity[-1]["p_dn"]
        print(sym, "eğitim", meta[sym], "eşik up", blk[1][2], "dn", blk[-1][2])
    (ROOT / "ITB_Skor.pine").write_text(pine_source(cfg, models, meta, demo))
    pd.concat(parity).to_csv(ROOT / "parity_check.csv", index=False, float_format="%.6f")
    print("Yazıldı:", ROOT / "ITB_Skor.pine", "ve parity_check.csv", "(DEMO)" if demo else "")


if __name__ == "__main__":
    main()

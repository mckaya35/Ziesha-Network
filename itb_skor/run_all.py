"""Tüm akış, her deney için ayrı: veri -> walk-forward -> rapor (+ mühürlü holdout) -> (geçerse) Pine.

python run_all.py                  # config'teki tüm deneyler (main = 1h/10 coin, exp4h = 4h/20 coin)
python run_all.py --exp main
python run_all.py --synthetic      # AĞ/VERİ YOK: sahte rastgele-yürüyüş CSV'leriyle akış denemesi (out_synth/)
python run_all.py --force-pine     # kriter geçmese de DEMO Pine
"""
import argparse
from pathlib import Path

import data as data_mod
import export_pine
import report
import walkforward
from common import ROOT, experiments, load_config

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default=None)
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--force-pine", action="store_true")
ap.add_argument("--no-holdout", action="store_true", help="mühürlü holdout'u bu çalıştırmada açma")
a = ap.parse_args()

overrides, banner = None, ""
if a.synthetic:
    root = data_mod.make_synthetic(ROOT / "data_synth" / "python_port")
    overrides = {"local_data_dir": str(root), "api_fill": False, "data_dir": "data_synth", "out_dir": "out_synth"}
    banner = "> ⚠️ **SENTETİK VERİ (rastgele yürüyüş). Yalnızca akış testidir, GERÇEK SONUÇ DEĞİLDİR.**\n"

for exp in [a.exp] if a.exp else experiments():
    cfg = load_config(exp, overrides=overrides)
    print(f"\n===== {exp} ({cfg['interval']}) =====")
    data = walkforward.prepare(cfg)
    if not data:
        print(exp, ": veri yok, atlandı"); continue
    preds, models, data = walkforward.run(cfg, data)
    ok = report.main(cfg, data, preds, models, banner, holdout=not a.no_holdout)
    if a.synthetic:
        print("Sentetik mod: Pine üretilmez.")
    elif ok or a.force_pine:
        export_pine.export(cfg, demo=not ok, data=data)
    else:
        print(f"{exp}: GEÇMEDİ -> Pine üretilmedi (demo: python export_pine.py --exp {exp} --force)")

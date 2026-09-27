"""Tüm akış: veri -> walk-forward -> backtest/rapor -> (geçerse) Pine.
python run_all.py              # gerçek Binance verisi
python run_all.py --synthetic  # AĞ YOK: sahte rastgele-yürüyüş verisiyle sadece akış denemesi (data_synth/, out_synth/)
"""
import argparse

import data
import report
import walkforward
from common import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--force-pine", action="store_true", help="kriter geçmese de DEMO Pine üret")
a = ap.parse_args()

cfg = load_config()
banner = ""
if a.synthetic:
    cfg["data_dir"] = cfg["data_dir"] + "_synth"
    cfg["out_dir"] = cfg["out_dir"] + "_synth"
    import pathlib; pathlib.Path(cfg["out_dir"]).mkdir(exist_ok=True)
    data.make_synthetic(cfg)
    banner = "> ⚠️ **SENTETİK VERİ (rastgele yürüyüş). Bu rapor yalnızca akış testidir, GERÇEK SONUÇ DEĞİLDİR.**\n"
else:
    data.update(cfg)

walkforward.run(cfg)
ok = report.main(cfg, banner)
if a.synthetic:
    print("Sentetik mod: Pine üretilmez.")
elif ok or a.force_pine:
    import export_pine
    export_pine.export(cfg, demo=not ok)
else:
    print("GEÇMEDİ -> Pine üretilmedi (demo için: python export_pine.py --force)")

# ITB Skor — Lojistik regresyon → walk-forward → Pine v6

Binance USDT-M Futures 1h verisiyle `intelligent-trading-bot` tarzı basit bir lojistik regresyon sinyalini
**önce dürüstçe test eder**, yalnızca test geçerse TradingView Pine v6 indikatörüne çevirir.
Canlı emir gönderen kod yoktur; API anahtarı gerekmez.

## Kurulum (Mac M1)
```bash
brew install python@3.11          # yoksa
cd itb_skor
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                          # testler yeşil olmalı
```

## Tek komut
```bash
python run_all.py
```
Sırasıyla: `data.py` (indir/güncelle + kalite raporu) → `walkforward.py` (`out/folds.parquet`, `out/folds_models.parquet`)
→ `report.py` (`out/report.md` + grafikler) → kriter geçerse `export_pine.py` (`ITB_Skor.pine`, `parity_check.csv`).

- `python run_all.py --synthetic` — internetsiz **akış denemesi** (rastgele yürüyüş verisi, `out_synth/`). Sonuçları gerçek değildir;
  rastgele yürüyüşte doğru olarak **GEÇMEDİ** çıkmalıdır.
- `python run_all.py --force-pine` — kriter geçmese de **DEMO** Pine üretir (dosyada ve tabloda "DEMO — kriter GEÇMEDİ" yazar).

## Dosyalar
| Dosya | Görev |
|---|---|
| `config.yaml` | Tüm parametreler (test sonucuna bakarak değiştirme) |
| `data.py` | `fapi/v1/klines` + `fapi/v1/fundingRate`, 1000'lik sayfalar, artımlı parquet, kapanmamış mum atılır, eksik saat/dublike raporu |
| `features.py` | Oran özellikleri; her satırda Pine karşılığı yorumda |
| `labels.py` | highlow2 / triple-barrier etiketleri (aynı barda TP+SL → 0) |
| `walkforward.py` | 12 ay eğitim → 1 ay test; purge H + embargo H; scaler/model/eşik yalnızca eğitimde |
| `backtest.py` | Kötümser simülasyon, maliyetler, metrikler, rastgele giriş kıyası |
| `report.py` | Kıyaslar, başarı kriteri, `report.md` + grafikler |
| `export_pine.py` | Son 12 aya yeniden eğitim → `ITB_Skor.pine` + `parity_check.csv` |

## Yöntem notları (varsayımlar açıkça)
- **Model sembol başına** eğitilir (long ve short için ayrı iki model). Pine dosyası iki sembolün modelini de içerir, `syminfo.ticker`'a göre seçer.
- **Eşik seçimi:** eğitim dönemi içindeki son %20 doğrulama bölümünde, aynı kötümser simülasyonla; net ort. R > 0 ve ≥ `min_val_trades`
  işlem veren eşikler arasından toplam R'si en yüksek olan. Hiçbiri yoksa o fold'da o yönde **işlem yok**. Eşik seçildikten sonra
  model tüm eğitim verisine yeniden fit edilir.
- **Purge/embargo:** eğitim, test başlangıcından `H` bar önce biter (embargo) ve ayrıca son `H` bar atılır (purge). İç doğrulamada da purge var.
- **İşlem:** sinyal bar kapanışında, giriş sonraki barın açılışında; TP/SL giriş fiyatına göre; aynı barda TP ve SL → SL;
  SL'de fiyat boşluğu varsa daha kötü fiyat; TP'de iyileştirme yok. Funding: (giriş, çıkış] aralığındaki funding anları.
- **"SMA168'den iyi"** = işlem başına net ort. R'si daha yüksek. **"3 yılın 2'si"** = test yıllarının ≥ 2/3'ü pozitif toplam R.
- **Maliyet ×2** = komisyon, kayma ve funding birlikte iki katı.

## Pine eşleşme kontrolü (tolerans 1e-3)
1. `ITB_Skor.pine` içeriğini TradingView → Pine Editor'a yapıştır, "Add to chart".
2. Grafik: `BINANCE:BTCUSDT.P` (veya ETHUSDT.P), zaman dilimi **1h**. Grafik saat dilimini **UTC** yap (sağ alt köşe).
3. `parity_check.csv`'deki satırlar son 50 kapanmış barı içerir; `time_open_utc` barın **açılış** saatidir.
4. Aynı barların üzerine gel → Data Window'da `p_up`, `p_dn`, `Skor` değerlerini CSV ile karşılaştır. |fark| ≤ 1e-3 olmalı.
5. Fark büyükse olası nedenler: TradingView ve Binance API hacim/fiyat verisinin farklı olması (özellikle `rvol_24`), yetersiz geçmiş bar (≥ 169 bar gerekir),
   yanlış sembol (spot vs perpetual) veya saat dilimi.

## Aylık yeniden eğitim
```bash
python export_pine.py --retrain         # veriyi günceller, son 12 aya yeniden eğitir, yeni ITB_Skor.pine + parity_check.csv
```
Walk-forward raporu GEÇMEDİ ise komut Pine üretmez (`--force` ile DEMO üretir). Yeni `.pine`'ı TradingView'da eski sürümün yerine yapıştır.

## Sınırlar
- Pine dosyası otomatik üretilir; TradingView derleyicisinde test edilmesi ve `parity_check.csv` ile doğrulanması gerekir.
- Geçmiş test gelecekteki performansı garanti etmez. Yatırım tavsiyesi değildir.

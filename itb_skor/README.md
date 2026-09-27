# ITB Skor — Lojistik regresyon → walk-forward → Pine v6

`intelligent-trading-bot` tarzı basit bir lojistik regresyon sinyalini **önce dürüstçe test eder**, yalnızca test geçerse
TradingView Pine v6 indikatörüne çevirir. Canlı emir gönderen kod yoktur; API anahtarı gerekmez.

## Kurulum (Mac M1)
Proje `~/Desktop/pine/itb_skor/` altına kurulmalı; mevcut veri `~/Desktop/pine/python_port/` altında (yalnızca **okunur**, hiçbir dosyası değiştirilmez).
```bash
cp -R <repo>/itb_skor ~/Desktop/pine/itb_skor
cd ~/Desktop/pine/itb_skor
brew install python@3.11                 # yoksa
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                # testler yeşil olmalı
```

## Tek komut
```bash
python run_all.py                  # iki deney: main (1h / 10 coin) ve exp4h (4h / 20 coin), ayrı raporlar
python run_all.py --exp main       # yalnızca ana deney
python run_all.py --synthetic      # veri/ağ olmadan akış denemesi (rastgele yürüyüş, out_synth/) — GERÇEK SONUÇ DEĞİL
python run_all.py --force-pine     # kriter geçmese de DEMO Pine
```
Her deney için `out/<deney>/`: `data_report.md` (tarama + kaynak + kalite), `report.md` (dürüst sonuç + holdout),
grafikler, `folds.parquet` (tahminler + eşikler), `folds_models.parquet` (fold başına katsayı, scaler, eşik), işlem CSV'leri.
Kriter geçerse: `ITB_Skor.pine` + `parity_check.csv` (main), `ITB_Skor_4h.pine` + `parity_check_exp4h.csv` (exp4h).

## Veri akışı (`data.py`)
1. `config.yaml > local_data_dir` (varsayılan `../python_port`) taranır: CSV/Parquet, sembol (klasör/dosya adından), zaman dilimi
   (zaman damgası farklarından), başlangıç–bitiş, sütunlar, dublike/eksik bar, NaN, bozuk OHLC → `data_report.md`.
2. Sütunlar `timestamp, open, high, low, close, volume` biçimine eşlenir (zaman UTC; ms/s/ISO otomatik).
3. Hedef zaman dilimi dosyası varsa o, yoksa **daha küçük** zaman diliminden yeniden örnekleme (eksik alt barlı barlar atılır).
4. Eksik semboller ve yerel verinin bitişinden sonrası Binance USDT-M Futures API'den tamamlanır (`api_fill`).
   Örtüşen 200 barda yerel/API kapanış farkı raporlanır → yerel verinin spot mu futures mı olduğu anlaşılır.
5. Funding `fapi/v1/fundingRate`'ten. API'ye ulaşılamazsa funding 0 kabul edilir ve rapor başında **uyarı** çıkar.

## Yöntem (sabit, test sonucuna göre değiştirilmez)
- **Tek model**, deneyin tüm sembollerinin verisiyle (long ve short için iki ayrı LogisticRegression). Sonuçlar sembol bazında da raporlanır.
- **Walk-forward:** 12 ay eğitim → 1 ay test, 1 ay kaydır. İlk test ayı = en erken veri + 12 ay sonrası ilk ay başı (1h için ≈ 2022-08).
  Purge `H` + embargo `H` bar. Scaler/model/eşik yalnızca eğitimde.
- **Eşik:** eğitimin zaman olarak son %20'sinde (iç purge ile), aynı kötümser simülasyonla; tüm sembollerde toplam ≥ `min_val_trades`
  işlem ve net ort. R > 0 veren eşikler arasından toplam R'si en yüksek olan. Yoksa o ay o yönde işlem yok.
- **Mühürlü holdout:** son 90 gün. Walk-forward işlemleri holdout başlamadan kapanır (fiyatına bile bakılmaz — testle kanıtlı).
  En sonda bir kez değerlendirilir; her değerlendirme `holdout_log.json`'a config özetiyle yazılır, config değiştikten sonra tekrar
  bakılırsa raporda "artık tam mühürlü değil" uyarısı çıkar. Holdout başarı kriterini değiştirmez.
- **İşlem:** sinyal bar kapanışında, giriş sonraki barın açılışında; TP/SL giriş fiyatına göre; aynı barda TP+SL → SL; SL'de boşluk varsa
  daha kötü fiyat; sembol başına tek pozisyon. Funding (giriş, çıkış] aralığındaki anlar.
- **Yorumlar:** "SMA168'den iyi" = işlem başına net ort. R daha yüksek; "3 yılın 2'si" = test yıllarının ≥ 2/3'ü pozitif;
  "maliyet ×2" = komisyon + kayma + funding birlikte ×2.

## Önceki çalışmayla ilişki (`final_test_report.md`)
Raporun "Skor ile trend etkisi" bölümü: skorun trend göstergeleriyle Spearman korelasyonu, trend özellikleriyle açıklanan varyans (R²),
işlemlerin SMA168 yönünde olma oranı ve trend yönlü / trende karşı işlemlerin ayrı ort. R'si. Önceden sabit kural:
R² ≥ 0.5 **ve** trend yönlü işlem ≥ %70 → "büyük ölçüde aynı trend etkisi". Kontrol modelinin tahminleri
`symbol,time,score` CSV'si olarak `control_predictions`'a verilirse doğrudan korelasyon da hesaplanır; aksi halde
`final_test_report.md` ile elle karşılaştırılmalıdır (kod o raporu yorumlamaz).

## Pine eşleşme kontrolü (tolerans 1e-3)
1. `ITB_Skor.pine` içeriğini TradingView → Pine Editor'a yapıştır → "Add to chart".
2. Grafik: `BINANCE:<COIN>USDT.P` (eğitilen sembollerden biri), zaman dilimi **1h** (4h dosyası için 4h). Grafik saat dilimi **UTC**.
3. `parity_check.csv`: her sembol için son 50 kapanmış bar; `time_open_utc` barın **açılış** saati.
4. Aynı barlarda Data Window'daki `p_up`, `p_dn`, `Skor` değerlerini CSV ile karşılaştır: |fark| ≤ 1e-3.
5. Büyük fark nedenleri: TradingView ile yerel/API verisinin farklı olması (özellikle hacim → `rvol_24`; yerel veri spot ise
   perpetual grafikte fark beklenir), yetersiz geçmiş (≥ 169 bar), yanlış sembol/zaman dilimi.

## Aylık yeniden eğitim
```bash
python export_pine.py --retrain            # main: veriyi (API kuyruğu) günceller, son 12 aya yeniden eğitir, yeni .pine + parity CSV
python export_pine.py --exp exp4h --retrain
```
Walk-forward GEÇMEDİ ise Pine üretilmez (`--force` ile DEMO).

## H2 — haftalık kesitsel momentum (ön kayıtlı: `HYPOTHESIS_H2.md`)
H1 (lojistik skor) iki deneyde de GEÇMEDİ. H2, kod yazılmadan önce commit'lenen ön kayda birebir uyar (20 coin, 4h veri,
28 günlük getiriye göre top 4 long / bottom 4 short, haftalık).
```bash
python h2_momentum.py              # A: tarihsel test (holdout hariç) -> out/h2/report.md
python h2_momentum.py --holdout    # B: yalnızca A GEÇTİ ise, tek sefer
python h2_momentum.py --signal     # C: yalnızca A GEÇTİ ise; her pazartesi (2026-10-05'ten itibaren) çalıştır ve commit'le
python h2_momentum.py --forward    # C: kayıtlı sinyallerin gerçekleşen sonuçları
```
C için her pazartesi `forward/h2_signals.csv` git'e commit'lenmelidir (commit zamanı = sinyalin önceden kaydedildiğinin kanıtı).

## H3 — günlük trend takibi, Turtle Sistem 1 (ön kayıtlı: `HYPOTHESIS_H3.md`)
Klasik kurallar (20g kırılım, 10g çıkış, 2·ATR(20) stop), bu veride optimize edilmedi. 20 coin, 4h veriden günlük mum.
```bash
python h3_trend.py             # A: tarihsel test -> out/h3/report.md
python h3_trend.py --forward   # C: 2026-10-05 sonrası işlemler
```
`H3_Trend.pine`: aynı kuralların Pine v6 `strategy` karşılığı (sabitler gömülü, funding hariç).

## Sınırlar
- Pine dosyası otomatik üretilir; TradingView derleyicisinde denenmesi ve `parity_check.csv` ile doğrulanması gerekir.
- Geçmiş test gelecekteki performansı garanti etmez. Yatırım tavsiyesi değildir.

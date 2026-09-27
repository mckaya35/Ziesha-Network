# H3 — Günlük trend takibi (Turtle Sistem 1 kuralları) — ön kayıt

**Durum:** ÖN KAYIT. Kod yazılmadan ve H3 verisi test edilmeden önce commit'lendi. Sonuç görüldükten sonra DEĞİŞTİRİLEMEZ.

## 1. Gerekçe ve dürüstlük notu
- Trend takibi (zaman serisi momentumu) varlık sınıfları genelinde belgelenmiş bir etkidir; H3 **yeni parametre üretmez**,
  kamuya açık klasik Turtle Sistem 1 kurallarını olduğu gibi kullanır (20 gün kırılım, 10 gün çıkış, 2N stop, N = ATR(20)).
  Hiçbir parametre bu veride optimize edilmedi.
- Acrypto Weighted Strategy'ye göre farklar: 5 parametre yerine sabit klasik kurallar; stop gerçek stop emri (bar içi);
  pozisyon riske göre boyutlanır (sermayenin %100'ü değil); maliyet + funding dahil; sabitler `input()` değil (ayar yapılamaz).
- **Kirlenme:** Bu 20 coin'in 2021–2026 verisi H1/H2'de görüldü. H2'nin yıllık sonuçları (2021–23 pozitif, 2024–26 negatif)
  momentumla ilgili bilgi içeriyordu. H3 bu sonuçlardan türetilmedi, ama tarihsel test tam "kör" sayılamaz.
  Bu yüzden **asıl kanıt C (ileriye dönük) testtir**; A yalnızca ön elemedir.

## 2. Tanım (sabit)
| Öğe | Değer |
|---|---|
| Evren | exp4h'deki 20 coin (USDT-M perpetual), her coin bağımsız |
| Veri | Yerel 4h mumlardan UTC günlük mum (6 tam 4h barı olan günler). Funding gerçek (API) |
| N | ATR(20), Pine `ta.atr(20)` ile aynı (Wilder RMA, TR) |
| Long giriş | Günlük kapanış > önceki 20 günün en yüksek high'ı → ertesi gün açılışında |
| Short giriş | Günlük kapanış < önceki 20 günün en düşük low'u → ertesi gün açılışında |
| Stop | Sinyal günü kapanışı ∓ 2·N; gerçek stop emri, giriş günü dahil bar içi. Boşlukta açılış fiyatından |
| Kanal çıkışı | Long: kapanış < önceki 10 günün en düşük low'u; short: kapanış > önceki 10 günün en yüksek high'ı → ertesi açılış |
| Pozisyon | Aynı coin'de tek pozisyon, yalnızca düz iken giriş. 1R = giriş − stop mesafesi |
| Maliyet | %0,05 komisyon + %0,02 kayma / taraf; funding (giriş, çıkış] |
| Portföy ölçüsü | Her işlem sermayenin %0,5'i riskle; günlük piyasa değerine göre (mark-to-market) portföy getirisi |

## 3. Başarı kriteri — A (tarihsel, tüm veri; hepsi)
1. ≥ 200 işlem
2. İşlem başına net ort. R > 0 **ve** günlük portföy getirisinin Newey-West (5 gecikme) t ≥ 2,0
3. Takvim yıllarının ≥ 2/3'ünde pozitif toplam R
4. Maliyet ×2 (komisyon+kayma+funding) iken ort. R ≥ 0
5. Rastgele giriş kıyası: aynı coin/yön başına işlem sayısı ve aynı elde tutma süreleriyle rastgele günlerde giriş,
   1000 tekrar → modelin ort. R'si p95'ten büyük
6. Herhangi bir coin çıkarılınca (20 koşu) ort. R > 0

## 4. C — ileriye dönük (asıl kanıt)
2026-10-05'ten itibaren 26 hafta; yeni işlemler aynı kodla izlenir. Ölçüt: ort. R > 0 ve maliyet ×2 iken ≥ 0.
A geçmezse: "GEÇMEDİ" yazılır, parametre ayarlanmaz, C yapılmaz.

## 5. Pine
`H3_Trend.pine` (Pine v6 `strategy`) aynı kuralları içerir. Sabitler kodda gömülüdür (`input()` yok), TradingView'da
parametre optimizasyonu yapılmaması için. Funding Pine'da modellenmez → TradingView sonucu Python'dan iyimserdir.

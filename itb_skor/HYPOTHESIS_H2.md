# H2 — Haftalık kesitsel momentum (ön kayıt)

**Durum:** ÖN KAYIT. Bu dosya kod yazılmadan ve H2 verisi hiç test edilmeden önce commit'lendi. Commit zamanı = kayıt zamanı.
Aşağıdaki tanımlar, sonuçlar görüldükten sonra DEĞİŞTİRİLEMEZ. Değişiklik gerekirse H3 olarak yeni bir ön kayıt açılır.

## 1. Neden bu hipotez (H1'den bağımsız gerekçe)
- H1 (lojistik skor, 1h/4h, %2/%1 bariyer) GEÇMEDİ. H2, H1 sonuçlarından **türetilmedi**. Dayanağı literatür:
  Liu, Tsyvinski, Wu (2022), *Common Risk Factors in Cryptocurrency*, Journal of Finance 77(2). Kesitsel getirileri
  piyasa, büyüklük ve **momentum** faktörlerinin açıkladığını raporluyor.
- H1'den farkı yapısal: sembol başına zamanlama değil, sembollerin **birbirine göre** sıralanması (piyasa-nötr long/short).
  Haftada bir yeniden dengeleme, bu yüzden işlem maliyetinin getiriye oranı H1'e göre çok daha düşük.
- H1'den alınan tek ders yöntemsel: sık işlem + dar bariyer maliyet yüzünden baştan dezavantajlı. H2 bu yüzden düşük devir hızlı seçildi.
  Bu bir parametre ayarı değil, hipotez sınıfı seçimidir.

## 2. Tanım (sabit)
| Öğe | Değer |
|---|---|
| Evren | exp4h'deki 20 coin (USDT-M perpetual). Oluşum anında ≥ 29 günlük verisi olan coin'ler (SUI geç girer) |
| Veri | 4h mumlar (yerel `uzun_*_4h`), günlük kapanışlar 4h'den türetilir. Funding gerçek (API) |
| Oluşum | Her **Pazartesi 00:00 UTC**. Sinyal = son 28 günlük getiri: close(Pazar 20:00 4h barı) / close(28 gün önce aynı bar) − 1 |
| Portföy | En yüksek 4 → long, en düşük 4 → short. Eşit ağırlık (her bacak %50 brüt, coin başına %12,5) |
| Giriş / çıkış | Pazartesi 00:00 4h barının açılışında giriş, bir sonraki Pazartesi 00:00 açılışında çıkış (7 gün). Stop/TP yok |
| Maliyet | Komisyon %0,05 + kayma %0,02 taraf başına, yalnızca pozisyonu değişen coin'lerde. Funding gerçek oranlarla |
| Ölçü | Haftalık net portföy getirisi (long-short) |

## 3. Başarı kriteri (hepsi)
**A. Tarihsel test** (ilk oluşum = evrende ≥ 10 coin olduğu ilk Pazartesi → 2026-06-28; H1 holdout'u olan son 90 gün HARİÇ):
1. Haftalık net ortalama > 0 ve Newey-West (4 gecikme) t ≥ 2,0
2. Takvim yıllarının ≥ 2/3'ünde pozitif toplam
3. Maliyet ×2 iken ortalama ≥ 0
4. Rastgele sıralama kıyası: aynı takvimde 1000 rastgele 4/4 long-short portföyün ortalamasının %95'inden iyi
5. Tek coin'e bağımlı değil: herhangi bir coin evrenden çıkarılınca (20 ayrı koşu) ortalama hâlâ > 0

**B. Tarihsel holdout** (2026-06-29 → 2026-09-27, 13 hafta): yalnızca A geçerse, tek sefer. Ölçüt: ortalama > 0 (13 haftada istatistiksel güç düşük; kırılmayı yakalamak içindir).

**C. İleriye dönük (asıl kanıt):** 2026-10-05'ten itibaren 26 hafta, kâğıt üzerinde. Her Pazartesi sinyal ve sonuç
değiştirilemez şekilde commit'lenir. Ölçüt: ortalama > 0 ve maliyet ×2 iken ≥ 0.

A geçmezse: "GEÇMEDİ" yazılır, B/C yapılmaz, parametre ayarlanmaz.

## 4. Bilinen zayıflıklar (baştan kabul)
- **Hayatta kalma yanlılığı:** Evren bugün hâlâ büyük ve listeli olan 20 coin'den oluşuyor. Tarihsel testte momentumu olumlu yönde şişirebilir.
  C (ileriye dönük) bu yanlılıktan arınıktır, bu yüzden asıl kanıt C'dir.
- Makaledeki evren çok daha geniştir. 20 büyük coin'de etki daha zayıf olabilir ya da hiç olmayabilir.
- Veri aynı veri. H1 bu dönemi "gördü", ama H2'nin tanımı H1 sonuçlarına bakılarak seçilmedi. Yine de C olmadan kesin hüküm verilmez.
- Pine karşılığı tek grafikte değil, `request.security` ile 20 sembolün sıralaması gerektirir. Yalnızca C geçerse yapılır.

## 5. Alternatif (bu kayıtta TEST EDİLMEYECEK)
Funding/carry: Schmeling, Schrimpf, Todorov, *Crypto Carry* (BIS WP 1087, 2023). Spot + perp iki bacak gerektirir;
tek-grafik indikatörüne uymaz. İstenirse ayrı ön kayıtla (H3).

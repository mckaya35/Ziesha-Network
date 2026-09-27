# Kafes Momentum (Fabio tarzı) – Kullanım notu

Dosyalar:
- `kafes_momentum.pine` – indikatör (görseller, tablo, alarmlar)
- `kafes_momentum_strateji.pine` – aynı mantığın `strategy()` sürümü (görsel yok)

## Zaman dilimi
1–5 dk grafikler için tasarlandı. 15 dk üstünde ve kafes penceresi grafik zaman diliminden kısaysa tabloda uyarı çıkar.

## Kafes penceresi
- Kripto için **New York açılışı (09:30 NY, 30 dk)** önerilir; hacim ve yön genelde bu saatte gelir.
- Saatler `time(timeframe.period, seans, saat_dilimi)` ile hesaplanır, yaz/kış saati otomatik doğrudur.
- Süre (15/30/60 dk) seçimi üç hazır pencereye de uygulanır; "Özel" pencerede seans ve saat dilimi elle girilir.
- Gün, kafes penceresinin başlangıcıyla başlar, kafes aktif süresinin bitişiyle biter (günlük R, kayıp sayısı bu aralıkta sayılır).

## Repaint durumu
- Tüm kararlar yalnızca kapanmış mumda (`barstate.isconfirmed`) verilir, `request.security` kullanılmaz; geçmiş işaretler sonradan değişmez.
- Canlı mum kapanana kadar sinyal oluşmaz. Alarmlar da mum kapanışında tetiklenir ("Bar kapanışında bir kez" seçin).
- Günlük aralık (ADR) grafik zaman diliminde biriktirilir; grafiğin ilk günü eksik olabilir. Gün tipi filtresi en az 5 tamamlanmış gün ister, en fazla son 20 günü kullanır.

## Bu sistem ne DEĞİLDİR
- Gerçek order flow (emir defteri, işlem hızı, footprint) **yoktur**. İnisiyatif, emilim ve kabul; mum gövdesi, RVOL ve VWAP'tan yapılan **OHLCV tahminleridir**.
- Dengeli (yatay/dalgalı) günlerde sistem **para kaybeder**. Gün tipi filtresi, 2 başarısız müzayededen sonra kafesin kapanması ve günlük durma kuralı bunu sınırlamak içindir, ortadan kaldırmaz.

## Bias, maliyet ve minimum stop
- **Bias zorunlu:** LONG yalnızca kapanış > seans VWAP, SHORT yalnızca kapanış < seans VWAP iken. Emir kurulurken ve bekleyen emir her bar kapanışında yeniden kontrol edilir; bias bozulursa emir iptal edilir ("Son iptal: Bias ters"). VWAP hesaplanamıyorsa (hacim yok) işlem açılmaz.
- Dolum barının **kendi** kapanışı kullanılmaz: stop emri bar içinde dolar, kapanış ise ancak dolumdan sonra bilinir. Onu kullanmak geleceği görmek (repaint / hayali backtest) olurdu. Debug etiketi dolum barı kapanışındaki bias'ı ayrıca gösterir.
- **Maliyet filtresi:** 2 × komisyon% × fiyat + 2 × kayma × tick, stop mesafesinin %15'inden büyükse işleme girilmez ("Stop maliyete göre dar").
- **Minimum stop** = max(0,3 × ATR, fiyatın %0,35'i).
- Dikkat: Komisyon %0,05 iken gidiş-dönüş maliyeti fiyatın ≈ %0,10'udur. %15 kuralı bu yüzden stopun fiyatın en az ≈ %0,67'si (+ kayma) olmasını fiilen zorunlu kılar; %0,35'lik minimum stop tek başına yetmez ve bu işlemler maliyet filtresine takılır. Düşük komisyonlu (maker) hesapta eşik düşer.

## Risksiz (başa baş) mantığı
- Fiyat girişten 1R lehe gittikten sonra stop giriş ± 1 tick'e çekilir; **ancak** bar kapanışı bu seviyenin lehe tarafındaysa. Kapanış girişin gerisine dönmüşse stop orada kurulmaz (piyasanın yanlış tarafında kalan stop sonraki açılışta zararla dolar); ilk uygun kapanışta kurulur.
- Bu kuralla risksiz işlemin brüt sonucu ≥ +1 tick, net sonucu ≥ −maliyet olur. Tek istisna, açılışın stopun ötesinde olduğu gap'tir (7/24 kripto 1–5 dk'da nadir). Tablodaki "Risksiz" satırı bunu sayar: ihlal > 0 ise o işlemler gap kaynaklıdır.

## Debug
"Debug" açıkken her girişte küçük etiket: VWAP, bias (✓/✗), stop mesafesi (ATR) ve maliyet/R.

## Simülasyon varsayımları (tablo istatistiği)
- Stop emri, sinyal barından sonraki barlarda tetiklenir; bar girişin ötesinde açıldıysa giriş = açılış.
- R, **planlanan** risk (giriş − stop) üzerinden hesaplanır; gap kayması R'yi düşürür.
- Giriş barında stop'a değildiyse stop sayılır; aynı barda stop ve hedef varsa stop sayılır.
- Stop taşıma kararı bar kapanışında verilir, sonraki bardan geçerlidir.
- Net R = brüt R − (2 × komisyon% × giriş / 100 + 2 × kayma × mintick) / risk.
- "Riski yarıya indir" seçiliyse o işlemin R sonucu 0,5 ile çarpılır (stratejide adet yarıya iner).
- Günlük kayıp sayısı fiyat bazında zararlı işlemleri sayar (başa baş + maliyet kayıp sayılmaz).

## İndikatör ile strateji neden farklı çıkabilir
Strateji, indikatörle aynı iç durum makinesini çalıştırır ve emirleri onun aynası olarak gönderir. Adet = özsermaye × risk% / (giriş − stop), yani kâr/zarar yaklaşık R cinsindendir. Farklar:
1. **Giriş barındaki stop:** İndikatör giriş barında low ≤ stop ise her zaman stop sayar. TradingView broker emülatörü bar içi yolu (açılış → yakın uç → uzak uç) varsayar; fiyat önce stop'a sonra girişe gittiyse stop olmaz. Bu durumda iç simülasyon işlemi kapattığı için strateji pozisyonu o barın kapanışında kapatır ("Senkron"), sonuç farklıdır.
2. **Aynı barda stop + hedef:** İndikatör stop sayar; emülatör bar içi yola göre hedefi seçebilir.
3. **Maliyet:** İndikatör kaymayı giriş ve çıkışta 1'er tick maliyet olarak düşer; emülatör kaymayı yalnızca market/stop emirlerine uygular (limit hedefine uygulamaz).
4. **Özsermaye bileşiği:** Strateji adedi güncel özsermayeye göre hesapladığı için net kâr, R toplamıyla birebir orantılı değildir.
5. BM ve zaman çıkışları iki sürümde de ilgili mumun kapanışında yapılır (`strategy.close(..., immediately=true)`).

Sonuçlar yaklaşık tutarlı olmalıdır; büyük fark varsa önce 1. maddeyi kontrol edin (List of Trades'te "Senkron" çıkışlar).

## Canlıya geçmeden önce
- En az **100 işlemlik** test (tablodaki "(az veri)" 30 işlemin altını gösterir; 100 hedeftir).
- Komisyon ve kayma **dahil** ortalama net R **pozitif** olmalı; hem toplamda hem kullandığınız giriş modelinde (Momentum/Reload).
- Maks. düşüş (R) ve en uzun kayıp serisini günlük limitinizle karşılaştırın.
- Farklı piyasa dönemlerinde (trend ve yatay aylar) ayrı ayrı bakın.

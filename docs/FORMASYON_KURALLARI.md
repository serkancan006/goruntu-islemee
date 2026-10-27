# Formasyon etiketleme kuralları

Bu tanımlar projenin tekrarlanabilir etiketleme sözleşmesidir. TA-Lib ile bire bir eşdeğerlik iddiası yoktur. Kod: `mumvision/patterns.py`. Aynı mumda birden çok etiket bulunabilir.

Her mum için açılış O, en yüksek H, en düşük L ve kapanış C olsun:

- Toplam aralık: R = H − L.
- Gövde: B = |C − O|.
- Üst fitil: U = H − max(O, C).
- Alt fitil: D = min(O, C) − L.
- Boğa mumu: C > O; ayı mumu: C < O.
- R = 0 olduğunda tek mumluk formasyon üretilmez.

Trend, formasyonun **ilk mumundan önceki beş mumda** ölçülür. Bu beş mumun ilk ve son kapanış farkı Δ, ortalama H−L aralığı A olsun. Δ > 0,5A ise yükseliş; Δ < −0,5A ise düşüş; aksi durumda nötrdür. Formasyonun kendi mumları ve gelecekteki mumlar trend hesabına katılmaz. Yeterli geçmiş yoksa trend gerektiren etiket üretilmez.

| Sınıf | Geometri | Önceki trend |
|---|---|---|
| Doji | R > 0 ve B/R ≤ 0,10 | Aranmaz |
| Çekiç | 0,10 < B/R ≤ 0,35; D ≥ 2B; U/R ≤ 0,12 | Düşüş |
| Ters Çekiç | 0,10 < B/R ≤ 0,35; U ≥ 2B; D/R ≤ 0,12 | Düşüş |
| Asılı Adam | Çekiç geometrisi | Yükseliş |
| Kayan Yıldız | Ters çekiç geometrisi | Yükseliş |
| Boğa Yutan | Önceki mum ayı, son mum boğa; O₂ ≤ C₁; C₂ ≥ O₁; B₂ ≥ 1,05B₁ | Düşüş |
| Ayı Yutan | Önceki mum boğa, son mum ayı; O₂ ≥ C₁; C₂ ≤ O₁; B₂ ≥ 1,05B₁ | Yükseliş |

Sabah Yıldızı, üç mumluk **boşluksuz varyant** olarak tanımlanır:

1. İlk mum ayıdır ve B₁/R₁ ≥ 0,50 olur.
2. Ortadaki mumun B₂/R₂ ≤ 0,30 ve B₂ ≤ 0,50B₁ koşulları sağlanır.
3. Ortadaki gövdenin üstü, ilk gövdenin orta noktasında veya altındadır: **max(O₂,C₂) ≤ (O₁+C₁)/2**.
4. Son mum boğadır, B₃/R₃ ≥ 0,40 ve C₃ > (O₁+C₁)/2 olur.
5. Formasyon öncesinde düşüş trendi bulunur.

Akşam Yıldızı bunun simetriğidir: ilk mum boğa, son mum ayı; ortadaki gövdenin altı ilk gövdenin orta noktasında veya üzerindedir; son kapanış bu orta noktanın altındadır; önceki trend yükseliştir. Gövde oranları aynıdır.

Kesintisiz işlem gören kripto verilerinde klasik açılış boşlukları zorunlu tutulmaz. Bu nedenle yıldız etiketleri raporda boşluksuz varyant olarak belirtilir.

Pencere etiketi, formasyonu **son mumda tamamlayan** sınıfları içerir. Nesne etiketleri ise pencere içindeki tamamlanmış bütün formasyonları içerir. Trend için gereken geçmiş görüntü dışında kalırsa, o konumdaki trendli etiket nesne listesine yazılmaz. Kutular, formasyonun 1–3 mumunun fitilleriyle birlikte birleşimidir; geçmiş trend mumları kutuya dahil edilmez.

Bu kurallar yalnız veri setinin ham OHLC üzerinden otomatik etiketlenmesinde kullanılır. Notebook'taki CNN, görüntülerden bu etiketleri öğrenir; tahmin sırasında OHLC kurallarını çalıştırmaz. Çizimde piksel yuvarlaması ve görüntü küçültme, eşik yakınındaki örneklerde görsel bilgi kaybına neden olabilir. Test metrikleri otomatik etiketlerle uyumu ölçer; bağımsız uzman doğruluğunu veya yatırım getirisini ölçmez.

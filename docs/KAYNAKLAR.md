# Kaynak incelemesi ve veri seçimi

İnceleme tarihi: 25 Ekim 2026. Yalnız incelenen kaynaklar hakkında sonuç çıkarılmıştır; bütün formasyon veri setlerinin yokluğu ileri sürülmez.

| Kaynak | Doğrulanan bilgi | Projedeki kullanım / sınır |
|---|---|---|
| [SP500-Chart-Dataset](https://github.com/JaehyunAhn/SP500-Chart-Dataset) | Depo, S&P 500 için 1.374.694 görüntü ve altı gelecekteki getiri sınıfı bildiriyor. | Hazır büyük veri örneği. Sayılar depo sahibinin beyanıdır; veri indirilip bağımsız sayılmadı. Formasyon etiketi olmadığı için kullanılmadı. |
| [Binance Public Data](https://github.com/binance/binance-public-data) | Aylık/günlük ZIP, 12 OHLCV alanı, yan dosyada SHA-256; 2025 başından itibaren SPOT mikrosaniye zaman damgası. | Gerçek OHLC kaynağı. 216 arşiv indirilip checksum doğrulandı; kullanılan tam URL ve hash'ler `data/raw/sources.json` içinde. |
| [Binance veri koşulları](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md) | Veri için ayrı kullanım koşulları bulunuyor. | Depo kodunun lisansı ile verinin koşulları aynı kabul edilmez. |
| [OpenCV inRange](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html) | HSV aralıklarıyla renk nesnelerinin ayrıştırılması. | Yeşil/kırmızı mum maskeleri. |
| [OpenCV morfoloji](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) | Yapı elemanlarıyla erozyon, genişleme ve açma. | 1×3 yatay açma ile gövde ve fitil ayrımı. |
| [TA-Lib formasyon işlevleri](https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html) | Doji, hammer, engulfing, morning/evening star gibi işlevler OHLC girdisi alıyor. | İsimlendirme ve kapsam karşılaştırması. Paket kullanılmadı; kuralların TA-Lib ile aynı olduğu iddia edilmez. |

Veri kararı: Formasyonun sayısal tanımını ve görüntünün kaynağını denetlenebilir tutmak için gerçek OHLC'den yeni etiketli görüntüler üretildi. Fiyat uydurulmadı. Etiketlerin otomatik oluşması, bağımsız uzman değerlendirmesi ihtiyacını ortadan kaldırmaz.

Ölçüm iddiaları web kaynaklarından alınmaz. Çalıştırılan kodun ürettiği `outputs/metrics.json`, `outputs/test_predictions.csv` ve `outputs/audit.json` dosyaları yerel deney kanıtıdır.

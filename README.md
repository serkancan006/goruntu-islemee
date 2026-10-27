# MumVision — Mum grafiği veri seti ve CNN notebook'u

Proje iki bölümden oluşur:

- **`mumvision/`:** Yalnız veri dizininden sorumludur. Eksik sayısal OHLC verisini Binance'ten indirir, kurallarla formasyon etiketlerini belirler ve mum grafiği resimlerini oluşturur.
- **[model_egitimi.ipynb](model_egitimi.ipynb):** Veri analizi, temizleme, ön işleme, örnek görüntüler, CNN eğitimi, validation/test grafikleri, model kaydı ve yükleyerek tekli tahmin adımlarını içerir. Gerekli eğitim/tahmin kodları notebook'un içindedir.

Kapsam: Doji, Çekiç, Ters Çekiç, Asılı Adam, Kayan Yıldız, Boğa Yutan, Ayı Yutan, Sabah Yıldızı ve Akşam Yıldızı.

## Kurulum

Proje kökünde PowerShell ile:

```powershell
# .venv yoksa bir kez oluşturun.
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Bütün bağımlılıklar tek `requirements.txt` dosyasındadır.

## MumVision ile veri oluşturma

Mevcut `data/` veri seti hazırsa bu adımı tekrar çalıştırmanız gerekmez. Veri dizininin varsayılanı `data/` klasörüdür. İlk üretim tek komutla yapılabilir:

```powershell
.\.venv\Scripts\python.exe -m mumvision build --count 100000
```

`build`, gereken ham CSV dosyalarından eksik olanları otomatik indirir; mevcut olanları kullanır. Veri üretim dönemini kronolojik bölüm tarihlerinden belirler. Kısmi indirmede diğer varlıkların kaynak/checksum kayıtları korunur. Yeni bir üretim deneyi için `--data data_yeni` seçin; mevcut veri setinin üzerine yazılmaz.

Paketin iki komutu vardır: `download` ve `build`. Yalnız indirmek veya tarih aralığı seçmek isterseniz `python -m mumvision download --data data_yeni --start 2023-01 --end 2025-12` kullanabilirsiniz. Yardım için `python -m mumvision --help` çalıştırın.

MumVision'ın ürettiği tüm dosyalar seçilen veri dizininin `cache/`, `raw/` ve `dataset/` klasörlerindedir. Formasyon kuralları yalnız veri etiketlerini üretmek için çalıştırılır. Model kaydı, tahmin raporu, loss grafikleri ve `outputs/` klasöründeki yeni deney çıktıları notebook'un sorumluluğundadır.

İndirme kesilirse aynı komut doğrulanmış önbellekten devam eder. Mevcut `manifest.csv` üzerine yazılmaz. Küçük bir üretim denemesi için `--count 1000` kullanılabilir. `build --seed` ile örnek seçimi tekrarlanabilir. Tarih bölümleri `mumvision/dataset.py` içindeki `SPLITS` ile sabittir. Mevcut CSV dosyalarının tarih aralığını elle daralttıysanız gereken dönem için `download` komutunu yeniden çalıştırın; otomatik indirme, eksik CSV dosyalarını tamamlar.

`patterns.py` içindeki kurallar, ham fiyatlardan veri seti etiketleri üretmek için gereklidir. `imaging.py` mum resimlerini ve etiket kutularını çizer.

## Notebook ile eğitim ve tahmin

`model_egitimi.ipynb` dosyasını açın, `.venv/Scripts/python.exe` çekirdeğini seçin ve hücreleri sırayla çalıştırın. Veri klasörünü notebook'taki `config.data` ayarında seçin. `QUICK_RUN=True` küçük bir akış denemesi yapar; tam veriyle eğitim için `False` yapın. Her çalıştırmada yeni deney klasörü oluşturulur.

Akış: ham veri analizi → temizleme → kronolojik bölüm kontrolü → örnek resimler → RGB/resize/normalizasyon → CNN eğitimi → train/validation loss ve F1 → validation eşikleri → test özeti ve confusion matrix/PR/ROC → modeli kaydetme, yükleme ve tekli predict.

CNN, görüntünün son mumunda tamamlanan formasyonları çok etiketli olarak öğrenir. Sınıf ağırlıkları eğitimden, tahmin eşikleri validation'dan hesaplanır. Test son değerlendirmede kullanılır. Kayıtlar ve grafikler `outputs/cnn_notebook_.../` altındadır:

| Dosya | İçerik |
|---|---|
| `best_model.pt` | En düşük validation loss modeli, sınıflar, giriş boyutu ve validation eşikleri |
| `history.csv`, `figures/learning_curves.png` | Train/validation loss ve F1 |
| `TEST_OZETI.md`, `test_metrics.json`, `test_per_class.csv` | Test özeti ve sınıf bazında sonuçlar |
| `figures/` | Dağılımlar, örnekler, ön işleme, confusion matrisleri, PR/ROC ve tekli tahmin |
| `cleaned_raw/`, `raw_cleaning.csv`, `used_manifest.csv` | Temizleme kayıtları ve kullanılan örnekler |
| `test_predictions.csv`, `test_probabilities.npz` | Örnek bazında test sonuçları |

Notebook modelden kutu üretmez. Küçük denemenin ölçümleri tam veri başarısı olarak kullanılmamalıdır. `outputs/` kökündeki eski klasik yöntem raporları önceki çalışmalara aittir; güncel CNN sonuçları ilgili deney alt klasöründedir.

## Büyük veri seti

| Özellik | Değer |
|---|---|
| Kaynak | Binance Public Data, SPOT, 1 saat |
| Varlıklar | BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT, SOLUSDT |
| Dönem | 2023-01-01 – 2025-12-31, UTC |
| Ham veri | 157.818 gerçek OHLC mumu, 216 doğrulanmış aylık arşiv |
| Grafik sayısı | 100.000 |
| Her grafik | 24 mum, 512 × 320 piksel, PNG |
| Bölümler | 70.000 eğitim, 15.000 doğrulama, 15.000 test |
| Etiketler | Çok etiketli son-mum sınıfları + görünür formasyonların kutuları |
| Depolama | ZIP shard dosyaları; PNG, YOLO TXT ve kutu JSON |
| Çeşitlilik | Açık/koyu tema, değişen mum renkleri, gövde genişliği, kenar boşluğu ve ızgara |

Fiyatlar gerçek veridir; grafikleri proje çizer. Etiketler, belgelenmiş OHLC kurallarından otomatik üretilir. **İnsan tarafından doğrulanmış etiketler değildir.** Bir görüntü birden çok formasyon içerebilir; son mumda hiçbir formasyon yoksa `none` yazılır. `none`, YOLO için ayrı bir nesne sınıfı değildir.

Hazır büyük bir veri seti olan [SP500-Chart-Dataset](https://github.com/JaehyunAhn/SP500-Chart-Dataset), gelecekteki getiri sınıflarıyla etiketlenmiştir. Burada amaç mum formasyonu olduğundan doğrudan yeniden kullanılmadı. OHLC kaynağı [Binance'in resmi açık arşividir](https://github.com/binance/binance-public-data); veri kullanımında [kaynağın koşulları](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md) geçerlidir.

### Veri sızıntısını önleme

- Eğitim: 2023–2024; doğrulama: 2025 ilk yarısı; test: 2025 ikinci yarısı.
- Bölme, grafik üretiminden önce yapılır; 24 mumun tümü aynı bölümde kalır.
- Bölüm sınırını kesen veya arada eksik saat bulunan pencereler kullanılmaz.
- Aynı ham mum farklı bölümlerde paylaşılmaz. Aynı bölümde pencereler örtüşebilir; görüntüler istatistiksel olarak bağımsız değildir.
- Girdi görüntülerinde sınıf adı veya etiket kutusu bulunmaz; stil sınıfa göre seçilmez.
- Notebook tahmin eşiklerini yalnız doğrulama bölümünde seçer; test model veya eşik seçiminde kullanılmaz.

### Dosya düzeni

```text
mumvision/__main__.py      Yalnız download ve build komutları
mumvision/download.py      Sayısal piyasa verisi indirme ve checksum kontrolü
mumvision/dataset.py       Etiketli grafik veri seti ve kronolojik bölümler
mumvision/imaging.py       PNG çizimi, nesne kutuları ve YOLO etiketleri
mumvision/patterns.py      OHLC'den otomatik formasyon etiketleri
model_egitimi.ipynb        Analiz, temizleme, eğitim, test ve tekli tahmin
requirements.txt          Tek bağımlılık dosyası
tests/                    Veri indirme, etiketleme ve üretim testleri
docs/FORMASYON_KURALLARI.md Sayısal etiketleme tanımları
docs/KAYNAKLAR.md          Veri kaynakları
data/cache/               Kaynak ZIP arşivleri ve SHA-256 dosyaları
data/raw/                 Sayısal OHLC verileri ve sources.json
data/dataset/             Manifest, PNG arşivleri, etiketler ve metadata
outputs/cnn_notebook_.../  Notebook'un model, grafik ve test çıktıları
```

### Veri dosyalarının biçimi

Her ZIP içindeki `images/`, `labels/` ve `annotations/` dizinleri aynı örnek kimliğini kullanır. PNG'leri incelemek için notebook'taki örnek görüntü hücrelerini kullanın.

Nesne kutuları `xyxy` biçiminde, sağ/alt sınırlar hariç tutularak saklanır. YOLO TXT satırları `class_id x_center y_center width height` biçimindedir; koordinatlar görüntü boyutuna bölünmüştür. Sınıf sırası `classes.txt` içindedir. Bu etiketler, son-mum sınıflandırması için hazırlanan CNN'in kutu öğrendiği anlamına gelmez.

`geometry.npz` içindeki satır sırası `manifest.csv` ile aynıdır. `labels`: N×9 son-mum etiketleri; `ohlc_pixels`: N×24×4 çizim koordinatları; `boxes`: N×24×4 mum kutuları. Y ekseni yukarı yönlüdür; piksel OHLC değerleri mutlak piyasa fiyatı değildir.

## Kontroller ve sınırlar

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Veri indirilirken SHA-256, OHLC geometrisi ve tekrar eden zaman damgaları kontrol edilir. Üretimde eksik saat içeren veya bölüm sınırını geçen pencereler elenir. Notebook da seçilen görüntüleri ve ham veriyi eğitim öncesinde kontrol eder.

Görseller gerçek sayısal piyasa verilerinden programatik olarak çizilir; internetten hazır etiketli resimler indirilmez. Etiketler otomatik OHLC kurallarına dayanır ve uzman doğrulaması değildir. Piksel yuvarlama ve küçültme ince fitillerde bilgi kaybına yol açabilir. Test metrikleri bu zayıf etiketlerle uyumu ölçer; farklı platform görüntülerinde uzman doğruluğu veya finansal getiri göstermez.

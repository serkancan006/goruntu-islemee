# MumVision — Görüntü İşleme ile Mum Formasyonu Tespiti

Finansal mum grafiği **görüntüsünden** mum gövdelerini ve fitillerini çıkarıp 9 formasyonu tespit eden Python projesi. Büyük veri seti, gerçek piyasa OHLC verilerinin programatik olarak çizilmesiyle oluşturulur. Tespit sırasında CSV veya fiyat verisi okunmaz; yalnız görüntü pikselleri kullanılır.

**Kapsam:** Doji, Çekiç, Ters Çekiç, Asılı Adam, Kayan Yıldız, Boğa Yutan, Ayı Yutan, Sabah Yıldızı ve Akşam Yıldızı. OBO, üçgen ve çift tepe/dip gibi uzun grafik formasyonları bu sürümün kapsamı dışındadır.

## Hemen kullan

Hazırlanan ortamda **`CALISTIR.bat` dosyasına çift tıkla**. “Örneği aç” ile çalışan örneği incele veya “Görüntü aç” ile kendi grafiğini seç. İlk sekmede fareyle dikdörtgen çizerek yalnız fiyat panelini kırp. Girdi, renk maskesi ve işaretli sonuç ayrı sekmelerdedir.

Yeni bilgisayarda Python 3.12 kurulduktan sonra önce `KURULUM.bat`, sonra `CALISTIR.bat` çalıştırılır. Yerel ortam hazırsa yeniden kurulum gerekmez.

PowerShell üzerinden:

```powershell
.\.venv\Scripts\python.exe -m mumvision.gui

.\.venv\Scripts\python.exe -m mumvision detect outputs/demo_input.png
```

Kendi ekran görüntüsünden fiyat panelini sayısal olarak seçmek için:

```powershell
.\.venv\Scripts\python.exe -m mumvision detect "grafik.png" --roi 80 40 900 500 --output outputs/kendi_grafigim
```

`--roi`: sol üst x, sol üst y, genişlik, yükseklik. Çıktı kutuları kırpılan alana göredir. Çıktılar `annotated.png`, `mask.png`, `result.json` dosyalarıdır.

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
- Tespit eşikleri test verisine göre ayarlanmaz.

### Dosya düzeni

```text
mumvision/                 Veri indirme, üretim, tespit, değerlendirme ve arayüz
tests/                     Kurallar, geometri, zaman damgası, sızıntı kontrolleri
docs/FORMASYON_KURALLARI.md Sayısal etiketleme tanımları
docs/KAYNAKLAR.md          Araştırılan kaynaklar ve kullanım gerekçesi
notebooks/inceleme.ipynb   Veri yükleme ve görüntü işleme incelemesi
data/raw/                 Sıkıştırılmış gerçek OHLC ve sources.json
data/cache/               Orijinal ZIP arşivleri ve SHA-256 dosyaları
data/dataset/manifest.csv  Her görüntünün kaynağı, bölümü, konumu ve etiketleri
data/dataset/geometry.npz  Çizilen OHLC pikselleri ve mum kutuları
data/dataset/shards/       100.000 PNG + YOLO TXT + kutu JSON
data/dataset/examples/     Her formasyondan bağımsız açılabilen örnek PNG
data/dataset/metadata.json Dağılımlar, kurallar ve shard SHA-256 değerleri
outputs/RAPOR.md           Gerçekte ölçülen sonuçlarla deney raporu
outputs/metrics.json       F1, precision, recall, geometri ve bozulma deneyleri
outputs/demo_input.png     Kullanıma hazır örnek
outputs/demo_detected.png  Örneğin işaretlenmiş sonucu
```

### Veriyi Python ile aç

```python
import pandas as pd
from mumvision.dataset import ImageStore

manifest = pd.read_csv("data/dataset/manifest.csv", keep_default_na=False)
with ImageStore("data/dataset") as store:
    row = manifest.iloc[0]
    bgr_image = store.read(row)
    print(row["labels"], bgr_image.shape)
```

Her shard içindeki `images/`, `labels/` ve `annotations/` dizinleri aynı örnek kimliğini kullanır. Nesne kutuları `xyxy` biçiminde, sağ/alt sınırlar hariç tutulacak şekilde saklanır. YOLO TXT satırları `class_id x_center y_center width height` biçimindedir; koordinatlar görüntü boyutuna bölünmüştür. Sınıf sırası `classes.txt` dosyasındadır. YOLO kullanımı için shard'lar bölüm bazında açılmalıdır; bu projede YOLO eğitilmemiştir.

`geometry.npz` içindeki satır sırası `manifest.csv` ile aynıdır. `labels`: N×9 son-mum zayıf etiketleri; `ohlc_pixels`: N×24×4 çizim koordinatları; `boxes`: N×24×4 mum kutuları. Y ekseni yukarı yönlüdür; piksel OHLC değerleri mutlak piyasa fiyatı değildir.

## Yeniden üretme ve doğrulama

Mevcut veri seti hazırsa bu adımların tekrarı gerekmez. Yeni üretimde farklı bir veri dizini kullanılabilir:

```powershell
.\.venv\Scripts\python.exe -m mumvision download --data data_yeni
.\.venv\Scripts\python.exe -m mumvision build --data data_yeni --count 100000
.\.venv\Scripts\python.exe -m mumvision audit --data data_yeni --full
.\.venv\Scripts\python.exe -m mumvision evaluate --data data_yeni --output outputs_yeni
.\.venv\Scripts\python.exe -m pytest -q
```

İndirme kesilirse aynı komut doğrulanmış önbellekten devam eder. Mevcut `manifest.csv` dosyasının üzerine yazılmaz. Hızlı deneme için `--count 1000` kullanılabilir. Bölüm tarihleri `dataset.py` içindeki `SPLITS` ile sabittir; `download --start/--end` değiştirildiğinde bu aralıklar da uyumlu tutulmalıdır. Aynı aralıktaki mevcut verilerden yaklaşık 157 bin farklı pencere üretilebilir; daha fazlası için tarih veya varlık kapsamı genişletilmelidir.

Yalnız değerlendirme:

```powershell
.\.venv\Scripts\python.exe -m mumvision evaluate
.\.venv\Scripts\python.exe -m mumvision audit --full
```

Sonuçlar [deney raporunda](outputs/RAPOR.md) ve makine tarafından okunabilir [ölçüm dosyasında](outputs/metrics.json) bulunur. Kurulumdaki tam paket sürümleri `requirements-lock.txt` dosyasındadır.

Tarayıcıda açılabilen, görselleri içine gömülü paylaşılabilir rapor: `outputs/RAPOR.html`. Notebook için `.venv` çekirdeğini seçin; başka bir bilgisayarda gerekirse `python -m pip install -r requirements-notebook.txt` çalıştırın.

## Yöntemin sınırları

Klasik görüntü işleme hattı: **HSV → renk eşikleme → yatay morfolojik açma → bağlı bileşenler → gövde/fitil ölçümü → formasyon kuralları**. Çalışmak için model eğitimi veya API anahtarı gerekmez. Eğitim ve doğrulama verileri ileride öğrenen yöntemler için de hazırlanmıştır.

Kırmızı/yeşil, içi dolu mumlar ve doğrusal fiyat ekseni hedeflenir. Hacim çubukları, renkli indikatörler, yazılar ve arayüz simgeleri segmentasyonu etkileyebilir; fiyat paneli kırpılmalıdır. Logaritmik eksen, siyah/beyaz ve içi boş mumlar desteklenmez.

F1 değerleri otomatik OHLC kural etiketleriyle uyumu ölçer. Bu, farklı platformlardan uzman etiketli görüntülerde başarı veya yatırım getirisi anlamına gelmez. Görüntüye çizim sırasındaki piksel yuvarlaması bile eşik yakınındaki etiketleri değiştirebilir; raporda bu etki ayrı ölçülür. Sonraki bilimsel değerlendirme adımı bağımsız, elle etiketli ekran görüntüleriyle dış testtir.

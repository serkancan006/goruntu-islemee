"""Generate an evidence-linked Turkish experiment report from actual metrics."""
import base64
import html
import json
import re
from pathlib import Path

from .patterns import PATTERNS, TURKISH


def write_report(root, output, stats):
    root, output = Path(root), Path(output)
    meta = json.loads((root / "dataset" / "metadata.json").read_text(encoding="utf-8"))
    source = json.loads((root / "raw" / "sources.json").read_text(encoding="utf-8"))
    agreement = stats["agreement_with_ohlc_rules"]
    if (output / "demo_input.png").exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .detector import annotate, detect
        from .imaging import read_image
        original = read_image(output / "demo_input.png")
        detected = detect(original)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].imshow(original[:, :, ::-1])
        axes[1].imshow(detected.green_mask | detected.red_mask, cmap="gray")
        axes[2].imshow(annotate(original, detected)[:, :, ::-1])
        for axis, title in zip(axes, ("1. Mum grafiği", "2. HSV renk maskesi", "3. Formasyon tespiti")):
            axis.set_title(title)
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(output / "pipeline.png", dpi=150)
        plt.close(fig)
    lines = ["# MumVision deney raporu", "",
             "## Araştırma sorusu", "",
             "Finansal mum grafiği görüntülerinde, HSV renk ayrıştırma ve mum geometrisi ile 1–3 mumluk formasyonlar tespit edilebilir mi?", "",
             "## Veri", "",
             f"- Gerçek OHLC mum sayısı: **{source['total_candles']:,}**.",
             f"- Üretilen görüntü sayısı: **{meta['images']:,}**; her görüntü {meta['window']} mum, 512 × 320 piksel.",
             "- Kaynak: Binance SPOT, BTCUSDT / ETHUSDT / BNBUSDT / XRPUSDT / ADAUSDT / SOLUSDT, 1 saatlik, 2023–2025.",
             "- Fiyatlar gerçek veridir; grafikler programatik çizilmiştir. Rastgele üretilmiş piyasa fiyatı kullanılmadı.",
             "- Girdi PNG'lerinde sınıf adı, etiket, sembol veya işaretli kutu bulunmaz. Renk/tema sınıftan bağımsız seçilir.",
             "- Etiketler OHLC kurallarından otomatik çıkarıldı (**zayıf gözetim**). Uzman tarafından elle etiketlenmedi.", "",
             "| Bölüm | Görüntü | Dönem (UTC) |", "|---|---:|---|",
             *[f"| {split} | {meta['split_sizes'][split]:,} | {meta['split_definitions'][split][0]} – {meta['split_definitions'][split][1]} (bitiş hariç) |" for split in ("train", "val", "test")], "",
             "Bölme işlemi grafik üretiminden önce zaman ekseninde yapıldı. Pencerenin bütün mumları aynı bölümde bulunur; sınırı kesen ve eksik saat içeren pencereler alınmaz. Böylece aynı ham mum eğitim, doğrulama ve testte tekrar kullanılmaz. Bir bölümün içindeki pencereler örtüşebilir; 100.000 görüntü, 100.000 bağımsız piyasa olayı anlamına gelmez.", "",
             "## Yöntem", "",
             "1. BGR görüntüyü HSV uzayına çevir.",
             "2. Yeşil ve kırmızı mumları `inRange` ile maskele; kırmızı hue sarmalanmasını iki aralıkla ele al.",
             "3. Yatay 1×3 açma ile gövdeleri ince fitillerden ayır.",
             "4. Özgün maskede bağlı bileşenleri bul; açılmış maskeden gövde üst/alt sınırlarını ölç.",
             "5. Bileşenleri soldan sağa sırala, renk ve geometri üzerinden piksel OHLC çıkar.",
             "6. Gövde/fitil oranları, önceki 5 mumun trendi ve komşu mum ilişkilerinden formasyonları bul.",
             "7. Formasyonun kapsadığı mumların birleşik kutusunu görüntü üzerinde göster.", "",
             "Bu yöntem sabit kurallı klasik görüntü işlemedir; eğitilmiş bir sinir ağı veya sınıflandırıcı yoktur. Eğitim/doğrulama bölümleri ileride öğrenen yöntemlerle karşılaştırmak için de hazırdır. Teste bakılarak eşik optimizasyonu yapılmaz.", "",
             "![Girdi, HSV maskesi ve formasyon tespiti](pipeline.png)", "",
             "## Ölçülen sonuçlar", "",
             f"Test edilen görüntü: **{stats['test_images']:,}**.", "",
             f"- Son mumdaki çok etiketli mikro F1: **{agreement['micro_f1']:.4f}**.",
             f"- Son mumdaki çok etiketli makro F1: **{agreement['macro_f1']:.4f}**.",
             f"- Bütün son-mum etiketlerinin tam uyumu: **%{100*agreement['exact_match']:.2f}**.",
             f"- Hep 'formasyon yok' diyen yaklaşımın tam uyumu: **%{100*stats['always_none_baseline']['exact_match']:.2f}**.",
             f"- Bütün görünür formasyon kutularında IoU≥0,5 mikro F1: **{stats['all_instance_agreement']['micro_f1']:.4f}**.",
             f"- Tüm mumların doğru sayıda ve doğru yatay konumda çıkarıldığı görüntüler: **%{100*stats['reconstruction']['all_candles_recovered_fraction']:.2f}**.",
             f"- Hizalanan görüntülerde OHLC piksel MAE: **{stats['reconstruction']['pixel_ohlc_mae_when_aligned']:.4f}** piksel.",
             f"- Ortanca tespit süresi: **{stats['detector_latency_ms']['median']:.2f} ms** (disk/ZIP okuma hariç).", "",
             "| Formasyon | Precision | Recall | F1 | Son-mum örnek sayısı |", "|---|---:|---:|---:|---:|",
             *[f"| {tr} | {agreement['per_class'][name]['precision']:.3f} | {agreement['per_class'][name]['recall']:.3f} | {agreement['per_class'][name]['f1']:.3f} | {agreement['per_class'][name]['support']} |" for name, tr in zip(PATTERNS, TURKISH)], "",
             "![Sınıflara göre ölçülen F1](class_f1.png)", "",
             "Precision/recall/F1, ham OHLC üzerindeki kural etiketlerine göre hesaplanır. Görüntüdeki mumlar da aynı kural ailesiyle yorumlandığı için bu sonuç bağımsız uzman doğruluğu değildir; görüntüden geometri geri kazanımını ve kuralların tutarlılığını ölçer. Hiçbir fiyat tahmini veya yatırım getirisi değerlendirilmedi.", "",
             "'Yalnız piksel yuvarlama' karşılaştırması, gerçek OHLC yerine çizilen tam sayı piksel koordinatlarına aynı kuralları uygular. Eşik yakınındaki gövdeler, görüntüye çizilirken 1 piksel yuvarlama nedeniyle sınıf değiştirebilir. Bu, segmentasyon hatası bulunmasa da etiket uyuşmazlığına yol açar.", "",
             "## Görsel bozulmalara duyarlılık", "",
             "Testten eşit aralıklı seçilmiş sabit alt kümede, eşikler değiştirilmeden ölçüldü. Bu deneyler dış platform ekran görüntüsü doğrulamasının yerini tutmaz.", "",
             "| Dönüşüm | Görüntü | Mum sayısı korunumu | Mikro F1 | Makro F1 |", "|---|---:|---:|---:|---:|",
             *[f"| {kind} | {result['images']} | %{100*result['candle_count_recovery']:.1f} | {result['micro_f1']:.3f} | {result['macro_f1']:.3f} |" for kind, result in stats["robustness"].items()], "",
             "## Sınırlamalar ve sonraki deney", "",
             "- Kırmızı/yeşil, içi dolu mumlar ve yalnız fiyat paneli desteklenir. Hacim çubukları, renkli indikatörler ve arayüz simgeleri önceden kırpılmalıdır. Logaritmik fiyat ekseni, siyah/beyaz veya içi boş mumlar desteklenmez.",
             "- Veri, tek borsanın altı kripto varlığıyla sınırlıdır; BIST/hisse senedi genellemesi ölçülmedi.",
             "- Sabah/Akşam Yıldızı, kesintisiz piyasalara uygun boşluksuz varyant olarak tanımlıdır; klasik gap koşulu aranmaz. Diğer eşikler de projenin açık tanımlarıdır, TA-Lib ile bire bir eşdeğerlik iddiası yoktur.",
             "- Küçük sınıflarda örnek sayısını mutlaka dikkate alın; örnek yoksa F1 0 raporlanır.",
             "- Bağımsız değerlendirme için farklı platformlardan ekran görüntülerinin en az iki kişi tarafından etiketlenmesi ve anlaşmazlıkların kaydedilmesi gerekir.",
             "- OBO, üçgen, çift tepe/dip gibi uzun grafik formasyonları bu 1–3 mumluk kapsamda yer almaz.", "",
             "## Kaynaklar", "",
             "- [Binance Public Data: arşiv şeması, checksum ve 2025 mikrosaniye değişikliği](https://github.com/binance/binance-public-data)",
             "- [Binance veri kullanım koşulları](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)",
             "- [OpenCV HSV/inRange](https://docs.opencv.org/4.x/da/d97/tutorial_threshold_inRange.html)",
             "- [OpenCV morfolojik işlemler](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)",
             "- [TA-Lib formasyon işlevleri: isimlendirme karşılaştırması](https://ta-lib.github.io/ta-lib-python/func_groups/pattern_recognition.html)",
             "- [Hazır SP500-Chart-Dataset: getiri sınıfları, formasyon etiketleri yerine](https://github.com/JaehyunAhn/SP500-Chart-Dataset)", "",
             "Ham ölçümler: `metrics.json`, `per_class_metrics.csv`, `test_predictions.csv`. Kaynak arşivleri ve SHA-256 değerleri `data/raw/sources.json` dosyasında bulunur.", ""]
    (output / "RAPOR.md").write_text("\n".join(lines), encoding="utf-8")
    # Standalone HTML: the figures remain visible when this file is shared alone.
    def inline(text):
        safe = html.escape(text)
        safe = re.sub(r"\[([^\]]+)\]\((https://[^)]+)\)", r'<a href="\2">\1</a>', safe)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        return re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)

    html_parts, in_table, in_list = [], False, False
    for line in lines:
        if line.startswith("|"):
            if not in_table:
                html_parts.append("<table>")
                in_table = True
            if line.replace("|", "").replace("-", "").replace(":", "").strip():
                cells = line.strip("|").split("|")
                html_parts.append("<tr>" + "".join("<td>" + inline(cell.strip()) + "</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            html_parts.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append("<li>" + inline(line[2:]) + "</li>")
            continue
        if in_list:
            html_parts.append("</ul>")
            in_list = False
        if line.startswith("!["):
            filename = line.split("](", 1)[1].rstrip(")")
            encoded = base64.b64encode((output / filename).read_bytes()).decode("ascii")
            html_parts.append(f'<img alt="Sınıflara göre F1" src="data:image/png;base64,{encoded}">')
        elif line.startswith("## "):
            html_parts.append("<h2>" + html.escape(line[3:]) + "</h2>")
        elif line.startswith("# "):
            html_parts.append("<h1>" + html.escape(line[2:]) + "</h1>")
        elif line:
            html_parts.append("<p>" + inline(line) + "</p>")
    if in_table:
        html_parts.append("</table>")
    if in_list:
        html_parts.append("</ul>")
    document = '''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MumVision Deney Raporu</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f5f6fa;color:#1f2937;margin:0}
main{max-width:1050px;margin:32px auto;padding:35px;background:white;border-radius:12px}
h1{font-size:32px;color:#183d62}h2{color:#183d62;margin-top:34px}img{max-width:100%}
table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}td{border-bottom:1px solid #ddd;padding:9px;text-align:left}
tr:first-child{font-weight:bold;background:#edf3f9}li{margin:6px 0}p{overflow-wrap:anywhere}
@media print{body{background:white}main{margin:0;padding:0}h2{break-after:avoid}table{break-inside:avoid}}
</style><main>''' + "\n".join(html_parts) + "</main></html>"
    (output / "RAPOR.html").write_text(document, encoding="utf-8")

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="MumVision: görüntü işleme ile mum formasyonu tespiti")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Gerçek OHLC verisini indir, SHA-256 doğrula")
    download.add_argument("--data", default="data")
    download.add_argument("--start", default="2023-01")
    download.add_argument("--end", default="2025-12")
    download.add_argument("--workers", type=int, default=8)
    build = sub.add_parser("build", help="Etiketli grafikleri ve YOLO kutularını üret")
    build.add_argument("--data", default="data")
    build.add_argument("--count", type=int, default=100_000)
    build.add_argument("--seed", type=int, default=42)
    evaluate = sub.add_parser("evaluate", help="Piksel çıkarımını ve zayıf etiket uyumunu ölç")
    evaluate.add_argument("--data", default="data")
    evaluate.add_argument("--output", default="outputs")
    evaluate.add_argument("--limit", type=int, default=0, help="0: test bölümünün tamamı")
    detect_parser = sub.add_parser("detect", help="Grafik PNG/JPEG dosyasını analiz et")
    detect_parser.add_argument("image")
    detect_parser.add_argument("--output", default="outputs/detection")
    detect_parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Hacim/menü hariç fiyat paneli")
    audit = sub.add_parser("audit", help="Veri bütünlüğünü ve bölüm sınırlarını doğrula")
    audit.add_argument("--data", default="data")
    audit.add_argument("--full", action="store_true", help="Tüm shard SHA-256 ve ZIP CRC kontrolleri")
    args = parser.parse_args()
    if args.command == "download":
        from .download import download_market
        download_market(args.data, args.start, args.end, workers=args.workers)
    elif args.command == "build":
        from .dataset import build_dataset
        build_dataset(args.data, args.count, seed=args.seed)
    elif args.command == "evaluate":
        from .evaluation import evaluate_dataset
        evaluate_dataset(args.data, args.output, args.limit)
    elif args.command == "audit":
        from .evaluation import audit_dataset
        print(json.dumps(audit_dataset(args.data, args.full), indent=2, ensure_ascii=False))
    elif args.command == "detect":
        from .detector import annotate, detect
        from .imaging import read_image, write_image
        image = read_image(args.image)
        if args.roi:
            x, y, w, h = args.roi
            if x < 0 or y < 0 or w < 1 or h < 1 or x+w > image.shape[1] or y+h > image.shape[0]:
                parser.error("ROI görüntü sınırları içinde olmalı.")
            image = image[y:y+h, x:x+w]
        result = detect(image)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_image(output / "annotated.png", annotate(image, result))
        write_image(output / "mask.png", result.green_mask | result.red_mask)
        payload = result.to_dict()
        payload["roi"] = args.roi
        payload["bounding_box_coordinates"] = "relative to ROI when --roi is supplied"
        (output / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

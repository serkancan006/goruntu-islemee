import argparse


def main():
    parser = argparse.ArgumentParser(description="MumVision: gerçek piyasa verisinden etiketli mum grafiği veri seti üretimi")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Gerçek OHLC verisini indir, SHA-256 doğrula")
    download.add_argument("--data", default="data")
    download.add_argument("--start", default="2023-01")
    download.add_argument("--end", default="2025-12")
    download.add_argument("--workers", type=int, default=8)
    build = sub.add_parser("build", help="Eksik ham verileri indir, etiketli grafik veri setini oluştur")
    build.add_argument("--data", default="data")
    build.add_argument("--count", type=int, default=100_000)
    build.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "download":
        from .download import download_market
        download_market(args.data, args.start, args.end, workers=args.workers)
    elif args.command == "build":
        from .dataset import build_dataset
        build_dataset(args.data, args.count, seed=args.seed)


if __name__ == "__main__":
    main()

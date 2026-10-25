"""Verified public historical OHLC download, without an account or API key."""
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .patterns import validate_ohlc


SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "SOLUSDT")
SOURCE = "https://data.binance.vision/data/spot/monthly/klines"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def request_bytes(url):
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "MumVision-Research/1.0"})
            with urllib.request.urlopen(request, timeout=40) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"İndirilemedi: {url}: {last_error}") from last_error


def normalize_timestamps(values):
    """Binance SPOT changed from milliseconds to microseconds in January 2025."""
    a = np.asarray(values, dtype=np.int64)
    return np.where(a >= 100_000_000_000_000, a // 1000, a)


def fetch_archive(symbol, month, cache):
    name = f"{symbol}-1h-{month}.zip"
    url = f"{SOURCE}/{symbol}/1h/{name}"
    path = cache / name
    checksum_path = cache / (name + ".CHECKSUM")
    if not checksum_path.exists():
        checksum_path.write_bytes(request_bytes(url + ".CHECKSUM"))
    expected = checksum_path.read_text(encoding="utf-8").split()[0].lower()
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        payload = request_bytes(url)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"SHA-256 uyuşmazlığı: {url}")
        temporary = path.with_suffix(".zip.tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Arşiv tek CSV içermeli: {path}")
        frame = pd.read_csv(io.BytesIO(archive.read(csv_names[0])), header=None, names=COLUMNS)
    frame["timestamp_ms"] = normalize_timestamps(frame["open_time"])
    frame = frame[["timestamp_ms", "open", "high", "low", "close", "volume"]]
    validate_ohlc(frame[["open", "high", "low", "close"]].to_numpy())
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"Pozitif olmayan fiyat: {url}")
    return frame, {"symbol": symbol, "month": month, "url": url, "sha256": expected,
                   "rows": len(frame), "verified": True}


def download_market(root="data", start="2023-01", end="2025-12", symbols=SYMBOLS, workers=8):
    root = Path(root)
    cache, raw = root / "cache", root / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    months = [str(period) for period in pd.period_range(start, end, freq="M")]
    frames = {symbol: [] for symbol in symbols}
    records, errors = [], []
    total = len(symbols) * len(months)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(fetch_archive, symbol, month, cache): (symbol, month)
                   for symbol in symbols for month in months}
        for completed, future in enumerate(as_completed(pending), 1):
            symbol, month = pending[future]
            try:
                frame, record = future.result()
                frames[symbol].append(frame)
                records.append(record)
            except Exception as exc:
                errors.append({"symbol": symbol, "month": month, "error": str(exc)})
                print(f"HATA {symbol} {month}: {exc}", flush=True)
            if completed % 12 == 0 or completed == total:
                print(f"İndirilen/doğrulanan arşiv: {completed}/{total}", flush=True)
    (raw / "download_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        raise RuntimeError(f"{len(errors)} arşiv indirilemedi. Aynı komut önbellekten devam eder; download_errors.json dosyasına bakın.")
    summaries = []
    for symbol, parts in frames.items():
        frame = pd.concat(parts, ignore_index=True).sort_values("timestamp_ms").reset_index(drop=True)
        if frame["timestamp_ms"].duplicated().any():
            raise ValueError(f"Tekrarlanan zaman damgası: {symbol}")
        deltas = np.diff(frame.timestamp_ms.to_numpy())
        path = raw / f"{symbol}-1h.csv.gz"
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        summaries.append({"symbol": symbol, "rows": len(frame), "gaps": int(np.sum(deltas != 3_600_000)),
                          "start": pd.to_datetime(frame.timestamp_ms.min(), unit="ms", utc=True).isoformat(),
                          "end": pd.to_datetime(frame.timestamp_ms.max(), unit="ms", utc=True).isoformat(),
                          "file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    provenance = {"source": "Binance Public Data, SPOT 1h", "synthetic_prices": False,
                  "timestamp_unit": "milliseconds UTC (normalized)", "symbols": summaries,
                  "total_candles": sum(s["rows"] for s in summaries),
                  "terms_url": "https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md",
                  "archives": sorted(records, key=lambda item: (item["symbol"], item["month"]))}
    (raw / "sources.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Toplam {provenance['total_candles']:,} gerçek OHLC mumu hazır.", flush=True)
    return provenance

"""Build a sharded image dataset with purged, chronological partitions."""
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .download import SYMBOLS
from .imaging import pattern_objects, png_bytes, render_chart, write_image, yolo_text
from .patterns import PATTERNS, RuleConfig, label_names, pattern_matrix


SPLITS = {
    "train": ("2023-01-01", "2025-01-01", 0.70),
    "val": ("2025-01-01", "2025-07-01", 0.15),
    "test": ("2025-07-01", "2026-01-01", 0.15),
}


def eligible_endpoints(timestamps, window, start, end):
    """All input bars must be inside the split, and exactly hourly/contiguous."""
    ts = np.asarray(timestamps, np.int64)
    ends = np.arange(window - 1, len(ts))
    starts = ends - window + 1
    start_ms = pd.Timestamp(start, tz="UTC").value // 1_000_000
    end_ms = pd.Timestamp(end, tz="UTC").value // 1_000_000
    bad = np.r_[0, np.cumsum(np.diff(ts) != 3_600_000)]
    valid = (ts[starts] >= start_ms) & (ts[ends] < end_ms) & ((bad[ends] - bad[starts]) == 0)
    return ends[valid]


def build_dataset(root="data", count=100_000, window=24, seed=42, shard_size=2000):
    if count < 10 or window < 8 or shard_size < 1:
        raise ValueError("En az 10 örnek, 8 mum ve pozitif shard_size gerekli.")
    root = Path(root)
    target = root / "dataset"
    if (target / "manifest.csv").exists():
        raise FileExistsError("Veri seti mevcut. Yeni üretim için farklı --data dizini kullanın.")
    target.mkdir(parents=True, exist_ok=True)
    (target / "shards").mkdir(exist_ok=True)
    (target / "examples").mkdir(exist_ok=True)
    frames = {symbol: pd.read_csv(root / "raw" / f"{symbol}-1h.csv.gz") for symbol in SYMBOLS}
    arrays = {symbol: frame[["open", "high", "low", "close"]].to_numpy() for symbol, frame in frames.items()}
    rng = np.random.default_rng(seed)
    selections, available = [], {}
    allocated = 0
    for split, (start, end, fraction) in SPLITS.items():
        candidates = [(symbol, int(endpoint)) for symbol, frame in frames.items()
                      for endpoint in eligible_endpoints(frame.timestamp_ms.to_numpy(), window, start, end)]
        available[split] = len(candidates)
        wanted = count - allocated if split == "test" else int(count * fraction)
        if len(candidates) < wanted:
            raise ValueError(f"{split} için {wanted} örnek isteniyor, {len(candidates)} mevcut. Ham veri tarihleri/SPLITS ayarını kontrol edin.")
        selected = rng.choice(len(candidates), wanted, replace=False)
        # Chronological order keeps localities together; class is never a selection factor.
        chosen = sorted((candidates[int(i)] for i in selected), key=lambda pair: (frames[pair[0]].timestamp_ms.iloc[pair[1]], pair[0]))
        selections.extend((split, symbol, endpoint) for symbol, endpoint in chosen)
        allocated += wanted
    labels_store = np.zeros((count, len(PATTERNS)), np.uint8)
    pixel_store = np.zeros((count, window, 4), np.float32)
    box_store = np.zeros((count, window, 4), np.int16)
    counters = {split: Counter() for split in SPLITS}
    bbox_counts = {split: Counter() for split in SPLITS}
    totals, by_symbol = Counter(), {split: Counter() for split in SPLITS}
    limits, examples, shards = {}, [], []
    archive, shard_name, shard_index = None, None, -1
    current_split, shard_count = None, 0
    manifest_path = target / "manifest.csv.partial"
    try:
        with manifest_path.open("w", newline="", encoding="utf-8") as stream:
            columns = ["sample_id", "split", "symbol", "window_start_ms", "window_end_ms", "shard", "image", "labels", "objects", "style", "render_seed"]
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for index, (split, symbol, endpoint) in enumerate(selections):
                if split != current_split or shard_count >= shard_size:
                    if archive is not None:
                        archive.close()
                    shard_index = shard_index + 1 if split == current_split else 0
                    shard_name = f"{split}-{shard_index:04d}.zip"
                    archive = zipfile.ZipFile(target / "shards" / shard_name, "w", compression=zipfile.ZIP_STORED)
                    shards.append(shard_name)
                    current_split, shard_count = split, 0
                start_index = endpoint - window + 1
                raw = arrays[symbol][start_index:endpoint + 1]
                reference = pattern_matrix(raw)
                render_seed = int(seed + index * 17)
                image, pixels, boxes, style = render_chart(raw, seed=render_seed)
                objects = pattern_objects(reference, boxes)
                sample_id = f"{split}_{index:07d}"
                member = f"images/{sample_id}.png"
                archive.writestr(member, png_bytes(image))
                archive.writestr(f"labels/{sample_id}.txt", yolo_text(objects, image.shape[1], image.shape[0]))
                archive.writestr(f"annotations/{sample_id}.json", json.dumps({"sample_id": sample_id, "objects": objects}, separators=(",", ":")))
                shard_count += 1
                labels_store[index] = reference[-1]
                pixel_store[index] = pixels
                box_store[index] = boxes
                names = label_names(reference[-1])
                counters[split].update(names or ["none"])
                bbox_counts[split].update(obj["name"] for obj in objects)
                totals[split] += 1
                by_symbol[split][symbol] += 1
                start_ms = int(frames[symbol].timestamp_ms.iloc[start_index])
                end_ms = int(frames[symbol].timestamp_ms.iloc[endpoint])
                limits.setdefault(split, [start_ms, end_ms])
                limits[split] = [min(limits[split][0], start_ms), max(limits[split][1], end_ms)]
                writer.writerow({"sample_id": sample_id, "split": split, "symbol": symbol,
                                 "window_start_ms": start_ms, "window_end_ms": end_ms,
                                 "shard": shard_name, "image": member, "labels": "|".join(names) or "none",
                                 "objects": len(objects), "style": "dark" if style["dark"] else "light", "render_seed": render_seed})
                if len(examples) < 18 and names and all(example["label"] != names[0] for example in examples):
                    example_name = f"{names[0]}.png"
                    write_image(target / "examples" / example_name, image)
                    examples.append({"label": names[0], "sample_id": sample_id, "file": example_name})
                if index % 5000 == 0 or index == count - 1:
                    print(f"Grafik üretimi: {index + 1:,}/{count:,} ({split})", flush=True)
    finally:
        if archive is not None:
            archive.close()
    manifest_path.replace(target / "manifest.csv")
    np.savez_compressed(target / "geometry.npz", labels=labels_store, ohlc_pixels=pixel_store, boxes=box_store)
    for first, second in (("train", "val"), ("val", "test")):
        if first in limits and second in limits:
            assert limits[first][1] < limits[second][0], "Bölümler arasında tarih sızıntısı."
    metadata = {
        "name": "MumVision-100K", "version": 1, "images": count, "window": window,
        "image_size": [512, 320], "seed": seed, "classes": list(PATTERNS), "synthetic_prices": False,
        "label_source": "Weak supervision: explicit geometric rules on genuine OHLC; not expert annotations",
        "split_strategy": "Global chronological blocks; full window inside split; gaps excluded; no cross-split candle reuse",
        "split_definitions": SPLITS, "available_windows": available, "split_sizes": dict(totals),
        "last_candle_counts": {split: dict(counts) for split, counts in counters.items()},
        "all_instance_counts": {split: dict(counts) for split, counts in bbox_counts.items()},
        "symbol_counts": {split: dict(counts) for split, counts in by_symbol.items()},
        "timestamp_ranges_ms": limits, "rules": RuleConfig().to_dict(), "examples": examples,
        "limitations": ["Overlapping windows within a split are correlated and not independent samples.",
                        "Rule labels are not expert-confirmed patterns and do not establish predictive returns.",
                        "Rendered green/red filled candles are a restricted visual domain.",
                        "Star patterns are gap-free variants; not exact TA-Lib implementations."],
        "shards": [{"file": name, "sha256": hashlib.sha256((target / "shards" / name).read_bytes()).hexdigest(),
                    "bytes": (target / "shards" / name).stat().st_size} for name in shards],
    }
    (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "classes.txt").write_text("\n".join(PATTERNS) + "\n", encoding="utf-8")
    (target / "dataset.yaml").write_text("# Extract shard images/ and labels/ into train/, val/, test/ first.\npath: .\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(PATTERNS)), encoding="utf-8")
    print(f"{count:,} grafik, {sum(sum(c.values()) for c in bbox_counts.values()):,} formasyon kutusu üretildi.", flush=True)
    return metadata


class ImageStore:
    def __init__(self, dataset_dir):
        self.root = Path(dataset_dir)
        self.name, self.archive = None, None

    def read(self, row):
        import cv2
        if row["shard"] != self.name:
            self.close()
            self.name = row["shard"]
            self.archive = zipfile.ZipFile(self.root / "shards" / self.name)
        image = cv2.imdecode(np.frombuffer(self.archive.read(row["image"]), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Bozuk PNG: {row['image']}")
        return image

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        self.name = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

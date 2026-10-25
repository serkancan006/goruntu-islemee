"""Honest weak-label agreement, reconstruction and instance metrics."""
import hashlib
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .dataset import ImageStore
from .detector import annotate, detect
from .imaging import pattern_objects, write_image
from .patterns import PATTERNS, TURKISH, label_names, pattern_matrix


def multilabel_metrics(truth, predicted):
    p, r, f, s = precision_recall_fscore_support(truth, predicted, average=None, zero_division=0)
    _, _, micro, _ = precision_recall_fscore_support(truth, predicted, average="micro", zero_division=0)
    return {"micro_f1": float(micro), "macro_f1": float(f.mean()),
            "exact_match": float(np.mean(np.all(truth == predicted, axis=1))),
            "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                                 "f1": float(f[i]), "support": int(s[i])} for i, name in enumerate(PATTERNS)}}


def iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - intersection
    return intersection / union if union else 0.0


def instance_counts(reference, prediction, threshold=0.5):
    """One-to-one greedy matching by class and descending IoU, no confidence/AP."""
    counts = np.zeros((len(PATTERNS), 3), np.int64)  # TP, FP, FN
    for class_id in range(len(PATTERNS)):
        gt = [obj["bbox"] for obj in reference if obj["class_id"] == class_id]
        pred = [obj["bbox"] for obj in prediction if obj["class_id"] == class_id]
        pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(gt) for j, b in enumerate(pred)), reverse=True)
        used_gt, used_pred = set(), set()
        for overlap, i, j in pairs:
            if overlap >= threshold and i not in used_gt and j not in used_pred:
                used_gt.add(i)
                used_pred.add(j)
        counts[class_id] = len(used_gt), len(pred)-len(used_pred), len(gt)-len(used_gt)
    return counts


def instance_metrics(counts):
    result = {}
    for name, (tp, fp, fn) in zip(PATTERNS, counts):
        precision, recall = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
        result[name] = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": precision,
                        "recall": recall, "f1": 2*tp / max(2*tp+fp+fn, 1), "support": int(tp+fn)}
    tp, fp, fn = counts.sum(axis=0)
    return {"iou_threshold": 0.5, "micro_f1": float(2*tp / max(2*tp+fp+fn, 1)), "per_class": result}


def audit_dataset(root="data", full=False):
    root = Path(root)
    dataset = root / "dataset"
    meta = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(dataset / "manifest.csv", keep_default_na=False)
    geometry = np.load(dataset / "geometry.npz")
    labels = geometry["labels"]  # Decompress once, rather than once per manifest row.
    assert len(manifest) == meta["images"] == len(labels)
    assert manifest.sample_id.is_unique
    assert not manifest.duplicated(["symbol", "window_end_ms"]).any()
    assert np.all(manifest.window_end_ms - manifest.window_start_ms == (meta["window"]-1)*3_600_000)
    for split in ("train", "val", "test"):
        rows = manifest[manifest.split == split]
        assert len(rows) == meta["split_sizes"][split]
        assert Counter(rows.symbol) == Counter(meta["symbol_counts"][split])
        start, end, _ = meta["split_definitions"][split]
        assert rows.window_start_ms.min() >= pd.Timestamp(start, tz="UTC").value // 1_000_000
        assert rows.window_end_ms.max() < pd.Timestamp(end, tz="UTC").value // 1_000_000
    for a, b in (("train", "val"), ("val", "test")):
        assert manifest.loc[manifest.split == a, "window_end_ms"].max() < manifest.loc[manifest.split == b, "window_start_ms"].min()
    for i, text in enumerate(manifest.labels):
        assert text == ("|".join(label_names(labels[i])) or "none")
    image_count, object_count = 0, 0
    for shard in meta["shards"]:
        path = dataset / "shards" / shard["file"]
        if full:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"], f"SHA-256: {path}"
        with zipfile.ZipFile(path) as archive:
            if full:
                assert archive.testzip() is None, f"ZIP CRC: {path}"
            image_count += sum(name.startswith("images/") and name.endswith(".png") for name in archive.namelist())
            object_count += sum(name.startswith("annotations/") and name.endswith(".json") for name in archive.namelist())
            expected = manifest[manifest.shard == shard["file"]]
            assert len(expected)*3 == len(archive.namelist())
    assert image_count == len(manifest) == object_count
    sources = json.loads((root / "raw" / "sources.json").read_text(encoding="utf-8"))
    for record in sources["symbols"]:
        path = root / "raw" / record["file"]
        if full:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        frame = pd.read_csv(path)
        assert len(frame) == record["rows"]
        assert frame.timestamp_ms.is_unique and frame.timestamp_ms.is_monotonic_increasing
    return {"passed": True, "images": image_count, "shards": len(meta["shards"]),
            "raw_candles": sources["total_candles"], "sha256_and_crc_verified": full,
            "cross_split_candle_overlap": False, "manifest_geometry_labels_consistent": True}


def transformed(image, kind, rng):
    if kind == "jpeg_q65":
        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if kind == "noise_sigma5":
        return np.clip(image.astype(float) + rng.normal(0, 5, image.shape), 0, 255).astype(np.uint8)
    if kind == "blur_3x3":
        return cv2.GaussianBlur(image, (3, 3), 0.7)
    if kind == "resize_75pct":
        return cv2.resize(image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
    return image


def evaluate_dataset(root="data", output="outputs", limit=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root, output = Path(root), Path(output)
    dataset = root / "dataset"
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(dataset / "manifest.csv", keep_default_na=False)
    rows = manifest[manifest.split == "test"]
    if limit:
        rows = rows.sample(min(limit, len(rows)), random_state=43).sort_index()
    geometry = np.load(dataset / "geometry.npz")
    all_pixels, all_boxes = geometry["ohlc_pixels"], geometry["boxes"]
    truth = geometry["labels"][rows.index].astype(bool)
    prediction, ideal = np.zeros_like(truth), np.zeros_like(truth)
    recovery, errors, latencies = [], [], []
    instance_total = np.zeros((len(PATTERNS), 3), np.int64)
    details, gallery, failure_gallery = [], [], []
    # A fixed, evenly spaced 300-image subset is used for robustness.
    robust_positions = set(np.linspace(0, len(rows)-1, min(300, len(rows)), dtype=int))
    kinds = ["jpeg_q65", "noise_sigma5", "blur_3x3", "resize_75pct"]
    robust_truth = []
    robust_pred = {kind: [] for kind in kinds}
    robust_counts = Counter()
    rng = np.random.default_rng(44)
    with ImageStore(dataset) as store:
        for position, (index, row) in enumerate(rows.iterrows()):
            image = store.read(row)
            begin = time.perf_counter()
            result = detect(image)
            latencies.append((time.perf_counter() - begin) * 1000)
            same_count = len(result.ohlc) == len(all_pixels[index])
            aligned = same_count and np.all(np.abs(result.boxes[:, 0] - all_boxes[index, :, 0]) <= 1)
            recovery.append(aligned)
            if aligned:
                errors.extend(np.abs(result.ohlc - all_pixels[index]).ravel().tolist())
            if len(result.labels):
                prediction[position] = result.labels[-1]
            ideal_matrix = pattern_matrix(all_pixels[index])
            ideal[position] = ideal_matrix[-1]
            with store.archive.open(f"annotations/{row.sample_id}.json") as stream:
                reference_objects = json.load(stream)["objects"]
            instance_total += instance_counts(reference_objects, result.objects)
            exact = bool(np.array_equal(prediction[position], truth[position]))
            details.append({"sample_id": row.sample_id, "symbol": row.symbol, "recovered_candles": len(result.ohlc),
                            "expected_candles": len(all_pixels[index]), "exact_match": exact,
                            "truth": "|".join(label_names(truth[position])) or "none",
                            "prediction": "|".join(label_names(prediction[position])) or "none"})
            if truth[position].any() and len(gallery) < 9:
                gallery.append((image.copy(), annotate(image, result), row.sample_id))
            if not exact and len(failure_gallery) < 6:
                failure_gallery.append((image.copy(), annotate(image, result), row.sample_id))
            if position in robust_positions:
                robust_truth.append(truth[position])
                for kind in kinds:
                    transformed_result = detect(transformed(image, kind, rng))
                    robust_pred[kind].append(transformed_result.labels[-1] if len(transformed_result.labels) else np.zeros(len(PATTERNS), bool))
                    robust_counts[kind] += len(transformed_result.ohlc) == len(all_pixels[index])
            if (position+1) % 3000 == 0 or position+1 == len(rows):
                print(f"Test: {position+1:,}/{len(rows):,}", flush=True)
    metrics = multilabel_metrics(truth, prediction)
    stats = {"test_images": len(rows), "evaluation_scope": "Last-candle multi-label agreement plus all visible instance boxes",
             "label_source": "Rule-generated weak labels on genuine OHLC; no independent human test set",
             "detector": "Fixed HSV + horizontal morphology + connected components + explicit rules; no fitted model",
             "agreement_with_ohlc_rules": metrics,
             "rasterization_only_agreement": multilabel_metrics(truth, ideal),
             "all_instance_agreement": instance_metrics(instance_total),
             "always_none_baseline": multilabel_metrics(truth, np.zeros_like(truth)),
             "reconstruction": {"all_candles_recovered_fraction": float(np.mean(recovery)),
                                "pixel_ohlc_mae_when_aligned": float(np.mean(errors)) if errors else None,
                                "aligned_images": int(np.sum(recovery))},
             "detector_latency_ms": {"median": float(np.median(latencies)), "p95": float(np.percentile(latencies, 95)),
                                     "excludes": "disk IO, ZIP decode, rendering; includes HSV, morphology, rules"},
             "robustness": {kind: {"images": len(robust_truth), "candle_count_recovery": robust_counts[kind]/len(robust_truth),
                                     **multilabel_metrics(np.array(robust_truth), np.array(robust_pred[kind]))} for kind in kinds},
             "interpretation": "F1 measures agreement with weak labels, not expert accuracy or financial forecasting. Rasterization can cross rule thresholds."}
    (output / "metrics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(details).to_csv(output / "test_predictions.csv", index=False)
    table = pd.DataFrame(metrics["per_class"]).T
    table.index.name = "pattern"
    table.to_csv(output / "per_class_metrics.csv")
    # Comparison visual grounded in measured results.
    fig, ax = plt.subplots(figsize=(11, 5))
    xpos = np.arange(len(PATTERNS))
    ax.bar(xpos - 0.2, [metrics["per_class"][name]["f1"] for name in PATTERNS], 0.4, label="Görüntüden tespit")
    ax.bar(xpos + 0.2, [stats["rasterization_only_agreement"]["per_class"][name]["f1"] for name in PATTERNS], 0.4, label="Yalnız piksel yuvarlama")
    ax.set_xticks(xpos, TURKISH, rotation=25, ha="right")
    ax.set_ylabel("F1: OHLC kural etiketleriyle uyum")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "class_f1.png", dpi=160)
    plt.close(fig)
    for name, items in (("examples", gallery), ("failures", failure_gallery)):
        if items:
            grid = np.vstack([np.hstack([original, marked]) for original, marked, _ in items])
            write_image(output / f"{name}.png", grid)
    # Expose one self-contained input/output example for CLI and desktop GUI.
    if gallery:
        original, marked, _ = gallery[0]
        write_image(output / "demo_input.png", original)
        write_image(output / "demo_detected.png", marked)
    from .report import write_report
    write_report(root, output, stats)
    print(json.dumps({"images": len(rows), "micro_f1": metrics["micro_f1"], "macro_f1": metrics["macro_f1"],
                      "reconstruction": stats["reconstruction"]}, indent=2, ensure_ascii=False), flush=True)
    return stats

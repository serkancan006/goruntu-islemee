"""Chart rendering and image IO, including Unicode paths on Windows."""
from pathlib import Path

import cv2
import numpy as np

from .patterns import PATTERNS, SPANS


def read_image(path):
    result = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if result is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    return result


def png_bytes(image):
    ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 2])
    if not ok:
        raise ValueError("PNG kodlanamadı.")
    return buffer.tobytes()


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes(image))


def render_chart(ohlc, seed=42, width=512, height=320):
    """Render genuine OHLC with independent style randomization.

    No text/labels/colored annotations are included in model input images.
    Return BGR image and exact *drawn* pixel OHLC, in upward coordinates.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(ohlc, dtype=float)
    if len(a) < 3 or len(a) > width // 6:
        raise ValueError("Görüntü genişliğine uygun sayıda mum gerekli.")
    dark = bool(rng.integers(0, 2))
    base = int(rng.integers(12, 33) if dark else rng.integers(231, 251))
    image = np.full((height, width, 3), base, np.uint8)
    left, right = int(rng.integers(14, 28)), int(rng.integers(14, 28))
    top, bottom = int(rng.integers(18, 34)), int(rng.integers(18, 34))
    lower, upper = a[:, 2].min(), a[:, 1].max()
    spread = max(upper - lower, max(abs(upper), 1) * 1e-8)
    scale = (height - top - bottom) / spread
    pixels = np.rint(height - bottom - (a - lower) * scale).astype(int)
    xs = np.rint(np.linspace(left, width - right - 1, len(a))).astype(int)
    step = float(np.median(np.diff(xs)))
    half = max(2, int(step * rng.uniform(0.22, 0.32)))
    wick_width = 1  # Crisp wicks preserve endpoint measurement.
    grid = base + int(rng.integers(12, 24)) * (1 if dark else -1)
    if rng.random() < 0.75:
        for yy in np.linspace(top, height - bottom, 6).astype(int):
            cv2.line(image, (0, yy), (width - 1, yy), (grid,) * 3, 1)
        for xx in np.linspace(left, width - right, 7).astype(int):
            cv2.line(image, (xx, 0), (xx, height - 1), (grid,) * 3, 1)
    hues = (int(rng.integers(43, 86)), int(rng.choice([0, 3, 6, 172, 176])))
    colors = [tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, rng.integers(150, 246), rng.integers(155, 251)]]]), cv2.COLOR_HSV2BGR)[0, 0]) for hue in hues]
    boxes = []
    for i, (op, hi, lo, cl) in enumerate(pixels):
        color = colors[0 if a[i, 3] >= a[i, 0] else 1]
        x = int(xs[i])
        cv2.line(image, (x, int(hi)), (x, int(lo)), color, wick_width)
        cv2.rectangle(image, (x - half, int(min(op, cl))), (x + half, int(max(op, cl))), color, -1)
        boxes.append([x - half, int(hi), x + half + 1, int(lo) + 1])
    return image, -pixels.astype(np.float32), np.asarray(boxes, np.int16), {"dark": dark, "body_half_width": half}


def pattern_objects(labels, boxes):
    """Ground-truth instance boxes in xyxy, derived from OHLC weak labels."""
    objects = []
    for end, class_id in zip(*np.where(labels)):
        start = end - SPANS[class_id] + 1
        group = boxes[start:end + 1]
        objects.append({"class_id": int(class_id), "name": PATTERNS[class_id],
                        "start": int(start), "end": int(end),
                        "bbox": [int(group[:, 0].min()), int(group[:, 1].min()),
                                 int(group[:, 2].max()), int(group[:, 3].max())]})
    return objects


def yolo_text(objects, width, height):
    lines = []
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        lines.append(f"{obj['class_id']} {(x1+x2)/2/width:.7f} {(y1+y2)/2/height:.7f} {(x2-x1)/width:.7f} {(y2-y1)/height:.7f}")
    return "\n".join(lines) + ("\n" if lines else "")

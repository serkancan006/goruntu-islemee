"""HSV segmentation -> components -> body/wick geometry -> pattern rules."""
from dataclasses import dataclass

import cv2
import numpy as np

from .imaging import pattern_objects
from .patterns import RuleConfig, label_names, pattern_matrix


@dataclass
class Detection:
    ohlc: np.ndarray
    boxes: np.ndarray
    labels: np.ndarray
    objects: list
    green_mask: np.ndarray
    red_mask: np.ndarray
    warnings: list

    def to_dict(self):
        return {"candles": len(self.ohlc), "coordinate_system": "upward pixel coordinates; no absolute price calibration",
                "last_candle_patterns": label_names(self.labels[-1]) if len(self.labels) else [],
                "ohlc_pixels": self.ohlc.tolist(), "candle_boxes": self.boxes.tolist(),
                "patterns": self.objects, "warnings": self.warnings}


def detect(image, config=None):
    if image is None or image.ndim != 3 or min(image.shape[:2]) < 24:
        raise ValueError("En az 24x24 boyutunda renkli grafik gerekli.")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 75, 65), (100, 255, 255))
    red = cv2.bitwise_or(cv2.inRange(hsv, (0, 75, 65), (12, 255, 255)),
                        cv2.inRange(hsv, (165, 75, 65), (179, 255, 255)))
    entries = []
    height, width = image.shape[:2]
    for mask, bullish in ((green, True), (red, False)):
        # Horizontal opening separates wide candle bodies from thin wicks.
        # Component extraction uses the original mask so wicks are retained.
        body_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, 3), np.uint8))
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_id in range(1, count):
            x, y, w, h, area = (int(z) for z in stats[component_id])
            if w < 3 or h < 1 or area < 4 or w > width / 7 or h > height * 0.96:
                continue
            component = components[y:y+h, x:x+w] == component_id
            body = (body_mask[y:y+h, x:x+w] > 0) & component
            rows = np.flatnonzero(body.sum(axis=1) >= max(3, int(w * 0.6)))
            if not len(rows):
                continue
            body_top, body_bottom = y + int(rows.min()), y + int(rows.max())
            op, cl = ((-body_bottom, -body_top) if bullish else (-body_top, -body_bottom))
            entries.append((x + w / 2, [op, -y, -(y+h-1), cl], [x, y, x+w, y+h]))
    entries.sort(key=lambda item: item[0])
    ohlc = np.array([e[1] for e in entries], dtype=np.float64).reshape(-1, 4)
    boxes = np.array([e[2] for e in entries], dtype=np.int32).reshape(-1, 4)
    labels = pattern_matrix(ohlc, config or RuleConfig())
    warnings = []
    if len(ohlc) < 8:
        warnings.append("8'den az mum çıkarıldı; trend gerektiren formasyonlar için bağlam yetersiz olabilir.")
    if len(entries) >= 4:
        gaps = np.diff([e[0] for e in entries])
        if np.max(gaps) > 1.8 * np.median(gaps) or np.min(gaps) < 0.45 * np.median(gaps):
            warnings.append("Mum aralıkları düzensiz; eksik mum veya grafik dışı renkli nesne olabilir. Fiyat panelini kırpın.")
    return Detection(ohlc, boxes, labels, pattern_objects(labels, boxes), green, red, warnings)


def annotate(image, detection):
    result = image.copy()
    for index, box in enumerate(detection.boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1-1, y1-1), (x2, y2), (180, 130, 40), 1)
        cv2.putText(result, str(index + 1), (x1, max(10, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 130, 60), 1)
    for obj in detection.objects:
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(result, (x1-3, y1-3), (x2+3, y2+3), (0, 195, 245), 2)
        text = obj["name"]
        yy = max(12, y1 - 8 - 11 * (obj["class_id"] % 2))
        cv2.putText(result, text, (max(0, min(x1, result.shape[1]-len(text)*6)), yy), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 170, 220), 1, cv2.LINE_AA)
    return result

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from mumvision.dataset import eligible_endpoints
from mumvision.detector import detect
from mumvision.download import normalize_timestamps
from mumvision.evaluation import instance_counts
from mumvision.imaging import read_image, render_chart, write_image
from mumvision.patterns import PATTERNS, pattern_matrix


def descending():
    o = np.arange(100.0, 88.0, -1.5)
    return np.c_[o, o + 0.3, o - 1.0, o - 0.7]


def mirrored(a):
    return np.c_[200-a[:, 0], 200-a[:, 2], 200-a[:, 1], 200-a[:, 3]]


def sample(name):
    fixtures = {
        "doji": np.vstack([descending(), [86, 88, 84, 86.05]]),
        "hammer": np.vstack([descending(), [86.9, 87.35, 84.5, 87.2]]),
        "inverted_hammer": np.vstack([descending(), [85, 87.7, 84.8, 85.4]]),
        "bullish_engulfing": np.vstack([descending(), [88, 88.2, 86.8, 87], [86.9, 89.1, 86.7, 88.6]]),
        "morning_star": np.vstack([descending(), [88, 88.2, 85.8, 86], [85.8, 86.2, 85.4, 85.9], [85.8, 88.2, 85.6, 88]]),
    }
    mirrors = {"hanging_man": "inverted_hammer", "shooting_star": "hammer",
               "bearish_engulfing": "bullish_engulfing", "evening_star": "morning_star"}
    return mirrored(fixtures[mirrors[name]]) if name in mirrors else fixtures[name]


@pytest.mark.parametrize("name", PATTERNS)
def test_hand_authored_positive_pattern(name):
    assert pattern_matrix(sample(name))[-1, PATTERNS.index(name)]


def test_context_changes_hammer_name():
    a = sample("hammer")
    a[:-1] = mirrored(a[:-1])
    result = pattern_matrix(a)[-1]
    assert not result[PATTERNS.index("hammer")]
    assert result[PATTERNS.index("hanging_man")]


def test_flat_candles_are_not_doji():
    assert not pattern_matrix(np.full((24, 4), 100.0)).any()


def test_rule_labels_do_not_use_future():
    a = sample("morning_star")
    future = np.array([[89, 90, 88, 89.5], [90, 92, 89, 91]])
    assert np.array_equal(pattern_matrix(a), pattern_matrix(np.vstack([a, future]))[:len(a)])


def test_invalid_prices_are_rejected():
    with pytest.raises(ValueError):
        pattern_matrix([[100, 90, 80, 110]])


@pytest.mark.parametrize("seed", [1, 7, 42, 93, 201])
def test_renderer_roundtrip_all_geometry(seed):
    rng = np.random.default_rng(9)
    o = np.cumsum(rng.normal(size=24)) + 100
    c = o + rng.normal(size=24)
    h = np.maximum(o, c) + rng.uniform(0.1, 1.5, 24)
    l = np.minimum(o, c) - rng.uniform(0.1, 1.5, 24)
    image, pixels, boxes, _ = render_chart(np.c_[o, h, l, c], seed=seed)
    result = detect(image)
    assert len(result.ohlc) == 24
    np.testing.assert_array_equal(result.ohlc, pixels)
    np.testing.assert_array_equal(result.boxes, boxes)


def test_independent_pillow_drawing_and_hue_wrap():
    image = Image.new("RGB", (500, 240), "white")
    draw = ImageDraw.Draw(image)
    expected = []
    for i in range(24):
        x = 15 + 20*i
        top, bottom = 45 + i, 125 + i
        body_top, body_bottom = 65+i, 95+i
        # Include a one-pixel doji body and red hue near 180 degrees.
        if i == 12:
            body_bottom = body_top
        color = (15, 180, 90) if i % 2 == 0 else (220, 10, 35)
        draw.line((x, top, x, bottom), fill=color)
        draw.rectangle((x-4, body_top, x+4, body_bottom), fill=color)
        op, cl = (-body_bottom, -body_top) if i % 2 == 0 else (-body_top, -body_bottom)
        expected.append([op, -top, -bottom, cl])
    result = detect(np.asarray(image)[:, :, ::-1].copy())
    np.testing.assert_array_equal(result.ohlc, expected)


def test_unicode_image_paths(tmp_path):
    path = tmp_path / "Türkçe görüntü" / "çekiç.png"
    image, *_ = render_chart(sample("hammer"))
    write_image(path, image)
    np.testing.assert_array_equal(read_image(path), image)


def test_timestamp_units_across_2025_change():
    actual = normalize_timestamps([1704067200000, 1735689600000000])
    np.testing.assert_array_equal(actual, [1704067200000, 1735689600000])


def test_split_windows_are_purged_and_gaps_excluded():
    start = pd.Timestamp("2024-12-30", tz="UTC").value // 1_000_000
    ts = start + np.arange(144) * 3_600_000
    train = eligible_endpoints(ts, 24, "2024-01-01", "2025-01-01")
    val = eligible_endpoints(ts, 24, "2025-01-01", "2026-01-01")
    assert ts[train].max() < ts[val-23].min()
    gapped = np.delete(ts, 70)
    endpoints = eligible_endpoints(gapped, 24, "2024-01-01", "2026-01-01")
    for end in endpoints:
        assert np.all(np.diff(gapped[end-23:end+1]) == 3_600_000)


def test_one_prediction_cannot_match_two_ground_truth_boxes():
    obj = {"class_id": 0, "bbox": [0, 0, 10, 10]}
    result = instance_counts([obj, obj], [obj])
    np.testing.assert_array_equal(result[0], [1, 0, 1])


def test_blank_image_returns_no_detections():
    result = detect(np.full((240, 500, 3), 240, np.uint8))
    assert len(result.ohlc) == 0 and not result.objects
    assert result.warnings

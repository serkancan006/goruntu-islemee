import hashlib
import json
import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

from mumvision.__main__ import main
from mumvision.dataset import SPLITS, build_dataset, eligible_endpoints
from mumvision.download import COLUMNS, SYMBOLS, download_market, fetch_archive, normalize_timestamps
from mumvision.imaging import render_chart, write_image
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


def test_unicode_image_paths(tmp_path):
    path = tmp_path / "Türkçe görüntü" / "çekiç.png"
    image, *_ = render_chart(sample("hammer"))
    write_image(path, image)
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_array_equal(decoded, image)


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


def test_cached_archive_is_verified_and_normalized_without_network(tmp_path, monkeypatch):
    name = "BTCUSDT-1h-2025-01.zip"
    with zipfile.ZipFile(tmp_path / name, "w") as archive:
        frame = pd.DataFrame([[1735689600000000, 10, 12, 9, 11, 3, 1735693199999999, 0, 1, 0, 0, 0]], columns=COLUMNS)
        archive.writestr(name.replace(".zip", ".csv"), frame.to_csv(index=False, header=False))
    checksum = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    (tmp_path / (name + ".CHECKSUM")).write_text(checksum + " " + name, encoding="utf-8")

    def reject_network(url):
        pytest.fail(f"Doğrulanmış önbellek için ağ isteği yapılmamalı: {url}")

    monkeypatch.setattr("mumvision.download.request_bytes", reject_network)
    frame, record = fetch_archive("BTCUSDT", "2025-01", tmp_path)
    assert frame.timestamp_ms.tolist() == [1735689600000]
    assert record["verified"] and record["sha256"] == checksum
    assert frame[["open", "high", "low", "close"]].values.tolist() == [[10, 12, 9, 11]]


def write_raw_fixture(root, symbols=SYMBOLS):
    (root / "raw").mkdir(parents=True, exist_ok=True)
    timestamps = np.concatenate([
        pd.date_range(start, periods=48, freq="h", tz="UTC").as_unit("ms").asi8
        for start, _, _ in SPLITS.values()
    ])
    for i, symbol in enumerate(symbols):
        rng = np.random.default_rng(i)
        o = 100 + np.cumsum(rng.normal(0, .5, len(timestamps)))
        c = o + rng.normal(0, .5, len(o))
        frame = pd.DataFrame({"timestamp_ms": timestamps, "open": o, "high": np.maximum(o, c) + .4,
                              "low": np.minimum(o, c) - .4, "close": c, "volume": 1.})
        frame.to_csv(root / "raw" / f"{symbol}-1h.csv.gz", index=False, compression="gzip")


def test_build_dataset_writes_notebook_compatible_images_labels_and_splits(tmp_path, monkeypatch):
    root = tmp_path / "data"
    write_raw_fixture(root)

    def reject_download(*args, **kwargs):
        pytest.fail("Ham dosyalar mevcutken indirme yapılmamalı.")

    monkeypatch.setattr("mumvision.dataset.download_market", reject_download)
    metadata = build_dataset(root, count=20, shard_size=4)
    target = root / "dataset"
    manifest = pd.read_csv(target / "manifest.csv", keep_default_na=False)
    with np.load(target / "geometry.npz") as geometry:
        labels = geometry["labels"]
        assert geometry["ohlc_pixels"].shape == (20, 24, 4)
        assert geometry["boxes"].shape == (20, 24, 4)
    assert metadata["split_sizes"] == {"train": 14, "val": 3, "test": 3}
    assert metadata["classes"] == list(PATTERNS)
    assert manifest.sample_id.is_unique
    assert set(tmp_path.iterdir()) == {root}  # Veri üretimi outputs/ oluşturmaz.
    for shard in metadata["shards"]:
        path = target / "shards" / shard["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"]
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            for index, row in manifest.loc[manifest.shard == shard["file"]].iterrows():
                image = cv2.imdecode(np.frombuffer(archive.read(row.image), np.uint8), cv2.IMREAD_COLOR)
                assert image.shape == (320, 512, 3)
                annotation = json.loads(archive.read(f"annotations/{row.sample_id}.json"))
                yolo = archive.read(f"labels/{row.sample_id}.txt").decode().splitlines()
                assert len(annotation["objects"]) == row.objects == len(yolo)
                for line in yolo:
                    class_id, *coordinates = map(float, line.split())
                    assert class_id.is_integer() and 0 <= class_id < len(PATTERNS)
                    assert all(0 <= coordinate <= 1 for coordinate in coordinates)
                raw = pd.read_csv(root / "raw" / f"{row.symbol}-1h.csv.gz")
                window = raw.loc[raw.timestamp_ms.between(row.window_start_ms, row.window_end_ms)]
                assert len(window) == 24
                np.testing.assert_array_equal(labels[index], pattern_matrix(window[["open", "high", "low", "close"]])[-1])
    for first, second in (("train", "val"), ("val", "test")):
        assert manifest.loc[manifest.split == first, "window_end_ms"].max() < manifest.loc[manifest.split == second, "window_start_ms"].min()
    original = (target / "manifest.csv").read_bytes()
    with pytest.raises(FileExistsError):
        build_dataset(root, count=20)
    assert (target / "manifest.csv").read_bytes() == original


@pytest.mark.parametrize("missing", [("BTCUSDT",), SYMBOLS])
def test_build_downloads_only_missing_raw_files_and_writes_inside_data(tmp_path, monkeypatch, missing):
    root = tmp_path / "data"
    write_raw_fixture(root, symbols=[symbol for symbol in SYMBOLS if symbol not in missing])
    originals = {path: path.read_bytes() for path in (root / "raw").glob("*.gz")}
    calls = []

    def fake_download(destination, start, end, symbols):
        calls.append((destination, start, end, symbols))
        write_raw_fixture(destination, symbols)

    monkeypatch.setattr("mumvision.dataset.download_market", fake_download)
    build_dataset(root, count=10)
    assert calls == [(root, "2023-01", "2025-12", missing)]
    assert len(pd.read_csv(root / "dataset" / "manifest.csv")) == 10
    assert all(path.read_bytes() == before for path, before in originals.items())
    assert set(tmp_path.iterdir()) == {root}


def test_existing_dataset_does_not_trigger_download(tmp_path, monkeypatch):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "manifest.csv").write_text("existing", encoding="utf-8")

    def reject_download(*args, **kwargs):
        pytest.fail("Mevcut veri seti için ağ isteği yapılmamalı.")

    monkeypatch.setattr("mumvision.dataset.download_market", reject_download)
    with pytest.raises(FileExistsError):
        build_dataset(tmp_path, count=10)


def test_partial_download_preserves_other_sources_and_replaces_own_records(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    stamp = 1735689600000
    frame = pd.DataFrame({"timestamp_ms": [stamp], "open": [10], "high": [12], "low": [9], "close": [11], "volume": [3]})
    existing_path = raw / "ETHUSDT-1h.csv.gz"
    frame.to_csv(existing_path, index=False, compression="gzip")
    original = existing_path.read_bytes()
    old_summary = {"symbol": "ETHUSDT", "file": existing_path.name, "rows": 1,
                   "sha256": hashlib.sha256(original).hexdigest()}
    old_archive = {"symbol": "ETHUSDT", "month": "2025-01", "sha256": "old-checksum"}
    (raw / "sources.json").write_text(json.dumps({"symbols": [old_summary], "archives": [old_archive]}), encoding="utf-8")

    def fake_fetch(symbol, month, cache):
        assert symbol == "BTCUSDT" and month == "2025-01"
        return frame.copy(), {"symbol": symbol, "month": month, "sha256": "new-checksum", "verified": True}

    monkeypatch.setattr("mumvision.download.fetch_archive", fake_fetch)
    for _ in range(2):
        provenance = download_market(tmp_path / "data", start="2025-01", end="2025-01", symbols=("BTCUSDT",), workers=1)
        assert len(provenance["symbols"]) == len(provenance["archives"]) == 2
        assert provenance["total_candles"] == 2
        assert old_summary in provenance["symbols"] and old_archive in provenance["archives"]
        assert existing_path.read_bytes() == original
        assert json.loads((raw / "sources.json").read_text(encoding="utf-8")) == provenance
    assert set(tmp_path.iterdir()) == {tmp_path / "data"}


@pytest.mark.parametrize("command", ["detect", "evaluate", "audit"])
def test_cli_rejects_removed_commands(command, monkeypatch):
    monkeypatch.setattr("sys.argv", ["mumvision", command])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2

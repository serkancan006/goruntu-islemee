"""Explicit project rules for weak labelling of market OHLC windows.

These are geometric research definitions, not exact TA-Lib equivalents.
Multiple patterns may be true at the same candle. No future bars are used.
"""
from dataclasses import asdict, dataclass

import numpy as np


PATTERNS = (
    "doji", "hammer", "inverted_hammer", "hanging_man", "shooting_star",
    "bullish_engulfing", "bearish_engulfing", "morning_star", "evening_star",
)
SPANS = (1, 1, 1, 1, 1, 2, 2, 3, 3)


@dataclass(frozen=True)
class RuleConfig:
    doji_body_ratio: float = 0.10
    min_body_ratio: float = 0.10
    max_small_body_ratio: float = 0.35
    long_shadow_body: float = 2.0
    max_short_shadow_ratio: float = 0.12
    trend_lookback: int = 5
    trend_range_multiple: float = 0.5
    star_first_body_ratio: float = 0.50
    star_middle_body_ratio: float = 0.30
    star_middle_relative_body: float = 0.50
    star_last_body_ratio: float = 0.40

    def to_dict(self):
        return asdict(self)


def validate_ohlc(ohlc):
    a = np.asarray(ohlc, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("OHLC dizisi (N, 4) boyutunda olmalı: open, high, low, close.")
    if not np.isfinite(a).all():
        raise ValueError("OHLC değerleri sonlu olmalı.")
    if len(a) and (np.any(a[:, 1] < np.maximum(a[:, 0], a[:, 3])) or
                   np.any(a[:, 2] > np.minimum(a[:, 0], a[:, 3]))):
        raise ValueError("High/low değerleri gövdeyi kapsamıyor.")
    return a


def pattern_matrix(ohlc, config=None):
    """Return N x 9 bool labels, each describing a pattern ENDING at that row."""
    cfg = config or RuleConfig()
    a = validate_ohlc(ohlc)
    n = len(a)
    labels = np.zeros((n, len(PATTERNS)), dtype=bool)
    if not n:
        return labels
    o, h, l, c = a.T
    body = np.abs(c - o)
    ranges = h - l
    valid = ranges > 1e-12
    safe = np.maximum(ranges, 1e-12)
    ratio = body / safe
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    bull, bear = c > o, c < o
    labels[:, 0] = valid & (ratio <= cfg.doji_body_ratio)

    def trend(span):
        result = np.zeros(n, dtype=np.int8)
        ends = np.arange(cfg.trend_lookback + span - 1, n)
        starts = ends - span + 1
        cumulative = np.r_[0.0, np.cumsum(ranges)]
        mean_range = (cumulative[starts] - cumulative[starts - cfg.trend_lookback]) / cfg.trend_lookback
        delta = c[starts - 1] - c[starts - cfg.trend_lookback]
        cutoff = cfg.trend_range_multiple * mean_range
        result[ends] = np.where(delta > cutoff, 1, np.where(delta < -cutoff, -1, 0))
        return result

    trend1, trend2, trend3 = trend(1), trend(2), trend(3)
    small = valid & (ratio > cfg.min_body_ratio) & (ratio <= cfg.max_small_body_ratio)
    hammer_shape = small & (lower >= cfg.long_shadow_body * body) & (upper / safe <= cfg.max_short_shadow_ratio)
    inverted_shape = small & (upper >= cfg.long_shadow_body * body) & (lower / safe <= cfg.max_short_shadow_ratio)
    labels[:, 1] = hammer_shape & (trend1 == -1)
    labels[:, 2] = inverted_shape & (trend1 == -1)
    labels[:, 3] = hammer_shape & (trend1 == 1)
    labels[:, 4] = inverted_shape & (trend1 == 1)
    if n >= 2:
        large_enough = body[1:] >= 1.05 * body[:-1]
        labels[1:, 5] = (bear[:-1] & bull[1:] & (o[1:] <= c[:-1]) & (c[1:] >= o[:-1]) & large_enough & (trend2[1:] == -1))
        labels[1:, 6] = (bull[:-1] & bear[1:] & (o[1:] >= c[:-1]) & (c[1:] <= o[:-1]) & large_enough & (trend2[1:] == 1))
    if n >= 3:
        first_long = ratio[:-2] >= cfg.star_first_body_ratio
        middle_small = (ratio[1:-1] <= cfg.star_middle_body_ratio) & (body[1:-1] <= cfg.star_middle_relative_body * body[:-2])
        last_long = ratio[2:] >= cfg.star_last_body_ratio
        midpoint = (o[:-2] + c[:-2]) / 2
        # Gap-free variant: explicitly appropriate to continuously traded bars.
        labels[2:, 7] = (bear[:-2] & bull[2:] & first_long & middle_small & last_long &
                         (np.maximum(o[1:-1], c[1:-1]) <= midpoint) & (c[2:] > midpoint) & (trend3[2:] == -1))
        labels[2:, 8] = (bull[:-2] & bear[2:] & first_long & middle_small & last_long &
                         (np.minimum(o[1:-1], c[1:-1]) >= midpoint) & (c[2:] < midpoint) & (trend3[2:] == 1))
    return labels


def label_names(row):
    return [name for name, value in zip(PATTERNS, row) if value]

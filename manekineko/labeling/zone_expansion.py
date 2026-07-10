"""Utilities for converting sparse buy/sell points into tradable zones."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ZoneLabels:
    """Default label values used by the current ManekiNeko signal design."""

    hold: int = 1
    buy_zone: int = 2
    sell_zone: int = 3
    strong_buy: int = 4
    strong_sell: int = 6


def _clean_indices(indices: Iterable[int], length: int) -> list[int]:
    clean: list[int] = []
    for raw_idx in indices:
        idx = int(raw_idx)
        if 0 <= idx < length:
            clean.append(idx)
    return sorted(set(clean))


def expand_indices_to_zones(
    length: int,
    indices: Iterable[int],
    *,
    zone_label: int,
    strong_label: int | None = None,
    pre: int = 12,
    post: int = 2,
    base_label: int = 1,
) -> pd.Series:
    """Expand sparse event indices into a continuous zone label series.

    Args:
        length: Number of rows in the target time series.
        indices: Sparse event positions, such as primary local minima.
        zone_label: Label applied to rows around each event.
        strong_label: Optional label applied to the exact event index.
        pre: Number of bars before each event to include in the zone.
        post: Number of bars after each event to include in the zone.
        base_label: Label used outside all zones.

    Returns:
        A pandas Series of nullable integer labels.
    """

    if length < 0:
        raise ValueError("length must be non-negative")
    if pre < 0 or post < 0:
        raise ValueError("pre and post must be non-negative")

    labels = np.full(length, base_label, dtype=np.int64)

    for idx in _clean_indices(indices, length):
        start = max(0, idx - pre)
        end = min(length, idx + post + 1)
        labels[start:end] = zone_label
        if strong_label is not None:
            labels[idx] = strong_label

    return pd.Series(labels, dtype="Int64")


def expand_buy_sell_zones(
    length: int,
    buy_indices: Iterable[int],
    sell_indices: Iterable[int] | None = None,
    *,
    buy_pre: int = 12,
    buy_post: int = 2,
    sell_pre: int = 12,
    sell_post: int = 2,
    labels: ZoneLabels = ZoneLabels(),
) -> pd.Series:
    """Create a 1/2/3/4/6 zone label series from sparse buy/sell points.

    The function uses a priority mask so strong labels are not overwritten by
    weak zone labels. If buy and sell zones overlap, the later assignment can
    overwrite only labels with equal or lower priority. Strong buy/sell points
    always have the highest priority.
    """

    if length < 0:
        raise ValueError("length must be non-negative")

    out = np.full(length, labels.hold, dtype=np.int64)
    priority = np.zeros(length, dtype=np.int64)

    def assign(start: int, end: int, label: int, label_priority: int) -> None:
        mask = priority[start:end] <= label_priority
        segment = out[start:end]
        segment[mask] = label
        out[start:end] = segment
        priority_segment = priority[start:end]
        priority_segment[mask] = label_priority
        priority[start:end] = priority_segment

    for idx in _clean_indices(buy_indices, length):
        assign(max(0, idx - buy_pre), min(length, idx + buy_post + 1), labels.buy_zone, 10)
        assign(idx, idx + 1, labels.strong_buy, 100)

    if sell_indices is not None:
        for idx in _clean_indices(sell_indices, length):
            assign(max(0, idx - sell_pre), min(length, idx + sell_post + 1), labels.sell_zone, 10)
            assign(idx, idx + 1, labels.strong_sell, 100)

    return pd.Series(out, dtype="Int64")


def expand_existing_signal(
    signal: pd.Series | np.ndarray,
    *,
    labels: ZoneLabels = ZoneLabels(),
    buy_pre: int = 12,
    buy_post: int = 2,
    sell_pre: int = 12,
    sell_post: int = 2,
) -> pd.Series:
    """Expand existing strong buy/sell labels into surrounding zones.

    This is useful for migrating the old extrema-point labels into a zone-based
    target without changing the extrema detector immediately.
    """

    arr = np.asarray(signal)
    buy_indices = np.flatnonzero(arr == labels.strong_buy)
    sell_indices = np.flatnonzero(arr == labels.strong_sell)

    expanded = expand_buy_sell_zones(
        len(arr),
        buy_indices=buy_indices,
        sell_indices=sell_indices,
        buy_pre=buy_pre,
        buy_post=buy_post,
        sell_pre=sell_pre,
        sell_post=sell_post,
        labels=labels,
    )

    # Preserve existing weak labels where no expanded zone exists.
    original = pd.Series(arr, dtype="Int64")
    mask = expanded == labels.hold
    expanded.loc[mask] = original.loc[mask].fillna(labels.hold)
    return expanded.astype("Int64")

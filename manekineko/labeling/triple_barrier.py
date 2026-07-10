"""Triple-barrier style labels for long-only trading experiments.

The functions in this module deliberately generate labels from future market
outcomes. That is expected in supervised learning. The important rule is that
these labels must be used only as targets, never as model input features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

TieBreak = Literal["loss", "profit"]


@dataclass(frozen=True)
class TripleBarrierConfig:
    """Configuration for triple-barrier label generation.

    Attributes:
        horizon: Maximum number of future bars to inspect.
        profit_barrier: Net return threshold for a positive event. Example:
            ``0.012`` means +1.2% after round-trip costs.
        loss_barrier: Net return threshold for a negative event. Example:
            ``-0.006`` means -0.6% after round-trip costs.
        fee_rate_round_trip: Estimated buy+sell fee rate.
        slippage_round_trip: Estimated buy+sell slippage rate.
        hold_label: Label for no trade / timeout.
        positive_label: Label when the profit barrier is hit first.
        negative_label: Label when the loss barrier is hit first.
        tie_break: Conservative default is ``"loss"`` when profit and loss
            barriers are both touched inside the same candle.
        require_full_horizon: When true, the last ``horizon`` rows are marked
            as missing because there is not enough future data.
    """

    horizon: int = 48
    profit_barrier: float = 0.012
    loss_barrier: float = -0.006
    fee_rate_round_trip: float = 0.002
    slippage_round_trip: float = 0.001
    hold_label: int = 1
    positive_label: int = 2
    negative_label: int = 3
    tie_break: TieBreak = "loss"
    require_full_horizon: bool = True

    @property
    def round_trip_cost(self) -> float:
        return self.fee_rate_round_trip + self.slippage_round_trip

    def validate(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.profit_barrier <= 0:
            raise ValueError("profit_barrier must be positive")
        if self.loss_barrier >= 0:
            raise ValueError("loss_barrier must be negative")
        if self.round_trip_cost < 0:
            raise ValueError("round-trip cost must be non-negative")
        if self.tie_break not in {"loss", "profit"}:
            raise ValueError("tie_break must be either 'loss' or 'profit'")


def _as_float_array(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def triple_barrier_labels(
    close: pd.Series | np.ndarray,
    high: pd.Series | np.ndarray | None = None,
    low: pd.Series | np.ndarray | None = None,
    *,
    config: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Generate long-only triple-barrier labels.

    For each row ``t``, the function simulates entering at ``close[t]`` and
    scans the next ``horizon`` bars. The first touched barrier determines the
    target label:

    - profit barrier first -> ``positive_label``
    - loss barrier first -> ``negative_label``
    - neither barrier -> ``hold_label``

    Returns a DataFrame with:

    - ``tb_label``: nullable integer class label
    - ``tb_event``: ``profit``, ``loss``, ``timeout``, or ``incomplete``
    - ``tb_hit_offset``: bars until the event, nullable
    - ``tb_net_return``: event or timeout net return after round-trip cost
    """

    cfg = config or TripleBarrierConfig()
    cfg.validate()

    close_arr = _as_float_array(close, "close")
    high_arr = _as_float_array(high if high is not None else close_arr, "high")
    low_arr = _as_float_array(low if low is not None else close_arr, "low")

    if not (len(close_arr) == len(high_arr) == len(low_arr)):
        raise ValueError("close, high, and low must have the same length")

    n = len(close_arr)
    labels: list[int | pd._libs.missing.NAType] = [cfg.hold_label] * n
    events = np.full(n, "timeout", dtype=object)
    hit_offsets: list[int | pd._libs.missing.NAType] = [pd.NA] * n
    net_returns = np.full(n, np.nan, dtype=np.float64)

    for idx in range(n):
        if cfg.require_full_horizon and idx + cfg.horizon >= n:
            labels[idx] = pd.NA
            events[idx] = "incomplete"
            continue

        end = min(n - 1, idx + cfg.horizon)
        if end <= idx:
            labels[idx] = pd.NA
            events[idx] = "incomplete"
            continue

        entry = close_arr[idx]
        if entry <= 0:
            raise ValueError("close prices must be positive")

        resolved = False
        for future_idx in range(idx + 1, end + 1):
            high_ret = high_arr[future_idx] / entry - 1.0 - cfg.round_trip_cost
            low_ret = low_arr[future_idx] / entry - 1.0 - cfg.round_trip_cost

            hit_profit = high_ret >= cfg.profit_barrier
            hit_loss = low_ret <= cfg.loss_barrier

            if hit_profit and hit_loss:
                if cfg.tie_break == "loss":
                    labels[idx] = cfg.negative_label
                    events[idx] = "loss"
                    net_returns[idx] = low_ret
                else:
                    labels[idx] = cfg.positive_label
                    events[idx] = "profit"
                    net_returns[idx] = high_ret
                hit_offsets[idx] = future_idx - idx
                resolved = True
                break

            if hit_profit:
                labels[idx] = cfg.positive_label
                events[idx] = "profit"
                hit_offsets[idx] = future_idx - idx
                net_returns[idx] = high_ret
                resolved = True
                break

            if hit_loss:
                labels[idx] = cfg.negative_label
                events[idx] = "loss"
                hit_offsets[idx] = future_idx - idx
                net_returns[idx] = low_ret
                resolved = True
                break

        if not resolved:
            labels[idx] = cfg.hold_label
            events[idx] = "timeout"
            hit_offsets[idx] = end - idx
            net_returns[idx] = close_arr[end] / entry - 1.0 - cfg.round_trip_cost

    return pd.DataFrame(
        {
            "tb_label": pd.Series(labels, dtype="Int64"),
            "tb_event": events,
            "tb_hit_offset": pd.Series(hit_offsets, dtype="Int64"),
            "tb_net_return": net_returns,
        }
    )


def apply_triple_barrier(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    config: TripleBarrierConfig | None = None,
    prefix: str = "tb",
) -> pd.DataFrame:
    """Return a copy of ``df`` with triple-barrier columns appended."""

    missing = [col for col in (close_col, high_col, low_col) if col not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")

    labels = triple_barrier_labels(
        close=df[close_col],
        high=df[high_col],
        low=df[low_col],
        config=config,
    )

    out = df.copy()
    rename_map = {
        "tb_label": f"{prefix}_label",
        "tb_event": f"{prefix}_event",
        "tb_hit_offset": f"{prefix}_hit_offset",
        "tb_net_return": f"{prefix}_net_return",
    }
    labels = labels.rename(columns=rename_map)
    return pd.concat([out.reset_index(drop=True), labels], axis=1)

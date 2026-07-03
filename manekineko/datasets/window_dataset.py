"""Windowed datasets for time-series classifiers.

Important memory rule: this module keeps tensors on CPU. Move each batch to
GPU inside the training loop instead of moving the entire dataset to GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for turning a time series into supervised windows.

    Attributes:
        lookback: Number of historical rows per input sample.
        target_offset: Offset from the window end to the target label. Use
            ``0`` when the label at row ``t`` describes the future outcome from
            ``t`` onward. Use ``1`` when predicting the next row's label.
        drop_missing_labels: Drop windows whose target label is NA.
    """

    lookback: int = 240
    target_offset: int = 0
    drop_missing_labels: bool = True

    def validate(self) -> None:
        if self.lookback <= 0:
            raise ValueError("lookback must be positive")
        if self.target_offset < 0:
            raise ValueError("target_offset must be non-negative")


def make_windowed_arrays(
    df: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    label_column: str,
    config: WindowConfig | None = None,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create ``X, y, index`` arrays from a time-ordered DataFrame.

    ``X`` has shape ``[num_samples, lookback, feature_dim]``. The target ``y``
    is taken from ``window_end + target_offset``.

    Returns:
        ``X``: windowed features.
        ``y``: integer labels.
        ``target_indices``: original DataFrame row positions used as labels.
    """

    cfg = config or WindowConfig()
    cfg.validate()

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"missing feature columns: {missing_features}")
    if label_column not in df.columns:
        raise KeyError(f"missing label column: {label_column}")

    feature_df = df.loc[:, list(feature_columns)]
    if feature_df.isna().any().any():
        bad_cols = feature_df.columns[feature_df.isna().any()].tolist()
        raise ValueError(f"feature columns contain NaN values: {bad_cols}")

    features = feature_df.to_numpy(dtype=dtype, copy=True)
    labels = df[label_column].to_numpy(copy=True)

    windows: list[np.ndarray] = []
    targets: list[int] = []
    target_indices: list[int] = []

    max_window_start = len(df) - cfg.lookback - cfg.target_offset + 1
    for start in range(max(0, max_window_start)):
        end = start + cfg.lookback - 1
        target_idx = end + cfg.target_offset
        target = labels[target_idx]

        if pd.isna(target):
            if cfg.drop_missing_labels:
                continue
            raise ValueError(f"missing target label at row {target_idx}")

        windows.append(features[start : start + cfg.lookback])
        targets.append(int(target))
        target_indices.append(target_idx)

    if not windows:
        feature_dim = len(feature_columns)
        return (
            np.empty((0, cfg.lookback, feature_dim), dtype=dtype),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(windows).astype(dtype, copy=False),
        np.asarray(targets, dtype=np.int64),
        np.asarray(target_indices, dtype=np.int64),
    )


class TimeSeriesWindowDataset(Dataset):
    """PyTorch Dataset for ``[lookback, feature_dim]`` windows.

    The dataset stores CPU tensors only. In training code, use:

    ```python
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
    ```
    """

    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        *,
        indices: np.ndarray | torch.Tensor | None = None,
        return_index: bool = False,
    ) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")

        self.X = torch.as_tensor(X, dtype=torch.float32).cpu()
        self.y = torch.as_tensor(y, dtype=torch.long).cpu()
        self.return_index = return_index

        if indices is None:
            self.indices = torch.arange(len(self.y), dtype=torch.long)
        else:
            if len(indices) != len(y):
                raise ValueError("indices and y must contain the same number of samples")
            self.indices = torch.as_tensor(indices, dtype=torch.long).cpu()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if self.return_index:
            return self.X[idx], self.y[idx], self.indices[idx]
        return self.X[idx], self.y[idx]

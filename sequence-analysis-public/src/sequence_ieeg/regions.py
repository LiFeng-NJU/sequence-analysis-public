"""Channel-to-region selection without participant or clinical metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def region_channel_indices(
    channel_names: Sequence[str],
    channel_regions: Mapping[str, str],
) -> dict[str, np.ndarray]:
    """Map region labels to indices in an analysis matrix.

    ``channel_regions`` should contain only anonymous channel names and region
    labels. Clinical fields and participant identifiers are not required.
    """
    grouped: dict[str, list[int]] = {}
    for index, channel in enumerate(channel_names):
        if channel not in channel_regions:
            continue
        grouped.setdefault(str(channel_regions[channel]), []).append(index)
    return {region: np.asarray(indices, dtype=int) for region, indices in grouped.items()}


def select_region_epochs(epochs: np.ndarray, channel_indices: Sequence[int]) -> np.ndarray:
    """Select channels from ``(trials, channels, time)`` epochs."""
    values = np.asarray(epochs)
    indices = np.asarray(channel_indices, dtype=int)
    if values.ndim != 3:
        raise ValueError("epochs must have shape (trials, channels, time).")
    if indices.size == 0:
        raise ValueError("channel_indices cannot be empty.")
    return values[:, indices, :]


def select_region_continuous(signals: np.ndarray, channel_indices: Sequence[int]) -> np.ndarray:
    """Select channels from ``(channels, time)`` continuous signals."""
    values = np.asarray(signals)
    indices = np.asarray(channel_indices, dtype=int)
    if values.ndim != 2:
        raise ValueError("signals must have shape (channels, time).")
    if indices.size == 0:
        raise ValueError("channel_indices cannot be empty.")
    return values[indices, :]


def apply_by_region(
    data: np.ndarray,
    region_indices: Mapping[str, Sequence[int]],
    function: Callable[[np.ndarray], T],
    *,
    channel_axis: int,
) -> dict[str, T]:
    """Apply any feature function independently to each channel group."""
    values = np.asarray(data)
    output: dict[str, T] = {}
    for region, indices in region_indices.items():
        selector = [slice(None)] * values.ndim
        selector[channel_axis] = np.asarray(indices, dtype=int)
        output[region] = function(values[tuple(selector)])
    return output


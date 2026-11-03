"""Time-resolved representational similarity analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.distance import pdist

from .io import validate_epochs


@dataclass(frozen=True)
class RSAResult:
    times: np.ndarray
    repetitions: np.ndarray
    mean: np.ndarray


def moving_average_epochs(
    epochs: np.ndarray,
    sfreq: float,
    *,
    window_ms: float = 20.0,
) -> np.ndarray:
    """Apply a centered moving average along the time dimension."""
    values = validate_epochs(epochs)
    width = max(1, int(round(window_ms * sfreq / 1000.0)))
    return uniform_filter1d(values, size=width, axis=-1, mode="nearest")


def position_model(labels: np.ndarray) -> np.ndarray:
    """Return a condensed binary model RDM: 0=same, 1=different."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must have shape (trials,).")
    rows, columns = np.triu_indices(values.size, k=1)
    return (values[rows] != values[columns]).astype(float)


def _spearman_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = np.linalg.norm(first_rank) * np.linalg.norm(second_rank)
    if denominator == 0:
        return np.nan
    return float(np.dot(first_rank, second_rank) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with tied values sharing a rank."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        sorted_ranks[start:stop] = (start + 1 + stop) / 2.0
        start = stop
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def balanced_indices(
    labels: np.ndarray,
    classes: Sequence[int] | None,
    n_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample the same number of trials from each requested position."""
    values = np.asarray(labels)
    selected_classes = np.unique(values) if classes is None else np.asarray(classes)
    selected = []
    for label in selected_classes:
        candidates = np.flatnonzero(values == label)
        if candidates.size < n_per_class:
            raise ValueError(f"Position {label!r} has fewer than {n_per_class} trials.")
        selected.append(rng.choice(candidates, size=n_per_class, replace=False))
    return np.concatenate(selected)


def time_resolved_rsa(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    *,
    sfreq: float,
    classes: Sequence[int] | None = None,
    n_per_class: int | None = None,
    n_repeats: int = 100,
    window_ms: float = 20.0,
    random_state: int = 0,
) -> RSAResult:
    """Correlate Euclidean neural RDMs with the binary position RDM.

    Parameters
    ----------
    epochs
        Array shaped ``(trials, channels, time)``.
    labels
        Position label for each trial.
    times
        Time axis corresponding to the final array dimension.

    Returns
    -------
    RSAResult
        Repeat-level and repeat-averaged RSA time series.
    """
    values = moving_average_epochs(epochs, sfreq, window_ms=window_ms)
    labels = np.asarray(labels)
    times = np.asarray(times, dtype=float)
    if labels.shape != (values.shape[0],):
        raise ValueError("labels must match the trial dimension.")
    if times.shape != (values.shape[-1],):
        raise ValueError("times must match the time dimension.")

    selected_classes = np.unique(labels) if classes is None else np.asarray(classes)
    available = [np.count_nonzero(labels == label) for label in selected_classes]
    take = min(available) if n_per_class is None else int(n_per_class)
    if take < 2:
        raise ValueError("At least two trials per position are required.")

    rng = np.random.default_rng(random_state)
    output = np.empty((n_repeats, values.shape[-1]), dtype=float)
    for repeat in range(n_repeats):
        indices = balanced_indices(labels, selected_classes, take, rng)
        repeat_labels = labels[indices]
        model = position_model(repeat_labels)
        for time_index in range(values.shape[-1]):
            neural = pdist(values[indices, :, time_index], metric="euclidean")
            output[repeat, time_index] = _spearman_similarity(neural, model)
    return RSAResult(times=times.copy(), repetitions=output, mean=np.nanmean(output, axis=0))


def matched_condition_rsa(
    condition_epochs: Mapping[str, np.ndarray],
    condition_labels: Mapping[str, np.ndarray],
    times: np.ndarray,
    *,
    sfreq: float,
    classes: Sequence[int],
    n_repeats: int = 100,
    window_ms: float = 20.0,
    random_state: int = 0,
) -> dict[str, RSAResult]:
    """Run RSA after matching position counts across all conditions."""
    if condition_epochs.keys() != condition_labels.keys():
        raise ValueError("condition_epochs and condition_labels must share keys.")
    shared_count = min(
        np.count_nonzero(np.asarray(condition_labels[name]) == label)
        for name in condition_labels
        for label in classes
    )
    return {
        name: time_resolved_rsa(
            condition_epochs[name],
            condition_labels[name],
            times,
            sfreq=sfreq,
            classes=classes,
            n_per_class=shared_count,
            n_repeats=n_repeats,
            window_ms=window_ms,
            random_state=random_state + index,
        )
        for index, name in enumerate(condition_epochs)
    }

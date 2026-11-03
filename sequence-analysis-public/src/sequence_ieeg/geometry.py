"""Representational geometry, MDS coordinates, and position distances."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.manifold import MDS

from .io import validate_continuous, validate_epochs


@dataclass(frozen=True)
class DistanceResult:
    classes: np.ndarray
    class_pairs: np.ndarray
    within_repetitions: np.ndarray
    between_repetitions: np.ndarray
    within_mean: np.ndarray
    between_mean: np.ndarray


def spatial_whitener(resting_signals: np.ndarray, *, floor: float = 1e-12) -> np.ndarray:
    """Estimate one channel-space whitening transform from resting data.

    ``resting_signals`` must have shape ``(channels, time)``. The same returned
    transform should be used for every task condition.
    """
    rest = validate_continuous(resting_signals)
    covariance = LedoitWolf().fit(rest.T).covariance_
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    inverse_root = np.diag(1.0 / np.sqrt(np.maximum(eigenvalues, floor)))
    return eigenvectors @ inverse_root @ eigenvectors.T


def window_patterns(
    epochs: np.ndarray,
    times: np.ndarray,
    window: tuple[float, float],
    *,
    whitener: np.ndarray | None = None,
) -> np.ndarray:
    """Average an epoch window and optionally whiten its channel patterns."""
    values = validate_epochs(epochs)
    times = np.asarray(times, dtype=float)
    if times.shape != (values.shape[-1],):
        raise ValueError("times must match the epoch time dimension.")
    keep = (times >= window[0]) & (times <= window[1])
    if not np.any(keep):
        raise ValueError("The requested window contains no samples.")
    patterns = values[:, :, keep].mean(axis=-1)
    if whitener is not None:
        transform = np.asarray(whitener, dtype=float)
        if transform.shape != (values.shape[1], values.shape[1]):
            raise ValueError("whitener shape must match the channel count.")
        patterns = patterns @ transform.T
    return patterns


def remove_position_grand_mean(patterns: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Subtract the mean response across position-specific mean patterns."""
    values = np.asarray(patterns, dtype=float)
    labels = np.asarray(labels)
    if values.ndim != 2 or labels.shape != (values.shape[0],):
        raise ValueError("patterns and labels have incompatible shapes.")
    class_means = np.stack([values[labels == label].mean(axis=0) for label in np.unique(labels)])
    return values - class_means.mean(axis=0, keepdims=True)


def correlation_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the rescaled correlation distance ``(1 - r) / 2``."""
    first = np.asarray(first, dtype=float) - np.mean(first)
    second = np.asarray(second, dtype=float) - np.mean(second)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return np.nan
    return float((1.0 - np.dot(first, second) / denominator) / 2.0)


def _folds(indices: np.ndarray, n_folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    shuffled = np.asarray(indices).copy()
    rng.shuffle(shuffled)
    folds = [part for part in np.array_split(shuffled, n_folds) if part.size]
    if len(folds) != n_folds:
        raise ValueError("Each position must contain at least n_folds trials.")
    return folds


def position_distances(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    *,
    window: tuple[float, float],
    resting_signals: np.ndarray,
    classes: Sequence[int] | None = None,
    n_folds: int = 6,
    n_repeats: int = 100,
    random_state: int = 0,
) -> DistanceResult:
    """Compute within-position and between-position correlation distances.

    Between-position distances use six complementary train/held-out folds.
    Within-position distances use two non-overlapping trial halves. Both
    procedures are repeated with new assignments.
    """
    values = validate_epochs(epochs)
    labels = np.asarray(labels)
    if labels.shape != (values.shape[0],):
        raise ValueError("labels must match the trial dimension.")
    selected = np.unique(labels) if classes is None else np.asarray(classes)
    keep = np.isin(labels, selected)
    values = values[keep]
    labels = labels[keep]
    whitener = spatial_whitener(resting_signals)
    patterns = window_patterns(values, times, window, whitener=whitener)
    patterns = remove_position_grand_mean(patterns, labels)

    pair_array = np.asarray(list(combinations(selected.tolist(), 2)))
    within = np.empty((n_repeats, selected.size), dtype=float)
    between = np.empty((n_repeats, pair_array.shape[0]), dtype=float)
    rng = np.random.default_rng(random_state)

    for repeat in range(n_repeats):
        class_folds = {
            label: _folds(np.flatnonzero(labels == label), n_folds, rng)
            for label in selected
        }
        for class_index, label in enumerate(selected):
            indices = np.flatnonzero(labels == label).copy()
            rng.shuffle(indices)
            first, second = np.array_split(indices, 2)
            if first.size == 0 or second.size == 0:
                raise ValueError("Each position needs at least two trials.")
            within[repeat, class_index] = correlation_distance(
                patterns[first].mean(axis=0), patterns[second].mean(axis=0)
            )

        for pair_index, (first_label, second_label) in enumerate(pair_array):
            fold_values = []
            for fold_index in range(n_folds):
                first_test = class_folds[first_label][fold_index]
                second_test = class_folds[second_label][fold_index]
                first_train = np.concatenate(
                    [fold for index, fold in enumerate(class_folds[first_label]) if index != fold_index]
                )
                second_train = np.concatenate(
                    [fold for index, fold in enumerate(class_folds[second_label]) if index != fold_index]
                )
                fold_values.extend(
                    [
                        correlation_distance(
                            patterns[first_train].mean(axis=0),
                            patterns[second_test].mean(axis=0),
                        ),
                        correlation_distance(
                            patterns[second_train].mean(axis=0),
                            patterns[first_test].mean(axis=0),
                        ),
                    ]
                )
            between[repeat, pair_index] = np.nanmean(fold_values)

    return DistanceResult(
        classes=selected.copy(),
        class_pairs=pair_array,
        within_repetitions=within,
        between_repetitions=between,
        within_mean=np.nanmean(within, axis=0),
        between_mean=np.nanmean(between, axis=0),
    )


def split_half_rdm(
    patterns: np.ndarray,
    labels: np.ndarray,
    *,
    classes: Sequence[int] | None = None,
    n_repeats: int = 100,
    random_state: int = 0,
) -> np.ndarray:
    """Build the repeat-averaged 2N x 2N split-half position RDM."""
    values = np.asarray(patterns, dtype=float)
    labels = np.asarray(labels)
    if values.ndim != 2 or labels.shape != (values.shape[0],):
        raise ValueError("patterns must be (trials, features) with one label per trial.")
    selected = np.unique(labels) if classes is None else np.asarray(classes)
    rng = np.random.default_rng(random_state)
    rdms = np.empty((n_repeats, selected.size * 2, selected.size * 2), dtype=float)
    for repeat in range(n_repeats):
        half_patterns = []
        for half in range(2):
            for label in selected:
                indices = np.flatnonzero(labels == label).copy()
                rng.shuffle(indices)
                halves = np.array_split(indices, 2)
                if min(part.size for part in halves) == 0:
                    raise ValueError("Each position needs at least two trials.")
                half_patterns.append(values[halves[half]].mean(axis=0))
        for row in range(len(half_patterns)):
            for column in range(len(half_patterns)):
                rdms[repeat, row, column] = correlation_distance(
                    half_patterns[row], half_patterns[column]
                )
    return np.nanmean(rdms, axis=0)


def mds_coordinates(
    rdm: np.ndarray,
    *,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Return metric-MDS coordinates for a square dissimilarity matrix."""
    matrix = np.asarray(rdm, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rdm must be a square matrix.")
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    if not np.isfinite(matrix).all():
        raise ValueError("rdm must contain finite values.")
    model = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=8,
        max_iter=1000,
    )
    return model.fit_transform(matrix)


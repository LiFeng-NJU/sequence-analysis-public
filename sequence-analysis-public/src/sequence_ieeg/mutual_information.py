"""Contact-pair mutual-information matrices from stimulus-locked responses."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_regression

from .io import validate_epochs


def trial_averaged_contact_series(
    epochs: np.ndarray,
    times: np.ndarray,
    *,
    window: tuple[float, float] = (0.0, 0.168),
) -> np.ndarray:
    """Average trials and retain a post-stimulus contact-by-time matrix."""
    values = validate_epochs(epochs)
    times = np.asarray(times, dtype=float)
    if times.shape != (values.shape[-1],):
        raise ValueError("times must match the epoch time dimension.")
    keep = (times >= window[0]) & (times <= window[1])
    if np.count_nonzero(keep) < 2:
        raise ValueError("The selected window must contain at least two samples.")
    return values[:, :, keep].mean(axis=0)


def contact_pair_mi(
    first_region: np.ndarray,
    second_region: np.ndarray,
    *,
    n_neighbors: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    """Estimate a contact-by-contact k-nearest-neighbor MI matrix.

    Inputs are shaped ``(contacts, time samples)``. Time samples are the
    observations used by the estimator. Output is shaped
    ``(first contacts, second contacts)``.
    """
    first = np.asarray(first_region, dtype=float)
    second = np.asarray(second_region, dtype=float)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Both inputs must have shape (contacts, time).")
    if first.shape[1] != second.shape[1]:
        raise ValueError("Both regions must use the same time samples.")
    if first.shape[1] <= n_neighbors:
        raise ValueError("The time dimension must exceed n_neighbors.")

    matrix = np.empty((first.shape[0], second.shape[0]), dtype=float)
    for row in range(first.shape[0]):
        feature = first[row, :, None]
        for column in range(second.shape[0]):
            matrix[row, column] = mutual_info_regression(
                feature,
                second[column],
                n_neighbors=n_neighbors,
                random_state=random_state,
            )[0]
    return matrix


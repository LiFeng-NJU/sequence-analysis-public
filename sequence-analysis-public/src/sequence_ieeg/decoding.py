"""Time-resolved linear-SVM position decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .io import validate_epochs
from .rsa import moving_average_epochs


@dataclass(frozen=True)
class DecodingResult:
    times: np.ndarray
    fold_accuracy: np.ndarray
    mean_accuracy: np.ndarray
    pseudotrial_labels: np.ndarray


def make_pseudotrials(
    epochs: np.ndarray,
    labels: np.ndarray,
    *,
    classes: Sequence[int] | None = None,
    group_size: int = 3,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Average non-overlapping groups of raw trials within each position."""
    values = validate_epochs(epochs)
    labels = np.asarray(labels)
    if labels.shape != (values.shape[0],):
        raise ValueError("labels must match the trial dimension.")
    selected_classes = np.unique(labels) if classes is None else np.asarray(classes)
    minimum = min(np.count_nonzero(labels == label) for label in selected_classes)
    usable = minimum - (minimum % group_size)
    if usable < group_size:
        raise ValueError("Not enough trials to form one pseudotrial per position.")

    rng = np.random.default_rng(random_state)
    pseudotrials = []
    pseudolabels = []
    for label in selected_classes:
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        groups = indices[:usable].reshape(-1, group_size)
        pseudotrials.append(values[groups].mean(axis=1))
        pseudolabels.extend([label] * groups.shape[0])
    return np.concatenate(pseudotrials, axis=0), np.asarray(pseudolabels)


def decode_time_series(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    *,
    sfreq: float,
    classes: Sequence[int] | None = None,
    group_size: int = 3,
    n_splits: int = 6,
    window_ms: float = 20.0,
    c_value: float = 1.0,
    random_state: int = 0,
) -> DecodingResult:
    """Decode position at every time point with a linear SVM.

    Scaling is fitted within each training fold. The returned matrix is shaped
    ``(folds, time)`` and contains feature-extraction output only.
    """
    pseudo, pseudo_labels = make_pseudotrials(
        epochs,
        labels,
        classes=classes,
        group_size=group_size,
        random_state=random_state,
    )
    pseudo = moving_average_epochs(pseudo, sfreq, window_ms=window_ms)
    times = np.asarray(times, dtype=float)
    if times.shape != (pseudo.shape[-1],):
        raise ValueError("times must match the time dimension.")
    smallest_class = min(np.count_nonzero(pseudo_labels == value) for value in np.unique(pseudo_labels))
    if smallest_class < n_splits:
        raise ValueError("Each position needs at least n_splits pseudotrials.")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(np.zeros(pseudo.shape[0]), pseudo_labels))
    scores = np.empty((n_splits, pseudo.shape[-1]), dtype=float)
    for time_index in range(pseudo.shape[-1]):
        features = pseudo[:, :, time_index]
        for fold, (train, validation) in enumerate(splits):
            model = make_pipeline(
                StandardScaler(),
                SVC(C=c_value, kernel="linear", decision_function_shape="ovo"),
            )
            model.fit(features[train], pseudo_labels[train])
            scores[fold, time_index] = model.score(features[validation], pseudo_labels[validation])
    return DecodingResult(
        times=times.copy(),
        fold_accuracy=scores,
        mean_accuracy=scores.mean(axis=0),
        pseudotrial_labels=pseudo_labels,
    )


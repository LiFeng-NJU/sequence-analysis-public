"""Shared continuous-signal operations for phase-lag and PAC features."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.decomposition import PCA

from .io import validate_continuous


def bandpass(
    signals: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    *,
    order: int = 2,
) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter along time."""
    values = validate_continuous(signals)
    if not 0 < band[0] < band[1] < sfreq / 2.0:
        raise ValueError("band must lie strictly between 0 and Nyquist.")
    sos = butter(order, band, btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, values, axis=-1)


def shared_pc1(condition_signals: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fit one PC1 spatial projection across conditions and apply it to each.

    Every input is shaped ``(channels, time)`` and must have the same channel
    order. PC1 polarity is aligned to the corresponding mean contact signal.
    A single-contact region is returned unchanged.
    """
    if not condition_signals:
        raise ValueError("condition_signals cannot be empty.")
    checked = {name: validate_continuous(data) for name, data in condition_signals.items()}
    counts = {data.shape[0] for data in checked.values()}
    if len(counts) != 1:
        raise ValueError("Conditions must have the same channel count and order.")
    n_channels = next(iter(counts))
    if n_channels == 1:
        return {name: data[0].copy() for name, data in checked.items()}

    combined = np.concatenate(tuple(checked.values()), axis=1)
    model = PCA(n_components=1)
    combined_pc = model.fit_transform(combined.T)[:, 0]
    mean_contact = combined.mean(axis=0)
    if np.corrcoef(combined_pc, mean_contact)[0, 1] < 0:
        model.components_[0] *= -1.0
    return {name: model.transform(data.T)[:, 0] for name, data in checked.items()}


def epoch_series(
    series: np.ndarray,
    onset_samples: np.ndarray,
    sfreq: float,
    *,
    tmin: float = -1.0 / 6.0,
    tmax: float = 5.0 / 6.0,
) -> np.ndarray:
    """Extract half-open stimulus-locked epochs ``[tmin, tmax)``.

    Output shape is ``(trials, round((tmax - tmin) * sfreq))``.
    """
    values = np.asarray(series, dtype=float)
    onsets = np.asarray(onset_samples, dtype=int)
    if values.ndim != 1 or onsets.ndim != 1:
        raise ValueError("series and onset_samples must both be one-dimensional.")
    start_offset = int(round(tmin * sfreq))
    n_samples = int(round((tmax - tmin) * sfreq))
    epochs = []
    for onset in onsets:
        start = int(onset) + start_offset
        stop = start + n_samples
        if start < 0 or stop > values.size:
            continue
        epochs.append(values[start:stop])
    if not epochs:
        raise ValueError("No complete epochs could be extracted.")
    return np.stack(epochs)


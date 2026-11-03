"""Condition-specific inter-regional phase-lag matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.signal import hilbert

from .signals import bandpass, epoch_series, shared_pc1


def wrap_lag_ms(values: np.ndarray | float, frequency_hz: float) -> np.ndarray:
    """Wrap lag values to half a cycle around zero."""
    period = 1000.0 / frequency_hz
    values = np.asarray(values, dtype=float)
    return (values + period / 2.0) % period - period / 2.0


def phase_lag_ms(first_phase: np.ndarray, second_phase: np.ndarray, frequency_hz: float) -> float:
    """Convert the circular mean of ``first - second`` phase into milliseconds.

    Positive output means that the first (row) region leads the second (column)
    region. Inputs are shaped ``(trials, time)``.
    """
    first = np.asarray(first_phase, dtype=float)
    second = np.asarray(second_phase, dtype=float)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Phase arrays must share shape (trials, time).")
    mean_vector = np.mean(np.exp(1j * (first - second)))
    lag = np.angle(mean_vector) * 1000.0 / (2.0 * np.pi * frequency_hz)
    return float(wrap_lag_ms(lag, frequency_hz))


def phase_lag_matrix(
    phase_epochs: Mapping[str, np.ndarray],
    region_order: Sequence[str] | None = None,
    *,
    frequency_hz: float,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build an antisymmetric row-leading-column phase-lag matrix."""
    regions = tuple(phase_epochs) if region_order is None else tuple(region_order)
    matrix = np.zeros((len(regions), len(regions)), dtype=float)
    for row, first in enumerate(regions):
        for column in range(row):
            second = regions[column]
            lag = phase_lag_ms(phase_epochs[first], phase_epochs[second], frequency_hz)
            matrix[row, column] = lag
            matrix[column, row] = -lag
    return regions, matrix


def condition_phase_epochs(
    region_signals: Mapping[str, Mapping[str, np.ndarray]],
    onset_samples: Mapping[str, np.ndarray],
    sfreq: float,
    *,
    band: tuple[float, float],
    tmin: float = -1.0 / 6.0,
    tmax: float = 5.0 / 6.0,
) -> dict[str, dict[str, np.ndarray]]:
    """Filter, apply shared regional PC1, Hilbert transform, and epoch phases.

    ``region_signals[region][condition]`` is a ``(channels, continuous time)``
    matrix. PCA is fitted jointly across conditions within each region.
    """
    output: dict[str, dict[str, np.ndarray]] = {name: {} for name in onset_samples}
    for region, conditions in region_signals.items():
        filtered = {name: bandpass(data, sfreq, band) for name, data in conditions.items()}
        components = shared_pc1(filtered)
        for condition, component in components.items():
            phase = np.angle(hilbert(component))
            output[condition][region] = epoch_series(
                phase,
                onset_samples[condition],
                sfreq,
                tmin=tmin,
                tmax=tmax,
            )
    return output


def phase_lag_matrices(
    region_signals: Mapping[str, Mapping[str, np.ndarray]],
    onset_samples: Mapping[str, np.ndarray],
    sfreq: float,
    *,
    frequency_hz: float,
    band: tuple[float, float] | None = None,
    region_order: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
    """Compute one phase-lag matrix for each condition."""
    selected_band = band or (
        (1.5, 2.5) if np.isclose(frequency_hz, 2.0) else (5.5, 6.5)
    )
    epochs = condition_phase_epochs(region_signals, onset_samples, sfreq, band=selected_band)
    matrices: dict[str, np.ndarray] = {}
    regions: tuple[str, ...] | None = None
    for condition, condition_epochs in epochs.items():
        regions, matrices[condition] = phase_lag_matrix(
            condition_epochs,
            region_order,
            frequency_hz=frequency_hz,
        )
    if regions is None:
        raise ValueError("No conditions were provided.")
    return regions, matrices


def circular_condition_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    frequency_hz: float,
) -> np.ndarray:
    """Return a cycle-wrapped elementwise lag difference between conditions."""
    if np.shape(first) != np.shape(second):
        raise ValueError("Condition matrices must have the same shape.")
    return wrap_lag_ms(np.asarray(first) - np.asarray(second), frequency_hz)


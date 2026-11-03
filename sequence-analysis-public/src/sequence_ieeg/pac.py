"""Inter-regional Tort modulation-index matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.signal import hilbert

from .signals import bandpass, epoch_series, shared_pc1


def tort_modulation_index(
    phase: np.ndarray,
    amplitude: np.ndarray,
    *,
    n_bins: int = 18,
) -> float:
    """Compute the normalized Tort modulation index from paired samples."""
    phase = np.asarray(phase, dtype=float).ravel()
    amplitude = np.asarray(amplitude, dtype=float).ravel()
    if phase.shape != amplitude.shape:
        raise ValueError("phase and amplitude must contain the same samples.")
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_index = np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)
    means = np.asarray(
        [amplitude[bin_index == index].mean() if np.any(bin_index == index) else 0.0 for index in range(n_bins)]
    )
    total = means.sum()
    if total <= 0:
        return np.nan
    distribution = means / total
    positive = distribution > 0
    divergence = np.sum(distribution[positive] * np.log(distribution[positive] * n_bins))
    return float(divergence / np.log(n_bins))


def surrogate_standardized_mi(
    phase_epochs: np.ndarray,
    amplitude_epochs: np.ndarray,
    *,
    n_surrogates: int = 1000,
    n_bins: int = 18,
    random_state: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Standardize observed MI against within-micro-trial circular shifts.

    Each surrogate independently shifts the amplitude samples within every
    micro-trial by a non-zero offset, preserving the within-trial structure.
    """
    phase = np.asarray(phase_epochs, dtype=float)
    amplitude = np.asarray(amplitude_epochs, dtype=float)
    if phase.shape != amplitude.shape or phase.ndim != 2:
        raise ValueError("Inputs must share shape (trials, time).")
    if phase.shape[1] < 2:
        raise ValueError("At least two samples per micro-trial are required.")
    observed = tort_modulation_index(phase, amplitude, n_bins=n_bins)
    rng = np.random.default_rng(random_state)
    surrogate = np.empty(n_surrogates, dtype=float)
    for iteration in range(n_surrogates):
        shifted = np.empty_like(amplitude)
        for trial in range(amplitude.shape[0]):
            offset = int(rng.integers(1, amplitude.shape[1]))
            shifted[trial] = np.roll(amplitude[trial], offset)
        surrogate[iteration] = tort_modulation_index(phase, shifted, n_bins=n_bins)
    spread = surrogate.std(ddof=1)
    standardized = np.nan if spread == 0 else (observed - surrogate.mean()) / spread
    return float(standardized), float(observed), surrogate


def condition_pac_epochs(
    region_signals: Mapping[str, Mapping[str, np.ndarray]],
    onset_samples: Mapping[str, np.ndarray],
    sfreq: float,
    *,
    phase_band: tuple[float, float],
    gamma_band: tuple[float, float] = (60.0, 100.0),
    tmin: float = -1.0 / 6.0,
    tmax: float = 5.0 / 6.0,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    """Prepare low-frequency phase and gamma-amplitude micro-trials.

    Low-frequency and gamma spatial projections are fitted separately. Within
    each frequency range, one regional projection is shared across conditions.
    """
    phase_output: dict[str, dict[str, np.ndarray]] = {name: {} for name in onset_samples}
    amplitude_output: dict[str, dict[str, np.ndarray]] = {name: {} for name in onset_samples}
    for region, conditions in region_signals.items():
        low_filtered = {name: bandpass(data, sfreq, phase_band) for name, data in conditions.items()}
        gamma_filtered = {name: bandpass(data, sfreq, gamma_band) for name, data in conditions.items()}
        low_components = shared_pc1(low_filtered)
        gamma_components = shared_pc1(gamma_filtered)
        for condition in conditions:
            low_phase = np.angle(hilbert(low_components[condition]))
            gamma_amplitude = np.abs(hilbert(gamma_components[condition]))
            phase_output[condition][region] = epoch_series(
                low_phase, onset_samples[condition], sfreq, tmin=tmin, tmax=tmax
            )
            amplitude_output[condition][region] = epoch_series(
                gamma_amplitude, onset_samples[condition], sfreq, tmin=tmin, tmax=tmax
            )
    return phase_output, amplitude_output


def pac_matrix(
    phase_epochs: Mapping[str, np.ndarray],
    amplitude_epochs: Mapping[str, np.ndarray],
    *,
    region_order: Sequence[str] | None = None,
    n_surrogates: int = 1000,
    n_bins: int = 18,
    random_state: int = 0,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Build ordered driver-phase by receiver-gamma PAC matrices."""
    regions = tuple(phase_epochs) if region_order is None else tuple(region_order)
    standardized = np.full((len(regions), len(regions)), np.nan, dtype=float)
    observed = np.full_like(standardized, np.nan)
    for row, driver in enumerate(regions):
        for column, receiver in enumerate(regions):
            if row == column:
                continue
            standardized[row, column], observed[row, column], _ = surrogate_standardized_mi(
                phase_epochs[driver],
                amplitude_epochs[receiver],
                n_surrogates=n_surrogates,
                n_bins=n_bins,
                random_state=random_state + row * len(regions) + column,
            )
    return regions, standardized, observed


def pac_matrices(
    region_signals: Mapping[str, Mapping[str, np.ndarray]],
    onset_samples: Mapping[str, np.ndarray],
    sfreq: float,
    *,
    frequency_hz: float,
    phase_band: tuple[float, float] | None = None,
    gamma_band: tuple[float, float] = (60.0, 100.0),
    region_order: Sequence[str] | None = None,
    n_surrogates: int = 1000,
    random_state: int = 0,
) -> tuple[tuple[str, ...], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute surrogate-standardized and observed PAC matrices per condition."""
    selected_band = phase_band or (
        (1.5, 2.5) if np.isclose(frequency_hz, 2.0) else (5.5, 6.5)
    )
    phase, amplitude = condition_pac_epochs(
        region_signals,
        onset_samples,
        sfreq,
        phase_band=selected_band,
        gamma_band=gamma_band,
    )
    standardized: dict[str, np.ndarray] = {}
    observed: dict[str, np.ndarray] = {}
    regions: tuple[str, ...] | None = None
    for condition in phase:
        regions, standardized[condition], observed[condition] = pac_matrix(
            phase[condition],
            amplitude[condition],
            region_order=region_order,
            n_surrogates=n_surrogates,
            random_state=random_state,
        )
    if regions is None:
        raise ValueError("No conditions were provided.")
    return regions, standardized, observed


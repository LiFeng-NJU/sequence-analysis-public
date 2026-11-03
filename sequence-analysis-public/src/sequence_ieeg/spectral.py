"""SNR, ITPC, responsive-channel and cumulative-ITPC feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .io import validate_epochs


@dataclass(frozen=True)
class SpectrumFeatures:
    frequencies: np.ndarray
    amplitude: np.ndarray
    snr: np.ndarray


def _rfft_frequencies(n_times: int, sfreq: float) -> np.ndarray:
    return np.fft.rfftfreq(n_times, d=1.0 / sfreq)


def amplitude_spectrum(epochs: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Return trial-averaged Fourier magnitude for every channel.

    Input shape is ``(trials, channels, time)``; output shape is
    ``(channels, frequencies)``.
    """
    values = validate_epochs(epochs)
    evoked = values.mean(axis=0)
    amplitude = np.abs(np.fft.rfft(evoked, axis=-1))
    return _rfft_frequencies(values.shape[-1], sfreq), amplitude


def snr_from_amplitude(
    amplitude: np.ndarray,
    *,
    neighbor_offsets: Sequence[int] = (-7, -6, -5, -4, -3, 3, 4, 5, 6, 7),
) -> np.ndarray:
    """Divide each frequency bin by its designated neighboring-bin mean.

    With 10-s epochs, the default offsets correspond to the bands 0.3--0.7 Hz
    below and above the target bin while excluding the two adjacent bins.
    Edge bins without a complete neighborhood are returned as NaN.
    """
    values = np.asarray(amplitude, dtype=float)
    if values.ndim != 2:
        raise ValueError("amplitude must have shape (channels, frequencies).")
    offsets = np.asarray(tuple(neighbor_offsets), dtype=int)
    if offsets.size == 0 or np.any(offsets == 0):
        raise ValueError("neighbor_offsets must be non-zero frequency-bin offsets.")
    result = np.full(values.shape, np.nan, dtype=float)
    left = int(max(0, -offsets.min()))
    right = int(max(0, offsets.max()))
    for index in range(left, values.shape[1] - right):
        noise = values[:, index + offsets].mean(axis=1)
        result[:, index] = np.divide(
            values[:, index],
            noise,
            out=np.full(values.shape[0], np.nan),
            where=noise != 0,
        )
    return result


def snr_spectrum(epochs: np.ndarray, sfreq: float) -> SpectrumFeatures:
    """Compute Fourier magnitude and the study's neighboring-bin SNR."""
    frequencies, amplitude = amplitude_spectrum(epochs, sfreq)
    return SpectrumFeatures(frequencies, amplitude, snr_from_amplitude(amplitude))


def itpc_spectrum(epochs: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute ITPC as ``abs(mean(exp(1j * phase), trials))``.

    Output has shape ``(channels, frequencies)``.
    """
    values = validate_epochs(epochs)
    phase = np.angle(np.fft.rfft(values, axis=-1))
    itpc = np.abs(np.mean(np.exp(1j * phase), axis=0))
    return _rfft_frequencies(values.shape[-1], sfreq), itpc


def frequency_index(frequencies: np.ndarray, target_hz: float) -> int:
    """Return the nearest Fourier-bin index after checking bin alignment."""
    values = np.asarray(frequencies, dtype=float)
    index = int(np.argmin(np.abs(values - target_hz)))
    if values.size > 1:
        tolerance = np.median(np.diff(values)) / 2.0 + 1e-12
        if abs(values[index] - target_hz) > tolerance:
            raise ValueError(f"No Fourier bin is aligned with {target_hz:g} Hz.")
    return index


def responsive_channel_mask(
    snr: np.ndarray,
    frequencies: np.ndarray,
    *,
    target_hz: float = 6.0,
    reference_band: tuple[float, float] = (1.0, 125.0),
    percentile: float = 95.0,
) -> np.ndarray:
    """Apply the channel-wise spectral response rule used by the study.

    A channel is retained when its target-bin SNR exceeds the requested
    percentile of that channel's SNR values in the reference band. This is a
    deterministic feature-selection rule and does not perform group inference.
    """
    values = np.asarray(snr, dtype=float)
    freqs = np.asarray(frequencies, dtype=float)
    if values.ndim != 2 or values.shape[1] != freqs.size:
        raise ValueError("snr must have shape (channels, len(frequencies)).")
    band = (freqs >= reference_band[0]) & (freqs < reference_band[1])
    target = frequency_index(freqs, target_hz)
    thresholds = np.nanpercentile(values[:, band], percentile, axis=1)
    return np.isfinite(values[:, target]) & (values[:, target] > thresholds)


def responsive_union(
    condition_epochs: Mapping[str, np.ndarray],
    sfreq: float,
    *,
    target_hz: float = 6.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the union of condition-specific responsive-channel masks."""
    masks: dict[str, np.ndarray] = {}
    for condition, epochs in condition_epochs.items():
        features = snr_spectrum(epochs, sfreq)
        masks[condition] = responsive_channel_mask(
            features.snr,
            features.frequencies,
            target_hz=target_hz,
        )
    channel_counts = {mask.size for mask in masks.values()}
    if len(channel_counts) != 1:
        raise ValueError("All conditions must use the same channel order.")
    union = np.logical_or.reduce(tuple(masks.values()))
    return union, masks


def cumulative_itpc(
    epochs: np.ndarray,
    sfreq: float,
    *,
    start_trials: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ITPC after cumulatively adding epochs in acquisition order.

    Returns
    -------
    trial_counts
        Included epoch counts, from ``start_trials`` to all epochs.
    frequencies
        Fourier frequency axis.
    values
        Array shaped ``(steps, channels, frequencies)``.
    """
    values = validate_epochs(epochs)
    if not 2 <= start_trials <= values.shape[0]:
        raise ValueError("start_trials must be between 2 and the epoch count.")
    phase_vectors = np.exp(1j * np.angle(np.fft.rfft(values, axis=-1)))
    cumulative = np.cumsum(phase_vectors, axis=0)
    counts = np.arange(start_trials, values.shape[0] + 1)
    curves = np.abs(cumulative[counts - 1] / counts[:, None, None])
    return counts, _rfft_frequencies(values.shape[-1], sfreq), curves


def growing_itpc(
    epochs: np.ndarray,
    sfreq: float,
    *,
    start_trials: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Named entry point for the study's Growing-ITPC calculation."""
    return cumulative_itpc(epochs, sfreq, start_trials=start_trials)


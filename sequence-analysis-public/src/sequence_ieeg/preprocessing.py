"""Convert raw EDF recordings into cleaned MNE objects and epoch arrays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PreprocessingConfig:
    """Signal-processing settings used by the study."""

    target_sfreq: float = 250.0
    l_freq: float = 0.1
    h_freq: float = 125.0
    line_freq: float = 50.0
    butterworth_order: int = 2
    reference_types: tuple[str, ...] = ("seeg", "ecog", "eeg")


def load_edf(path: str | Path, *, preload: bool = True):
    """Read one EDF file as ``mne.io.Raw``.

    The EDF file itself is never copied or included in exported feature bundles.
    """
    import mne

    source = Path(path)
    if source.suffix.lower() != ".edf":
        raise ValueError("Expected an EDF input file.")
    return mne.io.read_raw_edf(source, preload=preload, verbose="error")


def _data_channel_types(raw, allowed: Sequence[str]) -> list[str]:
    present = set(raw.get_channel_types(unique=True))
    return [channel_type for channel_type in allowed if channel_type in present]


def preprocess_raw(
    raw,
    *,
    bad_channels: Collection[str] = (),
    config: PreprocessingConfig = PreprocessingConfig(),
):
    """Apply channel exclusion, common-average reference, filtering and resampling.

    Parameters
    ----------
    raw
        Preloaded ``mne.io.Raw`` object.
    bad_channels
        Channel names identified during recording-quality review. Missing names
        are rejected to prevent silent metadata mismatches.
    config
        Processing settings.

    Returns
    -------
    mne.io.Raw
        A new preloaded Raw object sampled at ``config.target_sfreq``.
    """
    cleaned = raw.copy().load_data()
    unknown = sorted(set(bad_channels).difference(cleaned.ch_names))
    if unknown:
        raise ValueError(f"Unknown bad-channel names: {unknown}")
    cleaned.info["bads"] = sorted(set(cleaned.info["bads"]).union(bad_channels))
    cleaned.pick(picks="data", exclude="bads")

    reference_types = _data_channel_types(cleaned, config.reference_types)
    if not reference_types:
        raise ValueError("No EEG, ECoG or SEEG channels remain after exclusion.")
    cleaned.set_eeg_reference(
        ref_channels="average",
        projection=False,
        ch_type=reference_types,
        verbose="error",
    )

    raw_nyquist = cleaned.info["sfreq"] / 2.0
    notch_frequencies = np.arange(
        config.line_freq,
        min(config.h_freq, raw_nyquist),
        config.line_freq,
    )
    if notch_frequencies.size:
        cleaned.notch_filter(
            freqs=notch_frequencies,
            method="fir",
            phase="zero",
            fir_design="firwin",
            fir_window="hamming",
            filter_length="auto",
            verbose="error",
        )

    cleaned.resample(config.target_sfreq, npad="auto", verbose="error")
    target_nyquist = cleaned.info["sfreq"] / 2.0
    upper = min(config.h_freq, np.nextafter(target_nyquist, 0.0))
    cleaned.filter(
        l_freq=config.l_freq,
        h_freq=upper,
        method="iir",
        iir_params={"order": config.butterworth_order, "ftype": "butter"},
        phase="zero",
        verbose="error",
    )
    return cleaned


def preprocess_edf(
    edf_path: str | Path,
    output_fif: str | Path,
    *,
    bad_channels: Collection[str] = (),
    config: PreprocessingConfig = PreprocessingConfig(),
) -> Path:
    """Process one EDF file and save a reusable MNE FIF recording."""
    destination = Path(output_fif)
    if not destination.name.endswith("_raw.fif"):
        raise ValueError("MNE Raw output names must end with '_raw.fif'.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned = preprocess_raw(
        load_edf(edf_path, preload=True),
        bad_channels=bad_channels,
        config=config,
    )
    cleaned.save(destination, overwrite=True, verbose="error")
    return destination


def make_epochs(
    raw,
    events: np.ndarray,
    event_id: Mapping[str, int] | int,
    *,
    tmin: float = -1.0 / 6.0,
    tmax: float = 5.0 / 6.0,
    picks: str | Sequence[str] = "data",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Segment an MNE Raw object and return anonymous matrix data.

    Returns
    -------
    epochs
        ``float64`` array shaped ``(trials, channels, time)``.
    labels
        Integer event codes shaped ``(trials,)``.
    times
        Time axis in seconds shaped ``(time,)``.
    channel_names
        Channel order corresponding to the second array dimension.
    """
    import mne

    epoched = mne.Epochs(
        raw,
        np.asarray(events, dtype=int),
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        picks=picks,
        reject_by_annotation=True,
        verbose="error",
    )
    return (
        epoched.get_data(copy=True).astype(np.float64, copy=False),
        epoched.events[:, 2].astype(np.int64, copy=False),
        epoched.times.astype(np.float64, copy=False),
        tuple(epoched.ch_names),
    )

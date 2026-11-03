from __future__ import annotations

import numpy as np

from sequence_ieeg.geometry import mds_coordinates, split_half_rdm
from sequence_ieeg.mutual_information import contact_pair_mi
from sequence_ieeg.pac import tort_modulation_index
from sequence_ieeg.phase_lag import phase_lag_ms
from sequence_ieeg.spectral import cumulative_itpc, itpc_spectrum, snr_spectrum


def test_spectral_shapes() -> None:
    rng = np.random.default_rng(1)
    epochs = rng.normal(size=(8, 3, 500))
    snr = snr_spectrum(epochs, 250.0)
    frequencies, itpc = itpc_spectrum(epochs, 250.0)
    counts, growing_frequencies, growing = cumulative_itpc(epochs, 250.0)
    assert snr.snr.shape == (3, frequencies.size)
    assert itpc.shape == snr.snr.shape
    assert growing.shape == (counts.size, 3, growing_frequencies.size)


def test_geometry_shapes() -> None:
    rng = np.random.default_rng(2)
    patterns = rng.normal(size=(18, 5))
    labels = np.repeat([1, 2, 3], 6)
    rdm = split_half_rdm(patterns, labels, n_repeats=3)
    coordinates = mds_coordinates(rdm)
    assert rdm.shape == (6, 6)
    assert coordinates.shape == (6, 2)


def test_pair_features() -> None:
    time = np.linspace(0.0, 4.0 * np.pi, 200)
    first = np.stack([np.sin(time), np.cos(time)])
    second = np.stack([np.sin(time + 0.2), np.cos(time + 0.2)])
    matrix = contact_pair_mi(first, second)
    assert matrix.shape == (2, 2)

    phase_first = np.tile(time, (4, 1))
    phase_second = phase_first - 0.25
    lag = phase_lag_ms(phase_first, phase_second, 2.0)
    assert np.isfinite(lag)

    mi = tort_modulation_index(phase_first, np.abs(np.cos(phase_first)))
    assert np.isfinite(mi)


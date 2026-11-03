"""Run the public feature interfaces on synthetic matrices only."""

from __future__ import annotations

import numpy as np

from sequence_ieeg.decoding import decode_time_series
from sequence_ieeg.geometry import mds_coordinates, split_half_rdm, window_patterns
from sequence_ieeg.io import save_array_bundle
from sequence_ieeg.rsa import time_resolved_rsa
from sequence_ieeg.spectral import cumulative_itpc, itpc_spectrum, snr_spectrum


def main() -> None:
    rng = np.random.default_rng(7)
    sfreq = 250.0

    epochs_10s = rng.normal(size=(12, 6, 2500))
    snr = snr_spectrum(epochs_10s, sfreq)
    frequencies, itpc = itpc_spectrum(epochs_10s, sfreq)
    counts, _, growing = cumulative_itpc(epochs_10s, sfreq, start_trials=3)

    times = np.arange(-1.0 / 6.0, 5.0 / 6.0, 1.0 / sfreq)
    labels = np.repeat([1, 2, 3], 18)
    micro_epochs = rng.normal(size=(labels.size, 6, times.size))
    rsa = time_resolved_rsa(
        micro_epochs,
        labels,
        times,
        sfreq=sfreq,
        classes=[1, 2, 3],
        n_repeats=4,
    )
    decoding = decode_time_series(
        micro_epochs,
        labels,
        times,
        sfreq=sfreq,
        classes=[1, 2, 3],
        n_splits=3,
    )
    patterns = window_patterns(micro_epochs, times, (-1.0 / 6.0, 5.0 / 6.0))
    rdm = split_half_rdm(patterns, labels, classes=[1, 2, 3], n_repeats=4)
    coordinates = mds_coordinates(rdm)

    save_array_bundle(
        "outputs/synthetic_features.npz",
        frequencies=frequencies,
        snr=snr.snr,
        itpc=itpc,
        cumulative_counts=counts,
        cumulative_itpc=growing,
        rsa=rsa.mean,
        svm_accuracy=decoding.mean_accuracy,
        split_half_rdm=rdm,
        mds_coordinates=coordinates,
    )


if __name__ == "__main__":
    main()


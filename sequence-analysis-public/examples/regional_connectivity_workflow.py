"""Synthetic regional inputs for MI, time-lag and PAC feature extraction."""

from __future__ import annotations

import numpy as np

from sequence_ieeg.mutual_information import contact_pair_mi
from sequence_ieeg.pac import pac_matrices
from sequence_ieeg.phase_lag import phase_lag_matrices


def synthetic_regional_signals(
    sfreq: float = 250.0,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(11)
    duration = 20.0
    time = np.arange(int(duration * sfreq)) / sfreq
    onsets = (np.arange(1.0, 19.0, 1.0) * sfreq).astype(int)
    region_signals: dict[str, dict[str, np.ndarray]] = {}
    for region_index, region in enumerate(("region_A", "region_B", "region_C")):
        region_signals[region] = {}
        for condition_index, condition in enumerate(("randomized", "sequential")):
            phase = region_index * 0.15 + condition_index * 0.05
            base = np.sin(2.0 * np.pi * 2.0 * time + phase)
            gamma = (1.0 + 0.2 * base) * np.sin(2.0 * np.pi * 80.0 * time)
            noise = rng.normal(scale=0.4, size=(2, time.size))
            region_signals[region][condition] = base + gamma + noise
    onset_samples = {"randomized": onsets, "sequential": onsets}
    return region_signals, onset_samples


def main() -> None:
    sfreq = 250.0
    signals, onsets = synthetic_regional_signals(sfreq)
    regions, lag = phase_lag_matrices(
        signals,
        onsets,
        sfreq,
        frequency_hz=2.0,
    )
    _, standardized_pac, observed_pac = pac_matrices(
        signals,
        onsets,
        sfreq,
        frequency_hz=2.0,
        region_order=regions,
        n_surrogates=10,  # use 1,000 for the method-aligned run
    )

    first = signals["region_A"]["sequential"][:, :42]
    second = signals["region_B"]["sequential"][:, :42]
    mi = contact_pair_mi(first, second)
    print(regions)
    print({name: matrix.shape for name, matrix in lag.items()})
    print({name: matrix.shape for name, matrix in standardized_pac.items()})
    print({name: matrix.shape for name, matrix in observed_pac.items()})
    print(mi.shape)


if __name__ == "__main__":
    main()


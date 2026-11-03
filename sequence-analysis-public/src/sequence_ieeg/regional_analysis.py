"""Convenience wrappers for region-specific RSA, MDS and distances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .geometry import (
    DistanceResult,
    mds_coordinates,
    position_distances,
    spatial_whitener,
    split_half_rdm,
    window_patterns,
)
from .regions import select_region_epochs
from .rsa import RSAResult, time_resolved_rsa


@dataclass(frozen=True)
class RegionalGeometry:
    rdm: np.ndarray
    coordinates: np.ndarray


def regional_rsa(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    region_indices: Mapping[str, Sequence[int]],
    *,
    sfreq: float,
    classes: Sequence[int],
    n_repeats: int = 100,
    random_state: int = 0,
) -> dict[str, RSAResult]:
    """Run the identical RSA calculation independently in each region."""
    return {
        region: time_resolved_rsa(
            select_region_epochs(epochs, indices),
            labels,
            times,
            sfreq=sfreq,
            classes=classes,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        for region, indices in region_indices.items()
    }


def regional_mds(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    resting_signals: np.ndarray,
    region_indices: Mapping[str, Sequence[int]],
    *,
    window: tuple[float, float],
    classes: Sequence[int],
    n_repeats: int = 100,
    random_state: int = 0,
) -> dict[str, RegionalGeometry]:
    """Compute a split-half RDM and MDS coordinates for every region."""
    output: dict[str, RegionalGeometry] = {}
    for region, indices in region_indices.items():
        indices = np.asarray(indices, dtype=int)
        region_epochs = select_region_epochs(epochs, indices)
        region_rest = np.asarray(resting_signals)[indices]
        patterns = window_patterns(
            region_epochs,
            times,
            window,
            whitener=spatial_whitener(region_rest),
        )
        rdm = split_half_rdm(
            patterns,
            labels,
            classes=classes,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        output[region] = RegionalGeometry(rdm=rdm, coordinates=mds_coordinates(rdm))
    return output


def regional_distances(
    epochs: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    resting_signals: np.ndarray,
    region_indices: Mapping[str, Sequence[int]],
    *,
    window: tuple[float, float],
    classes: Sequence[int],
    n_folds: int = 6,
    n_repeats: int = 100,
    random_state: int = 0,
) -> dict[str, DistanceResult]:
    """Compute within- and between-position distances for every region."""
    output: dict[str, DistanceResult] = {}
    for region, indices in region_indices.items():
        indices = np.asarray(indices, dtype=int)
        output[region] = position_distances(
            select_region_epochs(epochs, indices),
            labels,
            times,
            window=window,
            resting_signals=np.asarray(resting_signals)[indices],
            classes=classes,
            n_folds=n_folds,
            n_repeats=n_repeats,
            random_state=random_state,
        )
    return output


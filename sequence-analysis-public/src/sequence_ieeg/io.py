"""Small, participant-agnostic array I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_array_bundle(path: str | Path, **arrays: Any) -> Path:
    """Save named NumPy arrays in one compressed NPZ file.

    Parameters
    ----------
    path
        Destination ending in ``.npz``.
    **arrays
        Named arrays. Do not include participant identifiers or clinical fields.
    """
    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise ValueError("Array bundles must use the .npz extension.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def load_array_bundle(path: str | Path) -> dict[str, np.ndarray]:
    """Load a compressed NPZ bundle without allowing pickled objects."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def validate_epochs(epochs: np.ndarray) -> np.ndarray:
    """Validate an array with shape ``(trials, channels, time)``."""
    values = np.asarray(epochs, dtype=float)
    if values.ndim != 3:
        raise ValueError("epochs must have shape (trials, channels, time).")
    if min(values.shape) < 1:
        raise ValueError("epochs cannot contain an empty dimension.")
    if not np.isfinite(values).all():
        raise ValueError("epochs must contain finite values only.")
    return values


def validate_continuous(signals: np.ndarray) -> np.ndarray:
    """Validate an array with shape ``(channels, time)``."""
    values = np.asarray(signals, dtype=float)
    if values.ndim != 2:
        raise ValueError("signals must have shape (channels, time).")
    if min(values.shape) < 1:
        raise ValueError("signals cannot contain an empty dimension.")
    if not np.isfinite(values).all():
        raise ValueError("signals must contain finite values only.")
    return values


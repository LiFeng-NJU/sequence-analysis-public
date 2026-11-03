# Sequence-learning iEEG feature extraction

This repository contains participant-level signal-processing and feature-
extraction code for an intracranial sequence-learning study. It is designed for
methodological transparency without distributing clinical recordings or
participant metadata.

The package deliberately stops at derived feature matrices. It contains no
figure generation, group-level inference, multiple-comparison procedures, or
publication-specific result tables.

## Included modules

| Module | Input | Output |
|---|---|---|
| `preprocessing.py` | Raw EDF + externally supplied bad-channel names | Cleaned MNE FIF; trial × channel × time epochs |
| `spectral.py` | Trial × channel × time | SNR, ITPC, responsive-channel masks, cumulative ITPC |
| `rsa.py` | Trial × channel × time + position labels | RSA value × time |
| `decoding.py` | Trial × channel × time + position labels | SVM accuracy × time |
| `geometry.py` | Trial patterns + resting-state channel × time | split-half RDM, MDS coordinates, within/between distances |
| `regions.py` | Anonymous channel names + region labels | Region-specific channel indices and matrices |
| `mutual_information.py` | Contact × time matrices | Contact-pair MI matrix |
| `phase_lag.py` | Regional continuous matrices + stimulus onsets | Condition-specific regional time-lag matrices |
| `pac.py` | Regional continuous matrices + stimulus onsets | Condition-specific observed and surrogate-standardized PAC matrices |

Detailed array contracts are in [docs/data_formats.md](docs/data_formats.md).
Regional use is described in [docs/region_analysis.md](docs/region_analysis.md).

## Installation

```bash
python -m venv .venv
python -m pip install -e .
```

For local interface checks:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Minimal workflow

### 1. Preprocess one EDF recording

```python
from sequence_ieeg.preprocessing import preprocess_edf

preprocess_edf(
    "input_recording.edf",
    "outputs/recording_raw.fif",
    bad_channels=["channel_name_from_quality_review"],
)
```

Bad-channel names are supplied outside the repository. The preprocessing code
applies common-average reference, 50-Hz line-noise removal and harmonics,
0.1--125-Hz zero-phase filtering, and resampling to 250 Hz.

### 2. Segment anonymous matrices

```python
from sequence_ieeg.preprocessing import make_epochs

epochs, labels, times, channel_names = make_epochs(
    raw,
    events,
    event_id={"position_1": 1, "position_2": 2, "position_3": 3},
)
```

The resulting `epochs` matrix is the public interface used by the remaining
modules. Neither EDF recordings nor these participant-level matrices are part
of this repository.

### 3. Compute features

```python
from sequence_ieeg.spectral import itpc_spectrum, snr_spectrum
from sequence_ieeg.rsa import time_resolved_rsa

snr = snr_spectrum(epochs_10s, sfreq=250.0)
frequencies, itpc = itpc_spectrum(epochs_10s, sfreq=250.0)
rsa = time_resolved_rsa(
    micro_epochs,
    position_labels,
    micro_times,
    sfreq=250.0,
    classes=[1, 2, 3],
)
```

See [examples/array_feature_workflow.py](examples/array_feature_workflow.py) for
a complete synthetic-data example.

## Method-aligned defaults

- Sampling frequency after preprocessing: 250 Hz.
- SNR epoch length: 10 s, giving 0.1-Hz Fourier resolution.
- SNR noise bins: offsets −7 to −3 and +3 to +7 Fourier bins.
- Responsive-channel rule: target 6-Hz SNR exceeds the channel's 95th
  percentile across 1--125 Hz in either condition.
- ITPC: magnitude of the trial-averaged unit phase vector.
- Cumulative ITPC: epochs are added in acquisition order, beginning at three.
- RSA: Euclidean trial RDM and Spearman fit to a binary position RDM after a
  20-ms moving average and position-count matching.
- SVM: groups of three non-overlapping raw trials form one pseudotrial; linear
  SVM (`C=1`) with six stratified folds and fold-local feature scaling.
- Geometry windows: full micro-trial (−167 to 833 ms) or pre-stimulus
  (−66.67 to 0 ms), with one resting-state whitening transform shared across
  conditions.
- Time lag: 1.5--2.5-Hz or 5.5--6.5-Hz phase; positive matrix values mean that
  the row region leads the column region.
- PAC: row-region low-frequency phase to column-region 60--100-Hz amplitude;
  18 phase bins and 1,000 within-micro-trial circular-shift surrogates.

## Data protection

The `.gitignore` excludes common electrophysiology, array, spreadsheet and
cache formats. Before publishing, run the repository audit described in
[docs/privacy_and_scope.md](docs/privacy_and_scope.md). No patient identifiers,
clinical fields, electrode coordinate tables, or recordings should be committed.


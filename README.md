# Sequence-learning iEEG feature extraction

Participant-level signal processing and feature extraction for an intracranial
EEG study of temporal sequence learning.

This repository provides the computational steps that transform raw or
anonymized electrophysiological matrices into participant-level derived
features. It does not contain patient recordings, electrode coordinates,
participant identifiers, figure-generation code, group-level inference, or
publication-specific result tables.

## Scope

The package includes:

- EDF preprocessing and export to MNE FIF;
- signal-to-noise ratio (SNR), inter-trial phase coherence (ITPC), responsive-
  channel selection and Growing-ITPC;
- time-resolved representational similarity analysis (RSA);
- time-resolved linear-SVM position decoding;
- split-half representational dissimilarity matrices (RDMs), multidimensional
  scaling (MDS), and within- versus between-position distances;
- region-specific RSA, MDS and distance calculations;
- contact-pair mutual information (MI);
- inter-regional phase-derived time-lag matrices; and
- inter-regional phase-amplitude coupling (PAC) matrices.

The repository ends at participant-level feature matrices. Downstream plotting
and group-level analyses are intentionally outside its scope.

## Repository structure

```text
sequence-analysis-public/
|-- src/sequence_ieeg/
|   |-- preprocessing.py       # EDF to MNE FIF and anonymous epochs
|   |-- spectral.py            # SNR, ITPC and Growing-ITPC
|   |-- rsa.py                 # time-resolved RSA
|   |-- decoding.py            # linear-SVM decoding
|   |-- geometry.py            # RDM, MDS and distances
|   |-- regions.py             # anonymous channel-to-region selection
|   |-- regional_analysis.py   # regional RSA, MDS and distances
|   |-- mutual_information.py  # contact-pair MI
|   |-- phase_lag.py           # regional time-lag matrices
|   |-- pac.py                 # Tort MI and PAC matrices
|   |-- signals.py             # shared filtering and PCA helpers
|   `-- io.py                  # safe numerical array I/O
|-- examples/
|   |-- array_feature_workflow.py
|   `-- regional_connectivity_workflow.py
|-- docs/
|   |-- data_formats.md
|   |-- region_analysis.md
|   `-- privacy_and_scope.md
|-- tests/
`-- pyproject.toml
```

## System requirements

### Software

- Python 3.10 or later
- MNE-Python 1.6 or later
- NumPy 1.24 or later
- SciPy 1.10 or later
- scikit-learn 1.3 or later

Dependencies are declared in `pyproject.toml` and are installed automatically.

### Tested environment

The array-level feature interfaces and synthetic examples were tested with:

- Windows 11;
- Python 3.13.9;
- NumPy 2.3.5;
- SciPy 1.16.3;
- scikit-learn 1.7.2; and
- MNE-Python 1.11.0.

### Hardware

No non-standard hardware or GPU is required. Runtime and memory use depend on
the number of trials, channels, time samples and PAC surrogates. A multi-core
desktop with at least 8 GB RAM is suitable for the included synthetic demos.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone <REPOSITORY_URL>
cd sequence-analysis-public
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Installing the package and its dependencies typically takes approximately
5-15 minutes on a standard desktop with an active network connection. The exact
time depends on the local Python environment and package cache.

For the interface checks:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Expected test output:

```text
3 passed
```

## Demo

The demos generate all input arrays in memory. They do not use or download
participant data.

### Array-based feature demo

```bash
python examples/array_feature_workflow.py
```

Expected output:

```text
outputs/synthetic_features.npz
```

The NPZ bundle contains synthetic frequency axes, SNR, ITPC, Growing-ITPC,
RSA, SVM accuracy, a split-half RDM and MDS coordinates.

### Regional time-lag, PAC and MI demo

```bash
python examples/regional_connectivity_workflow.py
```

Expected terminal output lists the region order and the shapes of the derived
time-lag, PAC and MI matrices.

On the tested desktop, the array demo took approximately 3.2 s and the regional
connectivity demo approximately 1.2 s. These timings use small synthetic arrays
and ten PAC surrogates; method-aligned PAC analyses use 1,000 surrogates and
therefore take longer.

## Input data contracts

The package accepts numerical arrays with explicit axis order.

### Stimulus-locked epochs

```text
epochs: float[trials, channels, time]
labels: int[trials]
times:  float[time], seconds relative to stimulus onset
```

### Continuous regional signals

```text
signals:       float[channels, continuous_time]
onset_samples: int[trials]
sfreq:         float, samples per second
```

### Resting-state signals

```text
resting_signals: float[channels, resting_time]
```

The resting-state channel order must match the task-epoch channel order.

### Anonymous regional metadata

```python
channel_names = ["contact_001", "contact_002", "contact_003"]
channel_regions = {
    "contact_001": "region_A",
    "contact_002": "region_A",
    "contact_003": "region_B",
}
```

Detailed contracts and output shapes are documented in
[docs/data_formats.md](docs/data_formats.md). Regional workflows are documented
in [docs/region_analysis.md](docs/region_analysis.md).

## Minimal examples

### EDF preprocessing

```python
from sequence_ieeg.preprocessing import preprocess_edf

preprocess_edf(
    "input_recording.edf",
    "outputs/recording_raw.fif",
    bad_channels=["channel_name_from_quality_review"],
)
```

Bad-channel names are supplied outside the repository. The function applies
common-average referencing, zero-phase Hamming-windowed FIR removal of 50-Hz
line noise and harmonics, resampling to 250 Hz, and zero-phase second-order
Butterworth filtering from 0.1 Hz to the Nyquist limit.

### SNR and ITPC

```python
from sequence_ieeg.spectral import growing_itpc, itpc_spectrum, snr_spectrum

snr = snr_spectrum(epochs_10s, sfreq=250.0)
frequencies, itpc = itpc_spectrum(epochs_10s, sfreq=250.0)
trial_counts, frequencies, itpc_growth = growing_itpc(
    epochs_5s,
    sfreq=250.0,
    start_trials=3,
)
```

### RSA

```python
from sequence_ieeg.rsa import time_resolved_rsa

rsa = time_resolved_rsa(
    micro_epochs,
    position_labels,
    micro_times,
    sfreq=250.0,
    classes=[1, 2, 3],
)
```

### Linear-SVM decoding

```python
from sequence_ieeg.decoding import decode_time_series

decoding = decode_time_series(
    micro_epochs,
    position_labels,
    micro_times,
    sfreq=250.0,
    classes=[1, 2, 3],
)
```

### Regional time-lag and PAC matrices

```python
from sequence_ieeg.pac import pac_matrices
from sequence_ieeg.phase_lag import phase_lag_matrices

regions, lag_matrices = phase_lag_matrices(
    region_signals,
    onset_samples,
    sfreq=250.0,
    frequency_hz=2.0,
)

regions, standardized_pac, observed_pac = pac_matrices(
    region_signals,
    onset_samples,
    sfreq=250.0,
    frequency_hz=2.0,
    n_surrogates=1000,
)
```

## Method-aligned defaults

- Resampled frequency: 250 Hz.
- SNR epoch length: 10 s, yielding 0.1-Hz Fourier resolution.
- SNR noise-bin offsets: -7 to -3 and +3 to +7 Fourier bins.
- Responsive-channel rule: 6-Hz SNR above the channel-specific 95th percentile
  of the 1-125-Hz spectrum in either condition.
- ITPC: magnitude of the trial-averaged unit phase vector.
- Growing-ITPC: cumulative epochs in acquisition order, starting at three.
- RSA: 20-ms moving average, Euclidean neural RDM, binary position model RDM,
  Spearman fit and position-count matching.
- SVM: three non-overlapping raw trials per pseudotrial, linear kernel, `C=1`,
  six stratified folds and fold-local scaling.
- Geometry windows: -167 to 833 ms for the full micro-trial and -66.67 to 0 ms
  for the pre-stimulus interval.
- Geometry whitening: one resting-state channel covariance shared across task
  conditions.
- Time lag: 1.5-2.5-Hz or 5.5-6.5-Hz phase; positive cells indicate that the
  row region leads the column region.
- PAC: row-region low-frequency phase to column-region 60-100-Hz amplitude,
  18 phase bins and 1,000 independent within-micro-trial circular shifts.

## Data availability and privacy

Raw iEEG, MRI and electrode-localization data are not distributed with this
repository because they contain sensitive clinical information and may be
subject to participant-consent and institutional restrictions.

The `.gitignore` excludes common electrophysiology, imaging-derived array,
spreadsheet and cache formats. Before public release, follow the checklist in
[docs/privacy_and_scope.md](docs/privacy_and_scope.md) and inspect the complete
Git history.

The repository should contain only source code, documentation, tests and fully
synthetic examples. Do not commit participant identifiers, clinical variables,
electrode coordinates, local file paths or network paths.

## Reproducibility boundary

The repository demonstrates how participant-level features are computed from
documented matrix inputs. Because the underlying clinical recordings cannot be
distributed publicly, the synthetic demos validate software interfaces and
expected output structure rather than reproducing the manuscript's numerical
results.

## License

No license has yet been assigned. Before making the repository public, add an
OSI-approved license such as MIT or BSD-3-Clause and replace this section with
the selected license name and a link to the `LICENSE` file.

## Citation

If you use this code, please cite the associated article. Complete the citation
below after the article receives its final bibliographic information:

```text
<AUTHORS>. <ARTICLE TITLE>. <JOURNAL> (<YEAR>).
```

## Contact

For questions about the analysis code, open an issue in the GitHub repository.


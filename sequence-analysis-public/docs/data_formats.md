# Array input and output formats

All arrays use NumPy axis order. Units must be documented by the caller.

## Core inputs

### Continuous recording

```text
signals: float[channels, continuous_time]
sfreq: float, samples/second
```

For regional phase-lag and PAC calculations:

```python
region_signals = {
    "region_A": {
        "randomized": float[n_channels_A, n_time_randomized],
        "sequential": float[n_channels_A, n_time_sequential],
    },
    "region_B": {
        "randomized": float[n_channels_B, n_time_randomized],
        "sequential": float[n_channels_B, n_time_sequential],
    },
}

onset_samples = {
    "randomized": int[n_trials_randomized],
    "sequential": int[n_trials_sequential],
}
```

Channel order must remain fixed within one region across conditions. Regional
recordings within one condition must share the same continuous time base and
stimulus onsets.

### Stimulus-locked epochs

```text
epochs: float[trials, channels, time]
labels: int[trials]
times: float[time], seconds relative to stimulus onset
```

RSA and SVM use three position labels. Randomized trials should be restricted
to the same three physical positions used in the sequential condition before
calling the functions.

### Resting-state signals

```text
resting_signals: float[channels, resting_time]
```

The channel order must match the task epochs. Estimate one whitening transform
and reuse it across task conditions.

### Anonymous channel-to-region map

```python
channel_names = ["contact_001", "contact_002", "contact_003"]
channel_regions = {
    "contact_001": "region_A",
    "contact_002": "region_A",
    "contact_003": "region_B",
}
```

No participant name, medical-record number, acquisition date or clinical field
is needed.

## Derived outputs

### SNR and ITPC

```text
frequencies: float[frequency]
amplitude: float[channel, frequency]
snr: float[channel, frequency]
itpc: float[channel, frequency]
responsive_mask: bool[channel]
cumulative_itpc: float[included_epoch_count, channel, frequency]
```

### RSA and SVM

```text
rsa_repetitions: float[resampling_repeat, time]
rsa_mean: float[time]
svm_fold_accuracy: float[fold, time]
svm_mean_accuracy: float[time]
```

### Geometry and distance

```text
split_half_rdm: float[2 * positions, 2 * positions]
mds_coordinates: float[2 * positions, dimensions]
within_distance: float[repeat, position]
between_distance: float[repeat, position_pair]
```

The split-half RDM order is all positions from the first half followed by all
positions from the second half.

### Mutual information

```text
mi_matrix: float[contacts_in_first_region, contacts_in_second_region]
```

### Time lag

```text
lag_matrix: float[region, region], milliseconds
```

The matrix is antisymmetric. A positive cell `(i, j)` means region `i` leads
region `j`. At 2 Hz values are wrapped to [−250, 250) ms; at 6 Hz they are
wrapped to approximately [−83.33, 83.33) ms.

### PAC

```text
observed_mi: float[driver_region, receiver_region]
surrogate_standardized_mi: float[driver_region, receiver_region]
```

The diagonal is NaN because only inter-regional ordered pairs are returned.

## Saving outputs

Use `sequence_ieeg.io.save_array_bundle` to save numeric arrays in compressed
NPZ form. Do not serialize MNE objects, tables containing identifiers, or Python
objects into public feature bundles.


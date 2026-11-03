# Region-specific analyses

The computational algorithms are identical for all-channel and regional
analyses. Regional analysis changes only the channel subset supplied to the
feature function.

## 1. Prepare an anonymous region map

Create a mapping from anonymous contact labels to anatomical region labels:

```python
channel_regions = {
    "contact_001": "visual_region_1",
    "contact_002": "visual_region_1",
    "contact_003": "parietal_region_1",
}
```

The mapping should be produced locally from the electrode-localization record.
Only the anonymous mapping is passed to the analysis code.

## 2. Convert labels to matrix indices

```python
from sequence_ieeg.regions import region_channel_indices

indices = region_channel_indices(channel_names, channel_regions)
```

## 3. Run the same algorithm on each subset

### Regional RSA

```python
from sequence_ieeg.regions import select_region_epochs
from sequence_ieeg.rsa import time_resolved_rsa

regional_rsa = {}
for region, channel_index in indices.items():
    regional_rsa[region] = time_resolved_rsa(
        select_region_epochs(epochs, channel_index),
        labels,
        times,
        sfreq=250.0,
        classes=[1, 2, 3],
    )
```

### Regional MDS and distances

Use the selected epoch channels and the matching selected resting-state
channels. `window_patterns`, `split_half_rdm`, `mds_coordinates` and
`position_distances` then operate without anatomical special cases.

```python
from sequence_ieeg.geometry import (
    mds_coordinates,
    position_distances,
    split_half_rdm,
    spatial_whitener,
    window_patterns,
)

region_epochs = epochs[:, channel_index, :]
region_rest = resting_signals[channel_index, :]
whitener = spatial_whitener(region_rest)
patterns = window_patterns(
    region_epochs,
    times,
    window=(-1.0 / 6.0, 5.0 / 6.0),
    whitener=whitener,
)
rdm = split_half_rdm(patterns, labels, classes=[1, 2, 3])
coordinates = mds_coordinates(rdm)
distances = position_distances(
    region_epochs,
    labels,
    times,
    window=(-1.0 / 15.0, 0.0),
    resting_signals=region_rest,
    classes=[1, 2, 3],
)
```

For pooled anatomical groups, concatenate the channel indices of their
constituent regions before selection. Preserve a fixed channel order within a
participant and use the identical pooled definition across conditions.


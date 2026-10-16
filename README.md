# Shapelet Space Representation Library
`shapelet_space` is a Python library for time series analysis using shapelets. This package provides tools for shapelet discovery, dynamic time warping, and transforming time series data into shapelet space.
Basic functions for transforming short time-series into a user-defined shapelet-space representation vector. 
Longer time-series can be transformed into a matrix where each column represents the shapelet-space vector for a rolling window of subsequences.

## Paper Link
https://arxiv.org/pdf/2209.04035.pdf (accepted at IEEE BigData 2022)

## Installation
```bash
pip install shapelet-space
```

## Function Descriptions

### find_shapelet_space_ts

- **Description**: This function finds shapelets in your time series data.
- **Usage**:
    ```python
    find_shapelet_space_ts(time_series, flatness_param)
    ```
- **Parameters**:
    - `time_series`: Your input data.
    - `flatness_param`: A threshold used during shapelet discovery.
- **Example**:
```python
from shapelet_space import shapelet

# Create an instance of the ShapeletSpace class
shapelet_transformer = shapelet.ShapeletSpace()
# # To initialize shaplet object with custom params
# shapelet_transformer = shapelet.ShapeletSpace(Number_of_shapelets, Shapelet_array_length, Shapelet_array)

# Define your time series data
time_series = [/* your time series data here */]

# Set a flatness parameter
flatness_param = 100000  # for example

# Generate the shapelet space representation of the time series
reps = shapelet_transformer.find_shapelet_space_ts(time_series, flatness_param)
```
### dtw_cons_md

- **Description**: This function computes a similarity matrix between time series sequences using the DTW method.
- **Usage**:
    ```python
    dtw_cons_md(sequence_1, sequence_2, window_size, metric)
    ```
- **Parameters**:
    - `sequence_1` and `sequence_2`: The sequences you wish to compare.
    - `window_size`: Determines the constraint on how much the sequences can be stretched.
    - `metric`: The distance metric ('euclidean', 'manhattan', etc.).
- **Example**:
```python
from shapelet_space import dtw

# Prepare your data
sequence_1 = [/* your first sequence here */]
sequence_2 = [/* your second sequence here */]
window_size = 5  # for example

# Calculate the DTW distance
distance = dtw.dtw_cons_md(sequence_1, sequence_2, window_size, 'euclidean')
```
### dtw_cons_md_path

- **Description**: Same as `dtw_cons_md`, but also returns the optimal warping path.
- **Usage**:
    ```python
    distance, path_1, path_2 = dtw.dtw_cons_md_path(sequence_1, sequence_2, window_size, metric)
    ```
- **Returns**: the DTW distance and two integer arrays. `path_1[k]` and `path_2[k]` are the (0-based) indices of `sequence_1` and `sequence_2` that are matched at step `k` of the path.

## Ensemble (consensus of several time series)

The `ensemble` module finds the consensus of a set of time series with DTW+S. It is a Python port of the MATLAB code in [DTW_S_apps/ensemble](https://github.com/scc-usc/DTW_S_apps/tree/main/ensemble).
A point-wise mean of unaligned series flattens and widens the peaks. The DTW+S consensus first aligns the series in shapelet space and then averages the matched points, so the shape of the individual series is kept.

### dtw_s_ensemble

- **Description**: End-to-end consensus of a set of time series. The steps are:
    1. estimate the slope threshold of the shapelet transform (`estimate_slope_threshold`),
    2. transform every series to shapelet space at every time point (`shape_ts_transform`),
    3. align all series to a reference with DTW+S (`find_dtw_matchings`). By default the reference starts as the medoid and is refined with DTW Barycenter Averaging (`dba_s_alignments`),
    4. average the matched points (`dtw_mean_ensemble`).
- **Usage**:
    ```python
    dtw_s_ensemble(dd, shapelet_array=DEFAULT_SHAPELETS, slope_thres=None, win=np.inf, dist_metric='euc', num_iter=10, base=None)
    ```
- **Parameters**:
    - `dd`: array of shape `(n_series, T)`, one time series per row.
    - `shapelet_array`: one shapelet per row. The default is the increase, peak, surge, and flat shapelets. Pass `None` to align the raw values with plain DTW.
    - `slope_thres`: slope threshold of the shapelet transform. It is estimated from the data when omitted.
    - `win`: DTW window. Default: no constraint.
    - `dist_metric`: `'euc'` or `'cos'`.
    - `num_iter`: number of DBA iterations. `0` aligns the series to the medoid only.
    - `base`: index of a series to use as a fixed reference instead of DBA.
- **Returns**: an `EnsembleResult` with these fields:
    - `mean_time`, `mean_curve`: the consensus as a (time, value) curve, one point per time step of the reference. Times are 0-based and can be fractional.
    - `consensus`: the consensus resampled on the regular time grid `0 .. T-1`.
    - `matches`: array of shape `(n_series, T)`. `matches[j, t]` is the time index of series `j` that is matched to time `t` of the reference.
    - `all_values`, `all_times`: value and time of every matched point, shape `(n_series, T)`.
    - `reference`: the reference series. `all_refs` holds the reference of every DBA iteration.
    - `reps`: the shapelet-space representations, shape `(n_series, T, d)`.
- **Example**:
```python
import numpy as np
from shapelet_space import ensemble

dd = np.loadtxt('data/flu_sims.csv', delimiter=',')   # shape (75, 26)
result = ensemble.dtw_s_ensemble(dd, win=30, num_iter=20)

print(dd.mean(axis=0).max())          # peak of the point-wise mean (flattened)
print(result.consensus.max())         # peak of the DTW+S consensus

import matplotlib.pyplot as plt
plt.plot(dd.T, color='b', alpha=0.1)
plt.plot(dd.mean(axis=0), 'r', label='Mean')
plt.plot(result.mean_time, result.mean_curve, 'k', label='DTW+S DBA')
plt.legend()
```
See `examples/ensemble_example.py` for a complete script.

### Lower-level functions

- `shape_ts_transform(dd, shapelet_array, slope_thres)`: shapelet-space representation of every time point of every series, shape `(n_series, T, d)`. The window is centred on the time point and the ends are padded.
- `estimate_slope_threshold(dd, window)`: slope threshold used in the MATLAB scripts (median over series of the maximum moving-mean absolute slope).
- `find_dtw_matchings(all_reps, dtw_win, dtw_metric, base_m)`: align every series to series `base_m`. Returns the integer matches of shape `(n_series, T)`.
- `dtw_mean_ensemble(dd, dtw_matches)`: average the matched points. Returns `mean_curve`, `mean_time`, `all_values`, `all_times`.
- `dba_s_alignments(dd, all_reps, shapelet_array, slope_thres, win, dist_metric, num_iter)`: iterative refinement of the reference (DTW Barycenter Averaging in shapelet space). Returns the matches, the final reference, and the reference of every iteration.
- `pairwise_dtw_distances(all_reps, win, dist_metric)` and `find_medoid(...)`: DTW+S distance matrix between all pairs of series and the medoid. The distance matrix can be used to cluster the series before building the consensus of one cluster.

## Tests
```bash
pip install numpy scipy pandas pytest
python -m pytest tests
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


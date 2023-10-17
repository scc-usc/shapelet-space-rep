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
## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


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
Example
```python
from shapelet_space import ShapeletSpace

# Create an instance of the ShapeletSpace class
shapelet_transformer = ShapeletSpace()

# Define your time series data
time_series = [/* your time series data here */]

# Set a flatness parameter
flatness_param = 100000  # for example

# Generate the shapelet space representation of the time series
reps = shapelet.find_shapelet_space_ts(time_series, flatness_param)
```

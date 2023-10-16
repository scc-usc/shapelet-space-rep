import numpy as np
import pandas as pd
from numpy import diff
from numpy import corrcoef as pcor
from numpy import exp
from numpy import mean
from numpy import log
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine

class ShapeletSpace:
    def __init__(self, Shapelet_length=4, Number_of_shapelets=4, shapelet_array = [[1, 2, 3, 4], [1, 2, 2, 1], [1, 2, 4, 8], [0, 0, 0, 0]]):
        self.Shapelet_length = Shapelet_length
        self.Number_of_shapelets = Number_of_shapelets
        self.shapelet_array = shapelet_array
        self.shapelet_names = ["Inc", "Peak", "Flat", "Dec"]
        
    def similarity_non_flat(self, vector1, vector2):
        if np.std(vector1) < 1e-100 or np.std(vector2) < 1e-100:
            similarity_value = 0
        else:
            similarity_value = pcor(vector1, vector2)[0][1]
        return similarity_value

    def return_best_shapelet(self, vector, slope_thres=0.0005):
        corrs = self.shapelet_space_representation(vector, slope_thres)
        scenario = corrs.index(max(corrs))
        return self.shapelet_names[scenario]

    def _shapelet_space_representation_gen(self, vector, slope_thres=0.0005):
        coords = []
        beta = -log(0.1) / slope_thres
        m0 = 0
        slope = mean(abs(diff(vector)))
        if slope < m0:
            flatness = 1
        else:
            flatness = exp(-beta * (slope - m0))
        for i in range(len(self.shapelet_array)):
            if not any(self.shapelet_array[i]):
                score = 2 * flatness - 1
            else:
                score = (1 - flatness) * self.similarity_non_flat(self.shapelet_array[i], vector)
            coords.append(score)
        return coords

    def _shapelet_space_representation_slope_based(self, vector, slope_thres=0.0005):
        coords = []
        flatness = 0
        if abs(vector[-1] - vector[0]) < slope_thres:
            flatness = 1
        for i in range(len(self.shapelet_array)):
            score = 0
            if not any(self.shapelet_array[i]):
                score = flatness
            else:
                if (vector[-1] - vector[0]) * (self.shapelet_array[i][-1] - self.shapelet_array[i][0]) > 0:
                    score = 1 - flatness
            coords.append(score)
        return coords

    def shapelet_space_representation(self, vector, slope_thres=0.0005, type='pearson'):
        if type.lower() == 'slope-based':
            return self._shapelet_space_representation_slope_based(vector, slope_thres)
        elif type.lower() == 'pearson':
            return self._shapelet_space_representation_gen(vector, slope_thres)
        else:
            raise Exception("type must be either 'slope-based' or 'pearson'!")

    def cosine_sim(self, coords1, coords2):
        return 1 - cosine(coords1, coords2)

    def gen_sampled_shapes(self, quant_forecast, num_shapes):
        sampled_shapes = np.zeros(shape=(num_shapes, len(quant_forecast)))
        for w in range(0, len(quant_forecast)):
            df = quant_forecast[w]
            if not isinstance(df, pd.DataFrame):
                raise Exception("Input must be a vector of pandas DataFrames!")
            df.dropna()
            if 'quantile' not in df or 'value' not in df:
                raise Exception("Input DataFrame must contain 'quantile' and 'value' columns!")
            min_q = df['quantile'].min()
            max_q = df['quantile'].max()
            f = interp1d(df['quantile'], df['value'], kind='linear')
            r_list = np.random.rand(num_shapes)
            for n in range(0, num_shapes):
                r = min_q + (max_q - min_q) * r_list[n]
                sampled_shapes[n, w] = f(r)
        return sampled_shapes

    def shapelet_space_representation_prob(self, quant_forecast, slope_thres=0.0005, type='pearson', num_shapes=100):
        shape_list = self.gen_sampled_shapes(quant_forecast, num_shapes)
        coords = [0] * self.Number_of_shapelets
        for s in range(0, num_shapes):
            vector = shape_list[s]
            if type == 'slope-based':
                these_coords = self._shapelet_space_representation_slope_based(vector, slope_thres)
            else:
                these_coords = self._shapelet_space_representation_gen(vector, slope_thres)

            for i in range(0, len(coords)):
                coords[i] = coords[i] + these_coords[i]
        for i in range(0, len(coords)):
            coords[i] = coords[i] / num_shapes
        return coords

    def find_shapelet_space_ts(self, time_series, slope_thres=0.0005):
        reps = np.full(shape=(self.Number_of_shapelets, len(time_series)), fill_value=np.nan)
        for i in range(0, len(time_series) - self.Shapelet_length + 1):
            this_shape = time_series[i:i + self.Shapelet_length]
            this_rep = self.shapelet_space_representation(this_shape, slope_thres)
            reps[:, i] = this_rep
        nan_mask = np.isnan(reps)
        idx = np.where(~nan_mask, np.arange(nan_mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        reps[nan_mask] = reps[np.nonzero(nan_mask)[0], idx[nan_mask]]
        return reps

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from shapelet_space import *
import dtw
from numpy import corrcoef as pcor
from numpy import exp
from numpy import mean
from numpy import log
import random
import numpy as np
import math
import pandas as pd
from numpy import diff
from scipy.interpolate import interp1d
from scipy.spatial.distance import cosine
from shapelet_space import config
from config import *
import time
import importlib

parser = argparse.ArgumentParser(description='Process input.')
parser.add_argument('csv_filename', help='The path to the CSV file to process')
args = parser.parse_args()
csv_filename = args.csv_filename
dd = pd.read_csv(csv_filename).values


dd = np.genfromtxt("flu_sims.csv", delimiter=',') 
plt.plot(dd.T, color=(0, 0, 1, 0.2))
plt.plot(np.nanmean(dd, axis=0), 'o', color='r')
plt.savefig('TimeSeriesPlot.png')


d, w = Number_of_shapelets, len(shapelet_array[0])
ns, T = dd.shape

# Calculate slope thresholds
slope_time = T
slope_thres = np.zeros(ns)
for cid in range(ns):
    slope_thres[cid] = np.nanmax(np.convolve(np.abs(np.diff(dd[cid, :slope_time])), np.ones(d)/d, mode='valid'))
slope_thres= np.nanmedian(slope_thres)


#shapelet transformation
Number_of_time_series = dd.shape[0]
Time_series_length = dd.shape[1]

start_time = time.time()
all_reps = np.zeros((Number_of_time_series, Time_series_length, Number_of_shapelets))
for i in range(Number_of_time_series):
    time_series = dd[i]
    reps = shapelet.find_shapelet_space_ts(time_series,slope_thres)
    nan_mask = np.isnan(reps)
    idx = np.where(~nan_mask, np.arange(nan_mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    reps[nan_mask] = reps[np.nonzero(nan_mask)[0], idx[nan_mask]]
    all_reps[i, :, :] = reps.T 
end_time = time.time()
print(f"Time taken for shapelet transformation: {end_time - start_time} seconds")


cid = np.random.randint(0, ns)
fig, ax = plt.subplots(2, 1)
ax[0].plot(dd[cid, :])
ax[0].set_xlim([0, all_reps.shape[1]])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Value')
ax[0].set_title('Time-series')

img = ax[1].imshow(np.squeeze(all_reps[cid, :, :]).T, aspect='auto', extent=[0, all_reps.shape[1], 0, 4])
fig.colorbar(img, ax=ax[1], orientation='vertical')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Shapelet dimensions')
ax[1].set_yticks([0.5, 1.5, 2.5, 3.5])
ax[1].set_yticklabels(['Flat', 'Surge', 'Peak', 'Inc'])

ax[1].set_title('Shapelet space visualization')
plt.tight_layout()
plt.savefig('Shapelet_Visualization.png')


# Find similarity matrix
win = 30
sim_mat = np.full((ns, ns), np.nan)

start_time = time.time()
print("Computing Similarity Matrix, Please Wait ......................................")
for ii in range(ns):
    for jj in range(ii+1):
        sim_mat[ii, jj] = dtw.dtw_cons_md(np.squeeze(all_reps[ii, :, :]).T, np.squeeze(all_reps[jj, :, :]).T, win, 'euc')
end_time = time.time()

print(f"Time taken for similarity matrix computation: {end_time - start_time} seconds")



pd.DataFrame(sim_mat).to_csv('similarity_matrix.csv', index=False)





"""Consensus and quantiles of the flu simulations with DTW+S.

Run from the repository root::

    python examples/ensemble_example.py

The script writes ``ensemble_example.png``.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..'))  # use the package from this repository

from shapelet_space import ensemble  # noqa: E402

dd = np.loadtxt(os.path.join(here, '..', 'data', 'flu_sims.csv'), delimiter=',')

result = ensemble.dtw_s_ensemble(dd, win=30, num_iter=20, verbose=True)

print(f"slope threshold: {result.slope_thres:.2f}")
print(f"point-wise mean peak: {dd.mean(axis=0).max():.1f}")
print(f"DTW+S consensus peak: {result.consensus.max():.1f}")
print(f"median of the individual peaks: {np.median(dd.max(axis=1)):.1f}")

# Quantiles: rows are the 2.5%, 25%, 50%, 75% and 97.5% quantiles.
qq = (0.025, 0.25)
Q = ensemble.get_dbas_ensemble(dd, qq=qq, win=10, seed=0)
levels = ensemble.quantile_levels(qq)
print('quantile levels:', levels)
print(f"median peak: {Q[2].max():.1f}, 97.5% quantile peak: {Q[4].max():.1f}")

t = np.arange(dd.shape[1])
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(dd.T, color='b', alpha=0.1)
ax[0].plot(dd.mean(axis=0), color='r', lw=1.5, label='Mean')
ax[0].plot(result.mean_time, result.mean_curve, color='k', lw=1.5, label='DTW+S DBA')
ax[0].set_xlabel('time')
ax[0].set_ylabel('value')
ax[0].legend()
ax[0].set_title('Consensus')
ax[1].plot(dd.T, color='b', alpha=0.1)
ax[1].fill_between(t, Q[0], Q[4], color='r', alpha=0.2, label='95%')
ax[1].fill_between(t, Q[1], Q[3], color='r', alpha=0.4, label='50%')
ax[1].plot(t, Q[2], color='k', lw=1.5, label='median')
ax[1].set_xlabel('time')
ax[1].legend()
ax[1].set_title('DBA+S quantiles')
plt.savefig('ensemble_example.png', dpi=100)

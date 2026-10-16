"""Consensus of the flu simulations with DTW+S.

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

plt.figure(figsize=(8, 5))
plt.plot(dd.T, color='b', alpha=0.1)
plt.plot(dd.mean(axis=0), color='r', lw=1.5, label='Mean')
plt.plot(result.mean_time, result.mean_curve, color='k', lw=1.5, label='DTW+S DBA')
plt.xlabel('time')
plt.ylabel('value')
plt.legend()
plt.savefig('ensemble_example.png', dpi=100)

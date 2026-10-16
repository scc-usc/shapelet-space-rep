"""Consensus (ensemble) of a set of time series with DTW+S.

Python port of the MATLAB code in ``DTW_S_apps/ensemble``
(``find_dtw_matchings.m``, ``dtw_mean_ensemble.m``, ``DBA_S_alignments.m``)
plus the helper ``shape_ts_transform.m``.

Given several time series (for example simulations or forecasts of the same
quantity), a plain point-wise mean flattens and widens the peaks when the
series are not aligned in time. The functions in this module first align all
series to a reference with Dynamic Time Warping in shapelet space (DTW+S)
and then average the matched points. The result is a consensus curve that
keeps the shape of the individual series.

Typical use::

    from shapelet_space import ensemble
    result = ensemble.dtw_s_ensemble(dd)       # dd has shape (n_series, T)
    result.mean_time, result.mean_curve        # consensus in (time, value) form
    result.consensus                           # consensus on the time grid 0..T-1

All time indices in this module are 0-based.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

from .dtw import dtw_cons_md, dtw_cons_md_path
from .shapelet import ShapeletSpace

#: Shapelets used in the MATLAB scripts: increase, peak, surge, flat.
DEFAULT_SHAPELETS = np.array([[1, 2, 3, 4],
                              [1, 2, 2, 1],
                              [1, 2, 4, 8],
                              [0, 0, 0, 0]], dtype=float)


def _as_2d_float(dd):
    dd = np.asarray(dd, dtype=float)
    if dd.ndim == 1:
        dd = dd[None, :]
    if dd.ndim != 2:
        raise ValueError(f"expected an array of shape (n_series, T), got shape {dd.shape}")
    return dd


def _series_rep(all_reps, i):
    """Return series ``i`` of ``all_reps`` as a ``(dims, T)`` array."""
    rep = all_reps[i]
    if rep.ndim == 1:
        return rep[None, :]
    return rep.T


def estimate_slope_threshold(dd, window=4):
    """Slope threshold used by the shapelet transform, as in the MATLAB scripts.

    For every series the absolute first differences are smoothed with a
    forward moving mean of ``window`` points. The maximum of that smoothed
    slope is the threshold of the series. The median over all series is
    returned.

    Parameters
    ----------
    dd : array_like, shape (n_series, T)
    window : int, optional
        Length of the forward moving-mean window. Default 4.

    Returns
    -------
    float
    """
    dd = _as_2d_float(dd)
    window = int(window)
    thres = np.full(dd.shape[0], np.nan)
    for cid in range(dd.shape[0]):
        slopes = np.abs(np.diff(dd[cid]))
        if slopes.size == 0:
            continue
        # movmean(x, [0 window-1]): mean of x[i : i + window], shrunk at the end.
        smoothed = np.array([np.nanmean(slopes[i:i + window]) for i in range(slopes.size)])
        thres[cid] = np.nanmax(smoothed)
    est = float(np.nanmedian(thres))
    if not np.isfinite(est) or est <= 0:
        # All series are flat: any positive value gives flatness 1 everywhere.
        est = np.finfo(float).eps
    return est


def shape_ts_transform(dd, shapelet_array=DEFAULT_SHAPELETS, slope_thres=0.0005):
    """Shapelet-space representation of every time point of every series.

    Port of ``shape_ts_transform.m``. The window used for time point ``t`` is
    centred on ``t``: it covers ``t - l .. t + r`` with ``l = ceil(w/2) - 1``
    and ``r = w - ceil(w/2)``. The first ``l`` and the last ``r`` points copy
    the nearest computed representation.

    Parameters
    ----------
    dd : array_like, shape (n_series, T) or (T,)
    shapelet_array : array_like, shape (d, w), optional
        One shapelet per row. Default :data:`DEFAULT_SHAPELETS`.
    slope_thres : float or array_like of shape (n_series,), optional
        Slope that gets a flatness of 0.1. A scalar is used for all series.
        See :func:`estimate_slope_threshold`.

    Returns
    -------
    ndarray, shape (n_series, T, d)
    """
    dd = _as_2d_float(dd)
    A = np.asarray(shapelet_array, dtype=float)
    if A.ndim == 1:
        A = A[None, :]
    ns, T = dd.shape
    d, w = A.shape

    slope_thres = np.asarray(slope_thres, dtype=float).ravel()
    if slope_thres.size == 1:
        slope_thres = np.repeat(slope_thres, ns)
    if slope_thres.size < ns:
        raise ValueError("slope_thres must be a scalar or have one value per series")

    if w == 1:
        # One-point shapelets: the representation is the series itself.
        return np.repeat(dd[:, :, None], d, axis=2)
    if T < w:
        raise ValueError(f"series length {T} is shorter than the shapelet length {w}")

    half = int(np.ceil(w / 2))
    l = half - 1
    r = w - half

    transformer = ShapeletSpace(Shapelet_length=w, Number_of_shapelets=d, shapelet_array=A)
    all_reps = np.full((ns, T, d), np.nan)
    for cid in range(ns):
        for tt in range(l, T - r):
            window = dd[cid, tt - l:tt + r + 1]
            all_reps[cid, tt, :] = transformer.shapelet_space_representation(window, slope_thres[cid])
        all_reps[cid, :l, :] = all_reps[cid, l, :]
        all_reps[cid, T - r:, :] = all_reps[cid, T - r - 1, :]
    return all_reps


def pairwise_dtw_distances(all_reps, win=np.inf, dist_metric='euc'):
    """Symmetric matrix of DTW(+S) distances between all pairs of series.

    Parameters
    ----------
    all_reps : array_like, shape (n_series, T, d) or (n_series, T)
        Shapelet-space representations (see :func:`shape_ts_transform`) or
        raw series.
    win : float or sequence of two floats, optional
        DTW window. See :func:`shapelet_space.dtw.dtw_cons_md_path`.
    dist_metric : str, optional
        ``'euc'`` or ``'cos'``.

    Returns
    -------
    ndarray, shape (n_series, n_series)
    """
    all_reps = np.asarray(all_reps, dtype=float)
    ns = all_reps.shape[0]
    sim_mat = np.zeros((ns, ns))
    for ii in range(ns):
        for jj in range(ii):
            sim_mat[ii, jj] = dtw_cons_md(_series_rep(all_reps, ii), _series_rep(all_reps, jj),
                                          win, dist_metric)
            sim_mat[jj, ii] = sim_mat[ii, jj]
    return sim_mat


def find_medoid(all_reps, win=np.inf, dist_metric='euc'):
    """Index of the series with the smallest mean DTW(+S) distance to the others."""
    sim_mat = pairwise_dtw_distances(all_reps, win, dist_metric)
    return int(np.argmin(np.mean(sim_mat, axis=0)))


def find_dtw_matchings(all_reps, dtw_win=np.inf, dtw_metric='euc', base_m=0):
    """Align every series to a base series with DTW(+S).

    Port of ``find_dtw_matchings.m``. Every series is warped against the base
    series. Time point ``t`` of the base defines one "event". For each other
    series the event is placed at the first time point that the warping path
    matches to ``t``. The last event is always placed at the last time point.

    Parameters
    ----------
    all_reps : array_like, shape (n_series, T, d) or (n_series, T)
        Shapelet-space representations (see :func:`shape_ts_transform`) or
        raw series.
    dtw_win : float or sequence of two floats, optional
        DTW window. See :func:`shapelet_space.dtw.dtw_cons_md_path`.
    dtw_metric : str, optional
        ``'euc'`` or ``'cos'``.
    base_m : int, optional
        Index of the base series. Default 0.

    Returns
    -------
    ndarray of int, shape (n_series, T)
        ``dtw_matches[j, t]`` is the time index in series ``j`` that matches
        time ``t`` of the base series. Row ``base_m`` is ``0 .. T-1``.
    """
    all_reps = np.asarray(all_reps, dtype=float)
    ns, T = all_reps.shape[:2]
    if not 0 <= base_m < ns:
        raise ValueError(f"base_m must be in 0..{ns - 1}, got {base_m}")

    base_rep = _series_rep(all_reps, base_m)
    dtw_matches = np.zeros((ns, T), dtype=int)
    dtw_matches[base_m] = np.arange(T)
    for j in range(ns):
        if j == base_m:
            continue
        _, ix, iy = dtw_cons_md_path(base_rep, _series_rep(all_reps, j), dtw_win, dtw_metric)
        # ix covers every base index 0..T-1 in increasing order; keep the first
        # match of every base index.
        _, first = np.unique(ix, return_index=True)
        dtw_matches[j] = iy[first]
        dtw_matches[j, T - 1] = iy[-1]
    return dtw_matches


def dtw_mean_ensemble(dd, dtw_matches):
    """Average the matched points of all series, event by event.

    Port of ``dtw_mean_ensemble.m``.

    Parameters
    ----------
    dd : array_like, shape (n_series, T)
    dtw_matches : array_like of int, shape (n_points, n_events)
        Output of :func:`find_dtw_matchings`. ``n_points`` must be a multiple
        of ``n_series``; row ``i`` indexes into series ``i % n_series``.

    Returns
    -------
    mean_curve : ndarray, shape (n_events,)
        Mean value of the matched points of every event.
    mean_time : ndarray, shape (n_events,)
        Mean (0-based) time of the matched points of every event.
    all_values : ndarray, shape (n_points, n_events)
        Value of every matched point.
    all_times : ndarray, shape (n_points, n_events)
        Time of every matched point (a float copy of ``dtw_matches``).
    """
    dd = _as_2d_float(dd)
    dtw_matches = np.asarray(dtw_matches)
    ns = dd.shape[0]
    n_points = dtw_matches.shape[0]
    if n_points % ns != 0:
        raise ValueError("the number of rows of dtw_matches must be a multiple of the number of series")
    rows = np.arange(n_points) % ns
    all_values = dd[rows[:, None], dtw_matches]
    all_times = dtw_matches.astype(float)
    return all_values.mean(axis=0), all_times.mean(axis=0), all_values, all_times


def resample_to_grid(mean_time, mean_curve, T, clip_negative=True):
    """Interpolate a (time, value) curve on the regular grid ``0 .. T-1``.

    Uses the modified Akima method (``'makima'`` in MATLAB) with
    extrapolation, as in ``DBA_S_alignments.m``. Repeated times keep their
    first occurrence.
    """
    mean_time = np.asarray(mean_time, dtype=float)
    mean_curve = np.asarray(mean_curve, dtype=float)
    uniq_t, first = np.unique(mean_time, return_index=True)
    uniq_y = mean_curve[first]
    grid = np.arange(T, dtype=float)
    if uniq_t.size == 1:
        curve = np.full(T, uniq_y[0])
    else:
        try:
            interp = Akima1DInterpolator(uniq_t, uniq_y, method='makima', extrapolate=True)
        except TypeError:  # SciPy < 1.13 has no 'makima' method
            interp = PchipInterpolator(uniq_t, uniq_y, extrapolate=True)
        curve = interp(grid)
    if clip_negative:
        curve = np.where(curve < 0, 0.0, curve)
    return curve


def dba_s_alignments(dd, all_reps=None, shapelet_array=DEFAULT_SHAPELETS, slope_thres=0.0005,
                     win=np.inf, dist_metric='euc', num_iter=10, tol=1e-10, verbose=False):
    """DTW Barycenter Averaging in shapelet space (DBA+S).

    Port of ``DBA_S_alignments.m``. The reference series starts as the medoid
    of the set. In every iteration all series are aligned to the reference
    (:func:`find_dtw_matchings`), the matched points are averaged
    (:func:`dtw_mean_ensemble`), and the average is resampled on the regular
    time grid to become the new reference.

    Parameters
    ----------
    dd : array_like, shape (n_series, T)
        Raw time series.
    all_reps : array_like, shape (n_series, T, d), optional
        Shapelet-space representations of ``dd``. Computed with
        :func:`shape_ts_transform` when omitted.
    shapelet_array : array_like, shape (d, w) or None, optional
        Shapelets used to transform the reference in every iteration (and
        ``dd`` when ``all_reps`` is omitted). ``None`` uses the raw values,
        that is plain DTW.
    slope_thres : float, optional
        Slope threshold of the shapelet transform.
    win : float or sequence of two floats, optional
        DTW window.
    dist_metric : str, optional
        ``'euc'`` or ``'cos'``.
    num_iter : int, optional
        Maximum number of iterations. With 0 the series are aligned to the
        medoid only.
    tol : float, optional
        Stop when the squared change of the reference is below this value.
    verbose : bool, optional
        Print a message on early termination.

    Returns
    -------
    dtw_matches : ndarray of int, shape (n_series, T)
        Alignment of every series to the final reference.
    ref_ts : ndarray, shape (T,)
        Final reference series.
    all_refs : ndarray, shape (n_iter + 1, T)
        Reference series of every iteration, starting with the medoid.
    """
    dd = _as_2d_float(dd)
    ns, T = dd.shape
    use_shapelets = shapelet_array is not None
    if all_reps is None:
        all_reps = shape_ts_transform(dd, shapelet_array, slope_thres) if use_shapelets else dd[:, :, None]
    all_reps = np.asarray(all_reps, dtype=float)
    if all_reps.ndim == 2:
        all_reps = all_reps[:, :, None]

    med_idx = find_medoid(all_reps, win, dist_metric)
    ref_ts = dd[med_idx].copy()
    all_refs = [ref_ts]
    dtw_matches = find_dtw_matchings(all_reps, win, dist_metric, med_idx)

    for _ in range(int(num_iter)):
        if use_shapelets:
            ref_rep = shape_ts_transform(ref_ts, shapelet_array, slope_thres)
        else:
            ref_rep = ref_ts[None, :, None]
        temp_reps = np.concatenate([all_reps, ref_rep], axis=0)
        dtw_matches = find_dtw_matchings(temp_reps, win, dist_metric, base_m=ns)[:ns]
        mean_curve, mean_time, _, _ = dtw_mean_ensemble(dd, dtw_matches)
        ref_ts = resample_to_grid(mean_time, mean_curve, T)
        d_change = float(np.sum((ref_ts - all_refs[-1]) ** 2))
        if d_change < tol:
            if verbose:
                print('Early Termination')
            break
        all_refs.append(ref_ts)
    return dtw_matches, ref_ts, np.array(all_refs)


@dataclass
class EnsembleResult:
    """Output of :func:`dtw_s_ensemble`."""
    #: Mean value of every event, shape (T,).
    mean_curve: np.ndarray
    #: Mean (0-based) time of every event, shape (T,).
    mean_time: np.ndarray
    #: Consensus curve resampled on the time grid ``0 .. T-1``, shape (T,).
    consensus: np.ndarray
    #: Time index in every series of every event, shape (n_series, T).
    matches: np.ndarray
    #: Value of every matched point, shape (n_series, T).
    all_values: np.ndarray
    #: Time of every matched point, shape (n_series, T).
    all_times: np.ndarray
    #: Reference series that the others were aligned to, shape (T,).
    reference: np.ndarray
    #: Shapelet-space representations of the input, shape (n_series, T, d).
    reps: np.ndarray
    #: Slope threshold used by the shapelet transform.
    slope_thres: float
    #: Reference series of every DBA iteration (empty when ``base`` was given).
    all_refs: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))


def dtw_s_ensemble(dd, shapelet_array=DEFAULT_SHAPELETS, slope_thres=None, win=np.inf,
                   dist_metric='euc', num_iter=10, base=None, tol=1e-10, verbose=False):
    """Consensus of a set of time series with DTW+S.

    This is the end-to-end procedure of the MATLAB scripts ``DTW_ensemble.m``
    and ``DBA_S_ensemble.m``:

    1. estimate the slope threshold (:func:`estimate_slope_threshold`),
    2. transform every series to shapelet space (:func:`shape_ts_transform`),
    3. align all series to a reference (:func:`find_dtw_matchings`, or
       iteratively with :func:`dba_s_alignments`),
    4. average the matched points (:func:`dtw_mean_ensemble`).

    Parameters
    ----------
    dd : array_like, shape (n_series, T)
        One time series per row. All series must have the same length and
        must not contain NaN.
    shapelet_array : array_like, shape (d, w) or None, optional
        One shapelet per row. Default :data:`DEFAULT_SHAPELETS`. ``None``
        aligns the raw values with plain DTW.
    slope_thres : float, optional
        Slope threshold of the shapelet transform. Estimated from the data
        when omitted.
    win : float or sequence of two floats, optional
        DTW window. Default: no constraint.
    dist_metric : str, optional
        ``'euc'`` (default) or ``'cos'``.
    num_iter : int, optional
        Number of DBA iterations. 0 aligns the series to the medoid only.
        Ignored when ``base`` is given.
    base : int, optional
        Index of the series to use as the fixed reference. When omitted the
        reference is found with :func:`dba_s_alignments`.
    tol : float, optional
        Convergence tolerance of the DBA iterations.
    verbose : bool, optional
        Print a message on early termination of the DBA iterations.

    Returns
    -------
    EnsembleResult
    """
    dd = _as_2d_float(dd)
    ns, T = dd.shape
    if ns < 2:
        raise ValueError("at least two time series are needed")
    if np.isnan(dd).any():
        raise ValueError("dd must not contain NaN")

    use_shapelets = shapelet_array is not None
    if use_shapelets:
        A = np.asarray(shapelet_array, dtype=float)
        if A.ndim == 1:
            A = A[None, :]
        if slope_thres is None:
            slope_thres = estimate_slope_threshold(dd, A.shape[1])
        all_reps = shape_ts_transform(dd, A, slope_thres)
    else:
        A = None
        slope_thres = float('nan') if slope_thres is None else slope_thres
        all_reps = dd[:, :, None]

    if base is not None:
        matches = find_dtw_matchings(all_reps, win, dist_metric, int(base))
        reference = dd[int(base)].copy()
        all_refs = np.empty((0, T))
    else:
        matches, reference, all_refs = dba_s_alignments(
            dd, all_reps, A, slope_thres, win, dist_metric, num_iter, tol, verbose)

    mean_curve, mean_time, all_values, all_times = dtw_mean_ensemble(dd, matches)
    consensus = resample_to_grid(mean_time, mean_curve, T)
    return EnsembleResult(mean_curve=mean_curve, mean_time=mean_time, consensus=consensus,
                          matches=matches, all_values=all_values, all_times=all_times,
                          reference=reference, reps=all_reps, slope_thres=float(slope_thres),
                          all_refs=all_refs)

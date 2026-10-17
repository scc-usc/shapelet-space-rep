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

Uncertainty around the consensus is described by envelopes
(:func:`find_envelopes`, port of ``find_envelops.m``) and by quantile curves
derived from them (:func:`envelope_quantiles`).

All time indices in this module are 0-based.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.spatial import ConvexHull, QhullError

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
    curve = _makima(mean_time, mean_curve, np.arange(T, dtype=float))
    if clip_negative:
        curve = np.where(curve < 0, 0.0, curve)
    return curve


def _makima(x, y, grid):
    """MATLAB ``interp1(x, y, grid, 'makima')`` with extrapolation.

    Repeated ``x`` values keep their first occurrence.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    uniq_x, first = np.unique(x, return_index=True)
    uniq_y = y[first]
    if uniq_x.size == 1:
        return np.full(np.shape(grid), uniq_y[0])
    try:
        interp = Akima1DInterpolator(uniq_x, uniq_y, method='makima', extrapolate=True)
    except TypeError:  # SciPy < 1.13 has no 'makima' method
        interp = PchipInterpolator(uniq_x, uniq_y, extrapolate=True)
    return interp(grid)


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


# ---------------------------------------------------------------------------
# Envelopes and quantiles (port of find_envelops.m and its helpers)
# ---------------------------------------------------------------------------

def _matlab_round(x):
    """MATLAB ``round``: halves are rounded away from zero."""
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def calculate_covering_lengths(x_values, y_values, fraction=0.95):
    """Lengths in x and y that cover ``fraction`` of the points.

    Port of ``calculate_covering_lengths.m``. The points are sorted along
    each axis and the smallest distance between points that are
    ``round(fraction * n)`` positions apart is returned. The lengths are used
    to normalise distances in :func:`convex_hull_degen_confidence`.

    Returns
    -------
    x_norm, y_norm : float
        0 when there are not enough points to measure a covering length.
    """
    x_values = np.asarray(x_values, dtype=float).ravel()
    y_values = np.asarray(y_values, dtype=float).ravel()
    if x_values.size != y_values.size:
        raise ValueError('Number of x-values must be equal to the number of y-values.')
    step = max(_matlab_round(fraction * x_values.size), 1)
    x_sel = np.sort(x_values)[::step]
    y_sel = np.sort(y_values)[::step]
    x_norm = float(np.min(np.diff(x_sel))) if x_sel.size > 1 else 0.0
    y_norm = float(np.min(np.diff(y_sel))) if y_sel.size > 1 else 0.0
    return x_norm, y_norm


def _hull_vertices(points):
    """Vertices of the convex hull of ``points``, in order around the hull.

    Degenerate point sets (a line or a single point) give the four extreme
    points, as in the MATLAB code.
    """
    x = points[:, 0]
    y = points[:, 1]
    if points.shape[0] >= 3 and np.ptp(x) > 1e-10 and np.ptp(y) > 1e-10:
        try:
            return points[ConvexHull(points).vertices]
        except QhullError:
            pass
    idx = [int(np.argmin(x)), int(np.argmin(y)), int(np.argmax(x)), int(np.argmax(y))]
    return points[idx]


def convex_hull_degen_confidence(x2, y2, x_norm, y_norm, centroid_x, centroid_y, confidence):
    """Convex hull of the points closest to a centroid.

    Port of ``convex_hull_degen_confidence.m``. When ``x_norm`` and
    ``y_norm`` are positive, the ``round(confidence * n)`` points with the
    smallest normalised distance to the centroid are kept. Otherwise the
    ``ceil(confidence * n)`` points closest to the centroid in y and the same
    number closest in x are kept.

    Parameters
    ----------
    x2, y2 : array_like
        Point coordinates.
    x_norm, y_norm : float
        Normalisation lengths. See :func:`calculate_covering_lengths`.
    centroid_x, centroid_y : float
    confidence : float
        Fraction of the points to keep, between 0 and 1.

    Returns
    -------
    ndarray, shape (n_vertices, 2)
        Vertices of the convex hull of the kept points.
    """
    if not 0 <= confidence <= 1:
        raise ValueError('confidence must be between 0 and 1')
    x2 = np.asarray(x2, dtype=float).ravel()
    y2 = np.asarray(y2, dtype=float).ravel()
    points = np.column_stack([x2, y2])
    n = points.shape[0]

    if x_norm > 0 and y_norm > 0:
        n_keep = _matlab_round(confidence * n)
        distances = np.sqrt(((x2 - centroid_x) / x_norm) ** 2 + ((y2 - centroid_y) / y_norm) ** 2)
        keep = np.sort(np.argsort(distances, kind='stable')[:n_keep])
    else:
        n_keep = int(np.ceil(confidence * n))
        sel = np.zeros(n, dtype=bool)
        sel[np.argsort(np.abs(centroid_y - y2), kind='stable')[:n_keep]] = True
        sel[np.argsort(np.abs(centroid_x - x2), kind='stable')[:n_keep]] = True
        keep = np.flatnonzero(sel)
    points = points[keep]
    if points.shape[0] == 0:
        return points
    return _hull_vertices(points)


def generate_midpoints(points, num_interpolations):
    """Points spaced along the closed polygon through ``points``.

    Port of ``generateMidpoints.m``. For every edge (including the edge from
    the last to the first point) ``num_interpolations + 1`` points are
    produced: the start vertex and ``num_interpolations`` points along the
    edge. ``points`` may be a 1-D array of one coordinate or a 2-D array of
    shape ``(n, k)``.
    """
    points = np.asarray(points, dtype=float)
    nxt = np.roll(points, -1, axis=0)
    t = np.arange(num_interpolations + 1) / (num_interpolations + 1)
    if points.ndim == 1:
        out = (1 - t)[None, :] * points[:, None] + t[None, :] * nxt[:, None]
        return out.ravel()
    out = (1 - t)[None, :, None] * points[:, None, :] + t[None, :, None] * nxt[:, None, :]
    return out.reshape(-1, points.shape[1])


def is_intersect_with_sides(p1, p2, points, min_idx, max_idx, direction):
    """Whether the segment ``p2 -> p1`` cuts through the profile ``points``.

    Port of ``isIntersectWithSides.m``. The profile points with index in
    ``min_idx + 1 .. max_idx`` and x strictly between ``p2[0]`` and
    ``p1[0]`` are tested. For ``direction='max'`` a point above the segment
    is an intersection, for ``direction='min'`` a point below it.
    """
    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
    intercept = p1[1] - p1[0] * slope
    for i in range(min_idx + 1, max_idx + 1):
        px, py = points[i, 0], points[i, 1]
        if px >= p1[0]:
            break
        if px <= p2[0]:
            continue
        line = px * slope + intercept
        if direction == 'max' and py > line:
            return True
        if direction == 'min' and py < line:
            return True
    return False


def _thin_profile(vertices_sorted, interval, direction):
    """Keep one point per x-interval: the lowest ('min') or highest ('max')."""
    xs = vertices_sorted[:, 0]
    ys = vertices_sorted[:, 1]
    n = xs.size
    pick = np.argmin if direction == 'min' else np.argmax
    out = []
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] < xs[i] + interval:
            j += 1
        out.append(vertices_sorted[i + int(pick(ys[i:j]))])
        i = j
    return np.array(out)


def _trace_boundary(profile, window, direction):
    """Greedy piecewise-linear boundary along a thinned profile.

    Starting from the first profile point, every step moves to the farthest
    point within ``window`` in x that can be reached without cutting through
    the profile (see :func:`is_intersect_with_sides`).
    """
    xs = profile[:, 0]
    ys = profile[:, 1]
    n = xs.size
    bx = [xs[0]]
    by = [ys[0]]
    if n == 1:
        return np.array(bx), np.array(by)
    idx = 1
    last_idx = 0
    sel = 0
    interval_end = xs[0] + window
    while idx < n:
        while xs[idx] < interval_end:
            if not is_intersect_with_sides(profile[idx], (bx[-1], by[-1]), profile, last_idx, idx, direction):
                sel = idx
            idx += 1
            if idx >= n:
                break
        bx.append(xs[sel])
        by.append(ys[sel])
        if idx >= n:
            break
        last_idx = idx
        interval_end = xs[idx] + window
    return np.array(bx), np.array(by)


def _envelope_curve(all_vertices_sorted, T, direction, method, thin_interval, window, strict_step):
    profile = _thin_profile(all_vertices_sorted, thin_interval, direction)
    bx, by = _trace_boundary(profile, window, direction)
    xq = np.arange(T, dtype=float)
    vq = _makima(bx, by, xq)

    if method == 'strict':
        sign = -1.0 if direction == 'min' else 1.0
        for i in range(T - 1):
            while is_intersect_with_sides((xq[i + 1], vq[i + 1]), (xq[i], vq[i]), profile, 0,
                                          profile.shape[0] - 1, direction):
                vq[i + 1] += sign * strict_step
                vq[i] += sign * strict_step
    elif method == 'combo':
        xq = np.concatenate([xq, bx])
        vq = np.concatenate([vq, by])
        order = np.argsort(xq, kind='stable')
        xq = xq[order]
        vq = vq[order]
    elif method != 'none':
        raise ValueError("method must be 'none', 'strict' or 'combo'")
    return xq, vq, profile


@dataclass
class EnvelopeResult:
    """Output of :func:`find_envelopes`."""
    #: Confidence level of the envelope.
    confidence: float
    #: Time of every point of the lower envelope.
    lower_time: np.ndarray
    #: Value of every point of the lower envelope.
    lower: np.ndarray
    #: Time of every point of the upper envelope.
    upper_time: np.ndarray
    #: Value of every point of the upper envelope.
    upper: np.ndarray
    #: Convex hull vertices of every event, shape (n_vertices, 2) each.
    hulls: list = field(default_factory=list)


def find_envelopes(dd, dtw_matches, confidence=0.95, method='none', poly_method='', jitter=0.1,
                   num_interpolations=50, thin_interval=0.025, window=2.0, strict_step=1.0,
                   seed=None):
    """Lower and upper envelopes of the matched points around the consensus.

    Port of ``find_envelops.m``. For every event (time step of the reference)
    the matched points ``(time, value)`` of all series are jittered, the
    points farthest from the consensus point are dropped so that a fraction
    ``confidence`` remains, and the convex hull of the rest is taken. The
    lower (upper) envelope follows the bottom (top) of the union of these
    hulls and is interpolated on the time grid ``0 .. T-1``.

    Parameters
    ----------
    dd : array_like, shape (n_series, T)
    dtw_matches : array_like of int, shape (n_series, T)
        Output of :func:`find_dtw_matchings` (or ``EnsembleResult.matches``).
    confidence : float, optional
        Fraction of the matched points of every event kept inside the hull.
        Default 0.95.
    method : {'none', 'strict', 'combo'}, optional
        ``'none'`` returns the interpolated envelope on the time grid.
        ``'strict'`` shifts grid segments outward in steps of ``strict_step``
        until no hull point lies outside. ``'combo'`` merges the grid points
        with the boundary points (the time arrays are then not the grid).
    poly_method : str, optional
        A value containing ``'2d'`` selects the points per axis instead of
        by normalised distance. See :func:`convex_hull_degen_confidence`.
    jitter : float, optional
        Relative random jitter applied to the points before the hull is
        taken (MATLAB uses 0.1). Times are jittered as 1-based values, as in
        MATLAB. Use 0 to disable.
    num_interpolations : int, optional
        Points generated along every hull edge. Default 50.
    thin_interval : float, optional
        Width (in time units) of the bins that thin the hull points. Default 0.025.
    window : float, optional
        Look-ahead (in time units) when tracing the boundary. Default 2.
    strict_step : float, optional
        Shift (in value units) used by ``method='strict'``. Default 1.
    seed : int or numpy.random.Generator, optional
        Seed for the jitter.

    Returns
    -------
    EnvelopeResult
    """
    dd = _as_2d_float(dd)
    T = dd.shape[1]
    rng = np.random.default_rng(seed)
    mean_curve, mean_time, all_values, all_times = dtw_mean_ensemble(dd, dtw_matches)

    hulls = []
    borders = []
    for mm in range(all_times.shape[1]):
        x1 = all_times[:, mm]
        y1 = all_values[:, mm]
        good = ~(np.isnan(x1) | np.isnan(y1))
        x1 = x1[good]
        y1 = y1[good]
        if x1.size == 0:
            continue
        if jitter:
            # MATLAB jitters 1-based times: x .* (1 + jitter * rand).
            x1 = (x1 + 1) * (1 + jitter * rng.random(x1.size)) - 1
            y1 = y1 * (1 + jitter * rng.random(y1.size))
        if '2d' in poly_method:
            x_norm = y_norm = 0.0
        else:
            x_norm, y_norm = calculate_covering_lengths(x1, y1)
        k = convex_hull_degen_confidence(x1, y1, x_norm, y_norm, mean_time[mm], mean_curve[mm], confidence)
        if k.shape[0] == 0:
            continue
        hulls.append(k)
        borders.append(generate_midpoints(k, num_interpolations))

    all_vertices = np.vstack(borders)
    all_vertices = all_vertices[np.argsort(all_vertices[:, 0], kind='stable')]

    lo_x, lo_y, _ = _envelope_curve(all_vertices, T, 'min', method, thin_interval, window, strict_step)
    hi_x, hi_y, _ = _envelope_curve(all_vertices, T, 'max', method, thin_interval, window, strict_step)
    return EnvelopeResult(confidence=float(confidence), lower_time=lo_x, lower=lo_y,
                          upper_time=hi_x, upper=hi_y, hulls=hulls)


def envelope_quantiles(dd, dtw_matches, quantiles=(0.025, 0.25, 0.5, 0.75, 0.975), method='strict',
                       poly_method='dense', median=None, clip_negative=False, seed=None, **kwargs):
    """Quantile curves of the ensemble on the time grid ``0 .. T-1``.

    This is the quantile computation of ``get_DBAS_ensemble.m``. A quantile
    ``q`` below 0.5 is the lower envelope at confidence ``1 - 2q``, a
    quantile above 0.5 the upper envelope at confidence ``2q - 1``, and
    ``q = 0.5`` is the DTW+S consensus. The envelopes come from
    :func:`find_envelopes`; extra keyword arguments are passed to it. Every
    envelope uses the same jitter.

    The envelopes are built independently for every confidence level, so the
    quantile curves are not guaranteed to be nested everywhere. Small
    crossings can occur, mostly at the first and last time steps.

    Parameters
    ----------
    dd : array_like, shape (n_series, T)
    dtw_matches : array_like of int, shape (n_series, T)
    quantiles : sequence of float, optional
        Quantile levels between 0 and 1.
    method : {'none', 'strict', 'combo'}, optional
        Envelope method. Default ``'strict'`` as in the MATLAB code. With
        ``'combo'`` the envelope is interpolated back on the grid.
    poly_method : str, optional
        Point selection of the hulls. Default ``'dense'`` as in the MATLAB
        code. See :func:`find_envelopes`.
    median : array_like, shape (T,), optional
        Curve used for ``q = 0.5``. Default: the mean of the matched points
        resampled on the grid (this equals the final DBA reference).
    clip_negative : bool, optional
        Clip the curves at 0. Default False, as in the MATLAB code.
    seed : int, optional
        Seed for the jitter.

    Returns
    -------
    ndarray, shape (len(quantiles), T)
    """
    dd = _as_2d_float(dd)
    T = dd.shape[1]
    grid = np.arange(T, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    if np.any((quantiles < 0) | (quantiles > 1)):
        raise ValueError('quantiles must be between 0 and 1')
    seed = np.random.SeedSequence().entropy if seed is None else seed

    out = np.empty((quantiles.size, T))
    cache = {}
    for i, q in enumerate(quantiles):
        if np.isclose(q, 0.5):
            if median is not None:
                out[i] = np.asarray(median, dtype=float)
            else:
                mean_curve, mean_time, _, _ = dtw_mean_ensemble(dd, dtw_matches)
                out[i] = resample_to_grid(mean_time, mean_curve, T)
            continue
        conf = abs(2 * q - 1)
        if conf not in cache:
            cache[conf] = find_envelopes(dd, dtw_matches, conf, method=method, poly_method=poly_method,
                                         seed=seed, **kwargs)
        env = cache[conf]
        if q < 0.5:
            x, y = env.lower_time, env.lower
        else:
            x, y = env.upper_time, env.upper
        out[i] = y if (x.size == T and np.array_equal(x, grid)) else np.interp(grid, x, y)
    if clip_negative:
        out = np.where(out < 0, 0.0, out)
    return out


def quantile_levels(qq):
    """Quantile levels of the rows of :func:`get_dbas_ensemble`.

    ``qq`` lists the lower quantiles. The result is ``qq``, then 0.5, then
    the mirrored upper quantiles ``1 - qq`` in reverse order.
    """
    qq = np.sort(np.asarray(qq, dtype=float).ravel())
    if qq.size and (qq[0] < 0 or qq[-1] >= 0.5):
        raise ValueError('qq must contain lower quantiles in [0, 0.5)')
    return np.concatenate([qq, [0.5], 1 - qq[::-1]])


def get_dbas_ensemble(dd, qq=(0.025, 0.25), win=10, shapelet_array=DEFAULT_SHAPELETS, slope_thres=None,
                      num_iter=None, seed=None, return_result=False):
    """Quantiles of a set of time series with the DBA+S ensemble.

    Port of ``get_DBAS_ensemble.m``. The series are aligned with
    :func:`dba_s_alignments` (medoid start, ``win`` iterations by default)
    and the quantiles are read from the envelopes of the matched points
    (:func:`envelope_quantiles` with ``method='strict'`` and
    ``poly_method='dense'``). The median row is the final DBA reference.

    Parameters
    ----------
    dd : array_like, shape (n_series, T)
        One time series per row.
    qq : sequence of float, optional
        Lower quantiles in ``[0, 0.5)``. The upper quantiles are added by
        symmetry and 0.5 is always included. ``(0.05, 0.1, 0.25)`` gives the
        rows ``0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95``.
    win : float, optional
        DTW window. Default 10.
    shapelet_array : array_like, optional
        Shapelets. Default :data:`DEFAULT_SHAPELETS`.
    slope_thres : float, optional
        Slope threshold. Estimated from the data when omitted.
    num_iter : int, optional
        DBA iterations. Default: ``win`` (as in the MATLAB code), or 10 when
        ``win`` is not finite.
    seed : int, optional
        Seed for the jitter of the envelopes.
    return_result : bool, optional
        Also return the :class:`EnsembleResult` of the alignment.

    Returns
    -------
    Q : ndarray, shape (2 * len(qq) + 1, T)
        One quantile curve per row, from the lowest to the highest quantile.
        See :func:`quantile_levels` for the level of every row.
    result : EnsembleResult
        Only when ``return_result`` is True.
    """
    dd = _as_2d_float(dd)
    levels = quantile_levels(qq)
    if num_iter is None:
        num_iter = int(win) if np.isfinite(win) else 10
    result = dtw_s_ensemble(dd, shapelet_array=shapelet_array, slope_thres=slope_thres, win=win,
                            dist_metric='euc', num_iter=num_iter)
    Q = envelope_quantiles(dd, result.matches, levels, method='strict', poly_method='dense',
                           median=result.reference, seed=seed)
    if return_result:
        return Q, result
    return Q

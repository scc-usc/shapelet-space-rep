import numpy as np


def dtw_cons_md_path(t, r, win=np.inf, dist_metric='eu'):
    """Constrained multi-dimensional Dynamic Time Warping with the warping path.

    Port of ``dtw_cons_md.m``.

    Parameters
    ----------
    t, r : array_like
        Sequences to align. Each is either a 1-D array of length ``N`` (``M``)
        or a 2-D array of shape ``(dims, N)`` (``(dims, M)``). Every column is
        one time point and every row is one dimension (for example one
        shapelet dimension).
    win : float or sequence of two floats, optional
        Sakoe-Chiba window. A time point ``n`` of ``t`` may only be matched to
        a time point ``m`` of ``r`` when ``-win[0] < n - m < win[1]``. A scalar
        is used for both sides. Default: no constraint.
    dist_metric : str, optional
        ``'eu'`` (squared Euclidean, default) or a string starting with
        ``'cos'`` (cosine distance, used only when ``dims > 1``).

    Returns
    -------
    dist : float
        DTW distance between ``t`` and ``r``.
    w1, w2 : ndarray of int
        Warping path. ``w1[k]`` is the (0-based) index into ``t`` and
        ``w2[k]`` the index into ``r`` of the k-th step of the optimal path.
    """
    t = np.atleast_2d(np.asarray(t, dtype=float))
    r = np.atleast_2d(np.asarray(r, dtype=float))
    if t.shape[0] != r.shape[0]:
        raise ValueError("t and r must have the same number of dimensions (rows), "
                         f"got {t.shape[0]} and {r.shape[0]}")

    N = t.shape[1]
    M = r.shape[1]

    win = np.asarray(win, dtype=float).ravel()
    if win.size == 1:
        win = np.repeat(win, 2)

    # Local cost matrix, restricted to the window.
    n_idx = np.arange(N)[:, None]
    m_idx = np.arange(M)[None, :]
    diff = n_idx - m_idx
    in_win = (diff > -win[0]) & (diff < win[1])

    if dist_metric.startswith('cos') and t.shape[0] > 1:
        t_norm = np.sqrt(np.sum(t ** 2, axis=0))
        r_norm = np.sqrt(np.sum(r ** 2, axis=0))
        d = 1.0 - (t.T @ r) / np.outer(t_norm, r_norm)
    else:
        d = np.sum((t.T[:, None, :] - r.T[None, :, :]) ** 2, axis=2)
    d = np.where(in_win, d, np.inf)

    # Accumulated cost matrix.
    D = np.empty((N, M))
    D[0, 0] = d[0, 0]
    for n in range(1, N):
        D[n, 0] = d[n, 0] + D[n - 1, 0]
    for m in range(1, M):
        D[0, m] = d[0, m] + D[0, m - 1]
    for n in range(1, N):
        Dn = D[n]
        Dp = D[n - 1]
        dn = d[n]
        for m in range(1, M):
            Dn[m] = dn[m] + min(Dp[m], Dp[m - 1], Dn[m - 1])

    dist = D[N - 1, M - 1]

    # Backtrack the optimal path. Ties are broken in the same order as the
    # MATLAB code: step in t, then step in r, then diagonal.
    n = N - 1
    m = M - 1
    path = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            m -= 1
        elif m == 0:
            n -= 1
        else:
            choice = int(np.argmin([D[n - 1, m], D[n, m - 1], D[n - 1, m - 1]]))
            if choice == 0:
                n -= 1
            elif choice == 1:
                m -= 1
            else:
                n -= 1
                m -= 1
        path.append((n, m))
    path.reverse()
    w = np.array(path, dtype=int)

    return dist, w[:, 0], w[:, 1]


def dtw_cons_md(t, r, win=np.inf, dist_metric='eu'):
    """Constrained multi-dimensional Dynamic Time Warping distance.

    See :func:`dtw_cons_md_path` for the parameters. Only the distance is
    returned.
    """
    return dtw_cons_md_path(t, r, win, dist_metric)[0]

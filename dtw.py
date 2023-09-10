import numpy as np

def dtw_cons_md(t, r, win=np.inf, dist_metric='eu'):
    N = t.shape[1]  # If single source/dest
    M = r.shape[1]  # If single source/dest

    if np.isscalar(win):
        win = np.ones((2, 1)) * win

    Dist = np.inf
    d = np.full((N, M), np.inf)
    w1 = 0
    w2 = 0

    if dist_metric.startswith('cos') and t.shape[0] > 1:
        t_norm = np.sqrt(np.sum(np.square(t), axis=0))
        r_norm = np.sqrt(np.sum(np.square(r), axis=0))
        for n in range(N):
            for m in range(M):
                if (-win[0] < (n - m) < win[1]):
                    d[n, m] = 1 - np.dot(t[:, n], r[:, m]) / (t_norm[n] * r_norm[m])
    else:
        for n in range(N):
            for m in range(M):
                if (-win[0] < (n - m) < win[1]):
                    d[n, m] = np.sum((t[:, n] - r[:, m]) ** 2)

    
    D = np.zeros(d.shape)
    D[0,0] = d[0,0]

    for n in range(1, N):
        D[n, 0] = d[n, 0] + D[n-1, 0]

    for m in range(1, M):
        D[0, m] = d[0, m] + D[0, m-1]

    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = d[n, m] + min(D[n-1, m], D[n-1, m-1], D[n, m-1])

    Dist = D[N-1, M-1]


    n = N
    m = M
    k = N + M
    w = np.zeros((N+M, 2), dtype=int)
    w[N+M-1, :] = [N, M]

    while (n+m) != 2:
        if (n-1) == 0:
            m = m-1
        elif (m-1) == 0:
            n = n-1
        else:
            number = np.argmin([D[n-2, m-1], D[n-1, m-2], D[n-2, m-2]]) + 1
            if number == 1:
                n = n-1
            elif number == 2:
                m = m-1
            elif number == 3:
                n = n-1
                m = m-1

        k = k-1
        w[k-1, :] = [n, m]

    w = w[k:, :]
    w1, w2 = w[:, 0], w[:, 1]

    return Dist, w1, w2




   

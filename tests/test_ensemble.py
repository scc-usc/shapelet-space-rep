import os

import numpy as np
import pytest

from shapelet_space import dtw, ensemble

DATA = os.path.join(os.path.dirname(__file__), '..', 'data', 'flu_sims.csv')


def gaussian_bump(T, center, height=1.0, width=3.0):
    t = np.arange(T, dtype=float)
    return height * np.exp(-0.5 * ((t - center) / width) ** 2)


def shifted_bumps(n=6, T=40):
    """Identical bumps that peak at different times."""
    return np.array([gaussian_bump(T, 15 + 2 * k) for k in range(n)])


class TestDTW:
    def test_identical_series_give_diagonal_path(self):
        x = np.random.default_rng(0).random((2, 9))
        dist, w1, w2 = dtw.dtw_cons_md_path(x, x)
        assert dist == 0
        np.testing.assert_array_equal(w1, np.arange(9))
        np.testing.assert_array_equal(w2, np.arange(9))

    def test_path_cost_equals_distance(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            a = rng.random((3, rng.integers(2, 12)))
            b = rng.random((3, rng.integers(2, 12)))
            dist, w1, w2 = dtw.dtw_cons_md_path(a, b, win=4)
            if not np.isfinite(dist):
                continue
            cost = sum(np.sum((a[:, i] - b[:, j]) ** 2) for i, j in zip(w1, w2))
            assert np.isclose(cost, dist)
            assert (w1[0], w2[0]) == (0, 0)
            assert (w1[-1], w2[-1]) == (a.shape[1] - 1, b.shape[1] - 1)
            assert np.all(np.diff(w1) >= 0) and np.all(np.diff(w1) <= 1)
            assert np.all(np.diff(w2) >= 0) and np.all(np.diff(w2) <= 1)
            assert np.all(np.diff(w1) + np.diff(w2) >= 1)

    def test_one_dimensional_input(self):
        a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        b = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        assert dtw.dtw_cons_md(a, b) == 0
        assert dtw.dtw_cons_md(a, b) == dtw.dtw_cons_md_path(a, b)[0]

    def test_window_restricts_matches(self):
        a = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        assert dtw.dtw_cons_md(a, b) == 0
        assert dtw.dtw_cons_md(a, b, win=1) == 2  # diagonal only


class TestShapeTransform:
    def test_shape_and_padding(self):
        dd = np.random.default_rng(2).random((3, 12))
        reps = ensemble.shape_ts_transform(dd, ensemble.DEFAULT_SHAPELETS, 0.1)
        assert reps.shape == (3, 12, 4)
        assert not np.isnan(reps).any()
        # w = 4: l = 1 point copied at the start, r = 2 points copied at the end.
        np.testing.assert_array_equal(reps[:, 0], reps[:, 1])
        np.testing.assert_array_equal(reps[:, 10], reps[:, 9])
        np.testing.assert_array_equal(reps[:, 11], reps[:, 9])

    def test_increasing_segment_scores_increase_shapelet(self):
        dd = np.arange(10, dtype=float)[None, :] * 10
        reps = ensemble.shape_ts_transform(dd, ensemble.DEFAULT_SHAPELETS, 0.1)
        assert reps[0, 4].argmax() == 0  # 'Inc' shapelet

    def test_flat_segment_scores_flat_shapelet(self):
        dd = np.ones((1, 10))
        reps = ensemble.shape_ts_transform(dd, ensemble.DEFAULT_SHAPELETS, 0.1)
        assert reps[0, 4].argmax() == 3  # all-zero 'Flat' shapelet
        assert reps[0, 4, 3] == pytest.approx(1.0)

    def test_estimate_slope_threshold(self):
        dd = np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        # per-series thresholds: 1, 2, 0 -> median 1
        assert ensemble.estimate_slope_threshold(dd, 4) == pytest.approx(1.0)
        assert ensemble.estimate_slope_threshold(np.ones((2, 5)), 4) > 0


class TestMatchings:
    def test_matchings_are_valid_alignments(self):
        dd = shifted_bumps()
        reps = ensemble.shape_ts_transform(dd, ensemble.DEFAULT_SHAPELETS, 0.05)
        matches = ensemble.find_dtw_matchings(reps, base_m=2)
        ns, T = dd.shape
        assert matches.shape == (ns, T)
        np.testing.assert_array_equal(matches[2], np.arange(T))
        assert np.all(matches[:, 0] == 0)
        assert np.all(matches[:, -1] == T - 1)
        assert np.all(np.diff(matches, axis=1) >= 0)

    def test_identical_series_align_to_identity(self):
        dd = np.tile(gaussian_bump(30, 12), (4, 1))
        matches = ensemble.find_dtw_matchings(dd)
        np.testing.assert_array_equal(matches, np.tile(np.arange(30), (4, 1)))
        mean_curve, mean_time, all_values, all_times = ensemble.dtw_mean_ensemble(dd, matches)
        np.testing.assert_allclose(mean_curve, dd[0])
        np.testing.assert_allclose(mean_time, np.arange(30))
        assert all_values.shape == (4, 30)
        assert all_times.shape == (4, 30)

    def test_mean_ensemble_indexes_each_series(self):
        dd = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
        matches = np.array([[0, 1, 2], [0, 0, 2]])
        mean_curve, mean_time, all_values, _ = ensemble.dtw_mean_ensemble(dd, matches)
        np.testing.assert_allclose(all_values, [[1, 2, 3], [10, 10, 30]])
        np.testing.assert_allclose(mean_curve, [5.5, 6, 16.5])
        np.testing.assert_allclose(mean_time, [0, 0.5, 2])

    def test_mean_ensemble_rejects_bad_row_count(self):
        with pytest.raises(ValueError):
            ensemble.dtw_mean_ensemble(np.zeros((3, 4)), np.zeros((4, 4), dtype=int))


class TestEnsemble:
    def test_shifted_bumps_keep_peak_height(self):
        dd = shifted_bumps()
        res = ensemble.dtw_s_ensemble(dd)
        # The point-wise mean flattens the peak, the DTW+S consensus keeps it.
        assert dd.mean(axis=0).max() < 0.8
        assert res.mean_curve.max() == pytest.approx(1.0, abs=0.05)
        assert res.consensus.max() == pytest.approx(1.0, abs=0.05)
        assert res.matches.shape == dd.shape
        assert res.mean_time.shape == (dd.shape[1],)
        assert np.all(np.diff(res.mean_time) >= 0)

    def test_fixed_base_reference(self):
        dd = shifted_bumps()
        res = ensemble.dtw_s_ensemble(dd, base=1)
        np.testing.assert_array_equal(res.reference, dd[1])
        np.testing.assert_array_equal(res.matches[1], np.arange(dd.shape[1]))
        assert res.all_refs.shape == (0, dd.shape[1])

    def test_plain_dtw_without_shapelets(self):
        dd = shifted_bumps()
        res = ensemble.dtw_s_ensemble(dd, shapelet_array=None, num_iter=3)
        assert res.reps.shape == dd.shape + (1,)
        assert res.consensus.max() == pytest.approx(1.0, abs=0.05)

    def test_dba_iterations_are_recorded(self):
        dd = shifted_bumps()
        res0 = ensemble.dtw_s_ensemble(dd, num_iter=0)
        assert res0.all_refs.shape == (1, dd.shape[1])
        res = ensemble.dtw_s_ensemble(dd, num_iter=5)
        assert 1 <= res.all_refs.shape[0] <= 6

    def test_input_validation(self):
        with pytest.raises(ValueError):
            ensemble.dtw_s_ensemble(np.zeros((1, 10)))
        dd = shifted_bumps()
        dd[0, 0] = np.nan
        with pytest.raises(ValueError):
            ensemble.dtw_s_ensemble(dd)

    def test_flu_sims(self):
        dd = np.loadtxt(DATA, delimiter=',')
        res = ensemble.dtw_s_ensemble(dd, win=30, num_iter=20)
        assert res.consensus.shape == (dd.shape[1],)
        assert np.all(res.consensus >= 0)
        # The consensus peak is much closer to the typical individual peak
        # than the point-wise mean is.
        median_peak = np.median(dd.max(axis=1))
        assert abs(res.consensus.max() - median_peak) < abs(dd.mean(axis=0).max() - median_peak)


class TestEnvelopes:
    def test_covering_lengths(self):
        x = np.arange(100, dtype=float)
        y = 2 * x
        x_norm, y_norm = ensemble.calculate_covering_lengths(x, y)
        assert x_norm == pytest.approx(95)
        assert y_norm == pytest.approx(190)
        assert ensemble.calculate_covering_lengths([1.0, 2.0], [1.0, 2.0]) == (0.0, 0.0)
        with pytest.raises(ValueError):
            ensemble.calculate_covering_lengths([1.0, 2.0], [1.0])

    def test_convex_hull_degen_confidence_keeps_closest_points(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(200, 2))
        far = np.array([[50.0, 50.0], [-50.0, -50.0]])
        x = np.concatenate([pts[:, 0], far[:, 0]])
        y = np.concatenate([pts[:, 1], far[:, 1]])
        hull = ensemble.convex_hull_degen_confidence(x, y, 1.0, 1.0, 0.0, 0.0, 0.9)
        assert hull.shape[1] == 2
        assert np.abs(hull).max() < 10  # the far points were dropped
        # Full confidence keeps every point, so the far points are vertices.
        hull_all = ensemble.convex_hull_degen_confidence(x, y, 1.0, 1.0, 0.0, 0.0, 1.0)
        assert np.abs(hull_all).max() == 50
        # 2-D selection (x_norm = 0) also returns a hull.
        hull_2d = ensemble.convex_hull_degen_confidence(x, y, 0.0, 0.0, 0.0, 0.0, 0.5)
        assert hull_2d.shape[1] == 2

    def test_convex_hull_degenerate_points(self):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.zeros(4)
        hull = ensemble.convex_hull_degen_confidence(x, y, 1.0, 1.0, 1.5, 0.0, 1.0)
        assert hull.shape == (4, 2)

    def test_generate_midpoints(self):
        pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        border = ensemble.generate_midpoints(pts, 3)
        assert border.shape == (12, 2)
        np.testing.assert_allclose(border[:4], [[0, 0], [0.25, 0], [0.5, 0], [0.75, 0]])
        # The closing edge from the last to the first vertex is included.
        np.testing.assert_allclose(border[8], [1, 1])
        np.testing.assert_allclose(border[11], [0.25, 0.25])
        np.testing.assert_allclose(ensemble.generate_midpoints(np.array([0.0, 2.0]), 1), [0, 1, 2, 1])

    def test_is_intersect_with_sides(self):
        profile = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 5.0], [3.0, 0.0]])
        p2 = (0.0, 0.0)
        p1 = (3.0, 0.0)
        assert ensemble.is_intersect_with_sides(p1, p2, profile, 0, 3, 'min')
        assert ensemble.is_intersect_with_sides(p1, p2, profile, 0, 3, 'max')
        flat = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert not ensemble.is_intersect_with_sides(p1, p2, flat, 0, 3, 'min')
        assert not ensemble.is_intersect_with_sides(p1, p2, flat, 0, 3, 'max')

    def test_envelopes_bracket_the_series(self):
        dd = shifted_bumps()
        res = ensemble.dtw_s_ensemble(dd)
        for method in ('none', 'strict'):
            env = ensemble.find_envelopes(dd, res.matches, 1.0, method=method, jitter=0.0)
            assert env.lower.shape == (dd.shape[1],)
            np.testing.assert_array_equal(env.lower_time, np.arange(dd.shape[1]))
            assert np.all(env.upper >= env.lower - 1e-9)
            # Without jitter and at full confidence the band covers the consensus.
            inner = slice(2, -2)
            assert np.all(env.upper[inner] >= res.consensus[inner] - 1e-6)
            assert np.all(env.lower[inner] <= res.consensus[inner] + 1e-6)
        combo = ensemble.find_envelopes(dd, res.matches, 0.9, method='combo', seed=0)
        assert combo.lower_time.size >= dd.shape[1]
        assert np.all(np.diff(combo.lower_time) >= 0)
        with pytest.raises(ValueError):
            ensemble.find_envelopes(dd, res.matches, 0.9, method='bogus')

    def test_wider_band_for_higher_confidence(self):
        dd = np.loadtxt(DATA, delimiter=',')
        res = ensemble.dtw_s_ensemble(dd, win=10, num_iter=5)
        env50 = ensemble.find_envelopes(dd, res.matches, 0.5, seed=1)
        env95 = ensemble.find_envelopes(dd, res.matches, 0.95, seed=1)
        assert np.mean(env95.upper - env95.lower) > np.mean(env50.upper - env50.lower)
        assert env95.upper.max() > res.consensus.max()
        assert len(env95.hulls) == dd.shape[1]

    def test_envelope_quantiles(self):
        dd = np.loadtxt(DATA, delimiter=',')
        res = ensemble.dtw_s_ensemble(dd, win=10, num_iter=5)
        qs = (0.025, 0.25, 0.5, 0.75, 0.975)
        Q = ensemble.envelope_quantiles(dd, res.matches, qs, seed=0)
        assert Q.shape == (5, dd.shape[1])
        np.testing.assert_allclose(Q[2], res.consensus)
        inner = slice(2, -2)
        assert np.all(Q[4][inner] >= Q[3][inner])
        assert np.all(Q[3][inner] >= Q[2][inner])
        assert np.all(Q[2][inner] >= Q[1][inner])
        Q2 = ensemble.envelope_quantiles(dd, res.matches, qs, seed=0)
        np.testing.assert_array_equal(Q, Q2)  # the seed makes the result repeatable
        Qc = ensemble.envelope_quantiles(dd, res.matches, qs, method='combo', seed=0)
        assert Qc.shape == Q.shape
        with pytest.raises(ValueError):
            ensemble.envelope_quantiles(dd, res.matches, (1.5,))

    def test_get_dbas_ensemble(self):
        dd = np.loadtxt(DATA, delimiter=',')
        np.testing.assert_allclose(ensemble.quantile_levels((0.05, 0.1, 0.25)),
                                   [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
        with pytest.raises(ValueError):
            ensemble.quantile_levels((0.6,))
        Q, res = ensemble.get_dbas_ensemble(dd, (0.05, 0.25), win=10, seed=0, return_result=True)
        assert Q.shape == (5, dd.shape[1])
        np.testing.assert_allclose(Q[2], res.reference)
        assert res.all_refs.shape[0] <= 11
        # The median keeps the peak and the 95% band is wide around it.
        assert Q[2].max() > 1.3 * dd.mean(axis=0).max()
        assert Q[4].max() > Q[2].max() > Q[0].max()
        Q_only = ensemble.get_dbas_ensemble(dd, (0.05, 0.25), win=10, seed=0)
        np.testing.assert_allclose(Q_only, Q)

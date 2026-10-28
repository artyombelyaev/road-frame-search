import numpy as np
import pytest

from src import metrics


def test_precision_at_k():
    rel = np.array([1, 0, 1, 1, 0])
    assert metrics.precision_at_k(rel, 1) == 1.0
    assert metrics.precision_at_k(rel, 4) == 0.75


def test_r_precision():
    # три релевантных, в топ-3 два из них
    assert metrics.r_precision(np.array([1, 0, 1, 1, 0])) == pytest.approx(2 / 3)
    assert metrics.r_precision(np.array([0, 0, 0])) == 0.0


def test_average_precision_by_hand():
    # релевантные на позициях 1, 3, 4: (1/1 + 2/3 + 3/4) / 3
    assert metrics.average_precision(np.array([1, 0, 1, 1, 0])) == pytest.approx((1 + 2 / 3 + 3 / 4) / 3)


def test_average_precision_extremes():
    assert metrics.average_precision(np.array([1, 1, 0, 0])) == 1.0
    assert metrics.average_precision(np.array([0, 0, 0])) == 0.0
    # единственный релевантный последним из N - AP = 1/N
    assert metrics.average_precision(np.array([0, 0, 0, 1])) == pytest.approx(0.25)


def test_random_ap_single_relevant_is_mean_reciprocal_rank():
    n = 7
    assert metrics.random_average_precision(1, n) == pytest.approx(np.mean(1 / np.arange(1, n + 1)))


def test_random_ap_all_relevant():
    assert metrics.random_average_precision(5, 5) == pytest.approx(1.0)


@pytest.mark.parametrize("n_rel, n", [(3, 20), (10, 200), (50, 1000)])
def test_random_ap_matches_simulation(n_rel, n):
    rng = np.random.default_rng(0)
    rel = np.zeros(n)
    rel[:n_rel] = 1
    sims = [metrics.average_precision(rng.permutation(rel)) for _ in range(20_000)]
    assert np.mean(sims) == pytest.approx(metrics.random_average_precision(n_rel, n), rel=0.01)


def test_random_precision_matches_simulation():
    rng = np.random.default_rng(0)
    rel = np.zeros(500)
    rel[:60] = 1
    sims = [metrics.precision_at_k(rng.permutation(rel), 50) for _ in range(5_000)]
    assert np.mean(sims) == pytest.approx(metrics.random_precision(60, 500), rel=0.02)

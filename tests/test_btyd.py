import numpy as np
import pytest

from clv.btyd import BGNBD, GammaGamma


def simulate_bgnbd(n, r, alpha, a, b, T, seed=0):
    """Simulate the BG/NBD story directly: Poisson purchases, dropout after each one."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    x = np.zeros(n)
    tx = np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T:
                break
            x[i] += 1
            tx[i] = t
            if rng.random() < p[i]:
                break
    return x, tx, np.full(n, T)


def test_bgnbd_recovers_simulated_parameters():
    x, tx, T = simulate_bgnbd(6000, r=0.6, alpha=4.0, a=0.8, b=2.5, T=40)
    m = BGNBD().fit(x, tx, T)
    assert m.r == pytest.approx(0.6, rel=0.2)
    assert m.alpha == pytest.approx(4.0, rel=0.3)
    assert m.a / (m.a + m.b) == pytest.approx(0.8 / 3.3, rel=0.25)  # mean dropout probability


def test_bgnbd_predictions_behave():
    m = BGNBD(r=0.5, alpha=5, a=0.7, b=2.5)
    # a customer who never repeated is certainly alive in the model
    assert m.p_alive(0, 0, 30) == pytest.approx(1.0)
    # same frequency, but a long silence since the last purchase lowers P(alive)
    assert m.p_alive(5, 29, 30) > m.p_alive(5, 5, 30)
    # more expected purchases for an active frequent buyer than for a lapsed one
    assert m.expected_purchases(26, 10, 29, 30) > m.expected_purchases(26, 10, 5, 30)
    # expected purchases grow with the horizon
    assert m.expected_purchases(52, 3, 20, 30) > m.expected_purchases(26, 3, 20, 30)


def test_gamma_gamma_shrinks_toward_population_mean():
    rng = np.random.default_rng(1)
    n = 4000
    nu = rng.gamma(3.0, 1 / 15.0, n)                 # customer-level rate
    x = rng.integers(1, 15, n)
    m = np.array([rng.gamma(6.0 * k, 1 / nu_i, 1)[0] / k for k, nu_i in zip(x, nu)])
    gg = GammaGamma().fit(x, m)
    pop = gg.p * gg.v / (gg.q - 1)
    one, many = gg.expected_value([1, 14], [3 * pop, 3 * pop])
    # a high average from one purchase is trusted less than the same average from fourteen
    assert pop < one < many < 3 * pop


def test_conditional_expectation_matches_simulation():
    """Simulate calibration + holdout; true parameters should predict holdout purchases."""
    rng = np.random.default_rng(3)
    r, alpha, a, b, T, t = 0.7, 3.0, 0.9, 3.0, 30.0, 26.0
    n = 20000
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    x, tx, hold = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        s, alive = 0.0, True
        while alive:
            s += rng.exponential(1 / lam[i])
            if s > T + t:
                break
            if s <= T:
                x[i] += 1
                tx[i] = s
            else:
                hold[i] += 1
            alive = rng.random() >= p[i]
    m = BGNBD(r=r, alpha=alpha, a=a, b=b)
    pred = m.expected_purchases(t, x, tx, np.full(n, T))
    assert pred.sum() == pytest.approx(hold.sum(), rel=0.05)
    # and per customer history: frequent recent buyers buy more
    hot = (x >= 5) & (tx > T - 5)
    assert pred[hot].mean() == pytest.approx(hold[hot].mean(), rel=0.1)

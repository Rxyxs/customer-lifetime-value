"""BG/NBD and Gamma-Gamma models, fitted by maximum likelihood with scipy.

BG/NBD (Fader, Hardie & Lee, 2005): while alive, a customer buys at a Poisson
rate lambda ~ Gamma(r, alpha); after each purchase they drop out with
probability p ~ Beta(a, b). Per customer it needs only
  x   = repeat purchases (days with a purchase after the first one),
  t_x = time of the last purchase, measured from the first,
  T   = time from the first purchase to the end of the calibration period.

Gamma-Gamma (Fader, Hardie & Lee, 2005b): the average value of a customer's
transactions is independent of how often they buy, and is Gamma-distributed
around a customer-level mean that is itself Gamma across customers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1


@dataclass
class BGNBD:
    r: float = 1.0
    alpha: float = 1.0
    a: float = 1.0
    b: float = 1.0

    @staticmethod
    def _ll(params, x, tx, T):
        r, alpha, a, b = np.exp(params)
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            a4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-12)) - (r + x) * np.log(alpha + tx),
                          -np.inf)
        return a1 + a2 + np.logaddexp(a3, a4)

    def fit(self, x, tx, T, penalizer: float = 0.0):
        x, tx, T = (np.asarray(v, dtype=float) for v in (x, tx, T))

        def nll(p):
            return -self._ll(p, x, tx, T).sum() + penalizer * np.sum(np.exp(p) ** 2)

        best = None
        for start in ([0.0, 0.0, 0.0, 0.0], [-1.0, 1.0, -1.0, 1.0], [0.5, 2.0, 0.0, 1.0]):
            res = minimize(nll, np.array(start), method="Nelder-Mead",
                           options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
            if best is None or res.fun < best.fun:
                best = res
        self.r, self.alpha, self.a, self.b = np.exp(best.x)
        self.loglik_ = -best.fun
        return self

    def _odds_dead(self, x, tx, T):
        with np.errstate(divide="ignore", invalid="ignore"):
            d = self.a / (self.b + x - 1) * ((self.alpha + T) / (self.alpha + tx)) ** (self.r + x)
        return np.where(x > 0, d, 0.0)

    def p_alive(self, x, tx, T):
        x, tx, T = (np.asarray(v, dtype=float) for v in (x, tx, T))
        return 1.0 / (1.0 + self._odds_dead(x, tx, T))

    def expected_purchases(self, t, x, tx, T):
        """Expected purchases in (T, T + t] for a customer with history (x, tx, T)."""
        x, tx, T = (np.asarray(v, dtype=float) for v in (x, tx, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        z = t / (alpha + T + t)
        num = 1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp2f1(r + x, b + x, a + b + x - 1, z)
        return (a + b + x - 1) / (a - 1) * num / (1 + self._odds_dead(x, tx, T))

    def expected_purchases_new(self, t):
        """Expected purchases in (0, t] for a newly acquired customer."""
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        z = t / (alpha + t)
        return (a + b - 1) / (a - 1) * (1 - (alpha / (alpha + t)) ** r * hyp2f1(r, b, a + b - 1, z))


@dataclass
class GammaGamma:
    p: float = 1.0
    q: float = 1.0
    v: float = 1.0

    @staticmethod
    def _ll(params, x, m):
        p, q, v = np.exp(params)
        return (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v)
                + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(v + m * x))

    def fit(self, x, m):
        """x: number of transactions behind the mean, m: mean transaction value (x > 0)."""
        x, m = np.asarray(x, dtype=float), np.asarray(m, dtype=float)
        res = minimize(lambda p: -self._ll(p, x, m).sum(), np.array([0.0, 1.0, 1.0]), method="Nelder-Mead",
                       options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
        self.p, self.q, self.v = np.exp(res.x)
        return self

    def expected_value(self, x, m):
        x, m = np.asarray(x, dtype=float), np.asarray(m, dtype=float)
        population = self.p * self.v / (self.q - 1)
        shrunk = self.p * (self.v + x * m) / (self.p * x + self.q - 1)
        return np.where(x > 0, shrunk, population)

"""Numerical regression checks for the educational drift integration."""
import unittest
import numpy as np
from scipy.integrate import quad
from src.diagnostics import alpha, sample_proposal
from src.drift_quadrature import (H, P, log_proposal, log_acceptance,
                                  proposal_mean, integrate_drift)


class DriftQuadratureTests(unittest.TestCase):
    def test_kernel_against_density_ratio_and_sampler(self):
        for x in (1., 6., 12., 16., 19., 21.):
            y = np.r_[np.linspace(-30, 30, 201), x, -x,
                      proposal_mean(x) + H*np.linspace(-5, 5, 51)]
            direct = np.minimum(np.longdouble(x)**4 - np.asarray(y, dtype=np.longdouble)**4
                                + log_proposal(y, x) - log_proposal(x, y), 0)
            np.testing.assert_allclose(log_acceptance(x, y), direct, rtol=2e-12, atol=1e-9)
            visible = log_acceptance(x, y) > -600
            np.testing.assert_allclose(np.exp(log_acceptance(x, y[visible])),
                                       alpha(x, y[visible], H, P), rtol=2e-9, atol=1e-14)
            rng1, rng2 = np.random.default_rng(17), np.random.default_rng(17)
            np.testing.assert_allclose(sample_proposal(x, H, P, rng1, 100),
                                       proposal_mean(x) + H*rng2.normal(size=100))

    def test_easy_points_against_unscaled_quadrature(self):
        for x in (1., 6., 12.):
            result = integrate_drift(x)
            lo, hi = proposal_mean(x)-12*H, proposal_mean(x)+12*H
            points = sorted([v for v in (-x, x) if lo < v < hi] + list(np.linspace(lo, hi, 121)[1:-1]))
            def density(y):
                return np.exp(float(log_proposal(x, y))) * alpha(x, y, H, P)
            a, _ = quad(density, lo, hi, points=points, epsabs=1e-12, epsrel=1e-11, limit=400)
            d, _ = quad(lambda y: (y*y-x*x)*density(y), lo, hi, points=points, epsabs=1e-12, epsrel=1e-11, limit=400)
            self.assertAlmostEqual(np.exp(result['acceptance'].log_value), a, places=9)
            self.assertAlmostEqual(result['sign']*np.exp(result['log_abs_drift']), d, places=8)

    def test_tail_stability_and_nonzero_logarithms(self):
        for x in (16., 21.):
            a = integrate_drift(x)
            b = integrate_drift(x, rtol=1e-10, padding=8, drop=60, extra_splits=True)
            self.assertLess(a['acceptance'].log_value, -745)
            self.assertTrue(a['sign_resolved'])
            self.assertEqual(a['sign'], -1)
            for key in ('acceptance', 'inward', 'outward'):
                self.assertTrue(np.isfinite(a[key].log_value))
                self.assertLess(abs(a[key].log_value-b[key].log_value), 1e-7)
                self.assertLess(a[key].relative_error_indicator, 1e-7)
                self.assertFalse(a[key].warnings)


if __name__ == '__main__':
    unittest.main()

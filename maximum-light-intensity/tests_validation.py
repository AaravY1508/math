"""Meaningful independent checks of interval containment and global certificates."""
from decimal import Decimal, localcontext
import unittest

import numpy as np
from validation import _box_bound, intensity, intensity_interval, validate


class ValidationTests(unittest.TestCase):
    def assertContains(self, result, objective, tol=1e-7):
        self.assertTrue(result['valid'])
        self.assertTrue(result['certified'], result)
        self.assertLessEqual(result['objective_lower'], objective)
        self.assertGreaterEqual(result['objective_upper'], objective)
        self.assertLessEqual(result['gap'], tol)

    def test_one_light_analytic_positions(self):
        for p in ([.5, .5], [0., 0.], [.1, .7], [1., .5]):
            # For one light, the weakest point is its farthest square corner.
            expected = 1 / (max(p[0], 1-p[0])**2 + max(p[1], 1-p[1])**2)
            r = validate([p], numerical=False, tolerance=1e-7)
            self.assertContains(r, expected)

    def test_two_lights_analytic(self):
        r = validate([[.25, .5], [.75, .5]], numerical=False, tolerance=1e-7)
        self.assertContains(r, 16/5 + 16/13)

    def test_coincident_lights_are_well_defined(self):
        r = validate([[.5, .5]] * 36, numerical=False, tolerance=1e-7)
        self.assertContains(r, 72.)
        self.assertEqual(intensity_interval([[.5, .5]], [.5, .5]), [float('inf')]*2)

    def test_geometry_rejections(self):
        for p in ([], [[-.01, .5]], [[.5, 1.01]], [[np.nan, .5]], [[.1, .2, .3]]):
            r = validate(p)
            self.assertFalse(r['valid'])
            self.assertFalse(r['certified'])

    def test_point_intervals_against_decimal(self):
        rng = np.random.default_rng(174983)
        with localcontext() as ctx:
            ctx.prec = 80
            for n in (1, 2, 36, 50):
                for _ in range(25):
                    points = rng.uniform(0, 1, (n, 2))
                    q = rng.uniform(0, 1, 2)
                    qd = [Decimal.from_float(x) for x in q]
                    exact = sum(Decimal(1) / sum((qd[k]-Decimal.from_float(p[k]))**2 for k in range(2)) for p in points)
                    lo, hi = intensity_interval(points, q)
                    self.assertLessEqual(Decimal.from_float(lo), exact)
                    self.assertGreaterEqual(Decimal.from_float(hi), exact)

    def test_rectangle_bounds_against_random_samples(self):
        # Exercise singular rectangles, boundary rectangles, and tiny rectangles
        # where cancellation and cubic Taylor terms matter.
        rng = np.random.default_rng(728436)
        for n in (1, 2, 36, 50):
            points = rng.uniform(0, 1, (n, 2))
            for width in (1., .2, .01, 1e-5):
                for _ in range(10):
                    lower = rng.uniform(0, 1-width, 2)
                    upper = lower + width
                    box = (lower[0], upper[0], lower[1], upper[1])
                    bound, ub, center = _box_bound(points, box)
                    corners = [[x, y] for x in (lower[0], upper[0]) for y in (lower[1], upper[1])]
                    samples = np.vstack((corners, center, rng.uniform(lower, upper, (30, 2))))
                    for q in samples:
                        self.assertLessEqual(bound, intensity(points, q) * (1+1e-14))
                    self.assertGreaterEqual(ub, intensity(points, center) * (1-1e-14))

    def test_near_poles_and_ulp_scale_rectangles(self):
        # Spacing around 0.5 is about 1e-16; all boxes below remain representable.
        p = np.array([[.5, .5], [np.nextafter(.5, 1.), .5], [0., 0.]])
        for width in (2.**-12, 2.**-35, 2.**-50):
            for offset in (0., 2.*width, -3.*width):
                x = .5 + offset
                box = (x-width, x+width, .5-width, .5+width)
                lb, ub, q = _box_bound(p, box)
                self.assertFalse(np.isnan(lb))
                self.assertFalse(np.isnan(ub))
                for t in (0., .125, .5, .75, 1.):
                    for u in (0., .3, 1.):
                        site = [box[0]+t*(box[1]-box[0]), box[2]+u*(box[3]-box[2])]
                        self.assertLessEqual(lb, intensity(p, site)*(1+1e-14))
        # Underflow in a squared distance must widen the enclosure, never claim
        # a finite upper certificate at a near-pole point.
        q = [np.nextafter(0., 1.), 0.]
        lo, hi = intensity_interval([[0., 0.]], q)
        self.assertEqual(lo, 0.)
        self.assertEqual(hi, float('inf'))

    def test_third_derivative_spectral_constant(self):
        for theta in np.linspace(0, 2*np.pi, 501):
            ux, uy = np.cos(theta), np.sin(theta)
            derivative = 8*np.array([[-3*ux, uy], [uy, ux]])
            self.assertLessEqual(np.max(np.abs(np.linalg.eigvalsh(derivative))), 24*(1+1e-15))

    def test_grid36_global_certificate(self):
        p = np.array([[(i+.5)/6, (j+.5)/6] for i in range(6) for j in range(6)])
        r = validate(p, numerical=False, tolerance=1e-7)
        self.assertContains(r, intensity(p, [0., 0.]))

    def test_time_limit_keeps_true_lower_bound(self):
        p = np.random.default_rng(4).uniform(0, 1, (36, 2))
        r = validate(p, numerical=False, tolerance=1e-10, time_limit=0)
        self.assertTrue(r['valid'])
        self.assertFalse(r['certified'])
        self.assertEqual(r['status'], 'time_limit')
        self.assertLessEqual(r['objective_lower'], min(intensity(p, q) for q in [[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]]))
        self.assertGreaterEqual(r['objective_upper'], r['objective_lower'])


if __name__ == '__main__':
    unittest.main()

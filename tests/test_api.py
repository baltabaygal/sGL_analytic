"""Public input contracts and density conversion, without expensive solves."""
import unittest
from unittest.mock import patch

import numpy as np

from sgl_analytic import generate_pdf, generate_pdf_lnmu
from sgl_analytic.pipeline import make_cosmology, on


class PublicAPI(unittest.TestCase):
    def test_invalid_configuration(self):
        for cfg in ("", "garbage", "+typo", "-typo"):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                on(cfg)

    def test_invalid_grid_fails_before_solver(self):
        with patch("sgl_analytic.api.run") as solver:
            for grid in ([0, 0], [1, 0], [0, np.nan], [[0, 1]], [0]):
                with self.subTest(grid=grid), self.assertRaises(ValueError):
                    generate_pdf_lnmu(1, lnmu=grid)
            with self.assertRaises(ValueError):
                generate_pdf(1, mu=[0, 1])
            solver.assert_not_called()

    def test_cosmology_validation(self):
        for cp in (dict(h=0), dict(Om=np.nan), dict(Ob=0.4),
                   dict(zeq=-1), dict(sigma8=.8, s8=.8)):
            with self.subTest(cp=cp), self.assertRaises(ValueError):
                make_cosmology(cp)

    def test_parameter_forwarding_and_jacobian(self):
        def fake_run(zs, cfg, **kw):
            self.assertEqual(zs, 2)
            self.assertEqual(kw["cosmo"], dict(h=.7, Om=.3, sigma8=.9,
                                              Ob=.05, ns=.97, zeq=3450))
            x = kw["xi_out"]
            return dict(xi=x.tolist(), P_s=np.ones_like(x).tolist(),
                        grid=dict(xi_min=-.5, xi_max=2)), {}
        mu = np.array([.5, 1, 2])
        with patch("sgl_analytic.api.run", side_effect=fake_run):
            x, p, info = generate_pdf(2, .7, .3, .9, .05, .97, 3450,
                                      mu=mu, return_info=True)
        np.testing.assert_array_equal(x, mu)
        np.testing.assert_array_equal(p, 1 / mu)
        self.assertEqual(info["moment_interval"], [-.5, 2])

    def test_negative_residual_is_visible(self):
        output = dict(xi=[0, 1], P_s=[-.1, 1], grid=dict(xi_min=0, xi_max=1))
        with patch("sgl_analytic.api.run", return_value=(output, {})):
            with self.assertWarns(RuntimeWarning):
                _, p, info = generate_pdf_lnmu(1, lnmu=[0, 1], return_info=True)
        self.assertEqual(p[0], -.1)
        self.assertAlmostEqual(info["diagnostics"]["negative_mass"], .05)


if __name__ == "__main__":
    unittest.main()

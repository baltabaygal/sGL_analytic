"""Public input contracts and density conversion, without expensive solves."""
import unittest
import warnings
from unittest.mock import patch

import numpy as np

from sgl_analytic import generate_pdf, generate_pdf_lnmu
from sgl_analytic.pipeline import make_cosmology, on, run


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

    def test_cosmology_reaches_engine(self):
        with patch("sgl_analytic.pipeline.sgl.Cosmology") as engine:
            make_cosmology(dict(h=.7, Om=.3, sigma8=.9, Ob=.05, ns=.97, zeq=3450))
        engine.assert_called_once_with(window="smoothk", transfer="eh98", anchor="tophat",
                                       conc_model=16, h=.7, Om=.3, s8=.9,
                                       Ob=.05, ns=.97, zeq=3450, Ok=0.0, w0=-1.0,
                                       wa=0.0, growth_mode="auto")

    def test_invalid_redshift_and_workers_fail_before_build(self):
        with patch("sgl_analytic.pipeline.make_cosmology") as cosmology:
            for zs in (0, -1, np.nan, np.inf):
                with self.subTest(zs=zs), self.assertRaises(ValueError):
                    run(zs, "full")
            for nproc in (0, -1, 1.5, True):
                with self.subTest(nproc=nproc), self.assertRaises(ValueError):
                    run(1, "full", nproc=nproc)
            cosmology.assert_not_called()

    def test_parameter_forwarding_and_jacobian(self):
        def fake_run(zs, cfg, **kw):
            self.assertEqual(zs, 2)
            self.assertEqual(kw["cosmo"], dict(h=.7, Om=.3, sigma8=.9,
                                              Ob=.05, ns=.97, zeq=3450, Ok=0.0,
                                              w0=-1.0, wa=0.0, growth_mode="auto"))
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

    def test_extension_validation(self):
        for cp in (dict(Ok=np.nan), dict(Ok=True), dict(w0=0.0), dict(wa=np.inf),
                   dict(wa=.8), dict(growth_mode="exact"), dict(Ok=-1.5),
                   dict(w0=-.9, growth_mode="legacy")):
            with self.subTest(cp=cp), self.assertRaises(ValueError):
                make_cosmology(cp)
        c = make_cosmology(dict(Ok=-.05, w0=-1.1, wa=.2))
        self.assertEqual((c.Ok, c.w0, c.wa, c.growth_mode), (-.05, -1.1, .2, "ode"))
        self.assertAlmostEqual(c.OL, 1 - .315 + .05)

    def test_extensions_forwarded(self):
        def fake_run(zs, cfg, **kw):
            self.assertEqual({k: kw["cosmo"][k] for k in ("Ok", "w0", "wa", "growth_mode")},
                             dict(Ok=.05, w0=-.9, wa=.2, growth_mode="ode"))
            x = kw["xi_out"]
            return dict(xi=x.tolist(), P_s=np.ones_like(x).tolist(),
                        grid=dict(xi_min=0, xi_max=1)), {}
        with patch("sgl_analytic.api.run", side_effect=fake_run) as solver:
            generate_pdf(1, mu=[1, 2], Ok=.05, w0=-.9, wa=.2, growth_mode="ode")
        solver.assert_called_once()

    def test_auto_growth_warning(self):
        # stop right after the cosmology is built; only the warning is under test
        with patch("sgl_analytic.pipeline.F.build_population", side_effect=RuntimeError("stop")):
            for cosmo, warns in ((dict(), False), (dict(Ok=.05), False), (dict(w0=-.9), True),
                                 (dict(wa=.3), True), (dict(w0=-.9, growth_mode="ode"), False)):
                with self.subTest(cosmo=cosmo), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaises(RuntimeError):
                        run(1, "halo", cosmo=cosmo)
                    hit = any("growth_mode='auto'" in str(w.message) for w in caught)
                    self.assertEqual(hit, warns)

    def test_negative_residual_is_visible(self):
        output = dict(xi=[0, 1], P_s=[-.1, 1], grid=dict(xi_min=0, xi_max=1))
        with patch("sgl_analytic.api.run", return_value=(output, {})):
            with self.assertWarns(RuntimeWarning):
                _, p, info = generate_pdf_lnmu(1, lnmu=[0, 1], return_info=True)
        self.assertEqual(p[0], -.1)
        self.assertAlmostEqual(info["diagnostics"]["negative_mass"], .05)


if __name__ == "__main__":
    unittest.main()

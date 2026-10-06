"""Curvature and CPL dark energy against independent references (fast, no solves).

Ported from the stochastic_wl_analytic tests (2026-10-06). The references
re-derive each quantity a different way: energy conservation instead of the
closed CPL density, the transverse Jacobi equation instead of sinh/sin, and
growth ODEs in another variable with another integrator and start time.
"""
import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.special import hyp2f1

from sgl_analytic import sgl
from sgl_analytic import sgl_full as F
from sgl_analytic import subhalos as S


@pytest.fixture(scope="module", params=[-.08, 0., .08])
def curved(request):
    return sgl.Cosmology(Ok=request.param, window="smoothk", transfer="eh98", conc_model=16)


def reference_distance(c, z1, z2):
    """Radial null ray, then the FLRW transverse Jacobi equation."""
    def expansion(z):
        return np.sqrt(c.Om*(1+z)**3 + c.Ok*(1+z)**2 + 1 - c.Om - c.Ok)
    radial = quad(lambda z: sgl.CKMS/c.H0/expansion(z), z1, z2, epsabs=1e-9)[0]
    def jacobi(chi, state):
        return state[1], c.Ok*(c.H0/sgl.CKMS)**2*state[0]
    ray = solve_ivp(jacobi, (0., radial), (0., 1.), rtol=2e-12, atol=1e-12)
    return ray.y[0, -1]/(1+z2)


def test_curved_lensing_geometry(curved):
    c = curved
    for zl, zs in ((.1, 1.), (.5, 3.), (2., 10.)):
        dl, ds, dls = (reference_distance(c, 0., zl), reference_distance(c, 0., zs),
                       reference_distance(c, zl, zs))
        np.testing.assert_allclose(c.DA(zs), ds, rtol=2e-11)
        np.testing.assert_allclose(c.DA_ls(zl, zs), dls, rtol=2e-11)
        expected = sgl.CKMS**2*ds/(4*np.pi*sgl.GNEWT*dl*dls)
        np.testing.assert_allclose(c.Sigma_cr(zl, zs), expected, rtol=3e-11)
        np.testing.assert_allclose(c.DL(zs), (1+zs)**2*c.DA(zs), rtol=1e-15)


def test_curved_growth_against_perturbation_ode(curved):
    c = curved
    def ode(loga, state):
        a = np.exp(loga)
        m, k, de = c.Om/a**3, c.Ok/a**2, c.OL
        e2 = m + k + de
        return state[1], -(2 + (-3*m - 2*k)/(2*e2))*state[1] + 1.5*m/e2*state[0]
    sol = solve_ivp(ode, (np.log(1e-6), 0.), (1e-6, 1e-6), rtol=2e-10, atol=1e-14,
                    dense_output=True)
    z = np.array([0., .1, 1., 3., 10.])
    expected = sol.sol(-np.log1p(z))[0]/sol.sol(0.)[0]
    np.testing.assert_allclose(c.D(z), expected, rtol=2e-7)
    # the bias sector keeps the Carroll-Press-Turner approximation for Lambda
    np.testing.assert_allclose([F.Dg(c, t) for t in z], expected, rtol=.004)
    assert c.D(0.) == 1.
    np.testing.assert_allclose(c.sigma_tophat8(), c.s8, rtol=1e-14)


def test_curved_worker_snapshot_and_matter_fraction(curved):
    c = curved
    rebuilt = sgl.Cosmology(**sgl._cosmology_kwargs(c))
    assert (rebuilt.Ok, rebuilt.OL) == (c.Ok, c.OL)
    np.testing.assert_array_equal(rebuilt.D([0., 1., 3.]), c.D([0., 1., 3.]))
    np.testing.assert_allclose(S._Om_z(c, 1.), c.Om*8/c.E(1.)**2, rtol=1e-15)


def test_flat_limit_and_zero_distance():
    flat = sgl.Cosmology()
    chi = np.array([0., 100., 10000.])
    np.testing.assert_array_equal(flat.f_K(chi), chi)
    for ok in (-1e-12, 1e-12):
        c = sgl.Cosmology(Ok=ok)
        assert c.f_K(0.) == 0.
        np.testing.assert_allclose(c.f_K(chi), chi, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(c.DA(3.), flat.DA(3.), rtol=2e-12)
        np.testing.assert_allclose(F.Dg(c, 1.), F.Dg(flat, 1.), rtol=2e-12)


def test_curvature_domain_guards():
    for bad in (np.nan, -1.5):          # non-finite; E^2 < 0 near z ~ 1
        with pytest.raises(ValueError):
            sgl.Cosmology(Ok=bad)
    c = sgl.Cosmology(Ok=-.05)
    antipode = np.pi*sgl.CKMS/c.H0/np.sqrt(.05)
    assert 0 < c.f_K(.999*antipode) < .01*c.f_K(.5*antipode)
    with pytest.raises(ValueError, match="antipode"):
        c.f_K(np.array([1e3, 1.01*antipode]))


@pytest.mark.parametrize("w0,wa,ok", [(-1., .3, 0.), (-1., -.3, 0.),
                                      (-.9, -.2, .04), (-1.1, .2, -.04)])
def test_cpl_background_and_growth(w0, wa, ok):
    c = sgl.Cosmology(w0=w0, wa=wa, Ok=ok)
    def rho(a):          # energy conservation, not the closed CPL density
        integral = quad(lambda x: 1 + w0 + wa*(1 - np.exp(x)), 0., np.log(a), epsabs=1e-12)[0]
        return c.OL*np.exp(-3*integral)
    def expansion(a):
        return np.sqrt(c.Om/a**3 + ok/a**2 + rho(a))
    z = np.array([0., .3, 1., 3., 10.])
    np.testing.assert_allclose(c.E(z), [expansion(1/(1+v)) for v in z], rtol=2e-13)
    def rhs(a, y):
        h2 = expansion(a)**2
        dh_da = (-3*c.Om/a**4 - 2*ok/a**3 - 3*(1 + w0 + wa*(1 - a))*rho(a)/a)/(2*h2)
        return [y[1], -(3/a + dh_da)*y[1] + 1.5*c.Om/(a**5*h2)*y[0]]
    sol = solve_ivp(rhs, (1e-7, 1.), (1e-7, 1.), rtol=2e-10, atol=1e-12,
                    method="RK45", dense_output=True)
    assert sol.success
    np.testing.assert_allclose(c.D(z), sol.sol(1/(1+z))[0]/sol.y[0, -1], rtol=3e-8)
    np.testing.assert_array_equal(F.Dg(c, z), c.D(z))
    np.testing.assert_allclose(S._Om_z(c, z), c.Om*(1+z)**3/c.E(z)**2, rtol=2e-15)
    rebuilt = sgl.Cosmology(**sgl._cosmology_kwargs(c))
    np.testing.assert_array_equal(rebuilt.D(z), c.D(z))
    assert (rebuilt.w0, rebuilt.wa, rebuilt.Ok) == (w0, wa, ok)


@pytest.mark.parametrize("w", [-1.2, -1.0, -0.8])
def test_constant_w_growth_hypergeometric(w):
    c = sgl.Cosmology(w0=w, growth_mode="ode")
    z = np.array([0., .1, .5, 1., 3., 10., 30.])
    def raw(a):
        return a*hyp2f1(-1/(3*w), (w - 1)/(2*w), 1 - 5/(6*w), -c.OL/c.Om*a**(-3*w))
    np.testing.assert_allclose(c.D(z), raw(1/(1+z))/raw(1.), rtol=2e-8)
    np.testing.assert_allclose(c.primordial_growth_amplitude(), raw(1.), rtol=2e-8)


def test_lambda_ode_recovers_legacy_integral_and_is_continuous():
    old, new = sgl.Cosmology(), sgl.Cosmology(growth_mode="ode")
    z = np.geomspace(.001, 30., 80)
    np.testing.assert_allclose(new.D(z), old.D(z), rtol=2e-8)
    # the ODE path is continuous through w = -1; "auto" is not (CPT bias growth)
    near = sgl.Cosmology(wa=1e-9)
    assert near.growth_mode == "ode" and old.growth_mode == "legacy"
    np.testing.assert_allclose([F.Dg(near, t) for t in (.5, 1., 3.)],
                               [F.Dg(new, t) for t in (.5, 1., 3.)], rtol=1e-9)


def test_zero_wa_and_flat_lambda_branches():
    for w0 in (-1., -.9):
        a, b = sgl.Cosmology(w0=w0), sgl.Cosmology(w0=w0, wa=0.)
        z = np.linspace(0, 3, 15)
        np.testing.assert_array_equal(a.E(z), b.E(z))
        np.testing.assert_array_equal(a.D(z), b.D(z))
    c = sgl.Cosmology(w0=-.8)
    z = np.array([0., .5, 1., 3.])
    np.testing.assert_allclose(c.E(z)**2, c.Om*(1+z)**3 + c.OL*(1+z)**.6, rtol=2e-15)
    np.testing.assert_allclose(sgl.Cosmology(wa=.3).w([0., 1., 3.]), [-1., -.85, -.775])


def test_geometry_growth_split_is_diagnostic_only():
    split = sgl.Cosmology(wa=.3, growth_w0=-1., growth_wa=0.)
    ref = sgl.Cosmology(growth_mode="ode")
    np.testing.assert_array_equal(split.D([0., 1., 3.]), ref.D([0., 1., 3.]))
    assert split.E(1.) != ref.E(1.)


@pytest.mark.parametrize("kw", [dict(w0=0.), dict(w0=np.nan), dict(wa=np.inf), dict(wa=.8),
                                dict(growth_wa=.8), dict(wa=.2, growth_mode="legacy"),
                                dict(w0=-.8, growth_mode="legacy")])
def test_unsupported_dark_energy_rejected(kw):
    with pytest.raises(ValueError):
        sgl.Cosmology(**kw)

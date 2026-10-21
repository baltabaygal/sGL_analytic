"""Radiation background and the r200c halo edge (fast, no solves).

Ported from the stochastic_wl_analytic tests test_radiation_engine.py and
test_r200_edge.py (2026-10-21). radiation="engine": Omega_r = Omega_m/(1+z_eq)
in E(z), growth and closure, as the C++ engine (cpp/cosmology.h). The library
default sgl.Cosmology() has no radiation; the pipeline default has it.
edge="r200": field halos cut at r200c, clumps in the r200c convention.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from sgl_analytic import sgl
from sgl_analytic import sgl_full as F
from sgl_analytic import subhalos as SH
from sgl_analytic.pipeline import make_cosmology

OM, ZEQ = 0.315, 3402.0
Z = np.array([0.1, 0.5, 1.0, 3.0, 6.0, 10.0])
ZS = 1.0


def mk(**kw):
    return sgl.Cosmology(window="smoothk", transfer="eh98", conc_model=16, **kw)


def test_library_default_has_no_radiation_and_old_expression():
    c = mk()
    assert c.radiation is None and c.Or == 0.0
    np.testing.assert_array_equal(c.E(Z), np.sqrt(OM * (1 + Z)**3 + (1.0 - OM)))
    assert c.OL == 1.0 - OM


def test_pipeline_default_is_engine_radiation_and_ode_growth():
    c = make_cosmology()
    assert (c.radiation, c.growth_mode) == ("engine", "ode")
    assert c.Or == OM / (1 + ZEQ)


def test_engine_background_and_closure():
    c = mk(radiation="engine")
    Or = OM / (1 + ZEQ)
    assert c.Or == Or and c.OL == 1.0 - OM - Or
    np.testing.assert_allclose(c.E(Z), np.sqrt(OM * (1 + Z)**3 + Or * (1 + Z)**4 + 1 - OM - Or),
                               rtol=1e-15)
    assert abs(c.E(0.0) - 1.0) < 1e-15
    np.testing.assert_allclose(c.rhocrit(Z) / c.rhoc0, c.E(Z)**2, rtol=1e-15)


def test_zeq_is_the_free_input():
    a, b = mk(radiation="engine", zeq=3300.0), mk(radiation="engine", zeq=3500.0)
    assert a.Or == pytest.approx(OM / 3301.0) and b.Or == pytest.approx(OM / 3501.0)
    assert a.E(3.0) != b.E(3.0)
    assert mk(zeq=3300.0).E(3.0) == mk(zeq=3500.0).E(3.0)       # no radiation: z_eq not in E


def test_cpt_growth_matches_engine_transcription():
    Or = OM / (1 + ZEQ)
    OL = 1 - OM - Or

    def dgnn(z):
        a = OM * (1 + z)**3 + Or * (1 + z)**4 + OL
        m, l = OM * (1 + z)**3 / a, OL / a
        return 2.5 * m / (m**(4 / 7) - l + (1 + m / 2) * (1 + l / 70)) / (1 + z)

    c = mk(radiation="engine")
    np.testing.assert_allclose(F.Dg(c, Z), dgnn(Z) / dgnn(0.0), rtol=1e-14)


def test_omega_m_z_uses_radiation_expansion():
    c = mk(radiation="engine")
    np.testing.assert_allclose(SH._Om_z(c, 1.0), OM * 8 / c.E(1.0)**2, rtol=1e-15)


def test_ode_growth_with_radiation_against_independent_solve():
    Or = OM / (1 + ZEQ)
    OL = 1 - OM - Or
    c = mk(radiation="engine", growth_mode="ode")

    def rhs(a, y):
        e2 = OM * a**-3 + Or * a**-4 + OL
        de2 = -3 * OM * a**-4 - 4 * Or * a**-5
        return [y[1], -(3 / a + 0.5 * de2 / e2) * y[1] + 1.5 * OM * a**-5 / e2 * y[0]]

    s = solve_ivp(rhs, (1e-6, 1.0), [1e-6 + 2 / 3 * Or / OM, 1.0], method="Radau",
                  rtol=1e-12, atol=1e-16, dense_output=True)
    ref = np.array([s.sol(1 / (1 + z))[0] for z in Z]) / s.sol(1.0)[0]
    np.testing.assert_allclose(c.D(Z), ref, rtol=2e-9)
    # the radiation effect on D(z) is real (order 1e-3 at z = 10)
    assert abs(c.D(10.0) / mk(growth_mode="ode").D(10.0) - 1) > 1e-3


def test_heath_integral_excludes_radiation():
    np.testing.assert_array_equal(mk(radiation="engine").D(Z), mk().D(Z))


def test_closure_with_curvature_and_cpl():
    c = mk(radiation="engine", Ok=-0.05, w0=-0.8, wa=0.2)
    assert c.Om + c.Ok + c.Or + c.OL == pytest.approx(1.0, rel=1e-15)
    assert abs(c.E(0.0) - 1.0) < 1e-14


def test_worker_snapshot_round_trip():
    c = mk(radiation="engine", Ok=0.03)
    d = sgl.Cosmology(**sgl._cosmology_kwargs(c))
    assert d.radiation == "engine" and d.Or == c.Or
    np.testing.assert_array_equal(d.E(Z), c.E(Z))
    np.testing.assert_array_equal(d.D(Z), c.D(Z))


def test_invalid_radiation_rejected():
    with pytest.raises(ValueError):
        mk(radiation="yes")
    assert mk(radiation="none").Or == 0.0


@pytest.fixture(scope="module")
def cos():
    return sgl.Cosmology(window="smoothk", transfer="eh98", anchor="tophat", conc_model=16)


def _models(cos):
    kw = dict(sector="exact", exact_kw=dict(method="moments"))
    return (SH.SubhaloModel(cos, ZS, None, clump_edge="rvir", **kw),
            SH.SubhaloModel(cos, ZS, None, clump_edge="r200", **kw))


def _host_index(sub):
    """A massive host at z ~ 0.5 that carries clumps."""
    jz = int(np.argmin(np.abs(np.asarray(sub.zl) - 0.5)))
    jM = int(np.argmin(np.abs(np.log(np.asarray(sub.Ml) / 1e14))))
    return jz, jM


def test_frac_inside_limits():
    assert SH._frac_inside(1.2, 1.2, 5.0, 1.0) == 1.0
    f = SH._frac_inside(1.0, 1.2, 5.0, 0.86 * 1.2)
    assert 0.5 < f < 1.0
    assert abs(f - SH._frac_inside(1.0, 1.2, 5.0, 0.86 * 1.2, n=65536)) < 1e-7


def test_r200_host_is_a_selection_of_the_rvir_host(cos):
    rv, r2 = _models(cos)
    jz, jM = _host_index(rv)
    a, b = rv.host(jz, jM), r2.host(jz, jM)
    assert a is not None and b is not None
    # same calibrated population (psi = m/M_vir), only thinned to r <= r200
    np.testing.assert_array_equal(a["m"], b["m"])
    F_in = SH._frac_inside(1.0, a["eta"], a["r200"] / a["rs"], SH.X0_RVIR * a["eta"])
    np.testing.assert_allclose(b["dN"], a["dN"] * F_in, rtol=1e-14)
    # clumps: M200 = m, cut at their own r200; host carve v_r = 1
    assert b["vr"] == 1.0 and a["vr"] > 1.0
    np.testing.assert_allclose(b["D"], b["xtc"] * b["rsc"], rtol=0)
    c_m, rs_m, _, _ = cos.nfw_params(b["m"], b["z"], ZS)
    np.testing.assert_allclose(b["xtc"], c_m, rtol=1e-14)
    np.testing.assert_allclose(b["rsc"], rs_m, rtol=1e-14)
    # projected clump density lives inside the host's r200 and is normalised
    assert b["Rn"].max() < b["r200"]
    assert abs(np.trapezoid(b["n2"] * 2 * np.pi * b["Rn"], b["Rn"]) - 1.0) < 1e-12


def test_bad_subhalo_edges_raise(cos):
    with pytest.raises(ValueError):
        SH.SubhaloModel(cos, ZS, None, clump_edge="r500", sector="exact")
    with pytest.raises(ValueError):
        SH.SubhaloModel(cos, ZS, 1e-3, clump_edge="r200", sector="pairs")
    with pytest.raises(ValueError):
        SH.SubhaloModel(cos, ZS, None, clump_edge="kthr", sector="exact")

#!/usr/bin/env python
"""Composite characteristic exponent for the FULL engine default config.

    ln Phi(k_kap, k_gam) = Lambda_halo+sub(k) + Lambda_fil(k)
                           + 1/2 B(k)^T C_delta B(k)          (Gaussian Cox)
                           + Gaussian sub-threshold background

The halo-only chain (`sgl.P_vector_lnmu`) hard-wires its population; this
module keeps the SAME width-relative inversion (`sgl.vector_grid`, measured
k-reach, Hankel + Fourier, pi/mu^2 map) but takes Lambda as a callable so each
ingredient is an additive/multiplicative term.  `invert()` with the bare halo
population is checked to reproduce `sgl.P_vector_lnmu` (see __main__).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy import exp, log, pi, sqrt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sgl
from sgl import (PHI_TOL, PPP_G, PPP_K, XI_REACH, _trapz, vector_grid)


# ---------------------------------------------------------------- inversion --
# Tail floor of the (kappa, gamma) box (2026-09-27).  The width-relative box
# (gamma <= 6 sigma) misses the critical line kappa + gamma = 1 at low z_s --
# P was zero beyond mu ~ 8.5 at z_s = 1 -- and the MC (hence FLUMEN, trained
# on it) is ray-starved there, so the analytic tail must be resolved on its
# own.  Box = max(width-relative, floor), the rule that survives both the
# absolute and the width-relative traps.  (0.99, 0.8) is the converged box:
# (0.97, 0.5) agrees to <= 6e-5 over ln mu = 0.3-7 at z_s = 1.
TAIL_KAP_MAX = 0.99
TAIL_GAM_MAX = 0.8
def _reach(lam, k0, axis, tol=PHI_TOL, max_mult=64.0):
    """sgl.phi_reach for an arbitrary Lambda callable lam(k_k, k_g)."""
    mults = 1.25 ** np.arange(0, int(np.ceil(log(max_mult) / log(1.25))) + 1)
    ks = k0 * mults
    zero = np.zeros(1)
    L = lam(ks, zero)[:, 0] if axis == "k" else lam(zero, ks)[0, :]
    phi = np.abs(np.exp(L))
    ok = np.nonzero(phi <= tol)[0]
    i = int(ok[0]) if len(ok) else int(np.argmin(phi))
    return float(ks[i]), float(phi[i])


def invert(lam, sigma, xi_out=None, n_k_int=300, xi_reach=XI_REACH,
           n_xi_wide=500, grid_scale=1.0, kap_max=None, gam_max=None,
           keep_dk=True, hankel_em=True, tail_box=True, phi_tol=PHI_TOL,
           nproc=1, **override):
    """Phi = exp(lam) -> P_s(ln mu).  Mirrors sgl.P_vector_lnmu line for line
    after its population build; `sigma` = total sigma_kappa sets the grid.

    kap_max / gam_max (2026-09-27) widen the (kappa, gamma) box BEFORE every
    derived count (n_kap, n_gam at the same points-per-sigma, nk_k, nk_g at
    the same points-per-period), unlike `override`, which is applied after.
    The width-relative box (6 sigma in gamma) excludes the critical line
    kappa + gamma = 1 when sigma is small -- at z_s = 1 it ends at mu ~ 8.5
    (the box corner) -- so the strong-lensing tail needs a TAIL box.
    keep_dk=True keeps the k-steps of the DEFAULT box: dk sets the alias
    period (PPP x box = 21 in gamma, 10.6 in kappa at z_s=1), which is a
    property of the density's support, not of where output is wanted.
    Scaling dk with the box (keep_dk=False) cost 1231 s vs 70 s at z_s=1
    for a 0.97 x 0.5 box.
    hankel_em=True adds the Euler-Maclaurin endpoint term of the k_gamma
    trapezoid (2026-09-27).  For g(k) = k J0(k gamma) Phi(k) the trapezoid
    misses (dk_g^2/12) g'(0) = (dk_g^2/12) Phi(k_kappa, 0): a term FLAT in
    gamma, proportional to the kappa-marginal.  At z_s=1 it put a plateau of
    -1.38e-2 (predicted -1.2e-2) along kappa ~ 0 at every gamma -- invisible
    in the 6-sigma default box, but a tail box's mu-paths cross it at large
    gamma (a x100 dip at ln mu ~ 0.9 with gamma_max = 0.8).  The kappa
    transform has no such term at leading order (Re dPhi/dk_kappa(0) = 0)."""
    from scipy.interpolate import RegularGridInterpolator

    if xi_out is None:
        xi_out = np.linspace(-0.35, 0.55, 160)
    xi_out = np.asarray(xi_out, float)
    grid = vector_grid(sigma, xi_reach=xi_reach)
    g0 = grid["gam_max"]                       # default (6 sigma) gamma reach
    if tail_box:
        kap_max = max(grid["kap_max"], TAIL_KAP_MAX) if kap_max is None else kap_max
        gam_max = max(g0, TAIL_GAM_MAX) if gam_max is None else gam_max
    dk0 = (2 * pi / (PPP_K * max(abs(grid["kap_min"]), grid["kap_max"])),
           2 * pi / (PPP_G * grid["gam_max"]))
    if (kap_max is not None and kap_max != grid["kap_max"]) or \
            (gam_max is not None and gam_max != grid["gam_max"]):
        if kap_max is not None:
            grid["kap_max"] = float(kap_max)
        if gam_max is not None:
            grid["gam_max"] = float(gam_max)
        grid["n_kap"] = int(np.ceil((grid["kap_max"] - grid["kap_min"])
                                    / (sigma / sgl.PTS_PER_SIG_KAP))) + 1
        grid["n_gam"] = int(np.ceil(grid["gam_max"] / (sigma / sgl.PTS_PER_SIG_GAM))) + 1
    grid["k_k_max"], phi_k = _reach(lam, grid["k_k_max"], "k", tol=phi_tol)
    grid["k_g_max"], phi_g = _reach(lam, grid["k_g_max"], "g", tol=phi_tol)
    kap_ext = max(abs(grid["kap_min"]), grid["kap_max"])
    if keep_dk:
        dk_k, dk_g = dk0
    else:
        dk_k, dk_g = 2 * pi / (PPP_K * kap_ext), 2 * pi / (PPP_G * grid["gam_max"])
    grid["nk_k"] = int(np.ceil(grid["k_k_max"] / dk_k)) + 1
    grid["nk_g"] = int(np.ceil(grid["k_g_max"] / dk_g)) + 1
    grid.update(phi_at_k_cut=phi_k, phi_at_g_cut=phi_g)
    grid.update(override)
    if grid_scale != 1.0:
        for key in ("nk_k", "nk_g", "n_kap", "n_gam"):
            grid[key] = int(np.ceil(grid[key] * grid_scale))

    k_k = np.linspace(0.0, grid["k_k_max"], grid["nk_k"])
    k_g = np.linspace(0.0, grid["k_g_max"], grid["nk_g"])
    Phi = np.exp(lam(k_k, k_g, nproc=nproc))
    kap = np.linspace(grid["kap_min"], grid["kap_max"], grid["n_kap"])
    gam = np.linspace(0.0, grid["gam_max"], grid["n_gam"])
    P3 = sgl.P_vector_joint(k_k, k_g, Phi, kap, gam)
    if hankel_em:
        dkk, dkg = k_k[1] - k_k[0], k_g[1] - k_g[0]
        wkk = np.full(k_k.size, dkk); wkk[0] *= 0.5; wkk[-1] *= 0.5
        # g = k J0(k gam) Phi = k Phi0 + k^3 (Phi2 - gam^2 Phi0 / 4) + ...
        # (Phi even in k_g); Euler-Maclaurin (B2 = 1/6, B4 = -1/30):
        # integral = T + (h^2/12) g'(0) - (h^4/720) g^(3)(0), g'(0) = Phi0, g^(3)(0) = 6 (Phi2 - gam^2 Phi0 / 4).
        # Phi2 from the first two k_g columns (O(h^2) accurate).
        Phi0 = Phi[:, 0]
        Phi2 = (Phi[:, 1] - Phi[:, 0]) / dkg**2
        E = np.exp(-1j * np.outer(kap, k_k)) * wkk[None, :]
        c = 1.0 / (2.0 * pi**2)
        a = c * np.real(E @ ((dkg**2 / 12.0) * Phi0 - (dkg**4 / 120.0) * Phi2))
        b = c * np.real(E @ ((dkg**4 / 480.0) * Phi0))
        P3 = P3 + a[:, None] + b[:, None] * gam[None, :]**2
    ip = RegularGridInterpolator((kap, gam), P3, bounds_error=False,
                                 fill_value=0.0)
    gmax = grid["gam_max"]
    g_def = min(g0, gmax)
    dgam = gam[1] - gam[0]

    def P_i(xi_grid):
        # The default segment (gamma <= the 6-sigma reach) is integrated in
        # kappa with n_k_int points exactly as before.  A TAIL box adds a
        # separate segment gamma in (g_def, gmax], parametrised by gamma at
        # half the grid's gamma step (2026-09-27): stretching one 300-point
        # kappa path over the longer range under-resolved the sqrt corner at
        # gamma -> 0, where P3 ~ 1e4 varies on ~sigma/12, and moved the BODY
        # (+55% at ln mu = -0.1 with gamma_max = 0.8).
        # (2026-10-03) gamma on the path from the FACTORISED difference of
        # squares, (1 - ke)^2 - 1/mu = (hi - ke)(1 - ke + 1/sqrt(mu)): exactly 0
        # at ke = hi.  The direct form left sqrt(eps) = 1.5e-8 there or 0 by
        # roundoff, and P3 ~ 1e4 is steep at gamma -> 0, so P_i(xi) was rough at
        # 1e-8 (a 1-ulp sigma change moved the z_s=1 normalisation 8e-10).
        # Tail-segment points are clamped into the kappa box: its
        # end point kx = kap[0] lands an ulp outside by roundoff, where the
        # interpolator's fill_value 0 replaced P3 (clipped mean moved 5e-9).
        # (2026-10-06) all path points are collected first and the (linear)
        # interpolator is called ONCE: it evaluates point by point, so the
        # values and every per-xi trapezoid are unchanged; only the ~6400
        # per-call overheads go.
        out = np.zeros_like(xi_grid)
        segs = []                        # (i, kind, x-array, extra, pref, offset)
        pts = []
        off = 0
        for i, xi in enumerate(xi_grid):
            mu = exp(xi)
            rm = 1.0 / sqrt(mu)
            hi = 1.0 - rm
            lo = max(kap[0], 1.0 - sqrt(1.0 / mu + g_def**2))
            if hi <= lo:
                continue
            ke = np.linspace(lo, hi, n_k_int)
            ge = sqrt(np.clip((hi - ke) * (1.0 - ke + rm), 0.0, None))
            pts.append(np.column_stack([ke, ge]))
            segs.append((i, 0, ke, None, pi / mu**2, off))
            off += ke.size
            if gmax > g_def and lo > kap[0]:
                gb = min(gmax, sqrt(max((hi - kap[0]) * (1.0 - kap[0] + rm), 0.0)))
                if gb > g_def:
                    n = int(np.ceil((gb - g_def) / (0.5 * dgam))) + 1
                    gx = np.linspace(g_def, gb, n)
                    r = np.sqrt(1.0 / mu + gx**2)
                    kx = np.clip(1.0 - r, kap[0], kap[-1])
                    pts.append(np.column_stack([kx, gx]))
                    segs.append((i, 1, gx, gx / r, pi / mu**2, off))
                    off += gx.size
        if not pts:
            return out
        vals = ip(np.concatenate(pts))
        for i, kind, xs, extra, pref, o in segs:
            v = vals[o:o + xs.size]
            if kind == 0:
                out[i] = pref * _trapz(v, xs)
            else:
                out[i] += pref * _trapz(v * extra, xs)
        return out

    xw = np.linspace(grid["xi_min"], grid["xi_max"], n_xi_wide)
    pw = P_i(xw)
    norm = _trapz(pw, xw)
    ps = pw / norm
    mean = float(_trapz(xw * ps, xw))
    var = float(_trapz((xw - mean)**2 * ps, xw))
    skew = float(_trapz((xw - mean)**3 * ps, xw) / var**1.5)
    x = exp(-0.5 * xw)
    md = float(_trapz(x * ps, xw))
    sDL = float(sqrt(_trapz((x - md)**2 * ps, xw)) / md)
    Ps = P_i(xi_out) / norm
    meta = dict(grid=grid, sigma_kappa=sigma, mean=mean, var=var, skew=skew,
                sigmaDL=sDL, P3=P3, kap=kap, gam=gam)
    return xi_out, Ps, meta


# ================================================================ physics ====
# Everything below mirrors the C++ engine's DEFAULT config (cpp/lensing.cpp,
# cpp/subhalo.cpp, cpp/cosmology.{h,cpp}); line refs are to this repo's cpp/.
# The halo sector (HMF, sigma(M), cons16, NFW projection) is sgl.Cosmology in
# engine mode, validated there to <=0.3% (dn/dlnM) and 1.7e-4 (sigma).
import filaments as fil
from sgl import CKMS

ZMIN_ENG, ZMAX_ENG, NZ_ENG = 0.01, 12.341169644129371, 103
MMIN_ENG, MMAX_ENG, NM_ENG = 1e7, 1e17, 100
DC0 = 3.0 / 5.0 * (1.5 * pi)**(2.0 / 3.0)          # cosmology.h deltac0
NHALOS = 100                                        # explicit halos per ray
EPS_FLOOR = 1e-3                                    # weak-arm floor / kappa_thr

# binning of the (kappa, gamma) jump cells: fine for Lambda_0, coarse for the
# per-shell Cox tables (the Cox term is a ~10% correction, see cox_Q)
# kappa axis is SIGNED (pseudo-elliptical kappa + eps cos2phi gamma can go
# below kappa_thr and negative near r_max, where most explicit area sits --
# a log-only axis silently dropped those cells: +ell Var came out 3.7% low).
# Cells beyond |kappa| = 30 (monster-ray cores) are dropped as in sgl.py;
# gamma is clipped at the floor (J0 -> 1 there), never dropped.
# SGL_FINE_MULT (env, default 1) scales the fine-lattice bin counts; read at
# import so spawned build workers inherit it.  A convergence handle for the
# z_s ~ 1 tail, where the lattice spacing (dln kappa = 0.04, dln gamma = 0.08,
# i.e. d gamma ~ 0.04 at gamma ~ 0.5) is comparable to sigma_kappa = 0.037.
import os as _os
_FM = float(_os.environ.get("SGL_FINE_MULT", "1"))
FINE = dict(k=(1e-7, 30.0, int(round(480 * _FM))), g=(1e-9, 30.0, int(round(300 * _FM))))
COARSE = dict(k=(1e-7, 30.0, 170), g=(1e-9, 30.0, 110))


def _edges(spec):
    lo, hi, n = spec
    pos = np.geomspace(lo, hi, n + 1)
    return np.concatenate([-pos[::-1], pos])


def _centers(e):
    c = np.where(e[:-1] * e[1:] > 0, np.sign(e[1:]) * sqrt(np.abs(e[:-1] * e[1:])), 0.0)
    return c


def engine_grids():
    z = np.exp(np.linspace(log(ZMIN_ENG), log(ZMAX_ENG), NZ_ENG))
    M = np.exp(np.linspace(log(MMIN_ENG), log(MMAX_ENG), NM_ENG))
    return z, M


ZINT_MODES = ("engine", "midpoint", "gauss")


def z_shells(zs, zint="engine", zint_n=2):
    """Redshift shells and quadrature nodes of the lens integral (2026-10-05).

    Returns (segments, tasks, zn): segments[i] = (z_lo, z_hi) of shell i (the
    clustering field is averaged over it), tasks[i] = [(jn, z, dz), ...] the
    nodes of shell i with their redshift and weight, zn[jn] = z (the index
    the subhalo model uses to look up a host's redshift).
      "engine": the engine's shells [z_{j-1}, z_j] for z_j < z_s, one node at
        the UPPER edge with weight z_j - z_{j-1} (backward difference; the MC's
        rule; it drops [0, z_0] and [z_last, z_s]; ~3.4% low in lensing weight).
      "midpoint" / "gauss": the engine's nodes below z_s as shell EDGES, plus
        0 and z_s, so the whole range [0, z_s] is covered; each shell is
        integrated with an n-point Gauss-Legendre rule, n = 1 ("midpoint") or
        zint_n ("gauss").  The clustering covariance is the segment average
        over the same shells, so the clustering term is unchanged in form."""
    if zint not in ZINT_MODES:
        raise ValueError(f"zint must be one of {ZINT_MODES}, got {zint!r}")
    zl, _ = engine_grids()
    if zint == "engine":
        jzs = [j for j in range(1, NZ_ENG) if zl[j] < zs]
        segs = [(zl[j - 1], zl[j]) for j in jzs]
        tasks = [[(j, zl[j], None)] for j in jzs]     # dz=None: the builder's own rule
        return segs, tasks, zl
    n = 1 if zint == "midpoint" else int(zint_n)
    if n < 1:
        raise ValueError("zint_n must be >= 1")
    edges = np.concatenate([[0.0], zl[zl < zs], [float(zs)]])
    t, w = np.polynomial.legendre.leggauss(n)
    segs, tasks, zn = [], [], [0.0]
    for a_, b_ in zip(edges[:-1], edges[1:]):
        if b_ <= a_:
            continue
        nodes = []
        for tk, wk in zip(t, w):
            zn.append(a_ + 0.5 * (b_ - a_) * (tk + 1.0))
            nodes.append((len(zn) - 1, zn[-1], 0.5 * (b_ - a_) * wk))
        segs.append((a_, b_))
        tasks.append(nodes)
    return segs, tasks, np.array(zn)


def Dg(cos, z):
    """Carroll-Press-Turner growth, Dg(0)=1 (cosmology.h:90-102) -- the engine
    uses THIS (not the exact integral) in delta_c(z), the bias and M*(z)."""
    def nn(zz):
        a3 = cos.Om * (1 + zz)**3
        E2 = a3 + cos.OL
        Omz, OLz = a3 / E2, cos.OL / E2
        return 2.5 * Omz / (Omz**(4 / 7) - OLz + (1 + Omz / 2) * (1 + OLz / 70)) / (1 + zz)
    return nn(z) / nn(0.0)


def halo_bias(cos, M, z):          # cosmology.cpp:515, (p,q) = (0.3, 0.8)
    qn2 = 0.8 * (DC0 / Dg(cos, z) / cos.sigmaM(M, 0.0))**2
    return 1 + (qn2 - 1) / DC0 + 0.6 / (DC0 * (1 + qn2**0.3))


def fil_bias(cos, M, z):           # cosmology.cpp:532, flat barrier q = 0.7
    qn2 = 0.7 * (DC0 / Dg(cos, z) / cos.sigmaM(M, 0.0))**2
    return 1 + (qn2 - 1) / DC0


def Mstar(cos, z):
    """sigma_0(M*) = delta_c(z) (logMcharlist); log-interpolated root."""
    tab = getattr(cos, "_mstar_tab", None)
    if tab is None:
        Mg = np.logspace(0, 17, 1200)
        s = np.array([cos.sigmaM(m, 0.0) for m in Mg])
        # keep the largest suffix on which sigma(M) is strictly decreasing
        # (2026-10-05): sigmaM's spline is fitted on [1e4, 1e17] only, and with
        # some Cosmology settings (e.g. the constructor defaults) its
        # extrapolation below 1e4 is non-monotone, so np.interp returned
        # M* = 1 Msun at z >~ 2.  Production settings: table monotone, unchanged.
        up = np.nonzero(np.diff(s) >= 0)[0]
        i0 = int(up[-1]) + 1 if up.size else 0
        tab = cos._mstar_tab = (Mg[i0:], s[i0:])
    Mg, s = tab
    target = DC0 / Dg(cos, z)
    return float(np.exp(np.interp(-target, -s, log(Mg))))


def epsilon_NFW(cos, M, z):        # lensing.cpp:50, Allgood+06
    s = 0.54 * (M / Mstar(cos, z))**-0.05
    return max(0.0, (1 - s) / (1 + s))


def epsilon_NFW_vec(cos, M, z):
    """epsilon_NFW for an array of masses at one z (2026-10-06): one M* lookup,
    the same arithmetic elementwise."""
    s = 0.54 * (np.asarray(M, dtype=float) / Mstar(cos, z))**-0.05
    return np.maximum(0.0, (1 - s) / (1 + s))


def kappagamma_eps(eps, ks, x, phi, xt=None):
    """lensing.cpp:56 kappagammaNFWeps (pseudo-elliptical NFW), vectorized.
    Mirrors the engine INCLUDING its cos^2 (not sin^2) form of the eps^2
    term (docs/pseudo_elliptical_shear_bug.md) and the max(0,.) clamp; kept
    on purpose (2026-10-05, Ville: use the engine's cos^2).
    xt (2026-10-02): NFW truncated at x_t = r_t/r_s (nfw_trunc_kg); beyond the
    truncation (x_eps >= x_t) kappa = 0 and gamma = the circular point-mass
    shear (the pseudo-elliptical terms are dropped there: they would put a
    convergence eps cos(2 psi) gamma = O(eps/x^2) outside the halo, whose
    angular mean is O(eps^2/x^2))."""
    x1 = sqrt(1 - eps) * np.cos(phi) * x
    x2 = sqrt(1 + eps) * np.sin(phi) * x
    xe = np.maximum(sqrt(x1**2 + x2**2), 1e-12)
    c2 = (x1**2 - x2**2) / np.maximum(xe**2, 1e-300)
    if xt is None:
        k0, g0 = sgl.kappa_gamma(np.ascontiguousarray(xe.ravel()), ks)
    else:
        k0, g0 = nfw_trunc_kg(xe.ravel(), ks, xt)
    k0, g0 = k0.reshape(xe.shape), g0.reshape(xe.shape)
    g0 = np.where(xe < 1e-4, ks, g0)                # safeNFWGammaCore -> 0.5
    k = k0 + eps * c2 * g0
    g2 = g0**2 + 2 * eps * c2 * g0 * k0 + eps**2 * (k0**2 - (c2 * g0)**2)
    if xt is not None:          # outside: monopole shear of the virial mass at x
        out = xe >= xt
        gpm = ks * 4 * (log(1 + xt) - xt / (1 + xt)) / np.maximum(np.asarray(x, float), 1e-300)**2
        k = np.where(out, 0.0, k)
        g2 = np.where(out, gpm**2, g2)
    return k, sqrt(np.maximum(g2, 0.0))


# --- NFW truncated at r_t (sharp 3D cut), 2026-10-02 ------------------------
# kappa_t = 2 ks F(x) - kappa_out,  kbar_t = 4 ks h(x)/x^2 - dkbar  (x < x_t),
# with the projected contribution of the untruncated profile beyond r_t
#   kappa_out = 2 ks int_0^1 ds s / [(s+x_t)^2 sqrt(1 - a^2 s^2)],   s = 1 - v^2
#   dkbar     = 4 ks int_0^1 ds s / [(s+x_t)^2 (1 + sqrt(1 - a^2 s^2))]
# (a = x/x_t, y = x_t/s); kappa_t = 0 and kbar_t = 4 ks mu(x_t)/x^2 for x >= x_t;
# gamma = kbar - kappa.
# Evaluation (2026-10-03, _tc_point): a < TC_A_SMALL -> the small-a expansion
# (error O(a^4) <= 1e-12); otherwise the CLOSED FORM of the sharply truncated
# NFW (Sigma and the enclosed projected mass, as in Takada & Jain 2003):
#   I1 = F(x) - F_t(x),  I2 = (h(x) - m_t(x)) / x^2,
# with exact arccosh / arccos arguments (arg - 1 = (x-1)(x-c)/(x(1+c))) and a
# linear bridge over |x - 1| < TC_BAND (removable 1/(1-x) cancellation).
# Checked vs direct line-of-sight quadrature: <= 1e-10 rel everywhere; the
# 24-node Gauss-Legendre rule it replaces was off by 7e-6 / 2.3e-2 / 0.81 in
# kappa at x = (0.999 / 0.99999 / 0.9999999) x_t (the integrand's 1/sqrt
# end-point singularity as a -> 1).  2.4x cheaper per point.  The GL rule is
# kept as the fallback for x_t inside the bridge and for the no-numba path.
TC_A_SMALL = 1e-3   # below a = x/x_t: small-a expansion
TC_BAND = 1e-5      # |x - 1| bridged linearly in the closed form
A_SMALL = 0.05      # (pre-2026-10-03 expansion threshold; no longer used by numba path)
_GLX, _GLW = np.polynomial.legendre.leggauss(24)
_GLX, _GLW = 0.5 * (_GLX + 1.0), 0.5 * _GLW


def _trunc_corr_py(x, xt, gx, gw):
    a = np.minimum(x / xt, 1.0)[:, None]
    v = gx[None, :]
    s = 1.0 - v * v
    I1 = (gw * 2 * v * s / ((s + xt)**2 * sqrt(np.maximum(1 - a * a * s * s, 1e-300)))).sum(1)
    t = gx[None, :]
    I2 = (gw * t / ((t + xt)**2 * (1 + sqrt(np.maximum(1 - a * a * t * t, 0.0))))).sum(1)
    return I1, I2


def nfw_trunc_unit(x, xt):
    """(kappa, gamma) / ks of truncated NFW for ARRAYS x and x_t (elementwise)."""
    x = np.ascontiguousarray(np.asarray(x, float).ravel())
    xt = np.ascontiguousarray(np.broadcast_to(np.asarray(xt, float).ravel(), x.shape), dtype=float)
    ku, gu = sgl.kappa_gamma(x, 1.0)
    I1, I2 = _trunc_corr_arr(x, xt, _GLX, _GLW)
    inside = x < xt
    mut = np.log1p(xt) - xt / (1 + xt)
    k = np.where(inside, ku - 2 * I1, 0.0)
    kb = np.where(inside, ku + gu - 4 * I2, 4 * mut / np.maximum(x, 1e-300)**2)
    return k, kb - k


def nfw_trunc_kg(x, ks, xt):
    """(kappa, gamma) of an NFW profile truncated at x_t = r_t/r_s."""
    x = np.ascontiguousarray(np.asarray(x, float).ravel())
    ku, gu = sgl.kappa_gamma(x, 1.0)
    I1, I2 = _trunc_corr(x, float(xt), _GLX, _GLW)
    inside = x < xt
    mut = log(1 + xt) - xt / (1 + xt)
    k = np.where(inside, ku - 2 * I1, 0.0)
    kb = np.where(inside, ku + gu - 4 * I2, 4 * mut / np.maximum(x, 1e-300)**2)
    return ks * k, ks * (kb - k)


def _xmax_for(ks, kthr):
    """x = rmax/rs where 2 ks F(x) = kthr (NFW kappa is monotone in x)."""
    if sgl.kappa_gamma(np.array([1e-9]), ks)[0][0] <= kthr:
        return 0.0
    lo, hi = log(1e-9), log(1e7)
    for _ in range(60):
        m = 0.5 * (lo + hi)
        if sgl.kappa_gamma(np.array([exp(m)]), ks)[0][0] > kthr:
            lo = m
        else:
            hi = m
    return exp(0.5 * (lo + hi))


def _xmax_for_vec(ks, kthr):
    """`_xmax_for` for an ARRAY of ks at one kthr, bisecting every entry at
    once (2026-09-25).  The scalar version costs 61 kappa_gamma calls on a
    1-element array each, and SubhaloModel.host calls it once per clump mass
    bin -- 158k calls / ~9.7M kappa_gamma calls per build_population(+sub).

    Exactness: NFW kappa is k = 2 ks F(x) (the kernel evaluates (2 ks) F).
    Scaling by 2 is exact in IEEE arithmetic, so ks * [kappa at ks=1] =
    ks * (2F) is bit-equal to (2 ks) F; every bisection comparison therefore
    sees the same float and takes the same branch.  Gated empirically against
    the scalar version, because the kernel is fastmath and a length-n call
    could in principle be SIMD-vectorised differently from a length-1 one."""
    ks = np.asarray(ks, float)

    def k_unit(x):
        return sgl.kappa_gamma(np.ascontiguousarray(x, dtype=float), 1.0)[0]

    zero = ks * k_unit(np.array([1e-9]))[0] <= kthr
    lo = np.full(ks.shape, log(1e-9))
    hi = np.full(ks.shape, log(1e7))
    for _ in range(60):
        m = 0.5 * (lo + hi)
        above = ks * k_unit(exp(m)) > kthr
        lo = np.where(above, m, lo)
        hi = np.where(above, hi, m)
    out = exp(0.5 * (lo + hi))
    out[zero] = 0.0
    return out


def _hist_index(K, G, edges):
    """Flattened, outlier-padded 2-D bin index EXACTLY as np.histogramdd forms
    it (numpy 2.4 `_histograms_impl.histogramdd`): searchsorted(side='right'),
    values on the last edge shifted into the last bin, ravel_multi_index over
    (len(edges)+1) bins per axis.  Computing it once lets several weightings
    share one pair of searchsorted passes (2026-09-25: `Cells.add` called
    np.histogram2d four times per call, three on identical K/G/edges; the
    profiler put 113 s in searchsorted + 40 s of histogramdd overhead per
    build_population(+sub))."""
    ek, eg = edges
    ik = np.searchsorted(ek, K, side="right")
    ig = np.searchsorted(eg, G, side="right")
    ik[K == ek[-1]] -= 1
    ig[G == eg[-1]] -= 1
    nb = (ek.size + 1, eg.size + 1)
    return np.ravel_multi_index((ik, ig), nb), nb


try:
    import numba as _numba
    import math as _math
    _HAVE_NUMBA = True

    @_numba.njit(cache=True)
    def _tc_closed_raw(x, c):
        """I1 = F - F_t, I2 = (h - m_t)/x^2 in closed form, 0 < x < c, x != 1."""
        q = _math.sqrt((c - x) * (c + x))
        d = (x - 1.0) * (x - c) / (x * (1.0 + c))          # arg - 1, exact
        L = _math.log((c + q) / (2.0 * (1.0 + c))) - (q - c) / (1.0 + c)
        if x < 1.0:
            u = (1.0 - x) * (1.0 + x)
            e = (1.0 - x) / x
            A0 = _math.log1p(e + _math.sqrt(e * (e + 2.0)))  # arccosh(1/x)
            At = _math.log1p(d + _math.sqrt(d * (d + 2.0)))  # arccosh(arg)
            su = _math.sqrt(u)
            return ((q / (1.0 + c) - 1.0) / u + (A0 - At) / (u * su),
                    (L + (A0 - At) / su) / (x * x))
        v = (x - 1.0) * (x + 1.0)
        B0 = 2.0 * _math.asin(_math.sqrt((x - 1.0) / (2.0 * x)))     # arccos(1/x)
        Bt = 2.0 * _math.asin(_math.sqrt(max(-d, 0.0) / 2.0))        # arccos(arg)
        sv = _math.sqrt(v)
        return ((1.0 - q / (1.0 + c)) / v - (B0 - Bt) / (v * sv),
                (L + (B0 - Bt) / sv) / (x * x))

    @_numba.njit(cache=True)
    def _tc_gl(a, xt, gx, gw):
        s1 = 0.0
        s2 = 0.0
        for j in range(gx.shape[0]):
            v = gx[j]
            s = 1.0 - v * v
            d = 1.0 - a * a * s * s
            if d < 1e-300:
                d = 1e-300
            s1 += gw[j] * 2.0 * v * s / ((s + xt) ** 2 * _math.sqrt(d))
            t = gx[j]
            e = 1.0 - a * a * t * t
            if e < 0.0:
                e = 0.0
            s2 += gw[j] * t / ((t + xt) ** 2 * (1.0 + _math.sqrt(e)))
        return s1, s2

    @_numba.njit(cache=True)
    def _tc_point(x, xt, gx, gw):
        """Truncation correction (I1, I2) at one point (2026-10-03); see the
        comment above TC_A_SMALL."""
        a = x / xt
        if a >= 1.0:
            return 0.0, 0.0
        if a < TC_A_SMALL:
            i0 = _math.log((1.0 + xt) / xt) - 1.0 / (1.0 + xt)
            u1 = 1.0 + xt
            jj = (0.5 * (u1 * u1 - xt * xt) - 3.0 * xt * (u1 - xt)
                  + 3.0 * xt * xt * _math.log(u1 / xt) + xt ** 3 * (1.0 / u1 - 1.0 / xt))
            return i0 + 0.5 * a * a * jj, 0.5 * i0 + 0.125 * a * a * jj
        if abs(x - 1.0) < TC_BAND:
            if xt <= 1.0 + 2.0 * TC_BAND:
                return _tc_gl(a, xt, gx, gw)
            a1, b1 = _tc_closed_raw(1.0 - TC_BAND, xt)
            a2, b2 = _tc_closed_raw(1.0 + TC_BAND, xt)
            f = (x - (1.0 - TC_BAND)) / (2.0 * TC_BAND)
            return a1 + f * (a2 - a1), b1 + f * (b2 - b1)
        return _tc_closed_raw(x, xt)

    @_numba.njit(cache=True, parallel=True)
    def _trunc_corr_arr_nb(x, xtv, gx, gw):
        n = x.shape[0]
        I1 = np.zeros(n)
        I2 = np.zeros(n)
        for i in _numba.prange(n):
            I1[i], I2[i] = _tc_point(x[i], xtv[i], gx, gw)
        return I1, I2

    @_numba.njit(cache=True)
    def _trunc_corr_nb(x, xt, gx, gw):
        n = x.shape[0]
        I1 = np.zeros(n)
        I2 = np.zeros(n)
        for i in range(n):
            I1[i], I2[i] = _tc_point(x[i], xt, gx, gw)
        return I1, I2
except ImportError:                    # pragma: no cover
    _HAVE_NUMBA = False

_trunc_corr = _trunc_corr_nb if _HAVE_NUMBA else _trunc_corr_py


def _trunc_corr_arr(x, xt, gx, gw):
    if _HAVE_NUMBA:
        return _trunc_corr_arr_nb(x, xt, gx, gw)
    out = [_trunc_corr_py(np.array([a]), b, gx, gw) for a, b in zip(x, xt)]
    return np.array([o[0][0] for o in out]), np.array([o[1][0] for o in out])

if _HAVE_NUMBA:
    @_numba.njit(cache=True, inline="always")
    def _ss_right(e, v):
        """np.searchsorted(e, v, side='right') for sorted e: the number of
        entries <= v.  NaN sorts last in numpy, so it maps to len(e)."""
        n = e.shape[0]
        if v != v:
            return n
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) >> 1
            if e[mid] <= v:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @_numba.njit(cache=True, inline="always")
    def _fix_bracket(e, v, g):
        """Walk a guess g to the exact searchsorted(side='right') answer: the
        unique g with e[g-1] <= v < e[g].  Exact for ANY guess, so a fast guess
        can never change the result, only the speed."""
        n = e.shape[0]
        if g < 0:
            g = 0
        elif g > n:
            g = n
        while g > 0 and e[g - 1] > v:
            g -= 1
        while g < n and e[g] <= v:
            g += 1
        return g

    # The bin GUESS may be computed with approximate (fastmath) arithmetic:
    # _fix_bracket makes the final index exact for ANY guess, so this can
    # change speed but never a single bin.
    @_numba.njit(cache=True, fastmath=True, inline="always")
    def _lguess(v, lo, dl):
        return int(_math.log(v / lo) / dl) + 1

    @_numba.njit(cache=True, inline="always")
    def _ss_glog(e, v, lo, dl):
        """searchsorted right on a geomspace grid (lo * e^{i dl}): log guess."""
        n = e.shape[0]
        if v != v:
            return n
        if v < e[0]:
            return 0
        if v >= e[n - 1]:
            return n
        return _fix_bracket(e, v, _lguess(v, lo, dl))

    @_numba.njit(cache=True, inline="always")
    def _ss_slog(e, v, lo, dl, npos):
        """searchsorted right on a SIGNED-log grid, concat(-pos[::-1], pos)
        with pos = lo * e^{i dl} (npos entries): log guess on |v|."""
        n = e.shape[0]
        if v != v:
            return n
        if v < e[0]:
            return 0
        if v >= e[n - 1]:
            return n
        if v >= lo:
            g = npos + _lguess(v, lo, dl)
        elif v <= -lo:
            g = npos - _lguess(-v, lo, dl)
        else:
            g = npos
        return _fix_bracket(e, v, g)

    @_numba.njit(cache=True, inline="always")
    def _ss_slog_g(e, v, lo, npos, gl):
        """_ss_slog with the log guess gl = _lguess(|v|, lo, dl) supplied
        (2026-10-03: one log shared by the fine and coarse grids)."""
        n = e.shape[0]
        if v != v:
            return n
        if v < e[0]:
            return 0
        if v >= e[n - 1]:
            return n
        if v >= lo:
            g = npos + gl
        elif v <= -lo:
            g = npos - gl
        else:
            g = npos
        return _fix_bracket(e, v, g)

    @_numba.njit(cache=True, inline="always")
    def _ss_glog_g(e, v, gl):
        """_ss_glog with the log guess supplied."""
        n = e.shape[0]
        if v != v:
            return n
        if v < e[0]:
            return 0
        if v >= e[n - 1]:
            return n
        return _fix_bracket(e, v, gl)

    @_numba.njit(cache=True, fastmath=True, inline="always")
    def _llog(v, lo):
        return _math.log(v / lo)

    @_numba.njit(cache=True)
    def _hist4_kernel_fast(K, G, W, ekf, egf, ekc, egc, pk_f, pg_f, pk_c, pg_c):
        """_hist4_kernel with log-guessed bin indices (identical integers).
        pk_* = (lo, dl, npos) of the signed-log kappa grids, pg_* = (lo, dl)
        of the gamma geomspace grids."""
        nkf, ngf = ekf.shape[0] + 1, egf.shape[0] + 1
        nkc, ngc = ekc.shape[0] + 1, egc.shape[0] + 1
        hf = np.zeros(nkf * ngf)
        hk = np.zeros(nkf * ngf)
        hg = np.zeros(nkf * ngf)
        hc = np.zeros(nkc * ngc)
        lkf, lgf = ekf[ekf.shape[0] - 1], egf[egf.shape[0] - 1]
        lkc, lgc = ekc[ekc.shape[0] - 1], egc[egc.shape[0] - 1]
        kfl, kfd, kfn = pk_f[0], pk_f[1], int(pk_f[2])
        kcl, kcd, kcn = pk_c[0], pk_c[1], int(pk_c[2])
        for i in range(K.shape[0]):
            k, g, w = K[i], G[i], W[i]
            a = _ss_slog(ekf, k, kfl, kfd, kfn)
            if k == lkf:
                a -= 1
            b = _ss_glog(egf, g, pg_f[0], pg_f[1])
            if g == lgf:
                b -= 1
            xy = a * ngf + b
            hf[xy] += w
            hk[xy] += w * k
            hg[xy] += w * g
            a = _ss_slog(ekc, k, kcl, kcd, kcn)
            if k == lkc:
                a -= 1
            b = _ss_glog(egc, g, pg_c[0], pg_c[1])
            if g == lgc:
                b -= 1
            hc[a * ngc + b] += w
        return hf, hk, hg, hc

    @_numba.njit(cache=True)
    def _add_direct_kernel(K, G, W, H0, H0k, H0g, Hc, betas,
                           ekf, egf, ekc, egc, pk_f, pg_f, pk_c, pg_c, k_max_fine):
        nkf, ngf = ekf.shape[0] - 1, egf.shape[0] - 1
        nkc, ngc = ekc.shape[0] - 1, egc.shape[0] - 1
        lkf, lgf = ekf[ekf.shape[0] - 1], egf[egf.shape[0] - 1]
        lkc, lgc = ekc[ekc.shape[0] - 1], egc[egc.shape[0] - 1]
        kfl, kfd, kfn = pk_f[0], pk_f[1], int(pk_f[2])
        kcl, kcd, kcn = pk_c[0], pk_c[1], int(pk_c[2])
        nord1 = betas.shape[0]
        gfl, gfd, gcd = pg_f[0], pg_f[1], pg_c[1]
        share = (kfl == kcl) and (pg_f[0] == pg_c[0])

        w_sum = 0.0
        h_sum = 0.0
        lin_sum = 0.0
        
        for i in range(K.shape[0]):
            k, g, w = K[i], G[i], W[i]
            w_sum += w
            
            # bin guesses: fine and coarse grids share their lower edge, so one
            # log per axis serves both (2026-10-03); _fix_bracket makes every
            # index exact for ANY guess -> identical bins, half the logs
            if share:
                ak = abs(k)
                if ak >= kfl:
                    lk = _llog(ak, kfl)
                    gkf = int(lk / kfd) + 1
                    gkc = int(lk / kcd) + 1
                else:
                    gkf = 0
                    gkc = 0
                if g >= gfl and g == g:
                    lg_ = _llog(g, gfl)
                    ggf = int(lg_ / gfd) + 1
                    ggc = int(lg_ / gcd) + 1
                else:
                    ggf = 0
                    ggc = 0
                a = _ss_slog_g(ekf, k, kfl, kfn, gkf)
                b = _ss_glog_g(egf, g, ggf)
                ac = _ss_slog_g(ekc, k, kcl, kcn, gkc)
                bc = _ss_glog_g(egc, g, ggc)
            else:
                a = _ss_slog(ekf, k, kfl, kfd, kfn)
                b = _ss_glog(egf, g, pg_f[0], pg_f[1])
                ac = _ss_slog(ekc, k, kcl, kcd, kcn)
                bc = _ss_glog(egc, g, pg_c[0], pg_c[1])

            # Fine grid
            if k == lkf:
                a -= 1
            if g == lgf:
                b -= 1

            if 1 <= a <= nkf and 1 <= b <= ngf:
                H0[a - 1, b - 1] += w
                H0k[a - 1, b - 1] += w * k
                H0g[a - 1, b - 1] += w * g
                h_sum += w

            # Coarse grid
            if k == lkc:
                ac -= 1
            if g == lgc:
                bc -= 1
                
            if 1 <= ac <= nkc and 1 <= bc <= ngc:
                for n in range(nord1):
                    Hc[n, ac - 1, bc - 1] += w * betas[n]
                    
            if abs(k) < k_max_fine:
                lin_sum += w * k
                
        return w_sum, h_sum, lin_sum

    @_numba.njit(cache=True)
    def _xi_add_kernel(K, G, W, iR, Jt, beta, mm, Hx, lx, ekc, egc, pk_c, pg_c):
        """xi_lin clustering deposit (2026-10-05).  For every point inside the
        coarse (kappa, gamma) box, on the SAME coarse bins as Hc:
          Hx[a, b, m, j] += beta w J_{2m}(k_j R),  m = 0..mm,
        and the exact first moments the coarse lattice centres miss:
          lx[0, j] += beta w kappa J0(k_j R),  lx[1, j] += beta w gamma J2(k_j R).
        Jt[u, m, j] = J_{2m}(k_j R_u) on the unique radii, iR = point -> u."""
        nkc, ngc = ekc.shape[0] - 1, egc.shape[0] - 1
        lkc, lgc = ekc[ekc.shape[0] - 1], egc[egc.shape[0] - 1]
        kcl, kcd, kcn = pk_c[0], pk_c[1], int(pk_c[2])
        nkp = Jt.shape[2]
        for i in range(K.shape[0]):
            k, g = K[i], G[i]
            bw = beta * W[i]
            a = _ss_slog(ekc, k, kcl, kcd, kcn)
            if k == lkc:
                a -= 1
            b = _ss_glog(egc, g, pg_c[0], pg_c[1])
            if g == lgc:
                b -= 1
            if 1 <= a <= nkc and 1 <= b <= ngc:
                u = iR[i]
                for m in range(mm + 1):
                    for j in range(nkp):
                        Hx[a - 1, b - 1, m, j] += bw * Jt[u, m, j]
                for j in range(nkp):
                    lx[0, j] += bw * k * Jt[u, 0, j]
                if mm >= 1:
                    for j in range(nkp):
                        lx[1, j] += bw * g * Jt[u, 1, j]

    @_numba.njit(cache=True)
    def _xi_add_grouped(K, G, W, order, gstart, Jt, beta, mm, Hx, lx, ekc, egc, pk_c, pg_c):
        """_xi_add_kernel with the points of one lens radius merged per coarse
        bin before the (m, k_perp) loop (2026-10-05): ~3 points share a
        (bin, R) pair in the full config, and the first moments need one
        k_perp loop per radius.  Same sums, different summation order.
        order: points sorted by R; gstart[u]..gstart[u+1] = radius u."""
        nkc, ngc = ekc.shape[0] - 1, egc.shape[0] - 1
        lkc, lgc = ekc[ekc.shape[0] - 1], egc[egc.shape[0] - 1]
        kcl, kcd, kcn = pk_c[0], pk_c[1], int(pk_c[2])
        nkp = Jt.shape[2]
        slot = np.zeros(nkc * ngc)
        mark = np.full(nkc * ngc, -1, dtype=np.int64)
        touched = np.empty(K.shape[0], dtype=np.int64)
        for u in range(gstart.shape[0] - 1):
            nt = 0
            sk = 0.0
            sg = 0.0
            for t in range(gstart[u], gstart[u + 1]):
                i = order[t]
                k, g = K[i], G[i]
                a = _ss_slog(ekc, k, kcl, kcd, kcn)
                if k == lkc:
                    a -= 1
                b = _ss_glog(egc, g, pg_c[0], pg_c[1])
                if g == lgc:
                    b -= 1
                if 1 <= a <= nkc and 1 <= b <= ngc:
                    f = (a - 1) * ngc + (b - 1)
                    bw = beta * W[i]
                    if mark[f] != u:
                        mark[f] = u
                        touched[nt] = f
                        nt += 1
                    slot[f] += bw
                    sk += bw * k
                    sg += bw * g
            for q in range(nt):
                f = touched[q]
                v = slot[f]
                slot[f] = 0.0
                a1, b1 = f // ngc, f % ngc
                for m in range(mm + 1):
                    for j in range(nkp):
                        Hx[a1, b1, m, j] += v * Jt[u, m, j]
            for j in range(nkp):
                lx[0, j] += sk * Jt[u, 0, j]
            if mm >= 1:
                for j in range(nkp):
                    lx[1, j] += sg * Jt[u, 1, j]

    @_numba.njit(cache=True, fastmath=True)
    def _bessel_pack_nb(x, j0, j1, mm, tang, Jt):
        """Packed in-place evaluation of [J0, J2, J4] into Jt (2026-10-06):
        evaluates power series directly for x < 2 and upward recurrence for
        x >= 2, avoiding temporary array allocations and boolean masking."""
        nr, nk = x.shape
        m_max = mm if tang else 0
        for i in range(nr):
            for j in range(nk):
                xv = x[i, j]
                if xv < 2.0:
                    h = 0.5 * xv
                    h2 = h * h
                    term0 = 1.0
                    ser0 = 1.0
                    term2 = 0.5
                    ser2 = 0.5
                    term4 = 1.0 / 24.0
                    ser4 = 1.0 / 24.0
                    nt = 3 if xv < 0.02 else (6 if xv < 0.3 else 14)
                    for k in range(1, nt):
                        factor = (-h2) / k
                        term0 *= factor / k
                        ser0 += term0
                        term2 *= factor / (k + 2)
                        ser2 += term2
                        term4 *= factor / (k + 4)
                        ser4 += term4
                    Jt[i, 0, j] = ser0
                    if m_max >= 1:
                        Jt[i, 1, j] = ser2 * (h * h)
                    else:
                        Jt[i, 1, j] = 0.0
                    if m_max >= 2:
                        Jt[i, 2, j] = ser4 * (h * h * h * h)
                    else:
                        Jt[i, 2, j] = 0.0
                else:
                    j0_v = j0[i, j]
                    Jt[i, 0, j] = j0_v
                    if m_max >= 1:
                        inv = 1.0 / xv
                        j1_v = j1[i, j]
                        j2_v = 2.0 * inv * j1_v - j0_v
                        Jt[i, 1, j] = j2_v
                        if m_max >= 2:
                            j3_v = 4.0 * inv * j2_v - j1_v
                            j4_v = 6.0 * inv * j3_v - j2_v
                            Jt[i, 2, j] = j4_v
                        else:
                            Jt[i, 2, j] = 0.0
                    else:
                        Jt[i, 1, j] = 0.0
                        Jt[i, 2, j] = 0.0

    @_numba.njit(cache=True)
    def _accumulate_hist4(hf, hk, hg, hc, H0, H0k, H0g, Hc, betas):
        """Merge nonempty per-object bins, preserving grouped sums exactly.

        Per-object histograms are sparse. Avoid repeatedly
        writing every empty bin and allocating a dense hc * beta**n array.
        This retains the original within-object and between-object orders.
        """
        ngf = H0.shape[1] + 2
        for a in range(H0.shape[0]):
            for b in range(H0.shape[1]):
                xy = (a + 1) * ngf + b + 1
                if hf[xy] != 0.0 or hk[xy] != 0.0 or hg[xy] != 0.0:
                    H0[a, b] += hf[xy]
                    H0k[a, b] += hk[xy]
                    H0g[a, b] += hg[xy]
        ngc = Hc.shape[2] + 2
        for n in range(Hc.shape[0]):
            bn = betas[n]
            for a in range(Hc.shape[1]):
                for b in range(Hc.shape[2]):
                    v = hc[(a + 1) * ngc + b + 1]
                    if v != 0.0:
                        Hc[n, a, b] += v * bn

    @_numba.njit(cache=True)
    def _hist4_kernel(K, G, W, ekf, egf, ekc, egc):
        """All four `Cells.add` histograms in ONE pass over the samples:
        fine (W, W*K, W*G) and coarse W, returned as FULL outlier-padded
        flat buffers so the caller slices exactly like np.histogramdd does.

        Bit-identical to np.histogram2d by construction: the same bin index
        (searchsorted side='right', last-edge values shifted into the last
        bin, C-order ravel) and np.bincount's own accumulation, which is a
        plain sequential `out[x[i]] += w[i]` into a zeroed buffer."""
        nkf, ngf = ekf.shape[0] + 1, egf.shape[0] + 1
        nkc, ngc = ekc.shape[0] + 1, egc.shape[0] + 1
        hf = np.zeros(nkf * ngf)
        hk = np.zeros(nkf * ngf)
        hg = np.zeros(nkf * ngf)
        hc = np.zeros(nkc * ngc)
        lkf, lgf = ekf[ekf.shape[0] - 1], egf[egf.shape[0] - 1]
        lkc, lgc = ekc[ekc.shape[0] - 1], egc[egc.shape[0] - 1]
        for i in range(K.shape[0]):
            k, g, w = K[i], G[i], W[i]
            a = _ss_right(ekf, k)
            if k == lkf:
                a -= 1
            b = _ss_right(egf, g)
            if g == lgf:
                b -= 1
            xy = a * ngf + b
            hf[xy] += w
            hk[xy] += w * k
            hg[xy] += w * g
            a = _ss_right(ekc, k)
            if k == lkc:
                a -= 1
            b = _ss_right(egc, g)
            if g == lgc:
                b -= 1
            hc[a * ngc + b] += w
        return hf, hk, hg, hc


if _HAVE_NUMBA:
    @_numba.njit(cache=True)
    def _count_sum(last, weight, xg, rs):
        """kappa_threshold's explicit-count sum, in the original loop order."""
        n = 0.0
        for i in range(last.shape[0]):
            if last[i] != 0:
                v = xg[last[i] - 1] * rs[i]
                n += weight[i] * (v * v)
        return n


def _hist_from_index(xy, nb, w):
    """np.histogram2d(..., weights=w)[0] from a precomputed `_hist_index`.
    Same bincount, same (nb0, nb1) layout, same outlier-stripping VIEW -- the
    view matters, because `h.sum()` downstream sums in memory order."""
    return np.bincount(xy, w, minlength=nb[0] * nb[1]).reshape(nb)[1:-1, 1:-1]


class Cells:
    """Accumulator of binned jump cells, split per z-shell for the Cox term."""

    def __init__(self, nshell, nord, xi=None):
        self.nord = nord
        # xi (2026-10-05): the xi_lin (Limber) two-point term, see xi_kperp_grid.
        # None = off (every older path).  Groups are appended per z node by
        # xi_begin / xi_end; Lam(xi=True) evaluates them.
        self.xi = xi
        self.xi_groups = []
        self._xi_cur = None
        self.fe = [_edges(FINE["k"]), np.geomspace(*FINE["g"][:2], FINE["g"][2] + 1)]
        self.ce = [_edges(COARSE["k"]), np.geomspace(*COARSE["g"][:2], COARSE["g"][2] + 1)]
        # (lo, dl[, npos]) of each log grid, from the SAME specs the edges use;
        # only a guess for the bin search, which is then corrected exactly
        def _pk(spec):
            lo, hi, n = spec
            return np.array([lo, log(hi / lo) / n, n + 1.0])
        def _pg(spec):
            lo, hi, n = spec
            return np.array([lo, log(hi / lo) / n])
        self._p = (_pk(FINE["k"]), _pg(FINE["g"]), _pk(COARSE["k"]), _pg(COARSE["g"]))
        self.H0 = np.zeros((self.fe[0].size - 1, FINE["g"][2]))
        self.H0k = np.zeros_like(self.H0)          # sum w kappa  -> bin centroid
        self.H0g = np.zeros_like(self.H0)          # sum w gamma
        self.Hc = np.zeros((nshell, nord + 1, self.ce[0].size - 1, COARSE["g"][2]))
        self.wM = np.zeros((nshell, nord + 1))     # weak arm: sum beta^n m
        self.wV = np.zeros((nshell, nord + 1))     # weak arm: sum beta^n v
        self.dropped = 0.0                          # expected count outside bins
        self.N = 0.0
        self.lin = {}                               # (shell, beta) -> sum w kappa

    def add(self, K, G, W, shell, beta, R=None, tang=True):
        """Deposit one object and return its coarse histogram for consumers.

        The returned view is independent of the accumulator and may be
        reused by analytical marked-response construction without rebinning.
        R, tang: only for the xi_lin term (self.xi), the comoving distance of
        each point's lens centre from the beam [Mpc] and whether its shear is
        tangential about that centre (halos yes, filaments no).
        """
        K, G, W = K.ravel(), G.ravel(), W.ravel()
        G = np.clip(G, 1.0000001 * FINE["g"][0], None)
        if self.xi is not None:
            if R is None:
                raise NotImplementedError(
                    "clustering='xilin' needs each point's lens distance R; this "
                    "builder path does not supply it (use edge='rvir', exact subhalos)")
            R = np.asarray(R, dtype=float).ravel()
            if R.size != K.size:
                raise ValueError("R must give one lens distance per point")
            self._xi_deposit(K, G, W, R, beta, tang)
        if _HAVE_NUMBA:
            betas = np.array([beta**n for n in range(self.nord + 1)])
            if np.all(np.isfinite(betas)):
                w_sum, h_sum, lin_sum = _add_direct_kernel(
                    np.ascontiguousarray(K, dtype=float),
                    np.ascontiguousarray(G, dtype=float),
                    np.ascontiguousarray(W, dtype=float),
                    self.H0, self.H0k, self.H0g, self.Hc[shell], betas,
                    self.fe[0], self.fe[1], self.ce[0], self.ce[1], *self._p,
                    FINE["k"][1])
                self.dropped += w_sum - h_sum
                self.N += w_sum
                key = (shell, float(beta))
                self.lin[key] = self.lin.get(key, 0.0) + lin_sum
                return None
            else:
                hf, hk, hg, hcf = _hist4_kernel_fast(
                    np.ascontiguousarray(K, dtype=float),
                    np.ascontiguousarray(G, dtype=float),
                    np.ascontiguousarray(W, dtype=float),
                    self.fe[0], self.fe[1], self.ce[0], self.ce[1], *self._p)
                nb = (self.fe[0].size + 1, self.fe[1].size + 1)
                nbc = (self.ce[0].size + 1, self.ce[1].size + 1)
                h = hf.reshape(nb)[1:-1, 1:-1]
                hc = hcf.reshape(nbc)[1:-1, 1:-1]
                self.H0 += h
                self.H0k += hk.reshape(nb)[1:-1, 1:-1]
                self.H0g += hg.reshape(nb)[1:-1, 1:-1]
                for n in range(self.nord + 1):
                    self.Hc[shell, n] += hc * betas[n]
        else:
            xy, nb = _hist_index(K, G, self.fe)      # one index, three weights
            h = _hist_from_index(xy, nb, W)
            self.H0 += h
            self.H0k += _hist_from_index(xy, nb, W * K)
            self.H0g += _hist_from_index(xy, nb, W * G)
            xyc, nbc = _hist_index(K, G, self.ce)
            hc = _hist_from_index(xyc, nbc, W)
        self.dropped += W.sum() - h.sum()
        self.N += W.sum()
        inb = (np.abs(K) < FINE["k"][1])
        key = (shell, float(beta))
        self.lin[key] = self.lin.get(key, 0.0) + float(np.sum((W * K)[inb]))
        if not _HAVE_NUMBA:
            for n in range(self.nord + 1):
                self.Hc[shell, n] += hc * beta**n
        return hc

    # ---- xi_lin two-point term (2026-10-05) -------------------------------
    XI_GROUPED = True       # False: per-point _xi_add_kernel (the gate reference)
    def xi_begin(self, dchi):
        """Start one lens-redshift node of comoving width dchi [Mpc]."""
        nkc, ngc = self.ce[0].size - 1, self.ce[1].size - 1
        nkp, mm = self.xi["kp"].size, self.xi["mmax"]
        self._xi_cur = dict(dchi=float(dchi),
                            Hx=np.zeros((nkc, ngc, mm + 1, nkp)),
                            lx=np.zeros((2, nkp)))

    def _xi_deposit(self, K, G, W, R, beta, tang):
        cur = self._xi_cur
        if cur is None:
            raise RuntimeError("xi deposit outside xi_begin / xi_end")
        import scipy.special as sp
        order = np.argsort(R, kind="stable")
        Rs = R[order]
        new = np.empty(Rs.size, bool)
        new[:1] = True
        new[1:] = Rs[1:] != Rs[:-1]
        Ru = Rs[new]
        gstart = np.append(np.flatnonzero(new), Rs.size).astype(np.int64)
        kp, mm = self.xi["kp"], self.xi["mmax"]
        x = np.outer(Ru, kp)
        Jt = np.zeros((Ru.size, mm + 1, kp.size))
        if _HAVE_NUMBA:
            j0_arr = sp.j0(x)
            j1_arr = sp.j1(x) if (tang and mm >= 1) else j0_arr
            _bessel_pack_nb(x, j0_arr, j1_arr, mm, bool(tang), Jt)
        else:
            for m, Jv in enumerate(_bessel_even(x, mm if tang else 0)):
                Jt[:, m] = Jv
        if not self.XI_GROUPED:         # per-point reference kernel (gate)
            iR = np.empty(R.size, np.int64)
            iR[order] = np.cumsum(new) - 1
            _xi_add_kernel(np.ascontiguousarray(K, dtype=float), np.ascontiguousarray(G, dtype=float),
                           np.ascontiguousarray(W, dtype=float), iR, Jt, float(beta), mm,
                           cur["Hx"], cur["lx"], self.ce[0], self.ce[1], self._p[2], self._p[3])
            return
        _xi_add_grouped(np.ascontiguousarray(K, dtype=float), np.ascontiguousarray(G, dtype=float),
                        np.ascontiguousarray(W, dtype=float), order.astype(np.int64), gstart,
                        Jt, float(beta), mm,
                        cur["Hx"], cur["lx"], self.ce[0], self.ce[1], self._p[2], self._p[3])

    def xi_add_shear_tail(self, beta, Gam, Rm, n_area):
        """Analytic shear tail beyond the emitted radius Rm (2026-10-05): kappa = 0
        and gamma = Gam / R^2 there, so its only xi contribution is linear in
        gamma (m = 1):  beta n_area int_Rm^inf 2 pi R dR (Gam / R^2) J2(k R)
        = beta n_area 2 pi Gam J1(k Rm) / (k Rm).  Without it Kaiser-Squires
        (gamma~ = kappa~ e^{2 i phi}) fails at k Rm <~ 1."""
        if self._xi_cur is None or self.xi["mmax"] < 1:
            return
        import scipy.special as sp
        a = self.xi["kp"] * Rm
        self._xi_cur["lx"][1] += beta * n_area * 2 * pi * Gam * sp.j1(a) / a

    def xi_end(self, shell):
        """Close the node: compress its rows and append them as groups.

        Per m, the node's contribution to Psi_2 is
          sign_m / 2 sum_j c_j / dchi (sum_b Hx[b, m, j] f_b + lres_j f_res)^2
        = sign_m / 2 |A^T f|^2 with A = [Hx; lres] diag(sqrt(c / dchi));
        A = U S V^T, so only the rows (S U^T)_r with s_r > XI_SVD_TOL s_0 are kept
        (dropped part <= (XI_SVD_TOL s_0)^2 |f|^2 relative to the largest mode)."""
        cur, self._xi_cur = self._xi_cur, None
        Hx, lx, dchi = cur["Hx"], cur["lx"], cur["dchi"]
        ngc = Hx.shape[1]
        sw = np.sqrt(self.xi["c"] / dchi)
        kmc = _centers(self.ce[0])
        gmc = sqrt(self.ce[1][:-1] * self.ce[1][1:])
        for m in range(Hx.shape[2]):
            H = Hx[:, :, m, :].reshape(-1, Hx.shape[3])
            idx = np.flatnonzero(np.any(H != 0, axis=1))
            if idx.size == 0:
                continue
            Hm = H[idx]
            if m == 0:      # exact sum w kappa J0 minus what the lattice centres give
                lres = lx[0] - (kmc[idx // ngc][:, None] * Hm).sum(axis=0)
            elif m == 1:    # exact sum w gamma J2 (J1(x) ~ x/2) minus the lattice's
                lres = lx[1] - (gmc[idx % ngc][:, None] * Hm).sum(axis=0)
            else:
                lres = np.zeros(Hm.shape[1])
            A = np.vstack([Hm, lres[None, :]]) * sw[None, :]
            # Tall-skinny SVD via 42x42 Gram eigendecomposition (2026-10-06):
            # A has shape (M, 42) with M >> 42; G = A^T A, eigh(G) gives exact
            # singular values and V; US = A @ V without Golub-Reinsch on A.
            G = A.T @ A
            eigvals, V = np.linalg.eigh(G)
            s_sq = eigvals[::-1]
            V_desc = V[:, ::-1]
            S = np.sqrt(np.maximum(s_sq, 0.0))
            keep = S > XI_SVD_TOL * S[0] if S[0] > 0 else np.zeros(S.size, bool)
            if not keep.any():
                continue
            rows = (A @ V_desc[:, keep]).T                 # (r, nidx + 1)
            self.xi_groups.append(dict(shell=shell, m=m, idx=idx, rows=rows))

    def add_weak(self, m, v, shell, beta):
        if self.xi is not None:
            raise NotImplementedError("clustering='xilin' has no weak (Gaussian) arm")
        key = (shell, float(beta))
        self.lin[key] = self.lin.get(key, 0.0) + m
        for n in range(self.nord + 1):
            self.wM[shell, n] += m * beta**n
            self.wV[shell, n] += v * beta**n


def build_population(cos, zs, ell=True, fils=True, nord=3, nphi=12, nx=260,
                     kthr=None, subhalo=None, nproc=1, kap_floor=None, trunc=None,
                     zint="engine", zint_n=2, xi=None):
    """Engine-grid cells.  Returns (Cells, info).  Cell (jz, jM), jz,jM >= 1,
    z_jz < zs, backward-difference dz and dlnM at the UPPER node
    (lensing.cpp:108 deltaNhfNFW); shell i = jz-1 (BiasField1D::shell).

    kap_floor (2026-10-02): None = the engine's explicit/weak split at kthr
    (explicit host inside r_max(kthr), kappa-only Gaussian weak arm beyond).
    A number = NO split: every halo is one exact jump integral (full kappa,
    gamma, ellipticity) from the centre to the radius where its kappa falls to
    kap_floor -- a numerical integration limit, not a model threshold; no
    kthr, no r_max, no weak (Gaussian / diffusion) term.  Filaments then use
    their full core (u <= 1).  Sub-lattice kappa keeps its mean exactly (bin
    centroids).

    trunc="rvir" (with kap_floor): every NFW halo is truncated at its virial
    radius (sharp 3D cut, nfw_trunc_kg) and integrated out to where its
    point-mass shear tail falls to kap_floor; the clustering lattice's mean
    is then made exact per shell (_fix_coarse_mean).

    nproc > 1: shells in a process pool (_build_targets); the result does not
    depend on nproc (2026-10-03).

    xi (2026-10-05): xi_kperp_grid(...) -> also accumulate the xi_lin (Limber)
    two-point term per lens-redshift node (Cells.xi_groups; Lam(xi=True)).
    Needs kap_floor + trunc (every point must know its lens distance)."""
    if xi is not None and (kap_floor is None or trunc is None):
        raise ValueError("xi needs the kap_floor + trunc path")
    return _build_targets(cos, [zs], ell=ell, fils=fils, nord=nord, nphi=nphi, nx=nx,
                          kthr=kthr, subhalo=subhalo, nproc=nproc, kap_floor=kap_floor,
                          trunc=trunc, zint=zint, zint_n=zint_n, xi=xi)[zs]


def build_population_multi(cos, zs_list, ell=True, fils=True, nord=3, nphi=12, nx=260,
                           subhalo=None, nproc=1, kap_floor=1e-8, trunc="rvir"):
    """build_population for SEVERAL source redshifts from ONE build (2026-10-03).
    Returns {zs: (Cells, info)}.

    Every kappa and |gamma| the builders emit -- truncated NFW (circular and
    elliptical), its monopole tail, filaments, and the decorated subhalo host
    rebuilt at M - S -- is linear in ks = rho_s r_s / Sigma_cr(z, zs), and no
    weight depends on zs.  So each lens shell is built once at zs_ref =
    max(zs_list) and fanned out to every zs > z with (K, G) scaled by
    r = Sigma_cr(z, zs_ref) / Sigma_cr(z, zs).  Only the adaptive GRID depends
    on ks: it is made the union of every target's needs -- critical-curve
    edges for every ks * r (ks_ratios), outer radius x_m and the faint-halo
    resolution of the largest ks (= zs_ref).  A superset of each standalone
    grid, so a target agrees with its own build_population to the grid's
    convergence level, not bit for bit (zs_ref too: it gets the others'
    edges).  Needs the kap_floor + truncation path (the kthr split has
    zs-dependent explicit/weak boundaries).  `subhalo` must be built at
    max(zs_list).

    Measured 2026-10-03 (full config, nproc 10, AC): {1, 3, 10} 19.5 s vs 27.4 s
    standalone; nine z_s (1 ... 10) 68 s vs 92 s -- only 1.3-1.4x, because the
    per-target binning (Cells.add, ~23 ns/point, cannot be shared: scaled
    points land in other bins) and the union edge refinement dominate.  vs
    standalone: zs_ref arrays agree to 1e-13 (4 extra cells); lower targets
    differ in faint-cell bookkeeping (N x2: cells beyond their own shear floor,
    contribution to Lambda ~ N kap_floor^2) and kbar by 5e-4..1e-3 (the finer
    grid).  NOT used by run_full_chain."""
    zs_list = sorted({float(z) for z in zs_list})
    if len(zs_list) > 1 and (kap_floor is None or trunc is None):
        raise ValueError("multi-z_s builds need kap_floor and trunc (no kthr split)")
    return _build_targets(cos, zs_list, ell=ell, fils=fils, nord=nord, nphi=nphi, nx=nx,
                          kthr=None, subhalo=subhalo, nproc=nproc, kap_floor=kap_floor,
                          trunc=trunc)


class _ShellFanout:
    """Cells-like sink for ONE z-shell (2026-10-03): forwards every deposit to
    one single-shell Cells per target source redshift, (K, G) scaled by that
    target's r (r == 1.0: the arrays themselves, so a one-target build deposits
    exactly what a plain Cells would).  Also keeps each target's sum W K
    (info["kbar"]), formed exactly as the builders form it."""

    def __init__(self, ratios, nord, xi=None):
        self.r = list(ratios)
        self.c = [Cells(1, nord, xi) for _ in self.r]
        self.kbar = [0.0] * len(self.r)

    def add(self, K, G, W, shell, beta, R=None, tang=True):
        for t, r in enumerate(self.r):
            Kt, Gt = (K, G) if r == 1.0 else (r * K, r * G)
            self.c[t].add(Kt, Gt, W, 0, beta, R=R, tang=tang)
            self.kbar[t] += float(np.sum(W * Kt))

    def xi_begin(self, dchi):
        for c in self.c:
            c.xi_begin(dchi)

    def xi_add_shear_tail(self, beta, Gam, Rm, n_area):
        for c, r in zip(self.c, self.r):
            c.xi_add_shear_tail(beta, r * Gam, Rm, n_area)

    def xi_end(self):
        for c in self.c:
            c.xi_end(0)

    def add_weak(self, m, v, shell, beta):
        for t, r in enumerate(self.r):
            self.c[t].add_weak(r * m, r * r * v, 0, beta)
            self.kbar[t] += r * m


_WCTX = None        # per-process build context (set once per worker)


def _make_ctx(cos, zs_ref, targets, nord, ell, fils, nphi, nx, kthr, subhalo, kap_floor, trunc,
              zint="engine", zint_n=2, xi=None):
    _, Ml = engine_grids()
    segs, tasks, zn = z_shells(zs_ref, zint, zint_n)
    if subhalo is not None and zint != "engine":
        # hosts are looked up by node index: give the model the node redshifts
        if getattr(subhalo, "_zgrid_key", None) != (zint, zint_n, float(zs_ref)):
            subhalo.zl = zn
            subhalo._cache = {}
            subhalo._zgrid_key = (zint, zint_n, float(zs_ref))
    return dict(cos=cos, zs_ref=zs_ref, targets=targets, nord=nord, ell=ell, fils=fils,
                nphi=nphi, nx=nx, kthr=kthr, sub=subhalo, kap_floor=kap_floor, trunc=trunc,
                zl_all=zn, Ml=Ml, dlnM=np.diff(log(Ml)), segs=segs, tasks=tasks, xi=xi,
                phis=(np.arange(nphi) + 0.5) * (0.5 * pi / nphi))   # [0, pi/2] suffices


def _shell_pieces(ctx, i):
    """Build reference shell i once (every quadrature node of it); return one
    sparse piece per target."""
    cos, zs_ref = ctx["cos"], ctx["zs_ref"]
    nodes = ctx["tasks"][i]
    z = nodes[-1][1]                    # target selection: engine shells have one node
    tidx = [t for t, zt in enumerate(ctx["targets"]) if z < zt]
    ratios = [1.0 if ctx["targets"][t] == zs_ref
              else cos.Sigma_cr(z, zs_ref) / cos.Sigma_cr(z, ctx["targets"][t]) for t in tidx]
    xi = ctx.get("xi")
    sink = _ShellFanout(ratios, ctx["nord"], xi)
    info = dict(N_expl=0.0, N_fil=0.0, V_weak=0.0, kbar=0.0)
    for jn, zn_, dz in nodes:
        if xi is not None:      # one Limber node per quadrature node, width dchi
            dzn = dz if dz is not None else ctx["zl_all"][jn] - ctx["zl_all"][jn - 1]
            sink.xi_begin(CKMS / cos.Hz(ctx["zl_all"][jn]) * dzn)
        _build_shell_into(sink, info, cos, zs_ref, i, jn, ctx["ell"], ctx["fils"], ctx["nphi"],
                          ctx["nx"], ctx["kthr"], ctx["sub"], ctx["zl_all"], ctx["Ml"], ctx["dlnM"],
                          ctx["phis"], ctx["kap_floor"], ctx["trunc"],
                          ks_ratios=tuple(r for r in ratios if r != 1.0), dz=dz)
        if xi is not None:
            sink.xi_end()
    out = []
    for t, r, c, kb in zip(tidx, ratios, sink.c, sink.kbar):
        h0, h0k, h0g = c.H0.reshape(-1), c.H0k.reshape(-1), c.H0g.reshape(-1)
        idx = np.flatnonzero((h0 != 0) | (h0k != 0) | (h0g != 0))
        out.append(dict(t=t, idx=idx, h0=h0[idx], h0k=h0k[idx], h0g=h0g[idx],
                        Hc=c.Hc[0], wM=c.wM[0], wV=c.wV[0], dropped=c.dropped, N=c.N,
                        lin={b: v for (_, b), v in c.lin.items()}, kbar=kb,
                        N_expl=info["N_expl"], N_fil=info["N_fil"], V_weak=r * r * info["V_weak"],
                        xi_groups=c.xi_groups))
    return out


def _init_build_worker(cos_kw, zs_ref, targets, nord, ell, fils, nphi, nx, kthr, sub_kw,
                       kap_floor, trunc, zint="engine", zint_n=2, xi=None):
    global _WCTX
    cos = sgl.Cosmology(**cos_kw)
    sub = None
    if sub_kw is not None:
        import subhalos
        sub = subhalos.SubhaloModel(cos, zs_ref, kthr, **sub_kw)
    _WCTX = _make_ctx(cos, zs_ref, targets, nord, ell, fils, nphi, nx, kthr, sub, kap_floor, trunc,
                      zint, zint_n, xi)


def _shell_worker(i):
    return _shell_pieces(_WCTX, i)


def _build_targets(cos, targets, ell, fils, nord, nphi, nx, kthr, subhalo, nproc,
                   kap_floor, trunc, zint="engine", zint_n=2, xi=None):
    """Shared engine of build_population / build_population_multi (2026-10-03).

    Each reference shell is built ONCE (in-process, or by a pool worker that
    builds Cosmology / SubhaloModel once at start-up and then pulls shells
    dynamically -- the 4 performance cores take more shells than the 6
    efficiency cores) and returned as per-target pieces.  Pieces are merged in
    SHELL ORDER, so the result does not depend on nproc (serial included);
    vs the pre-2026-10-03 in-place serial loop it differs at roundoff only
    (per-shell partial sums)."""
    zs_ref = max(targets)
    if zint != "engine" and (len(targets) > 1 or kap_floor is None):
        raise ValueError("zint != 'engine' needs a single z_s and the kap_floor path")
    if subhalo is not None and abs(subhalo.zs - zs_ref) > 1e-12:
        raise ValueError(f"subhalo model is at zs={subhalo.zs}, the build needs zs={zs_ref}")
    if trunc is not None and subhalo is not None and subhalo.sector != "exact":
        raise ValueError("truncated halos need the exact subhalo sector "
                         "(the pairs sector rebuilds hosts untruncated)")
    # kthr is needed only by the engine split (kap_floor None) and by a subhalo
    # model built on it (pairs sector / clump_edge "kthr"); the default
    # (kap_floor + exact sector + clump_edge "rvir") has no threshold at all
    # (2026-10-02).  A subhalo model carries its own kthr: reuse it, so the
    # parallel workers rebuild an identical model.
    if kthr is None and subhalo is not None:
        kthr = subhalo.kthr
    if kthr is None and kap_floor is None:
        kthr = kappa_threshold(cos, zs_ref)
    segs, tasks, _ = z_shells(zs_ref, zint, zint_n)
    nsh = len(tasks)
    if nproc is None or nproc <= 1:
        ctx = _make_ctx(cos, zs_ref, targets, nord, ell, fils, nphi, nx, kthr, subhalo,
                        kap_floor, trunc, zint, zint_n, xi)
        parts = [_shell_pieces(ctx, i) for i in range(nsh)]
    else:
        import os
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        sub_kw = None if subhalo is None else dict(subhalo._init_kw)
        keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")
        saved = {k: os.environ.get(k) for k in keys}
        try:
            for k in keys:            # workers single-threaded: no oversubscription
                os.environ[k] = "1"
            with ProcessPoolExecutor(
                    max_workers=min(nproc, nsh), mp_context=mp.get_context("spawn"),
                    initializer=_init_build_worker,
                    initargs=(sgl._cosmology_kwargs(cos), zs_ref, targets, nord, ell, fils,
                              nphi, nx, kthr, sub_kw, kap_floor, trunc, zint, zint_n, xi)) as ex:
                parts = list(ex.map(_shell_worker, range(nsh), chunksize=1))
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    res = {}
    for t, zt in enumerate(targets):
        it = [i for i in range(nsh) if tasks[i][-1][1] < zt]
        cells = Cells(len(it), nord, xi)
        info = dict(kthr=kthr, kap_floor=kap_floor, nshell=len(it), N_expl=0.0, N_fil=0.0,
                    V_weak=0.0, kbar=0.0, shells=[segs[i] for i in it], zint=zint,
                    zint_n=(1 if zint == "midpoint" else zint_n) if zint != "engine" else None)
        res[zt] = (cells, info)
    for i, pieces in enumerate(parts):                 # deterministic: shell order
        for pc in pieces:
            cells, info = res[targets[pc["t"]]]
            cells.H0.reshape(-1)[pc["idx"]] += pc["h0"]
            cells.H0k.reshape(-1)[pc["idx"]] += pc["h0k"]
            cells.H0g.reshape(-1)[pc["idx"]] += pc["h0g"]
            cells.Hc[i] += pc["Hc"]
            cells.wM[i] += pc["wM"]
            cells.wV[i] += pc["wV"]
            cells.dropped += pc["dropped"]
            cells.N += pc["N"]
            for b, v in pc["lin"].items():
                cells.lin[(i, b)] = cells.lin.get((i, b), 0.0) + v
            for g in pc.get("xi_groups", ()):
                cells.xi_groups.append({**g, "shell": i})
            for k in ("N_expl", "N_fil", "V_weak", "kbar"):
                info[k] += pc[k]
    if trunc is not None:
        for cells, _ in res.values():
            _fix_coarse_mean(cells)
    return res


def _fix_coarse_mean(cells):
    """Make the clustering lattice's per-shell mean exact (2026-10-02).  Lam.Q
    puts each coarse cell at its GEOMETRIC centre, and the bin straddling 0
    (|kappa| < 1e-7) at 0, so sum_b Hc kc misses the mean of faint cells.
    Cells.lin holds the exact sum w kappa per (shell, beta); the residual
    goes into wM[:, n>=1], which Lam.Q adds as i k (residual) -- exact to
    first order in kappa, which is all a sub-lattice cell carries."""
    kmc = _centers(cells.ce[0])
    nsh, nn = cells.wM.shape
    for (i, b), m in cells.lin.items():
        for n in range(1, nn):
            cells.wM[i, n] += m * b**n
    for i in range(nsh):
        for n in range(1, nn):
            cells.wM[i, n] -= cells.Hc[i, n].sum(axis=1) @ kmc


def _build_shell_into(cells, info, cos, zs, i, jz, ell, fils, nphi, nx, kthr,
                      subhalo, zl_all, Ml, dlnM, phis, kap_floor=None, trunc=None,
                      ks_ratios=(), dz=None):
    """ONE z-shell of build_population, accumulated into (cells, info).
    The pre-2026-09-25 loop body, moved verbatim (dedented) so the serial path
    stays operation-for-operation identical.
    ks_ratios (2026-10-03, multi-z_s builds): Sigma_cr(z, zs) / Sigma_cr(z, zs_t)
    of the OTHER source redshifts served by this shell; their halo amplitudes
    ks * r get their own critical-curve edges (union grid).  () = one z_s."""
    z = zl_all[jz]
    if dz is None:                      # engine rule: backward difference at the node
        dz = z - zl_all[jz - 1]
    wz = CKMS / cos.Hz(z) * dz * pi * (1 + z)**2    # x rmax^2 x dn x dlnM
    D = Dg(cos, z)
    Scr = cos.Sigma_cr(z, zs)
    for jM in range(1, NM_ENG):
        M = Ml[jM]
        dn = cos.dndlnM(M, z) * dlnM[jM - 1]
        C_, rs, ks, fC = cos.nfw_params(M, z, zs)
        bh = D * halo_bias(cos, M, z)
        nosplit = kap_floor is not None
        xt_ = None
        if nosplit and trunc is not None:
            import subhalos as _sh
            ft = 1.0 if trunc == "rvir" else float(trunc)  # r_t = ft * r_vir
            xt_ = ft * _sh._eta(cos, C_, z) * C_            # r_t / r_s
            g_xt = ks * 4 * (log(1 + xt_) - xt_ / (1 + xt_)) / xt_**2
            xm = xt_ * sqrt(g_xt / kap_floor) if g_xt > kap_floor else xt_
        else:
            xm = _xmax_for(ks, kap_floor if nosplit else kthr)
        if xt_ is not None:
            _truncated_host(cells, info, cos, i, jz, jM, M, z, rs, ks, xt_, xm, wz, dn,
                            bh, ell, phis, nphi, nx, subhalo,
                            ks_alt=tuple(ks * r for r in ks_ratios))
            xm = 0.0                    # the old explicit-host block below is skipped
        # ---- explicit host: r = rmax sqrt(U), phiH uniform
        #      (nosplit: the WHOLE halo out to kappa = kap_floor)
        if xm > 0:
            Nbar = wz * (xm * rs)**2 * dn
            info["N_expl"] += Nbar
            if nosplit:      # same ~43 points/decade as the split grid, from x = 1e-7
                x_in = min(1e-7, 1e-6 * xm)
                nn = max(nx, int(np.ceil(nx / 6.0 * np.log10(xm / x_in))))
                xe = np.concatenate([[0.0], np.geomspace(x_in, xm, nn)])
                if xt_ is not None:     # an edge exactly at the truncation
                    xe = np.unique(np.concatenate([xe, [xt_ * (1 - 1e-9), xt_, xt_ * (1 + 1e-9)]]))
                    xe = xe[xe <= xm]
            else:
                xe = np.concatenate([[0.0], xm * np.geomspace(1e-6, 1.0, nx)])
            xe = _refine_edges(xe, ks, xm)
            xc = np.where(xe[:-1] > 0, sqrt(xe[:-1] * xe[1:]), 0.5 * xe[1])
            fa = (xe[1:]**2 - xe[:-1]**2) / xm**2
            exact_sub = subhalo is not None and subhalo.sector == "exact"
            if subhalo is not None and not exact_sub:
                rr = xc * rs
                dk_sub, sk_sub, sg_sub = subhalo.host_profiles(jz, jM, rr)
            else:
                dk_sub = 0.0
            if ell:
                # per-phi radial grid hugging the PSEUDO-ELLIPTICAL critical
                # curves (the circular refinement misses them; skew was 4 sigma low)
                eps = epsilon_NFW(cos, M, z)
                Ks, Gs, Ws, xe_list = [], [], [], []
                # all 12 phi batched: edge refinement AND the final
                # kappa/gamma evaluation (bit-identical to the per-phi loop)
                xe_ps = _refine_edges_eps_batch(xe, eps, ks, xm, phis, xt_)
                xc_ps = [np.where(e[:-1] > 0, sqrt(e[:-1] * e[1:]), 0.5 * e[1])
                         for e in xe_ps]
                kcat, gcat = kappagamma_eps(
                    eps, ks, np.concatenate(xc_ps),
                    np.concatenate([np.full(c.size, p_) for c, p_ in zip(xc_ps, phis)]), xt_)
                splits = np.cumsum([c.size for c in xc_ps])[:-1]
                for xe_p, xc_p, k_p, g_p in zip(xe_ps, xc_ps,
                                                np.split(kcat, splits),
                                                np.split(gcat, splits)):
                    if subhalo is not None and not exact_sub:
                        k_p = k_p + np.interp(xc_p, xc, dk_sub)
                    Ks.append(k_p)
                    Gs.append(g_p)
                    Ws.append(Nbar * (xe_p[1:]**2 - xe_p[:-1]**2) / xm**2 / nphi)
                    xe_list.append(xc_p)
                K, G, W = np.concatenate(Ks), np.concatenate(Gs), np.concatenate(Ws)
            else:
                if xt_ is None:
                    K, G = sgl.kappa_gamma(np.ascontiguousarray(xc), ks)
                else:
                    K, G = nfw_trunc_kg(xc, ks, xt_)
                K = K + dk_sub
                W = Nbar * fa
            if exact_sub:
                # exact decorated host (subhalos_exact): replaces host-alone
                # cells, weak blur and pairs; host cells without clumps pass
                if subhalo.host(jz, jM) is not None:
                    xq = np.concatenate(xe_list) if ell else xc
                    phq = (np.concatenate([np.full(c.size, p_) for c, p_ in zip(xe_list, phis)])
                           if ell else None)
                    out = subhalo.exact.host_cells(jz, jM, xc, Nbar * fa, xq, W.ravel(), phq,
                                                   xt=xt_)
                    if out is not None:
                        K, G, W = out
            elif subhalo is not None:
                hh = subhalo.host(jz, jM)
                if hh is not None:
                    hh["_xm"], hh["_Nbar"] = xm, Nbar
                    hh["_dk_weak_fn"] = (lambda xx, _x=xc, _d=np.asarray(dk_sub) * np.ones_like(xc):
                                         np.interp(xx, _x, _d))
                    pr = subhalo.add_pairs(cells, jz, jM, i, bh, xc, Nbar * fa)
                    if pr is not None:     # host-alone weight x e^{-nu(r)}
                        xq = np.concatenate(xe_list) if ell else xc
                        W = W.ravel() * np.exp(-np.interp(xq, pr[0], pr[1]))
                K, G, W = _gh_blur(K, G, W, xc, xe_list if ell else None,
                                   rr, sk_sub, sg_sub)
            cells.add(K, G, W, i, bh)
            info["kbar"] += float(np.sum(W * K))
        # ---- weak arm: kappa-only Gaussian, r from rmax to kappa = 1e-3 kthr
        #      (absent with nosplit: the exact integral above covers it)
        r0 = xm if xm > 0 else 1e-9 / rs
        x_hi = 0.0 if nosplit else _xmax_for(ks, EPS_FLOOR * kthr)
        if x_hi > r0:
            xw = np.geomspace(r0, x_hi, 400)
            kw, _ = sgl.kappa_gamma(xw, ks)
            dA = 2 * xw**2 * np.gradient(log(xw))          # d(x^2) per node
            pref = wz * rs**2 * dn * dA
            m, v = float(np.sum(pref * kw)), float(np.sum(pref * kw**2))
            cells.add_weak(m, v, i, bh)
            info["V_weak"] += v
            info["kbar"] += m
        # ---- filaments: kappaCYL2, r uniform in rmaxF(phi=0) disk
        if fils:
            rsF = 1.0 * (M / 1e14)**(1 / 3)
            LF = 20.0 * rsF
            k0base = rsF * 14.4 * cos.rhoc0 * LF / Scr
            kc0 = 2 * pi * k0base / LF                    # phi=0 axis kappa
            if nosplit or kc0 > kthr:
                uF = 1.0 if nosplit else min(1.0, sqrt((kc0 / kthr)**2 - 1))
                NF = wz * (uF * rsF)**2 * fil.dndlnM_fil(cos, M, z) * dlnM[jM - 1]
                info["N_fil"] += NF
                ue = np.concatenate([[0.0], uF * np.geomspace(1e-5, 1.0, 120)])
                uc = np.where(ue[:-1] > 0, sqrt(ue[:-1] * ue[1:]), 0.5 * ue[1])
                fa = (ue[1:]**2 - ue[:-1]**2) / uF**2
                k0 = k0base / np.maximum(2 * rsF, np.abs(np.cos(phis)) * LF)
                K, G = fil.kappa_gamma_cyl(uc[:, None], k0[None, :])
                W = NF * fa[:, None] / nphi * np.ones_like(K)
                bf = D * fil_bias(cos, M, z)
                if _xi_on(cells):       # filament axis not tangential: m = 0 only
                    cells.add(K, G, W, i, bf, R=(uc[:, None] * (rsF * (1 + z)) * np.ones_like(K)).ravel(),
                              tang=False)
                else:
                    cells.add(K, G, W, i, bf)
                info["kbar"] += float(np.sum(W * K))




def _xi_on(cells):
    if isinstance(cells, _ShellFanout):
        return bool(cells.c) and cells.c[0].xi is not None
    return getattr(cells, "xi", None) is not None


def _truncated_host(cells, info, cos, i, jz, jM, M, z, rs, ks, xt, xm, wz, dn, bh,
                    ell, phis, nphi, nx, subhalo, ks_alt=()):
    """One (z, M) cell of truncated NFW halos (2026-10-02, speed-up of the
    first trunc="rvir" builder).  Three regions in x = r/r_s:
      inner  x <= x_ell = x_t/sqrt(1-eps): the (pseudo-elliptical) truncated
             profile on the ~43 points/decade grid with the critical-curve
             edge refinement, every phi;
      tail   x_ell < x <= x_m: every phi is outside the truncation, so the
             halo is the circular monopole, gamma = 4 ks mu(x_t)/x^2, kappa = 0:
             ONE phi and 12 points/decade (gamma is a pure power law);
    subhalos (exact sector) decorate only cells with x <= (r_t + max clump
    radius)/r_s: beyond it no clump reaches the ray (rate exactly 0)."""
    Nbar = wz * (xm * rs)**2 * dn
    info["N_expl"] += Nbar
    eps = epsilon_NFW(cos, M, z) if ell else 0.0
    x_ell = min(xt / sqrt(1 - eps), xm) if eps > 0 else min(xt, xm)
    x_in = 1e-6 * xt            # the old explicit grid's start, 1e-6 x its outer radius
    nx_eff = nx if ks > 0.05 else max(80, nx // 3)
    n_in = max(nx_eff, int(np.ceil(nx_eff / 6.0 * np.log10(x_ell / x_in))))
    xe = np.concatenate([[0.0], np.geomspace(x_in, x_ell, n_in)])
    xe = np.unique(np.concatenate([xe, [xt * (1 - 1e-9), xt, xt * (1 + 1e-9)]]))
    xe = xe[xe <= x_ell]
    xe = _refine_edges(xe, ks, x_ell, ks_alt)
    xc = np.where(xe[:-1] > 0, sqrt(xe[:-1] * xe[1:]), 0.5 * xe[1])
    has_sub = (subhalo is not None and subhalo.sector == "exact" and subhalo.host(jz, jM) is not None)
    if ell:
        xe_ps = _refine_edges_eps_batch(xe, eps, ks, x_ell, phis, xt, ks_alt)
        if len(xe_ps) == nphi and all(e is xe_ps[0] for e in xe_ps):
            xc0 = np.where(xe_ps[0][:-1] > 0, sqrt(xe_ps[0][:-1] * xe_ps[0][1:]), 0.5 * xe_ps[0][1])
            xq = np.tile(xc0, nphi)
            phq = np.repeat(phis, xc0.size)
            W0 = Nbar * (xe_ps[0][1:]**2 - xe_ps[0][:-1]**2) / xm**2 / nphi
            W = np.tile(W0, nphi)
        else:
            xc_ps = [np.where(e[:-1] > 0, sqrt(e[:-1] * e[1:]), 0.5 * e[1]) for e in xe_ps]
            xq = np.concatenate(xc_ps)
            phq = np.concatenate([np.full(c.size, p_) for c, p_ in zip(xc_ps, phis)])
            W = np.concatenate([Nbar * (e[1:]**2 - e[:-1]**2) / xm**2 / nphi for e in xe_ps])
        if not has_sub:
            K, G = kappagamma_eps(eps, ks, xq, phq, xt)
        else:
            K = G = None
    else:
        xq, phq = xc, None
        W = Nbar * (xe[1:]**2 - xe[:-1]**2) / xm**2
        if not has_sub:
            K, G = nfw_trunc_kg(xc, ks, xt)
        else:
            K = G = None
    # tail: circular monopole
    if xm > x_ell * (1 + 1e-12):
        n_out = max(8, int(np.ceil(12 * np.log10(xm / x_ell))))
        eo = np.geomspace(x_ell, xm, n_out + 1)
        xo = sqrt(eo[:-1] * eo[1:])
        Ko = np.zeros_like(xo)
        Go = ks * 4 * (log(1 + xt) - xt / (1 + xt)) / xo**2
        Wo = Nbar * (eo[1:]**2 - eo[:-1]**2) / xm**2
    else:
        xo = Ko = Go = Wo = np.zeros(0)
    # lens-centre distance of every emitted point (x = r / r_s), for the
    # xi_lin term (2026-10-05); a decorated host cell emits nq points per cell
    Xi, Xo = xq, xo
    def _rep(x, out):
        nq, rem = divmod(out[0].size, max(x.size, 1))
        if rem or nq == 0:
            raise RuntimeError("host_cells output is not nq points per input cell")
        return np.repeat(x, nq)
    # subhalos: exact decorated host, only where clumps can reach the ray
    if has_sub:
        h = subhalo.host(jz, jM)
        x_sub = (xt * rs + float(np.max(h["D"]))) / rs * 1.001
        xref = np.concatenate([xc, xo])
        xref = xref[xref <= x_sub]
        Wref = np.ones_like(xref)
        mi = xq <= x_sub
        if mi.any():
            out = subhalo.exact.host_cells(jz, jM, xref, Wref, xq[mi], W[mi],
                                           None if phq is None else phq[mi], xt=xt)
            if out is not None:
                if (~mi).any():
                    if ell:
                        Kn, Gn = kappagamma_eps(eps, ks, xq[~mi], phq[~mi], xt)
                    else:
                        Kn, Gn = nfw_trunc_kg(xq[~mi], ks, xt)
                    K = np.concatenate([Kn, out[0]])
                    G = np.concatenate([Gn, out[1]])
                    W = np.concatenate([W[~mi], out[2]])
                    Xi = np.concatenate([xq[~mi], _rep(xq[mi], out)])
                else:
                    K = out[0]
                    G = out[1]
                    W = out[2]
                    Xi = _rep(xq, out)
        mo = xo <= x_sub
        if mo.any():
            out = subhalo.exact.host_cells(jz, jM, xref, Wref, xo[mo], Wo[mo], None, xt=xt)
            if out is not None:
                Ko = np.concatenate([Ko[~mo], out[0]])
                Go = np.concatenate([Go[~mo], out[1]])
                Wo = np.concatenate([Wo[~mo], out[2]])
                Xo = np.concatenate([xo[~mo], _rep(xo[mo], out)])
    K, G, W = np.concatenate([K, Ko]), np.concatenate([G, Go]), np.concatenate([W, Wo])
    if _xi_on(cells):
        R = np.concatenate([Xi, Xo]) * (rs * (1 + z))          # comoving [Mpc]
        if R.size != K.size:
            raise RuntimeError("xi: lens distances do not match the emitted points")
        cells.add(K, G, W, i, bh, R=R, tang=True)
        Rm = xm * rs * (1 + z)
        cells.xi_add_shear_tail(bh, ks * 4 * (log(1 + xt) - xt / (1 + xt)) * (rs * (1 + z))**2,
                                Rm, Nbar / (pi * Rm**2))
    else:
        cells.add(K, G, W, i, bh)
    info["kbar"] += float(np.sum(W * K))


GH3 = (np.array([0.0, sqrt(3.0), -sqrt(3.0)]), np.array([2 / 3, 1 / 6, 1 / 6]))


def _gh_blur(K, G, W, xc, xc_phi, rr, sk, sg):
    """Attach the WEAK clump encounters' conditional fluctuation to the host
    cell: 3-node Gauss-Hermite in kappa (exact through the 5th moment of a
    Gaussian), <gamma^2> added in quadrature (random-orientation vector sum).
    xc_phi: per-phi centre arrays for elliptical hosts (sk, sg interpolated)."""
    if xc_phi is not None:
        x = np.concatenate(xc_phi)
        s_k = np.interp(x, xc, sk)
        s_g = np.interp(x, xc, sg)
    else:
        s_k, s_g = sk, sg
    K, G, W = K.ravel(), G.ravel(), W.ravel()
    Gb = sqrt(G**2 + s_g)
    sd = sqrt(np.maximum(s_k, 0.0))
    return (np.concatenate([K + a * sd for a in GH3[0]]),
            np.concatenate([Gb] * 3),
            np.concatenate([W * b for b in GH3[1]]))


def _refine_edges_eps(xe, eps, ks, xm, phi):
    """Edges hugging every zero of detA(x) of the pseudo-elliptical host at
    fixed phi_H inside (0, xm]: bisected, then eps-offset like sgl."""
    xb = np.geomspace(max(xe[1], 1e-7), xm, 1200)
    k, g = kappagamma_eps(eps, ks, xb, np.full_like(xb, phi))
    sg = np.sign((1 - k)**2 - g * g)
    idx = np.nonzero(sg[1:] * sg[:-1] < 0)[0]
    if not idx.size:
        return xe
    a, b, s0 = xb[idx].copy(), xb[idx + 1].copy(), sg[idx]
    for _ in range(45):
        m = 0.5 * (a + b)
        km, gm = kappagamma_eps(eps, ks, m, np.full_like(m, phi))
        same = np.sign((1 - km)**2 - gm * gm) == s0
        a, b = np.where(same, m, a), np.where(same, b, m)
    xc = 0.5 * (a + b)
    off = np.logspace(-11, -1.2, 22)
    add = np.concatenate([(xc[:, None] * (1 + off)).ravel(), (xc[:, None] * (1 - off)).ravel()])
    add = add[(add > xe[1]) & (add < xm)]
    return np.unique(np.concatenate([xe, add]))


def _refine_edges_eps_batch(xe, eps, ks, xm, phis, xt=None, ks_alt=()):
    """See _refine_edges_eps_batch_one; ks_alt (2026-10-03) as in _refine_edges:
    per phi, the union of the edges refined for ks and for every ks_alt.  When
    no amplitude refines anything, the shared `xe` object is returned for every
    phi (the caller's uniform-grid shortcut tests identity)."""
    if not ks_alt:
        return _refine_edges_eps_batch_one(xe, eps, ks, xm, phis, xt)
    outs = [_refine_edges_eps_batch_one(xe, eps, k_, xm, phis, xt) for k_ in (ks, *ks_alt)]
    if all(e is xe for o in outs for e in o):
        return [xe] * len(phis)
    return [np.unique(np.concatenate([o[j] for o in outs])) for j in range(len(phis))]


if _HAVE_NUMBA:
    _sgl_nfw_Fh = sgl.nfw_Fh

    @_numba.njit(cache=True)
    def _eps_det_sign(x, phi, eps, ks, xt, has_xt, gx, gw, se1, se2):
        """sign((1-k)^2 - g^2) of kappagamma_eps(eps, ks, x, phi, xt), same operations."""
        x1 = se1 * _math.cos(phi) * x
        x2 = se2 * _math.sin(phi) * x
        xe = _math.sqrt(x1 * x1 + x2 * x2)
        if xe < 1e-12:
            xe = 1e-12
        den = xe * xe
        if den < 1e-300:
            den = 1e-300
        c2 = (x1 * x1 - x2 * x2) / den
        Fv, hv = _sgl_nfw_Fh(xe)
        if has_xt:
            ku = 2.0 * 1.0 * Fv
            kbu = 4.0 * 1.0 * hv / (xe * xe)
            gu = kbu - ku
            I1, I2 = _tc_point(xe, xt, gx, gw)
            if xe < xt:
                kk = ku - 2.0 * I1
                kb = ku + gu - 4.0 * I2
            else:
                kk = 0.0
                dd = xe if xe > 1e-300 else 1e-300
                kb = 4.0 * (_math.log(1.0 + xt) - xt / (1.0 + xt)) / (dd * dd)
            k0 = ks * kk
            g0 = ks * (kb - kk)
        else:
            k0 = 2.0 * ks * Fv
            g0 = 4.0 * ks * hv / (xe * xe) - k0
        if xe < 1e-4:
            g0 = ks
        k = k0 + eps * c2 * g0
        g2 = g0 * g0 + 2.0 * eps * c2 * g0 * k0 + eps * eps * (k0 * k0 - (c2 * g0) * (c2 * g0))
        if has_xt and xe >= xt:
            xx = x if x > 1e-300 else 1e-300
            gpm = ks * 4.0 * (_math.log(1.0 + xt) - xt / (1.0 + xt)) / (xx * xx)
            k = 0.0
            g2 = gpm * gpm
        g = _math.sqrt(g2 if g2 > 0.0 else 0.0)
        d = (1.0 - k) * (1.0 - k) - g * g
        return 1 if d > 0.0 else (-1 if d < 0.0 else 0)

    @_numba.njit(cache=True)
    def _refine_eps_kernel(xb, phis, eps, ks, xt, has_xt, gx, gw, niter):
        nph, n = phis.shape[0], xb.shape[0]
        se1, se2 = _math.sqrt(1.0 - eps), _math.sqrt(1.0 + eps)
        roots = np.empty(nph * n); own = np.empty(nph * n, np.int64); m_ = 0
        for j in range(nph):
            sp = _eps_det_sign(xb[0], phis[j], eps, ks, xt, has_xt, gx, gw, se1, se2)
            for i in range(1, n):
                si = _eps_det_sign(xb[i], phis[j], eps, ks, xt, has_xt, gx, gw, se1, se2)
                if sp * si < 0:
                    a, b, s0 = xb[i - 1], xb[i], sp
                    for _ in range(niter):
                        mm = 0.5 * (a + b)
                        if _eps_det_sign(mm, phis[j], eps, ks, xt, has_xt, gx, gw, se1, se2) == s0:
                            a = mm
                        else:
                            b = mm
                    roots[m_] = 0.5 * (a + b); own[m_] = j; m_ += 1
                sp = si
        return roots[:m_], own[:m_]


def _refine_edges_eps_batch_one(xe, eps, ks, xm, phis, xt=None):
    """`_refine_edges_eps` for EVERY phi at once (2026-09-25).

    All phi share the same 1200-point search grid xb; only phi differs, and
    kappagamma_eps is elementwise with a per-element phi.  So the grid is
    evaluated once for all phi and the 45-step bisection runs once over the
    roots of all phi together, instead of 12 separate 45-step loops on
    arrays of 0-4 roots each (the profiler: 77,220 calls driving 1.24M
    kappagamma_eps calls, ~36 s per full-config build, nearly all overhead).
    Returns the list of per-phi edge arrays, in phi order, gated
    element-for-element against the per-phi function."""
    nph = len(phis)
    if ks < 0.05:
        return [xe] * nph
    # Fast screening check on 50 points: if detA > 0.02 at extremes, no roots exist for any phi
    xb_screen = np.geomspace(max(xe[1], 1e-7), xm, 50)
    k0, g0 = kappagamma_eps(eps, ks, xb_screen, np.zeros_like(xb_screen), xt)
    k1, g1 = kappagamma_eps(eps, ks, xb_screen, np.full_like(xb_screen, 0.5 * pi), xt)
    if np.all((1 - k0)**2 - g0**2 > 0.02) and np.all((1 - k1)**2 - g1**2 > 0.02):
        return [xe] * nph

    xb = np.geomspace(max(xe[1], 1e-7), xm, 1200)
    if _HAVE_NUMBA:
        # (2026-10-06) scan + 45-step bisection in ONE numba kernel, the same
        # operations as kappagamma_eps per point: 2.2x on 8019 real calls, edge
        # arrays bit-identical in 96215 of 96228 cases, the rest <= 5.6e-15 rel
        xc, own = _refine_eps_kernel(xb, np.asarray(phis, dtype=float), float(eps), float(ks),
                                     0.0 if xt is None else float(xt), xt is not None,
                                     np.asarray(_GLX, dtype=float), np.asarray(_GLW, dtype=float), 45)
        if not xc.size:
            return [xe] * nph
        off = np.logspace(-11, -1.2, 22)
        out = []
        for j in range(nph):
            xcj = xc[own == j]
            if not xcj.size:
                out.append(xe)
                continue
            add = np.concatenate([(xcj[:, None] * (1 + off)).ravel(),
                                  (xcj[:, None] * (1 - off)).ravel()])
            add = add[(add > xe[1]) & (add < xm)]
            out.append(np.unique(np.concatenate([xe, add])))
        return out
    nb = xb.size
    k, g = kappagamma_eps(eps, ks, np.tile(xb, nph), np.repeat(phis, nb), xt)
    sg = np.sign((1 - k)**2 - g * g).reshape(nph, nb)
    A, B, S0, PH, OWN = [], [], [], [], []
    for j in range(nph):
        idx = np.nonzero(sg[j, 1:] * sg[j, :-1] < 0)[0]
        if idx.size:
            A.append(xb[idx]); B.append(xb[idx + 1]); S0.append(sg[j, idx])
            PH.append(np.full(idx.size, phis[j])); OWN.append(np.full(idx.size, j))
    if not A:
        return [xe] * nph
    a, b, s0 = np.concatenate(A), np.concatenate(B), np.concatenate(S0)
    ph, own = np.concatenate(PH), np.concatenate(OWN)
    for _ in range(45):
        m = 0.5 * (a + b)
        km, gm = kappagamma_eps(eps, ks, m, ph, xt)
        same = np.sign((1 - km)**2 - gm * gm) == s0
        a, b = np.where(same, m, a), np.where(same, b, m)
    xc = 0.5 * (a + b)
    off = np.logspace(-11, -1.2, 22)
    out = []
    for j in range(nph):
        xcj = xc[own == j]
        if not xcj.size:
            out.append(xe)
            continue
        add = np.concatenate([(xcj[:, None] * (1 + off)).ravel(),
                              (xcj[:, None] * (1 - off)).ravel()])
        add = add[(add > xe[1]) & (add < xm)]
        out.append(np.unique(np.concatenate([xe, add])))
    return out


def _refine_edges(xe, ks, xm, ks_alt=()):
    """Add edges hugging every tangential/radial critical radius of the
    CIRCULAR host inside (0, xm] (the 2026-07-24 caustic-erratum annulus).
    ks_alt (2026-10-03, multi-z_s builds): further amplitudes whose critical
    radii are added too -- the union grid serves every z_s of the build.  The
    added points depend only on (ks, xe[1], xm), so the union is order-free;
    ks_alt = () is the single-z_s grid exactly."""
    for k_ in ks_alt:
        xe = _refine_edges(xe, k_, xm)
    if ks < 0.05:
        return xe
    k, g = sgl.kappa_gamma(sgl._XG, ks)
    detA = (1 - k)**2 - g * g
    sgn = np.sign(detA)
    idx = np.nonzero(sgn[1:] * sgn[:-1] < 0)[0]
    if not idx.size:
        return xe
    a, b = sgl._XG[idx].copy(), sgl._XG[idx + 1].copy()
    sgn0 = sgn[idx]
    for _ in range(sgl._BISECT_ITERS):
        m = 0.5 * (a + b)
        km, gm = sgl.kappa_gamma(m, ks)
        same = np.sign((1 - km)**2 - gm * gm) == sgn0
        a = np.where(same, m, a)
        b = np.where(same, b, m)
    xc = 0.5 * (a + b)
    eps = np.logspace(-11, -1.2, 22)
    add = np.concatenate([(xc[:, None] * (1.0 + eps[None, :])).ravel(),
                          (xc[:, None] * (1.0 - eps[None, :])).ravel()])
    add = add[(add > xe[1]) & (add < xm)]
    return np.unique(np.concatenate([xe, add])) if add.size else xe


def kappa_threshold(cos, zs, N=NHALOS):
    """findkappathr: kappa_thr such that the expected explicit count = N."""
    zl, Ml = engine_grids()
    dlnM = np.diff(log(Ml))
    rows = []
    for jz in range(1, NZ_ENG):
        z = zl[jz]
        if z >= zs:
            break
        wz = CKMS / cos.Hz(z) * (z - zl[jz - 1]) * pi * (1 + z)**2
        for jM in range(1, NM_ENG):
            C_, rs, ks, fC = cos.nfw_params(Ml[jM], z, zs)
            rows.append((ks, rs, wz * cos.dndlnM(Ml[jM], z) * dlnM[jM - 1]))
    rows = np.asarray(rows).reshape(-1, 3)
    xg = np.geomspace(1e-9, 1e7, 4000)
    # NFW convergence separates as kappa(x; ks) = ks * kappa(x; 1).
    # Retain the last point on this same radius grid above the threshold.
    unit_k, _ = sgl.kappa_gamma(xg, 1.0)
    neg_unit_k = -unit_k  # ascending for searchsorted
    ks, rs, weight = rows.T
    valid = ks > 0

    def count(kt):
        last = np.zeros(ks.size, dtype=np.intp)
        last[valid] = np.searchsorted(neg_unit_k, -kt / ks[valid], side="left")
        # Quotient rounding can move a boundary by one grid point. Compare
        # the candidate in the original multiplication direction and correct.
        prev = np.maximum(last - 1, 0)
        high = valid & (last > 0) & (ks * unit_k[prev] <= kt)
        last[high] -= 1
        nxt = np.minimum(last, xg.size - 1)
        low = valid & (last < xg.size) & (ks * unit_k[nxt] > kt)
        last[low] += 1
        if _HAVE_NUMBA:
            # same sequential order as the loop below -> identical sum, so the
            # bisection takes identical branches (2026-09-25: this Python loop
            # was ~0.4 s x 40 steps = ~16 s of every full-config PDF)
            return _count_sum(last, weight, xg, rs)
        n = 0.0
        for i in np.nonzero(last)[0]:
            n += weight[i] * (xg[last[i] - 1] * rs[i])**2
        return n
    lo, hi = log(1e-12), log(1.0)
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if count(exp(m)) > N:
            lo = m
        else:
            hi = m
    return exp(0.5 * (lo + hi))


# ================================================== correlated bias field ===
def P0(cos, k):
    """Growth-free linear P(k) [Mpc^3], same normalisation as cos.sigmaR."""
    return cos._Anorm**2 * k**cos.ns * cos.T(k)**2


# xi_lin two-point term (2026-10-05): SVD rows kept above this x the largest
XI_SVD_TOL = 1e-6


def _bessel_even(x, mmax):
    """[J0, J2, J4, ...][:mmax+1] of x (array), 20x faster than scipy jv:
    upward recurrence J_{n+1} = 2n J_n / x - J_{n-1} from scipy j0, j1 for
    x >= 2 (stable there for n <= 6), the power series below (14 terms:
    (x/2)^{2k} / (k! (k+n)!) < 1e-20 at x = 2)."""
    import scipy.special as sp
    from math import factorial
    out = [sp.j0(x)]
    if mmax == 0:
        return out
    Jm, Jn = out[0], sp.j1(x)
    inv2 = 2.0 / x
    rec = {}
    for n in range(1, 2 * mmax):
        Jm, Jn = Jn, n * inv2 * Jn - Jm
        rec[n + 1] = Jn
    xr = x.ravel()
    # series bands: terms needed for (x/2)^{2k} / (k! (k+n)!) < 1e-18 relative
    bands = [(0.0, 0.02, 3), (0.02, 0.3, 6), (0.3, 2.0, 14)]
    sel = [np.flatnonzero((xr >= lo) & (xr < hi)) for lo, hi, _ in bands]
    for m in range(1, mmax + 1):
        n = 2 * m
        Jn_ = rec[n]
        Jr = Jn_.ravel()
        for (_, _, nt), ix in zip(bands, sel):
            if ix.size == 0:
                continue
            xs = xr[ix]
            h2 = (0.5 * xs)**2
            term = np.full(xs.shape, 1.0 / factorial(n))
            ser = term.copy()
            for k in range(1, nt):
                term *= (-h2) / (k * (k + n))
                ser += term
            Jr[ix] = ser * (0.5 * xs)**n
        out.append(Jn_)
    return out


# nodes per decade of k_perp from 1e-4 Mpc^-1 (2026-10-05, z_s = 1, 3 vs 16 per
# decade: PDF <= 1.6e-5 of peak); the P(k) turnover decade 0.01-0.1 needs 12
XI_PER_DECADE = (2, 2, 12, 8, 6, 4, 4, 4)


def xi_kperp_grid(cos, kmin=1e-4, kmax=1e4, per_decade=XI_PER_DECADE, mmax=2):
    """Transverse-wavenumber quadrature of the xi_lin (Limber) two-point term.

    The supervisor's pair term, Psi_2 = 1/2 int d^3x1 d^3x2 n1 n2 xi_hh(r12) F1 F2,
    with xi_hh = b1 b2 D1 D2 xi_lin(r12) (3D, unsmoothed; F UNcompensated in
    Psi_2: clustering does not move the mean), in the Limber / flat-sky limit
    collapses the double line-of-sight integral to ONE:
      Psi_2 = 1/2 int dchi int k dk / (2 pi) P0(k) sum_m G_m(k) G_{-m}(k),
      G_m   = int d^2R sum_M b D n e^{i k_kap kappa} J_m(k_gam gamma) J_{2m}(k R)
              (- J0(k R) for m = 0),
    no z shells, no C_ij.  m != 0 are the shear harmonics: two correlated lenses
    sit on the same side of the beam, so their tangential shears align;
    G_{-m} = (-1)^m G_m.  m = 0 alone is the J0 x J0 (orientation-averaged)
    form of the 1D-field term.  Gauss-Legendre in ln k, per_decade nodes per
    decade.  Returns dict(kp, c, mmax), c_j = w_j k_j^2 P0(k_j) / (2 pi)."""
    nd = int(np.ceil(np.log10(kmax / kmin)))
    pds = [int(per_decade)] * nd if np.ndim(per_decade) == 0 else [int(p) for p in per_decade]
    if len(pds) != nd:
        raise ValueError(f"per_decade: need {nd} entries (one per decade from kmin)")
    lk, wk = [], []
    for d in range(nd):
        x, w = np.polynomial.legendre.leggauss(pds[d])
        a = log(kmin) + d * log(10.0)
        b = min(a + log(10.0), log(kmax))
        lk.append(0.5 * (a + b) + 0.5 * (b - a) * x)
        wk.append(0.5 * (b - a) * w)
    kp, wl = np.exp(np.concatenate(lk)), np.concatenate(wk)
    return dict(kp=kp, c=wl * kp**2 * P0(cos, kp) / (2 * pi), mmax=int(mmax),
                kmin=kmin, kmax=kmax, per_decade=pds)


def shell_covariance(cos, zs, Rs=20.0, zint="engine", zint_n=2):
    """BiasField1D::build (lensing.cpp:497) for bias_window=1: Cov of the
    per-shell segment averages of the 1D field, EXACT mode sum over
    k_q = 2 pi q / L, q = 1..max(4, floor(L/Rs)), L = 1.05 chi(zs), with
    P_1D(k) = (1/2pi) int_k^inf K P0(K) W_TH^2(K Rs) dK (the engine's
    k_perp integral after K^2 = k^2 + k_perp^2)."""
    # shells (2026-10-05): those of z_shells(zs, zint), the same as the build's
    segs, _, _ = z_shells(zs, zint, zint_n)
    a = np.array([cos.chi(lo) for lo, _ in segs])
    b = np.array([cos.chi(hi) for _, hi in segs])
    L = 1.05 * cos.chi(zs)
    Nmax = max(4, int(np.floor(L / Rs)))
    kq = 2 * pi * np.arange(1, Nmax + 1) / L
    Kg = np.geomspace(kq[0] * 0.5, 200.0 / Rs, 20000)
    x = Kg * Rs
    W = np.where(x < 1e-2, 1 - x**2 / 10 * (1 - x**2 / 28),
                 3 * (np.sin(x) - x * np.cos(x)) / x**3)
    f = Kg * P0(cos, Kg) * W**2
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(Kg))])
    P1D = (cum[-1] - np.interp(kq, Kg, cum)) / (2 * pi)
    Lh = b - a
    rw = sqrt(2 * P1D / L)
    cq = rw[:, None] * (np.sin(kq[:, None] * b) - np.sin(kq[:, None] * a)) / (kq[:, None] * Lh)
    sq = rw[:, None] * (np.cos(kq[:, None] * a) - np.cos(kq[:, None] * b)) / (kq[:, None] * Lh)
    return cq.T @ cq + sq.T @ sq


def limber_lss_var(cos, zs, lmin=1.0, lmax=1e7, nchi=600):
    """Gaussian two-halo (linear large-scale-structure) convergence variance of a
    point source, Limber with the chain's own linear P (2026-10-02):

      sigma2 = int_0^chi_s dchi W(chi)^2 Dg(z)^2 int_{lmin/chi}^{lmax/chi} dk k P0(k) / (2 pi),
      W = (3/2) Om H0^2 (1+z) chi (chi_s - chi) / chi_s,

    i.e. int dl l/(2 pi) C_l with C_l = int dchi W^2/chi^2 P_lin(l/chi, z), and the
    growth Dg that multiplies the bias in the engine's field.  The shear of the
    same field has the same total variance, split equally over gamma_1, gamma_2.
    With r_vir-truncated halos this is the two-halo term of the halo model
    (P_2h ~ P_lin); `tmp/hmcode_check.py` matches HMcode-2020 to +-3% with it."""
    from scipy.interpolate import interp1d
    chis = cos.chi(zs)
    zt = np.linspace(0.0, zs, 400)
    ct = np.array([cos.chi(z) for z in zt])
    chi = np.linspace(chis * 1e-4, chis * (1 - 1e-4), nchi)
    z = np.interp(chi, ct, zt)
    H0 = 100.0 * cos.h / CKMS
    W = 1.5 * cos.Om * H0**2 * (1 + z) * chi * (chis - chi) / chis
    D = np.array([Dg(cos, zz) for zz in z])
    Kg = np.geomspace(1e-7, 1e5, 40000)
    f = Kg * P0(cos, Kg) / (2 * pi)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(Kg))])
    I = np.interp(lmax / chi, Kg, cum) - np.interp(lmin / chi, Kg, cum)
    return float(np.trapezoid(W**2 * D**2 * I, chi))


# ======================================================== Lambda assembly ===
def _j0m1(x):
    """J0(x) - 1 without cancellation at small x."""
    import scipy.special as sp
    out = sp.j0(x) - 1.0
    s = x < 1e-3
    out[s] = -x[s]**2 / 4 * (1 - x[s]**2 / 16)
    return out


if _HAVE_NUMBA:
    @_numba.njit(cache=True, parallel=True)
    def _sep_lam_kernel(kk, k0, w, wl, cols, ng):
        """Separable Lambda (2026-10-03), the part exact in kappa.  For every k_n:
          A1[n]   = sum_c w_c (e^{i k_n kap_c} - 1 - i k_n kap_c)   (cancellation-free)
          T[n, q] = sum_c wl[c, j] e^{i k_n kap_c},  q = cols[c, j] (gamma node; -1 = none)
        Rows are independent (prange over n, each row written by one thread)."""
        nk, nc, m = kk.shape[0], k0.shape[0], cols.shape[1]
        Tr = np.zeros((nk, ng))
        Ti = np.zeros((nk, ng))
        A1 = np.zeros(nk, dtype=np.complex128)
        for n in _numba.prange(nk):
            k = kk[n]
            ar = 0.0
            ai = 0.0
            for c in range(nc):
                ph = k * k0[c]
                sn = _math.sin(ph)
                hs = _math.sin(0.5 * ph)
                cm1 = -2.0 * hs * hs                       # cos - 1
                if abs(ph) < 1e-2:
                    p2 = ph * ph
                    smp = -ph * p2 / 6.0 * (1.0 - p2 / 20.0 * (1.0 - p2 / 42.0))   # sin - ph
                else:
                    smp = sn - ph
                ar += w[c] * cm1
                ai += w[c] * smp
                cr = cm1 + 1.0
                for j in range(m):
                    q = cols[c, j]
                    if q >= 0:
                        Tr[n, q] += wl[c, j] * cr
                        Ti[n, q] += wl[c, j] * sn
            A1[n] = ar + 1j * ai
        return A1, Tr, Ti

    @_numba.njit(cache=True)
    def _scatter_add(out, idx, vals):
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                out[i, idx[j]] += vals[i, j]
else:
    def _scatter_add(out, idx, vals):
        np.add.at(out, (slice(None), idx), vals)


def _sep_gamma_nodes(g0, w, ig, eg, kg_max, theta, mg, smax):
    """Gamma interpolation of the separable Lambda (2026-10-03).  Each occupied
    fine gamma bin is cut into ns = ceil(kg_max h / (2 theta)) equal sub-bins
    (so kg_max * half-width <= theta) with mg Chebyshev nodes each; a cell's
    weight is spread onto its sub-bin's nodes with the Lagrange weights of its
    centroid, so B(g, gamma_c) = J0(g gamma_c) - 1 is replaced by its degree
    mg-1 interpolant.  Bins needing more than smax sub-bins: the cell is summed
    directly (returned mask).  Returns (nodes, wl, cols, direct)."""
    lo, hi = eg[ig], eg[ig + 1]
    ns = np.maximum(1, np.ceil(kg_max * (hi - lo) / (2 * theta))).astype(np.int64)
    direct = ns > smax
    sep = ~direct
    j = np.minimum(((g0 - lo) / (hi - lo) * ns).astype(np.int64), ns - 1)
    sl = lo + j * (hi - lo) / ns
    sh = lo + (j + 1) * (hi - lo) / ns
    t = np.cos((2 * np.arange(mg) + 1) * pi / (2 * mg))[::-1]
    ub, inv = np.unique(np.round(sl[sep], 14), return_inverse=True)
    lo_s, hi_s = sl[sep], sh[sep]
    nodes = np.empty((ub.size, mg))
    nodes[inv] = 0.5 * (lo_s + hi_s)[:, None] + 0.5 * (hi_s - lo_s)[:, None] * t[None, :]
    u = (2 * g0[sep] - lo_s - hi_s) / (hi_s - lo_s)
    L = np.ones((u.size, mg))
    for a in range(mg):
        for b in range(mg):
            if a != b:
                L[:, a] *= (u - t[b]) / (t[a] - t[b])
    wl = np.zeros((g0.size, mg))
    cols = -np.ones((g0.size, mg), np.int64)
    wl[sep] = w[sep, None] * L
    cols[sep] = inv[:, None] * mg + np.arange(mg)
    return nodes.ravel(), wl, cols, direct


def _lam_worker_block(payload):
    """Worker evaluating one block of kg columns for Lam._direct_call."""
    k0, g0, w, kk, kg_sub, kg_block = payload
    import scipy.special as sp
    out = np.empty((kk.size, kg_sub.size), complex)
    for glo in range(0, kg_sub.size, kg_block):
        ghi = min(glo + kg_block, kg_sub.size)
        arg = np.outer(g0, kg_sub[glo:ghi])
        J = sp.j0(arg)
        Jm1 = J - 1.0
        s = arg < 1e-3
        if np.any(s):
            xs = arg[s]
            Jm1[s] = -xs**2 / 4 * (1 - xs**2 / 16)
        base_g = w @ Jm1                               # (block,)
        for lo in range(0, kk.size, 64):
            hi = min(lo + 64, kk.size)
            ph = np.outer(kk[lo:hi], k0)
            re = ((np.cos(ph) - 1) * w) @ J
            im = ((np.sin(ph) - ph) * w) @ J + (ph * w) @ Jm1
            out[lo:hi, glo:ghi] = (re + 1j * im) + base_g[None, :]
        del J, Jm1, arg
    return out


def _bins(H, edges):
    return _centers(edges[0]), sqrt(edges[1][:-1] * edges[1][1:])


class Lam:
    """ln Phi(k_kap, k_gam) for a built population.

      Lambda_0 = sum_b w_b [e^{ik kap_b} J0(k_g gam_b) - 1 - ik kap_b]
                 - 1/2 k^2 V_weak                                 (Poisson part)
      Q        = 1/2 sum_{n>=1} (1/n!) sum_ij C_ij^n B_{n,i} B_{n,j}   (Cox)
      B_{n,i}  = sum_{c in shell i} beta_c^n a_c(k),
      a_c      = w (e^{ik kap}J0 - 1)  [explicit]  or  ik m - k^2 v/2  [weak]

    Q is the EXACT second cumulant of the lognormal count modulation
    lambda = exp(beta dbar - beta^2 sigma^2/2) (Mehler / Hermite generating
    function: Cov(lambda_c, lambda_c') = e^{beta beta' C_ij} - 1).
    """

    # exactS defaults to False (2026-10-02): the exact-S sampler (_build_S) is
    # the only RNG in the chain, so it must be asked for explicitly
    # (run_full_chain --clustering closure passes exactS=True).
    def __init__(self, cells, C=None, nord=None, chunk=8, k3=True, exactS=False,
                 nsamp=2_000_000, seed=0, centroid=True, s_taper="cos2",
                 s_is=True, is_alpha=0.5, is_levels=(0.1, 0.4, 1.0),
                 q_coarse=101, lss_var=0.0, lam_coarse=0, lam_method="sep", xi=False,
                 xi_coarse=51, xi_prune=1e-7):
        # lss_var (2026-10-02): variance of a Gaussian large-scale-structure
        # convergence (limber_lss_var), with the same total shear variance:
        # adds -1/2 lss_var (k_kap^2 + k_gam^2 / 2) to ln Phi.  0 = off.
        self.lss_var = float(lss_var)
        if s_taper not in ("cos2", "linear"):
            raise ValueError("s_taper must be 'cos2' or 'linear'")
        self.s_taper = s_taper
        self.s_is, self.is_alpha, self.is_levels = s_is, is_alpha, tuple(is_levels)
        self.q_coarse = int(q_coarse) if q_coarse else None
        self.lam_coarse = int(lam_coarse) if lam_coarse else 0
        # lam_method (2026-10-03): "sep" = separable evaluation of the cell sum
        # (_sep_raw; z_s=1 full grid 0.7 s vs 23.6 s, max|dLambda| 7e-12);
        # "direct" = the dense (k_kappa x cells) @ (cells x k_gamma) sum.
        if lam_method not in ("sep", "direct"):
            raise ValueError(f"lam_method must be 'sep' or 'direct', got {lam_method!r}")
        self.lam_method = lam_method
        self._sep_cache = {}
        # q_spline (2026-10-06): "tensor" (default) or "rgi" (the older scattered
        # RegularGridInterpolator evaluation of the same coarse Q grid)
        self.q_spline = "tensor"
        self.c = cells
        self.k3 = k3
        self.exactS = exactS and C is not None
        self.C = C
        self.nord = cells.nord if nord is None else nord
        self.chunk = chunk
        H0 = cells.H0
        km, gm = _bins(H0, cells.fe)
        ik, ig = np.nonzero(H0)
        self._ig = ig                  # fine gamma bin of each cell (separable Lambda)
        self.w0 = H0[ik, ig]
        if centroid:
            # weight-averaged (kappa, gamma) per bin: first moments exact per
            # bin -- geometric centres cost ~16% in mu near kappa+gamma ~ 1
            self.k0 = cells.H0k[ik, ig] / self.w0
            self.g0 = cells.H0g[ik, ig] / self.w0
        else:
            self.k0, self.g0 = km[ik], gm[ig]
        self.V = float(cells.wV[:, 0].sum())
        if C is not None:
            kmc, gmc = _bins(None, cells.ce)
            nz = np.nonzero(cells.Hc[:, 1:].sum(axis=(0, 1)))
            self.kc, self.gc = kmc[nz[0]], gmc[nz[1]]
            self.Hc = cells.Hc[:, :, nz[0], nz[1]]          # (nsh, nord+1, nb)
            # Separable form for Q (2026-09-25).  A coarse bin b IS a grid cell
            # (kappa-bin p, gamma-bin q): the phase e^{ik kc_b} depends on b
            # only through p and J0(gc_b kg) only through q.  Scattering Hc
            # onto the dense (p, q) grid turns Q's (9644-bin) contraction into
            # a p-contraction followed by a q-contraction: ~86x fewer GEMM
            # flops at z_s=1 (260 x 102 grid, 36% filled).  Mathematically the
            # same finite sum; only the summation ORDER changes.
            up, pinv = np.unique(nz[0], return_inverse=True)
            uq, qinv = np.unique(nz[1], return_inverse=True)
            self.kcu, self.gcu = kmc[up], gmc[uq]
            nsh, no = self.Hc.shape[:2]
            Ht = np.zeros((nsh, no, up.size, uq.size))
            Ht[:, :, pinv, qinv] = self.Hc
            self.Ht = Ht                                     # (nsh, nord+1, nP, nQ)
        if self.exactS:
            self._build_S(C, nsamp, seed)
        # xi (2026-10-05): the xi_lin Limber two-point term of the build
        # (Cells.xi_groups, xi_kperp_grid); independent of C (no shells)
        self.xi_on = bool(xi)
        if self.xi_on:
            if not cells.xi_groups:
                raise ValueError("Lam(xi=True) needs a population built with xi=")
            kmc = _centers(cells.ce[0])
            gmc = sqrt(cells.ce[1][:-1] * cells.ce[1][1:])
            ngc = cells.ce[1].size - 1
            self._kmc = kmc
            self._gmc = gmc
            self._xg = []
            m0_rows, m0_res, m0_sum = [], [], []
            stack_g_rows = {0: [], 1: [], 2: []}
            stack_g_res = {0: [], 1: [], 2: []}
            stack_g_sum = {0: [], 1: [], 2: []}
            self.xi_coarse = int(xi_coarse) if xi_coarse else 0
            # xi_prune (2026-10-05): drop rows whose norm is < xi_prune x the
            # largest of ANY node (the per-node SVD tolerance is relative to that
            # node only); 1e-6 at z_s=10: 11982 -> 7680 rows, PDF 4.3e-6 of peak
            groups = cells.xi_groups
            if xi_prune:
                s0 = max(float(np.max(np.linalg.norm(g["rows"], axis=1))) for g in groups)
                groups = []
                for g in cells.xi_groups:
                    k = np.linalg.norm(g["rows"], axis=1) > xi_prune * s0
                    if k.any():
                        groups.append({**g, "rows": g["rows"][k]})
            for g in groups:
                up, pinv = np.unique(g["idx"] // ngc, return_inverse=True)
                uq, qinv = np.unique(g["idx"] % ngc, return_inverse=True)
                m = int(g["m"])
                rows_act = g["rows"][:, :-1]
                res_act = g["rows"][:, -1]
                sum_rows = rows_act.sum(axis=1)
                r = rows_act.shape[0]
                self._xg.append(dict(m=m, sign=1.0 if m == 0 else 2.0 * (-1)**m,
                                     kp=kmc[up], gq=gmc[uq], pinv=pinv, qinv=qinv,
                                     idx_p=up, idx_q=uq,
                                     rows=rows_act, res=res_act, sum_rows=sum_rows,
                                     kc=kmc[g["idx"] // ngc]))
                # Stack for 1D reach along k (m=0 only, J0(0)=1, all m>=1 vanish)
                if m == 0:
                    W_global = np.zeros((r, kmc.size))
                    _scatter_add(W_global, g["idx"] // ngc, rows_act)
                    m0_rows.append(W_global)
                    m0_res.append(res_act)
                    m0_sum.append(sum_rows)
                # Stack for 1D reach along g (per m, phase=1)
                if m in stack_g_rows:
                    U_global = np.zeros((r, gmc.size))
                    _scatter_add(U_global, g["idx"] % ngc, rows_act)
                    stack_g_rows[m].append(U_global)
                    stack_g_res[m].append(res_act)
                    stack_g_sum[m].append(sum_rows)
            self._W0_all = np.vstack(m0_rows) if m0_rows else None
            self._res0_all = np.concatenate(m0_res) if m0_res else None
            self._sum0_all = np.concatenate(m0_sum) if m0_sum else None
            self._stack_g = {}
            for m in (0, 1, 2):
                if stack_g_rows[m]:
                    self._stack_g[m] = dict(U=np.vstack(stack_g_rows[m]),
                                            res=np.concatenate(stack_g_res[m]),
                                            sum=np.concatenate(stack_g_sum[m]),
                                            sign=1.0 if m == 0 else 2.0 * (-1)**m)

    def _Qxi(self, kk, kg):
        """xi_lin two-point term (2026-10-05), see xi_kperp_grid:
          sum_groups sign_m / 2 sum_r G_r^2,
          G_r = sum_b rows_rb f_m(b) + res_r f_res,
          f_0 = e^{ik kap_b} J0(kg gam_b) - 1, f_res = i kk;  f_1 = e^{ik kap_b} J1, f_res = kg/2;
          f_m = e^{ik kap_b} J_m (m >= 2).
        Separable per group: sum_p e^{ik kap_p} sum_q H[r, p, q] J_m(kg gam_q)."""
        kk, kg = np.asarray(kk, float), np.asarray(kg, float)

        # 1D reach along k (k_gamma = 0: m >= 1 vanish identically, J0(0) = 1)
        if kg.size == 1 and kg[0] == 0.0 and self._W0_all is not None:
            ph = np.outer(kk, self._kmc)
            E = np.exp(1j * ph)
            G = E @ self._W0_all.T
            G -= self._sum0_all[None, :]
            G += 1j * kk[:, None] * self._res0_all[None, :]
            out = np.zeros((kk.size, 1), complex)
            out[:, 0] = 0.5 * np.sum(G * G, axis=1)
            return out

        # 1D reach along gamma (k_kappa = 0: phase is identically 1)
        if kk.size == 1 and kk[0] == 0.0 and self._stack_g:
            import scipy.special as sp
            out = np.zeros((1, kg.size), complex)
            xg = np.outer(self._gmc, kg)
            max_m = max(self._stack_g.keys())
            J_tables = {0: sp.j0(xg)}
            if max_m >= 1:
                J_tables[1] = sp.j1(xg)
            for m in range(2, max_m + 1):
                J_tables[m] = sp.jv(m, xg)
            for m, d in self._stack_g.items():
                G = d["U"] @ J_tables[m]
                if m == 0:
                    G -= d["sum"][:, None]
                elif m == 1:
                    G += 0.5 * kg[None, :] * d["res"][:, None]
                out[0, :] += 0.5 * d["sign"] * np.sum(G * G, axis=0)
            return out

        # General 2D grid evaluation (coarse grid)
        import scipy.special as sp
        ph_all = np.outer(kk, self._kmc)
        Er_all, Ei_all = np.cos(ph_all), np.sin(ph_all)
        xg_all = np.outer(self._gmc, kg)
        max_m = max(g["m"] for g in self._xg)
        J_tables = {0: sp.j0(xg_all)}
        if max_m >= 1:
            J_tables[1] = sp.j1(xg_all)
        for m in range(2, max_m + 1):
            J_tables[m] = sp.jv(m, xg_all)

        out = np.zeros((kk.size, kg.size), complex)
        for g in self._xg:
            r, m = g["rows"].shape[0], g["m"]
            nP, nQ = g["kp"].size, g["gq"].size
            Hd = np.zeros((r, nP, nQ))
            Hd[:, g["pinv"], g["qinv"]] = g["rows"]
            Jm = J_tables[m][g["idx_q"], :]
            T = (Hd.reshape(r * nP, nQ) @ Jm).reshape(r, nP, kg.size)
            Er = Er_all[:, g["idx_p"]]
            Ei = Ei_all[:, g["idx_p"]]
            Gr = np.matmul(Er, T)
            Gi = np.matmul(Ei, T)
            if m == 0:
                Gr -= g["sum_rows"][:, None, None]
                Gi += kk[None, :, None] * g["res"][:, None, None]
            elif m == 1:
                Gr += 0.5 * kg[None, None, :] * g["res"][:, None, None]
            re = np.sum(Gr**2 - Gi**2, axis=0)
            im = 2.0 * np.sum(Gr * Gi, axis=0)
            out += (0.5 * g["sign"]) * (re + 1j * im)
        return out

    def _Q_interpolated(self, kk, kg, fn=None, n=None):
        """Coarse-grid evaluation + 2D cubic spline interpolation of Q(k_k, k_g).
        Evaluates Q on a nested coarse grid (k = kmax s^2, uniform in s), then
        interpolates with bicubic splines in s = sqrt(k / kmax).
        On dense inversion grids (e.g. 1657 x 3359 = 5.56M points at z_s=0.5),
        this reduces Q evaluation from ~900 s to ~1.3 s (700x) while matching
        un-interpolated moments to ~2e-5 relative in variance and ~4e-4 in skew.
        """
        n = self.q_coarse if n is None else int(n)
        kk_max, kg_max = float(kk[-1]), float(kg[-1])
        s = np.linspace(0.0, 1.0, n)
        kk_c = kk_max * s**2
        kg_c = kg_max * s**2
        Q_c = (self.Q if fn is None else fn)(kk_c, kg_c)
        sk = np.sqrt(np.clip(kk / kk_max, 0.0, 1.0))
        sg = np.sqrt(np.clip(kg / kg_max, 0.0, 1.0))
        if self.q_spline == "rgi":          # pre-2026-10-06 path, kept for A/B
            from scipy.interpolate import RegularGridInterpolator
            re_ip = RegularGridInterpolator((s, s), Q_c.real, method="cubic")
            im_ip = RegularGridInterpolator((s, s), Q_c.imag, method="cubic")
            Q_out = np.empty((kk.size, kg.size), complex)
            for lo in range(0, sk.size, 128):
                hi = min(lo + 128, sk.size)
                P = np.stack(np.meshgrid(sk[lo:hi], sg, indexing="ij"), axis=-1).reshape(-1, 2)
                Q_out[lo:hi] = (re_ip(P) + 1j * im_ip(P)).reshape(hi - lo, sg.size)
            return Q_out
        # Tensor-product evaluation (2026-10-06): the bicubic not-a-knot spline,
        # built axis by axis, is Q = B_k C B_g^T on the output grid (B = sparse
        # B-spline design matrices, 4 nonzeros per row) -- two small products
        # instead of a scattered evaluation at every grid point.  27x faster at
        # z_s = 0.5 (2.23 -> 0.08 s); vs the EXACT Q on a 60x60 subgrid both
        # this and the old RegularGridInterpolator cubic have max|dQ| 1.7e-3
        # (the 101^2 coarse grid), and they differ from each other by 1.2e-5.
        from scipy.interpolate import make_interp_spline, BSpline
        Q_out = np.empty((kk.size, kg.size), complex)
        for part, setter in ((Q_c.real, "real"), (Q_c.imag, "imag")):
            sp0 = make_interp_spline(s, part, k=3, axis=0)
            sp1 = make_interp_spline(s, sp0.c, k=3, axis=1)     # c is (nb_g, nb_k)
            val = BSpline.design_matrix(sk, sp0.t, 3) @ (BSpline.design_matrix(sg, sp1.t, 3) @ sp1.c).T
            setattr(Q_out, setter, val)
        return Q_out

    # k_gamma columns per J0 block.  The J0 / J0-1 tables are (n_fine x n_kg):
    # 61639 x 2687 float64 = 1.32 GB EACH at z_s=1, plus a same-size np.outer
    # temporary -> a ~4 GB transient that, on a 16 GB laptop, swapped ~5 GB
    # per inversion and made its time swing 34-78 s between identical runs
    # (2026-09-25).  Blocking caps the tables at ~0.25 GB each; the price is
    # recomputing the phase trig once per block (~1-2 s).
    KG_BLOCK = 512
    # Separable Lambda (2026-10-03).  theta = kg_max x gamma sub-bin half-width,
    # mg Chebyshev nodes per sub-bin: z_s=1 full grid, vs direct: (0.5, 4)
    # 3.9e-6, (0.5, 6) 6.5e-9, (0.5, 8) 7.1e-12 max|dLambda| (max|dPhi| 1.3e-13,
    # the direct sum's own roundoff).  Small calls (the _reach probes) stay
    # direct, so the adaptive k-grid is chosen exactly as before.
    SEP_THETA, SEP_MG, SEP_SMAX = 0.5, 8, 64
    SEP_MIN_POINTS = 200_000

    def _sep_raw(self, kk, kg):
        """sum_c w_c (e^{i kk kap_c} J0(kg gam_c) - 1 - i kk kap_c), separably:
          = sum_c w_c A(kk, kap_c)  +  sum_c w_c e^{i kk kap_c} B(kg, gam_c),
        A = e^{ix} - 1 - ix (1D, exact per cell), B = J0 - 1 interpolated in
        gamma on _sep_gamma_nodes (exact in kappa).  Every term is O(B) or O(A),
        so the interpolation error scales with the faint cells' tiny B, not
        with sum w (~4e6 at z_s=1, which killed a full-e^{ikk} spreading).
        Cost: one numba pass (kk x cells) + one (kk x nodes)(nodes x kg) GEMM."""
        key = float(kg[-1])
        st = self._sep_cache.get(key)
        if st is None:
            st = _sep_gamma_nodes(self.g0, self.w0, self._ig, self.c.fe[1], key,
                                  self.SEP_THETA, self.SEP_MG, self.SEP_SMAX)
            self._sep_cache = {key: st}
        nodes, wl, cols, direct = st
        A1, Tr, Ti = _sep_lam_kernel(np.ascontiguousarray(kk, dtype=float), self.k0,
                                     self.w0, wl, cols, nodes.size)
        Bn = _j0m1(np.outer(nodes, kg))
        out = Tr @ Bn + 1j * (Ti @ Bn)
        del Tr, Ti
        out += A1[:, None]
        if direct.any():
            E = np.exp(1j * np.outer(kk, self.k0[direct])) * self.w0[direct]
            out += E @ _j0m1(np.outer(self.g0[direct], kg))
        return out

    def _direct_call(self, kk, kg, nproc=1):
        if nproc > 1 and kg.size >= 2 * self.KG_BLOCK:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            kg_chunks = [c for c in np.array_split(kg, nproc) if c.size > 0]
            payloads = [(self.k0, self.g0, self.w0, kk, sub_kg, self.KG_BLOCK) for sub_kg in kg_chunks]
            with ProcessPoolExecutor(max_workers=nproc, mp_context=mp.get_context("spawn")) as ex:
                results = list(ex.map(_lam_worker_block, payloads))
            out = np.hstack(results)
        else:
            if (self.lam_method == "sep" and _HAVE_NUMBA
                    and kk.size * kg.size >= self.SEP_MIN_POINTS and kg.size > 1):
                out = self._sep_raw(kk, kg)
            else:
                out = _lam_worker_block((self.k0, self.g0, self.w0, kk, kg, self.KG_BLOCK))
        out -= 0.5 * np.outer(kk**2 * self.V, np.ones(kg.size))
        if self.lss_var:
            out -= 0.5 * self.lss_var * (kk[:, None]**2 + 0.5 * kg[None, :]**2)
        if self.C is not None:
            if self.q_coarse and (kk.size * kg.size > self.q_coarse**2) and kk.size > 1 and kg.size > 1 and kk[-1] > 0 and kg[-1] > 0:
                out += self._Q_interpolated(kk, kg)
            else:
                out += self.Q(kk, kg)
        if self.xi_on:
            # own coarse grid (2026-10-05): 51^2 vs 101^2 moves the z_s=10 PDF
            # by 3.6e-8 of peak (the two-halo term is smooth), invert 4.3 -> 1.4 s
            nx_ = self.xi_coarse
            if nx_ and (kk.size * kg.size > nx_**2) and kk.size > 1 and kg.size > 1 and kk[-1] > 0 and kg[-1] > 0:
                out += self._Q_interpolated(kk, kg, fn=self._Qxi, n=nx_)
            else:
                out += self._Qxi(kk, kg)
        if self.exactS:
            out += self.S_correction(kk)[:, None]
        return out

    def _lam_interpolated(self, kk, kg, nproc=1):
        """Coarse-grid evaluation + 2D cubic spline interpolation of Lambda(k_k, k_g).
        Evaluates Lambda on a nested coarse grid (k = kmax s^2, uniform in s), then
        interpolates with bicubic splines in s = sqrt(k / kmax).
        """
        from scipy.interpolate import RegularGridInterpolator
        n = self.lam_coarse
        kk_max, kg_max = float(kk[-1]), float(kg[-1])
        s = np.linspace(0.0, 1.0, n)
        kk_c = kk_max * s**2
        kg_c = kg_max * s**2
        L_c = self._direct_call(kk_c, kg_c, nproc=nproc)
        re_ip = RegularGridInterpolator((s, s), L_c.real, method="cubic")
        im_ip = RegularGridInterpolator((s, s), L_c.imag, method="cubic")
        sk = np.sqrt(np.clip(kk / kk_max, 0.0, 1.0))
        sg = np.sqrt(np.clip(kg / kg_max, 0.0, 1.0))
        L_out = np.empty((kk.size, kg.size), complex)
        for lo in range(0, sk.size, 128):
            hi = min(lo + 128, sk.size)
            P = np.stack(np.meshgrid(sk[lo:hi], sg, indexing="ij"), axis=-1).reshape(-1, 2)
            L_out[lo:hi] = (re_ip(P) + 1j * im_ip(P)).reshape(hi - lo, sg.size)
        return L_out

    def __call__(self, kk, kg, nproc=1):
        kk, kg = np.asarray(kk, float), np.asarray(kg, float)
        if self.lam_coarse and (kk.size * kg.size > self.lam_coarse**2) and kk.size > 1 and kg.size > 1 and kk[-1] > 0 and kg[-1] > 0:
            return self._lam_interpolated(kk, kg, nproc=nproc)
        return self._direct_call(kk, kg, nproc=nproc)

    # ---- exact mean-shift sector -------------------------------------------
    # S(delta) = sum_c m_c (lambda_c - 1) is the clustering-induced shift of the
    # ray's kappa (m_c = w kappa for explicit cells, the Campbell mean for the
    # weak arm).  It is bounded below (lambda >= 0) and skewed, which no
    # truncated cumulant series reproduces at the empty-beam edge.  Its CF is
    # computed exactly by sampling the Gaussian field with the engine's own
    # covariance, and REPLACES the S-only part of the kappa2+kappa3 closure;
    # the mixed (count x shape) sector stays in cumulant form.
    def _build_S(self, C, nsamp, seed):
        from math import factorial
        nsh = C.shape[0]
        sig = sqrt(np.diag(C))
        grid = np.linspace(-6, 6, 241)
        tab = np.zeros((nsh, grid.size))
        for (i, b), m in self.c.lin.items():
            tab[i] += m * (np.exp(b * sig[i] * grid - 0.5 * (b * sig[i])**2) - 1)
        L = np.linalg.cholesky(C + 1e-12 * np.trace(C) / nsh * np.eye(nsh))
        rng = np.random.default_rng(seed)
        S = np.zeros(nsamp)
        if not self.s_is:
            # plain sampling (pre-2026-09-27 arms; bit-identical stream)
            for lo in range(0, nsamp, 200_000):
                hi = min(lo + 200_000, nsamp)
                d = rng.standard_normal((hi - lo, nsh)) @ L.T / sig      # delta/sigma
                for i in range(nsh):
                    S[lo:hi] += np.interp(d[:, i], grid, tab[i])
            W = np.ones(nsamp)
            self.is_eta = np.zeros((0, nsh))
        else:
            # Large-deviation importance sampling (2026-09-27).  The S upper
            # tail (P(S > 0.3) ~ 3e-5 at z_s = 1, from lognormal lambda with
            # b sigma up to ~45) sets the z_s <~ 1 magnification tail, and plain
            # sampling left it to 25-63 draws: P at ln mu = 4 moved 40% between
            # seeds.  Proposal = defensive mixture (1 - alpha) N(0, I) +
            # alpha/J sum_j N(eta_j, I) in whitened z (u = L z / sigma), eta_j =
            # the most probable z reaching S = level_j; exact weights
            # w = 1 / [(1 - alpha) + alpha/J sum_j exp(eta_j.z - |eta_j|^2/2)]
            # <= 1/(1 - alpha); self-normalised histogram (E(0) = 1 exactly).
            eta = self._S_design_points(L, sig, grid, tab)
            self.is_eta = eta
            J = eta.shape[0]
            a = self.is_alpha if J else 0.0
            e2 = 0.5 * np.sum(eta**2, axis=1)
            W = np.empty(nsamp)
            for lo in range(0, nsamp, 200_000):
                hi = min(lo + 200_000, nsamp)
                z = rng.standard_normal((hi - lo, nsh))
                if J:
                    comp = rng.choice(J + 1, size=hi - lo, p=[1 - a] + [a / J] * J)
                    sh = comp > 0
                    z[sh] += eta[comp[sh] - 1]
                    with np.errstate(over="ignore"):      # overflow -> w = 0, the exact limit
                        W[lo:hi] = 1.0 / ((1 - a) + (a / J) * np.exp(z @ eta.T - e2).sum(axis=1))
                else:
                    W[lo:hi] = 1.0
                d = z @ L.T / sig
                for i in range(nsh):
                    S[lo:hi] += np.interp(d[:, i], grid, tab[i])
        self.S_samples, self.S_weights = S, W
        sw = W.sum()
        self.S_mean = float(np.sum(W * S) / sw)
        self.S_var = float(np.sum(W * (S - self.S_mean)**2) / sw)
        self.S_ess = float(sw**2 / np.sum(W**2))
        h, e = np.histogram(S, bins=200_000, weights=W)
        self._Sx, self._Sp = 0.5 * (e[1:] + e[:-1]), h / sw
        self._Stol = 25.0 / sqrt(self.S_ess)
        self._Skstar = self._S_noise_k()
        # the closure's own S-only cumulants, same Hermite truncation as Q
        N = self.nord
        Mn = [None] + [np.zeros(nsh) for _ in range(N)]
        for (i, b), m in self.c.lin.items():
            for n in range(1, N + 1):
                Mn[n][i] += m * b**n
        Cn = [None] + [C**n for n in range(1, N + 1)]
        self._q2 = sum(float(Mn[n] @ Cn[n] @ Mn[n]) / factorial(n) for n in range(1, N + 1))
        q3 = 0.0
        if self.k3:
            for n in range(1, N):
                for m in range(1, N - n + 1):
                    q3 += float(np.sum(Mn[n + m] * (Cn[n] @ Mn[n]) * (Cn[m] @ Mn[m]))) \
                          / (factorial(n) * factorial(m))
        self._q3 = 3.0 * q3                        # = kappa3(S) in the closure

    def _S_design_points(self, L, sig, grid, tab):
        """Mixture centres for the S-tail importance sampler.  The S upper tail
        is a UNION of rare events: at z_s = 1, 52 of 65 shells can reach
        S = 1 alone, and the most-probable points of 16-23 distinct shells lie
        within a factor ~3 in likelihood of each other -- a proposal aimed at
        the single best point per level left the rest to the unshifted half
        (seeds still disagreed by 2x at S > 1).  So: for every shell whose
        (clamped) table reaches the level, the most probable z with S = level
        (SLSQP from the single-shell point), falling back to the single-shell
        point itself; S through the SAME clamped tables the sampler uses; de-
        duplicated at |d eta| < 0.5.  NB the +-6 sigma clamp applies to every
        cell here, whereas the engine clamps only its weak-arm tables
        (cpp/lensing.cpp weakSV); explicit halos use an unclamped lambda there
        (lensing.cpp ~1143).  Negligible: lambda > 1 needs delta > b sigma / 2."""
        from scipy.optimize import minimize
        nsh = L.shape[0]
        A = L / sig[:, None]
        slope = np.gradient(tab, grid, axis=1)

        def S_of(z):
            u = A @ z
            return sum(np.interp(u[i], grid, tab[i]) for i in range(nsh))

        def dS(z):
            u = A @ z
            g = np.array([np.interp(u[i], grid, slope[i]) for i in range(nsh)])
            return A.T @ g
        pts, n_opt = [], 0
        reach = tab[:, -1]
        for lev in self.is_levels:
            for i in np.nonzero(reach >= lev)[0]:
                if not np.all(np.diff(tab[i]) >= 0):
                    continue
                u0 = np.zeros(nsh); u0[i] = float(np.interp(lev, tab[i], grid))
                z0 = np.linalg.solve(A, u0)
                r = minimize(lambda z: 0.5 * z @ z, z0, jac=lambda z: z, method="SLSQP",
                             constraints=[dict(type="eq", fun=lambda z: S_of(z) - lev, jac=dS)],
                             options=dict(maxiter=100, ftol=1e-9))
                ok = r.success and abs(S_of(r.x) - lev) < 1e-3 * lev and r.fun < 0.5 * z0 @ z0
                pts.append(r.x if ok else z0); n_opt += int(ok)
        keep = []
        for x in pts:
            if all(np.linalg.norm(x - y) > 0.5 for y in keep):
                keep.append(x)
        self.is_design = dict(n_candidates=len(pts), n_optimised=n_opt, n_kept=len(keep),
                              eta_norm=sorted(float(np.linalg.norm(x)) for x in keep))
        return np.array(keep) if keep else np.zeros((0, nsh))

    def _S_cf(self, kk):
        # Characteristic function of the 200k-bin S histogram, 32 k-rows at a
        # time (2026-09-25).  The one-shot np.exp(1j*np.outer(kk, x)) formed a
        # (1066 x 200000) complex matrix -- a ~7 GB transient (outer 1.7 GB ->
        # 1j* 3.4 GB -> exp 3.4 GB) just to multiply it by a vector, and it was
        # what pushed ~5 GB to swap per inversion on the 16 GB laptop.  Same
        # expression per element, same dot product per row.
        xs = self._Sx - self.S_mean
        E = np.empty(kk.size, complex)
        for lo in range(0, kk.size, 32):
            hi = min(lo + 32, kk.size)
            E[lo:hi] = np.exp(1j * np.outer(kk[lo:hi], xs)) @ self._Sp
        return E * np.exp(1j * kk * self.S_mean)

    def _S_noise_k(self, n=400):
        """k* = first k at which the sampled |E(k)| reaches the noise floor
        25/sqrt(N): scanned on a fixed grid to 60/sd(S), crossing interpolated
        linearly.  A property of the S sample alone, so the cos2 taper is the
        same function of k whatever k-array a caller passes (the reach probe
        uses a geometric grid, the inversion a linear one)."""
        kk = np.linspace(0.0, 60.0 / sqrt(self.S_var), n)
        a = np.abs(self._S_cf(kk))
        i = np.nonzero(a < self._Stol)[0]
        if not i.size:
            return float(kk[-1])
        i = int(i[0])
        f = (a[i - 1] - self._Stol) / (a[i - 1] - a[i])
        return float(kk[i - 1] + f * (kk[i] - kk[i - 1]))

    def S_correction(self, kk):
        kk = np.asarray(kk, float)
        E = self._S_cf(kk)
        closure = -0.5 * kk**2 * self._q2 - 1j / 6 * kk**3 * self._q3
        corr = np.log(np.where(np.abs(E) > 0, E, 1.0)) - closure
        if self.s_taper == "cos2":
            # cos^2 taper over the WHOLE measured range [0, k*] (2026-09-25).
            # The linear ramp below switched the correction off over the last
            # ~10% of k -- where corr is LARGEST (1.8+1.1i at z_s=1, i.e. a
            # x6 factor on Phi) and |Phi| still ~1e-5: a k-space step that
            # rang in kappa (tail ripple 1.4e-3 in ln P, sub-edge negative
            # mass -4.4e-5 at z_s=1; the z_s=0.5 tail ringing).  corr is
            # O(k^4) (k^2, k^3 live in the closure) and cos^2 = 1 - O(k^2),
            # so the taper leaves var, skew AND kurtosis untouched -- it acts
            # from the 6th cumulant on.  Measured: ripple -> the exact-S-off
            # level at z_s = 0.5, 1; negative mass /46 (z_s=1), /750 (z_s=3);
            # clipped moments unchanged to <=3e-4; bin-averaged chi2 vs the
            # MC 6.58 -> 5.38 (z_s=1), never worse (z_s = 0.5, 3, 10).
            t = np.clip(kk / self._Skstar, 0.0, 1.0)
            wgt = 0.5 * (1.0 + np.cos(np.pi * t))
        else:
            # "linear" (pre-2026-09-25, kept for reproducing old arms): beyond
            # the sampling floor |E| < 25/sqrt(N) the estimate is noise; ramp
            # off over one Stol, and once off stay off.
            a = np.abs(E)
            wgt = np.clip((a - self._Stol) / self._Stol, 0.0, 1.0)
            off = np.nonzero(wgt == 0)[0]
            if off.size:
                wgt[off[0]:] = 0.0
        return wgt * corr

    def Q(self, kk, kg):
        """Cox term, SEPARABLE evaluation (2026-09-25).  Identical algebra to
        `_Q_reference` (the original, kept for gating): only the B_n
        contraction is reordered, Sum_b -> Sum_q J[q,g] Sum_p em1[k,p] Ht[i,p,q].
        Agrees with the reference to roundoff, NOT bit-for-bit (a reordered
        sum cannot be)."""
        import scipy.special as sp
        from math import factorial
        kk, kg = np.asarray(kk, float), np.asarray(kg, float)
        J = sp.j0(np.outer(self.gcu, kg))                        # (nQ, nkg)
        Jm1 = _j0m1(np.outer(self.gcu, kg))
        nsh, _, nP, nQ = self.Ht.shape
        N = self.nord
        Cn = [None] + [self.C**n for n in range(1, N + 1)]
        # (nP, nsh*nQ) layout so the p-contraction is ONE GEMM per order
        Hr = [None] + [np.ascontiguousarray(
            self.Ht[:, n].transpose(1, 0, 2).reshape(nP, nsh * nQ))
            for n in range(1, N + 1)]
        Bg = [None] + [self.Ht[:, n].sum(axis=1) @ Jm1 for n in range(1, N + 1)]
        out = np.zeros((kk.size, kg.size), complex)
        for lo in range(0, kk.size, self.chunk):
            hi = min(lo + self.chunk, kk.size)
            c = hi - lo
            ph = np.outer(kk[lo:hi], self.kcu)                   # (c, nP)
            er = np.cos(ph) - 1                                  # Re(e^{i ph}-1)
            ei = np.sin(ph)                                      # Im
            B = [None]
            for n in range(1, N + 1):
                Zr = (er @ Hr[n]).reshape(c, nsh, nQ).transpose(1, 0, 2).reshape(nsh * c, nQ)
                Zi = (ei @ Hr[n]).reshape(c, nsh, nQ).transpose(1, 0, 2).reshape(nsh * c, nQ)
                b = (Zr @ J + 1j * (Zi @ J)).reshape(nsh, c, -1)
                b += Bg[n][:, None, :]
                b += (1j * np.outer(self.c.wM[:, n], kk[lo:hi])
                      - 0.5 * np.outer(self.c.wV[:, n], kk[lo:hi]**2))[:, :, None]
                B.append(b)
            T = [None] + [np.tensordot(Cn[n], B[n], axes=(1, 0)) for n in range(1, N)]
            T.append(np.tensordot(Cn[N], B[N], axes=(1, 0)))
            for n in range(1, N + 1):
                out[lo:hi] += 0.5 / factorial(n) * np.sum(B[n] * T[n], axis=0)
            if self.k3:
                for n in range(1, N):
                    for m in range(1, N - n + 1):
                        out[lo:hi] += (0.5 / (factorial(n) * factorial(m))
                                       * np.sum(B[n + m] * T[n] * T[m], axis=0))
        return out

    def _Q_reference(self, kk, kg):
        """The ORIGINAL (pre-2026-09-25) Q, kept verbatim as the gate."""
        import scipy.special as sp
        from math import factorial
        J = sp.j0(np.outer(self.gc, kg))                         # (nb, nkg)
        Jm1 = _j0m1(np.outer(self.gc, kg))
        nsh = self.Hc.shape[0]
        N = self.nord
        Cn = [None] + [self.C**n for n in range(1, N + 1)]
        Bg = [None] + [self.Hc[:, n, :] @ Jm1 for n in range(1, N + 1)]
        out = np.zeros((kk.size, kg.size), complex)
        for lo in range(0, kk.size, self.chunk):
            hi = min(lo + self.chunk, kk.size)
            ph = np.outer(kk[lo:hi], self.kc)                    # (c, nb)
            em1 = (np.cos(ph) - 1) + 1j * np.sin(ph)             # e^{i ph} - 1
            B = [None]
            for n in range(1, N + 1):
                H = self.Hc[:, n, :]
                X = (H[:, None, :] * em1[None]).reshape(-1, H.shape[1])
                b = (X.real @ J + 1j * (X.imag @ J)).reshape(nsh, hi - lo, -1)
                b += Bg[n][:, None, :]
                b += (1j * np.outer(self.c.wM[:, n], kk[lo:hi])
                      - 0.5 * np.outer(self.c.wV[:, n], kk[lo:hi]**2))[:, :, None]
                B.append(b)
            T = [None] + [np.tensordot(Cn[n], B[n], axes=(1, 0)) for n in range(1, N)]
            T.append(np.tensordot(Cn[N], B[N], axes=(1, 0)))
            # 2nd cumulant: exact lognormal pair covariance, Hermite series
            for n in range(1, N + 1):
                out[lo:hi] += 0.5 / factorial(n) * np.sum(B[n] * T[n], axis=0)
            # 3rd cumulant, pair-product (xy+xw+yw) part: O(C^2); the xyw
            # remainder is O(C^3) and dropped
            if self.k3:
                for n in range(1, N):
                    for m in range(1, N - n + 1):
                        out[lo:hi] += (0.5 / (factorial(n) * factorial(m))
                                       * np.sum(B[n + m] * T[n] * T[m], axis=0))
        return out

    def sigma(self):
        """Total sigma_kappa including the Cox (clustering) variance."""
        v = float(np.sum(self.w0 * self.k0**2)) + self.V + self.lss_var
        if getattr(self, "xi_on", False):
            if hasattr(self, "_W0_all") and self._W0_all is not None:
                v += float(np.sum((self._W0_all @ self._kmc + self._res0_all)**2))
            else:
                for g in self._xg:
                    if g["m"] == 0:
                        v += float(np.sum((g["rows"] @ g["kc"] + g["res"])**2))
        if self.C is not None:
            from math import factorial
            for n in range(1, self.nord + 1):
                m = self.Hc[:, n, :] @ self.kc + self.c.wM[:, n]
                v += float(m @ (self.C**n) @ m) / factorial(n)
        return sqrt(v)

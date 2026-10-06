"""Cylindrical filament jump measure R_fil(xi), companion to sgl.py's NFW
halo R_of_xi -- the C++ production engine's filament model
(``lensing.cpp::kappagammaCYL`` + the flat (p=0, q=0.7) filament barrier
``cosmology::pFCfil``/``FMFlistf``, docs/filament_bias_note.md).

Structure mirrors sgl.py exactly: same XI grid, same deposit-binned
cross-section trick (dsigma/dxi is exact per x-interval, not a Jacobian),
same line-of-sight z-integral. Two differences from the NFW halo case:

1. The filament mass function uses the flatter barrier (p, q) = (0, 0.7)
   instead of the halo's (0.3, 0.8) -- ``Cosmology.dndlnM_fil``.
2. The cylinder cross-section (``kappagammaCYL``) additionally depends on
   the filament's random orientation angle ``phi`` (isotropic per lens,
   entering only through the amplitude ``rhos``, not the shape). The C++
   draws one phi per realized filament; here the population's jump measure
   is the phi-AVERAGE of the per-phi cross-section (linear in the deposit
   weights, so averaging commutes with the z,M sum) -- a mean-field
   treatment that drops phi-realization covariance across lenses, matching
   the project's existing finding that orientation is a sub-dominant,
   isotropic effect (`docs/filament_bias_note.md` sec 4).

Verified against the production C++ directly (not asserted): see
``playground/filament_probe.cpp`` + ``verify_filaments.py`` -- kappa/gamma
agree to <1e-9 relative and dndlnM_fil to <1e-6 relative across the probe's
(M, z, phi, r) grid and the FMFlist table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy import log, exp, sqrt, pi

from . import sgl
from .sgl import XI, _XIE, _XIW, CKMS, Cosmology

# ------------------------- filament mass function ---------------------------
P_FIL, Q_FIL = 0.0, 0.7
A_FIL = 0.5   # 1/(1 + 2^-p * Gamma(0.5-p)/sqrt(pi)) at p=0: Gamma(0.5)=sqrt(pi)


def dndlnM_fil(cos: Cosmology, M, z):
    """Filament mass function -- mirrors ``Cosmology.dndlnM`` exactly but
    with the flat (p=0, q=0.7) barrier (``cosmology::pFCfil``/``FMFlistf``),
    same sigma(M) and chain rule."""
    nu = (cos.dc / cos.sigmaM(M, z))**2
    qn = Q_FIL * nu
    mult = A_FIL * 2.0 * sqrt(qn / (2 * pi)) * exp(-qn / 2)   # (1+qn^0)=2
    return (cos.rhom / M) * mult * (-2 * cos.dlnsig_dlnM(M))


# ------------------------- cylinder cross-section ----------------------------
def kappa_gamma_cyl(u, kappa0):
    """kappaCYL2/gammaCYL2 as a function of u = r/rs (dimensionless), given
    the amplitude kappa0 = rs*rhos/Sigma_cr. Reproduces the C++ truncation
    (kappa = 0 for u > 1; gamma is continuous, no truncation on its own
    leading term) verified against ``kappagammaCYL`` in filament_probe.cpp.
    ``(sqrt(1+u^2)-1)/u^2 = 1/(sqrt(1+u^2)+1)`` is used for the u->0 limit
    (exact identity, avoids the 0/0 cancellation)."""
    u = np.abs(np.asarray(u, float))
    s = sqrt(1.0 + u * u)
    kap_full = kappa0 * 2.0 * pi / s
    kap = np.where(u <= 1.0, kap_full, 0.0)
    gam = kappa0 * 4.0 * pi / (s + 1.0) - kap
    return kap, gam


def filament_params(cos: Cosmology, M, zl, zs, phi):
    """rs, kappa0 for a filament of mass M at (zl, zs, phi) -- mirrors
    ``kappagammaCYL``'s rs/L/rhos/kappa0 construction exactly (Mpc units
    throughout, matching sgl.py's Cosmology; the C++ uses kpc but the
    ratios are unit-consistent)."""
    rs = 1.0 * (M / 1.0e14)**(1.0 / 3.0)     # Mpc  (= 1000 kpc * (..)^1/3)
    L = 20.0 * (M / 1.0e14)**(1.0 / 3.0)     # Mpc  (= 20000 kpc * (..)^1/3)
    rhos = 14.4 * cos.rhoc0 * L / max(2.0 * rs, abs(np.cos(phi)) * L)
    kappa0 = rs * rhos / cos.Sigma_cr(zl, zs)
    return rs, kappa0


# u-grid: filament profile is smooth (no NFW-style near-caustic thin
# annulus), but still refine around any zero of detA(u) the same way.
_UG = np.logspace(-6, 3.5, 900)
_BISECT_ITERS = 45


def _refined_ugrid(kappa0):
    k, g = kappa_gamma_cyl(_UG, kappa0)
    detA = (1 - k)**2 - g * g
    sgn = np.sign(detA)
    idx = np.nonzero(sgn[1:] * sgn[:-1] < 0)[0]
    us = [_UG]
    if idx.size:
        a = _UG[idx].copy()
        b = _UG[idx + 1].copy()
        sgn0 = sgn[idx]
        for _ in range(_BISECT_ITERS):
            m = 0.5 * (a + b)
            km, gm = kappa_gamma_cyl(m, kappa0)
            same = np.sign((1 - km)**2 - gm * gm) == sgn0
            a = np.where(same, m, a)
            b = np.where(same, b, m)
        uc = 0.5 * (a + b)
        eps = np.logspace(-11, -1.2, 22)
        us.append((uc[:, None] * (1.0 + eps[None, :])).ravel())
        us.append((uc[:, None] * (1.0 - eps[None, :])).ravel())
    u = np.unique(np.concatenate(us))
    return u[u > 0]


def dsigma_bins_fil_phi(cos: Cosmology, M, zl, zs, phi):
    """Single-filament (fixed phi) deposit-binned cross-section dsigma/dxi
    on the shared XI grid [Mpc^2] -- same deposit-binning as
    ``sgl.dsigma_bins``, generalized from x=r/r_s(NFW) to u=r/rs(filament)."""
    rs, kappa0 = filament_params(cos, M, zl, zs, phi)
    u = _refined_ugrid(kappa0)
    k, g = kappa_gamma_cyl(u, kappa0)
    detA = (1 - k)**2 - g * g
    good = detA > 1e-300
    xi = np.where(good, -log(np.where(good, detA, 1.0)), np.nan)
    ua, ub = u[:-1], u[1:]
    fa, fb = xi[:-1], xi[1:]
    ok = np.isfinite(fa) & np.isfinite(fb)
    ua, ub, fa, fb = ua[ok], ub[ok], fa[ok], fb[ok]
    lo, hi = np.minimum(fa, fb), np.maximum(fa, fb)
    dsig = pi * rs * rs * (ub * ub - ua * ua)
    keep = (hi > _XIE[0]) & (lo < _XIE[-1]) & (dsig > 0)
    if not np.any(keep):
        return np.zeros_like(XI)
    lo, hi, dsig = lo[keep], hi[keep], dsig[keep]
    den = np.maximum(hi - lo, 1e-300)
    frac = np.clip((_XIE[:, None] - lo[None, :]) / den[None, :], 0.0, 1.0)
    return np.diff(frac @ dsig) / _XIW


def dsigma_bins_fil(cos: Cosmology, M, zl, zs, nphi=16):
    """Phi-averaged filament cross-section (isotropic orientation, uniform
    over one pi-period since kappa0 depends on phi only via |cos phi|)."""
    phis = np.linspace(0.0, pi, nphi, endpoint=False) + 0.5 * pi / nphi
    acc = np.zeros_like(XI)
    for phi in phis:
        acc += dsigma_bins_fil_phi(cos, M, zl, zs, phi)
    return acc / nphi


def R_of_xi_filaments(cos: Cosmology, zs, Mmin=1e7, Mmax=1e16, Nz=40, NM=48,
                       nphi=16):
    """Line-of-sight-integrated filament jump measure R_fil(xi; zs), on the
    SAME XI grid as sgl.R_of_xi -- since the compound-Poisson Levy exponent
    is linear in the jump measure, R_total = R_halo + R_fil exactly (two
    independent Poisson populations), so the two can just be added
    elementwise before calling sgl.P_of_xi / sgl.dP_dmu."""
    zgrid = np.linspace(1e-3, zs - 1e-3, Nz)
    dz = zgrid[1] - zgrid[0]
    Mgrid = np.logspace(log(Mmin) / log(10), log(Mmax) / log(10), NM)
    dlnM = log(Mgrid[1]) - log(Mgrid[0])
    R = np.zeros_like(XI)
    for z in zgrid:
        wz = (1 + z)**2 * CKMS / cos.Hz(z) * dz
        for M in Mgrid:
            R += wz * dndlnM_fil(cos, M, z) * dlnM * dsigma_bins_fil(cos, M, z, zs, nphi=nphi)
    return R

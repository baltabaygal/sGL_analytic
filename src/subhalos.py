#!/usr/bin/env python
"""Analytic subhalo term for the engine's production model 5 (cpp/subhalo.cpp).

Engine model (per EXPLICIT host, i.e. ray within the host's r_max):
  * clumps ~ Poisson, dN/dlnpsi = g psi^alpha exp(-beta psi^omega), psi = m/M_vir,
    psi in [m_floor/M_vir, 1]; g from f_s (JvdB14 eq. 23/26, z_f from the
    half-mass barrier crossing);
  * 3D positions dN/dx ~ x B(x)/(1+cx)^2, x = r/r200 <= eta = r_vir/r200,
    B = [1 + (x/x0)^-2.5]^-1/2, x0 = 0.86 eta; projected isotropically;
  * a clump is RENDERED iff the ray is within D(m) (kappa_clump(D) = 0.1 kappa_thr),
    as a full untruncated NFW with the field cons16 c(m);
  * the host is rebuilt at M_eff = M - sum(m_rendered)/vr (vr = M_vir/M200).

Analytic treatment (Neyman-Scott cluster process).  For a host hit at ray
offset r the exact exponent is  w_h [ e^{ik q_h(r)} exp(S(k; r)) - 1 ],  S the
clump Poisson exponent.  Clump ENCOUNTERS (a clump cell at ray distance d) are
split by their own kappa_c at ksplit = 10 kappa_thr:

  * WEAK encounters (numerous): their mean kappa (ring-averaged exactly over
    the projected Green+21 profile) and the mean carve of their mass shift the
    host cell's kappa; their conditional variance rides on the SAME host cell
    as a 3-node Gauss-Hermite blur in kappa (<gamma^2> added in quadrature).
    They therefore fluctuate only on rays that actually pass a host.
  * STRONG encounters (rare, <~0.5 per host-ray, carry the host-clump
    covariance and the skewness): exact first-order pairs, host-alone weight
    x e^{-nu(r)}, host+clump pair cells with probability 1 - e^{-nu}
    (probability-conserving multiplicity), pair kappa = host(r; carved by that
    clump) + kappa_c, pair |gamma| via Neumann's addition theorem.

Failed alternatives, kept because each one taught something (2026-09-24):
  - clumps as an independent compensated population + mean profile in the
    host: Var right (0.984) but the empty-beam edge slides down by E[kappa_s]
    ~ 0.009 -- every ray is smeared, not just host rays;
  - strong encounters as independent cells: loses the host-clump covariance,
    which is ~23% of the host variance and lives mostly in strong encounters;
  - first-order pairs without the n >= 2 mass: drops ~nu^2/2 of probability in
    cores (nu ~ 0.7-1.8 there), where kappa_h^2 is largest.
Residual approximations: weak-encounter blur is Gaussian; pair host uses the
circular kappa (ellipticity only on host-alone cells); mean-profile shear is
not modelled.  The engine's spin-1 host/clump shear convention together with
its placement density (clumps favoured on the host-centre side, where the two
shears align) is modelled for the STRONG pairs only, as an option:
SubhaloModel(pair_angle="engine") (2026-09-27; default "uniform").
"""
from __future__ import annotations

from math import gamma as Gfn

import numpy as np
from numpy import exp, log, pi, sqrt
from scipy.special import gammaincc

import sgl
import sgl_full as F

try:
    import numba as _numba
    import math as _math
    _HAVE_NUMBA = True
except ImportError:                    # pragma: no cover
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    # np.interp's C loop evaluates slope*(x - xp[j]) + fp[j] as ONE fused
    # multiply-add on this build (clang contracts it on arm64; verified 300/300
    # against exact single-rounding arithmetic, while the two-rounding form
    # disagrees at 25k/200k points).  Contraction is scoped to this helper so
    # that nothing else in the kernel -- in particular the R expression, which
    # numpy evaluates as separate un-fused ufunc passes -- is fused.
    @_numba.njit(cache=True, fastmath={"contract"})
    def _interp_fma(s, dx, f):
        return s * dx + f

    @_numba.njit(cache=True)
    def _pair_group_sums(grp, nr, wf, nuc, kh_dkw, extra, dkdM, cm, gh):
        """Segment sums replacing add_pairs' `S @ (...)` one-hot matmuls, for
        EVERY clump column of a host at once (2026-09-25).

        The one-hot S made each per-bin aggregation three tiny BLAS calls
        ((40 x 855) @ (855 x ~44)); a section profile put 43.5 s of an 80.7 s
        add_pairs (+sub, z_s=1) there -- almost all OpenBLAS thread start/sync
        overhead on ~140k tiny products, not arithmetic.

        Per element the values are built exactly as before:
          w     = wf[f] * nuc[f, j]                       (wt)
          khost = ((kh + dkh_weak) - cm_j * dkdM) + extra (kh + dkh_weak - carve + extra)
        Only the summation order over f changes (sequential here, BLAS-blocked
        before), so results agree to roundoff, not bit-for-bit."""
        nf, nj = nuc.shape
        den = np.zeros((nr, nj))
        nk = np.zeros((nr, nj))
        ng = np.zeros((nr, nj))
        for f in range(nf):
            g = grp[f]
            w0, a, e, d, gg = wf[f], kh_dkw[f], extra[f], dkdM[f], gh[f]
            for j in range(nj):
                w = w0 * nuc[f, j]
                kk = (a - cm[j] * d) + e
                den[g, j] += w
                nk[g, j] += w * kk
                ng[g, j] += w * gg
        return den, nk, ng

    # Otherwise NOT fastmath: every operation below is written to reproduce
    # the numpy _ring expression order (see _ring_numpy) so the result can be
    # gated against it bit-for-bit, and fastmath would license reassociation.
    @_numba.njit(cache=True, parallel=True)
    def _ring_kernel(r, d, cth, xp, fp):
        """Fused n_ring(r, d) = mean_theta n(|r - d|), (Nr, Nd).

        Replicates, in one pass and without the (Nr, Nd, nth) temporary:
          R = sqrt((r^2 + d^2) - ((2r) d) cos th)     -- numpy broadcast order
          v = np.interp(R, xp, fp, right=0.0)         -- incl. its edge rules
          mean over th = (0 + pairwise_sum(v)) / nth  -- numpy's add.reduce
        numpy's pairwise_sum for 8 <= n <= 128 on a contiguous axis is 8
        running partials, combined ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)), then
        the n % 8 remainder added sequentially.  (Verified against
        np.sum(axis=-1): identity-initialised, 20000/20000 bit matches.)"""
        nr, nd, nt = r.shape[0], d.shape[0], cth.shape[0]
        n = xp.shape[0]
        slopes = np.empty(n - 1)
        for j in range(n - 1):
            slopes[j] = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        x0, xl = xp[0], xp[n - 1]
        dx = (xl - x0) / (n - 1)
        out = np.empty((nr, nd))
        # Rows are independent and each is computed with an identical operation
        # sequence, so prange cannot change a single bit of the result.  The
        # scratch buffer MUST be per-row: a shared one is a data race here.
        for i in _numba.prange(nr):
            buf = np.empty(nt)
            ri = r[i]
            r2 = ri * ri
            tr = 2.0 * ri
            for k in range(nd):
                dk = d[k]
                c = r2 + dk * dk
                trd = tr * dk
                for t in range(nt):
                    x = _math.sqrt(c - trd * cth[t])
                    if x < x0:
                        v = fp[0]
                    elif x > xl:
                        v = 0.0
                    elif x == xl:
                        v = fp[n - 1]
                    else:
                        # arithmetic guess on the (near-)uniform grid, then
                        # corrected to the exact binary-search bracket
                        j = int((x - x0) / dx)
                        if j > n - 2:
                            j = n - 2
                        if j < 0:
                            j = 0
                        while j > 0 and xp[j] > x:
                            j -= 1
                        while j < n - 2 and xp[j + 1] <= x:
                            j += 1
                        if xp[j] == x:
                            v = fp[j]
                        else:
                            v = _interp_fma(slopes[j], x - xp[j], fp[j])
                    buf[t] = v
                if nt < 8:
                    s = 0.0
                    for t in range(nt):
                        s += buf[t]
                else:
                    p0, p1, p2, p3 = buf[0], buf[1], buf[2], buf[3]
                    p4, p5, p6, p7 = buf[4], buf[5], buf[6], buf[7]
                    t = 8
                    while t < nt - (nt % 8):
                        p0 += buf[t]
                        p1 += buf[t + 1]
                        p2 += buf[t + 2]
                        p3 += buf[t + 3]
                        p4 += buf[t + 4]
                        p5 += buf[t + 5]
                        p6 += buf[t + 6]
                        p7 += buf[t + 7]
                        t += 8
                    s = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
                    while t < nt:
                        s += buf[t]
                        t += 1
                out[i, k] = (0.0 + s) / nt
        return out

    @_numba.njit(cache=True, parallel=True, fastmath=True)
    def _pair_angle_table(r, d, cth, xp, fp):
        """P[i, k, t] = n(R_t) / sum_t' n(R_t'),  R_t = |r_i e_x - d_k e^{i th_t}|
        (pair_angle="engine", 2026-09-27): the conditional distribution of the
        host->ray / clump->ray angle on the ntheta midpoint nodes, for a clump
        at ray distance d_k from a ray at host radius r_i.  n = the host's
        projected clump density (h["n2"] on h["Rn"]), linear interpolation,
        0 beyond the last node (np.interp(..., right=0.0)).  Where every node
        has n = 0 the ring weight nu_c is 0 too; P is set uniform there so
        the row still sums to 1."""
        nr, nd, nt = r.shape[0], d.shape[0], cth.shape[0]
        n = xp.shape[0]
        slopes = np.empty(n - 1)
        for j in range(n - 1):
            slopes[j] = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        x0, xl = xp[0], xp[n - 1]
        dx = (xl - x0) / (n - 1)
        inv_dx = 1.0 / dx
        out = np.empty((nr, nd, nt))
        for i in _numba.prange(nr):
            ri = r[i]
            r2 = ri * ri
            tr = 2.0 * ri
            for k in range(nd):
                dk = d[k]
                c = r2 + dk * dk
                trd = tr * dk
                s = 0.0
                for t in range(nt):
                    arg = c - trd * cth[t]
                    x = _math.sqrt(arg) if arg > 0.0 else 0.0
                    if x <= x0:
                        v = fp[0]
                    elif x >= xl:
                        v = fp[n - 1] if x == xl else 0.0
                    else:
                        j = int((x - x0) * inv_dx)
                        if j > n - 2:
                            j = n - 2
                        elif j < 0:
                            j = 0
                        v = slopes[j] * (x - xp[j]) + fp[j]
                    out[i, k, t] = v
                    s += v
                if s > 0.0:
                    for t in range(nt):
                        out[i, k, t] /= s
                else:
                    for t in range(nt):
                        out[i, k, t] = 1.0 / nt
        return out

    @_numba.njit(cache=True, parallel=True, fastmath=True)
    def _pair_angle_mean_cos(r, d, cth, xp, fp):
        """ecd[i, k] = sum_t P[i, k, t] cth[t] directly, avoiding the 3D (nr, nd, nt)
        allocation and P @ cth matrix multiplication. Where sum_t n(R_t) == 0,
        the uniform row weights give sum_t (1/nt) cth[t] == 0 identically on
        midpoint cosine nodes."""
        nr, nd, nt = r.shape[0], d.shape[0], cth.shape[0]
        n = xp.shape[0]
        slopes = np.empty(n - 1)
        for j in range(n - 1):
            slopes[j] = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        x0, xl = xp[0], xp[n - 1]
        dx = (xl - x0) / (n - 1)
        inv_dx = 1.0 / dx
        out = np.empty((nr, nd))
        for i in _numba.prange(nr):
            ri = r[i]
            r2 = ri * ri
            tr = 2.0 * ri
            for k in range(nd):
                dk = d[k]
                c = r2 + dk * dk
                trd = tr * dk
                s = 0.0
                s_cth = 0.0
                for t in range(nt):
                    arg = c - trd * cth[t]
                    x = _math.sqrt(arg) if arg > 0.0 else 0.0
                    if x <= x0:
                        v = fp[0]
                    elif x >= xl:
                        v = fp[n - 1] if x == xl else 0.0
                    else:
                        j = int((x - x0) * inv_dx)
                        if j > n - 2:
                            j = n - 2
                        elif j < 0:
                            j = 0
                        v = slopes[j] * (x - xp[j]) + fp[j]
                    s += v
                    s_cth += v * cth[t]
                out[i, k] = s_cth / s if s > 0.0 else 0.0
        return out

    @_numba.njit(cache=True, fastmath=True)
    def _ring_and_pair_angle_mean_cos(r, d, cth, xp, fp):
        """Computes ring[i, k] = sum_t n(R_t) / nt and ecd[i, k] = sum_t n(R_t) cth[t] / sum_t n(R_t)
        in a single fused pass over the (nr, nd, nt) grid."""
        nr, nd, nt = r.shape[0], d.shape[0], cth.shape[0]
        n = xp.shape[0]
        slopes = np.empty(n - 1)
        for j in range(n - 1):
            slopes[j] = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
        x0, xl = xp[0], xp[n - 1]
        dx = (xl - x0) / (n - 1)
        inv_dx = 1.0 / dx
        inv_nt = 1.0 / nt
        ring = np.empty((nr, nd))
        ecd = np.empty((nr, nd))
        for i in range(nr):
            ri = r[i]
            r2 = ri * ri
            tr = 2.0 * ri
            for k in range(nd):
                dk = d[k]
                c = r2 + dk * dk
                trd = tr * dk
                s = 0.0
                s_cth = 0.0
                for t in range(nt):
                    arg = c - trd * cth[t]
                    x = _math.sqrt(arg) if arg > 0.0 else 0.0
                    if x <= x0:
                        v = fp[0]
                    elif x >= xl:
                        v = fp[n - 1] if x == xl else 0.0
                    else:
                        j = int((x - x0) * inv_dx)
                        if j > n - 2:
                            j = n - 2
                        elif j < 0:
                            j = 0
                        v = slopes[j] * (x - xp[j]) + fp[j]
                    s += v
                    s_cth += v * cth[t]
                ring[i, k] = s * inv_nt
                ecd[i, k] = s_cth / s if s > 0.0 else 0.0
        return ring, ecd

    @_numba.njit(cache=True, parallel=True)
    def _pair_angle_sums(grp, nr, wf, nuc, kcol, P):
        """Wt[g, j, t] = sum_{f in group g} wf[f] nuc[f, j] P[f, kcol[j], t]:
        the pair weight of clump column j, radial group g, angle node t.
        sum_t Wt[g, j, t] = den[g, j] of `_pair_group_sums` to roundoff (the
        rows of P sum to 1), so the ring-averaged nu_c and every marginal
        weight are unchanged; only the split over angle nodes moves.
        Columns are independent -> prange over j is race-free."""
        nf, nj = nuc.shape
        nt = P.shape[2]
        out = np.zeros((nr, nj, nt))
        for j in _numba.prange(nj):
            k = kcol[j]
            for f in range(nf):
                w = wf[f] * nuc[f, j]
                if w == 0.0:
                    continue
                g = grp[f]
                for t in range(nt):
                    out[g, j, t] += w * P[f, k, t]
        return out

ALPHA, BETA, OMEGA, PSI_RES, PSI_MAX = -0.82, 50.0, 4.0, 1e-4, 1.0
X0_RVIR, BIAS_EXP = 0.86, 2.5
M_FLOOR = 1e7
KTHR_FACTOR = 0.1


def _mu(y):
    return log(1 + y) - y / (1 + y)


def _Om_z(cos, z):
    a3 = cos.Om * (1 + z)**3
    return a3 / (a3 + cos.OL)


def _Dvir(cos, z):
    d = _Om_z(cos, z) - 1.0
    return 18 * pi**2 + 82 * d - 39 * d * d


if _HAVE_NUMBA:
    @_numba.njit(cache=True, fastmath=True, inline="always")
    def _mu_scalar(y):
        return _math.log(1.0 + y) - y / (1.0 + y)

    @_numba.njit(cache=True, fastmath=True)
    def _eta_root_nb(c, Dvir):
        """c_v = eta c solving c_v^3 / mu(c_v) = 200 c^3 / (mu(c) Dvir) on
        [c, 3c] (2026-10-06): safeguarded Newton on ln(m^3/mu(m)), bisection
        when a step leaves the bracket; converges in ~5 steps to the root the
        former 80-step bisection approached (agreement ~1 ulp)."""
        lt = _math.log(200.0 * c * c * c / _mu_scalar(c) / Dvir)
        lo, hi = c, 3.0 * c
        m = 0.5 * (lo + hi)
        for _ in range(60):
            mu = _mu_scalar(m)
            f = _math.log(m * m * m / mu) - lt
            if f < 0.0:
                lo = m
            else:
                hi = m
            dmu = m / ((1.0 + m) * (1.0 + m))
            fp = 3.0 / m - dmu / mu
            step = f / fp
            mn = m - step
            if not (lo < mn < hi):
                mn = 0.5 * (lo + hi)
            if abs(mn - m) <= 4e-16 * m or hi - lo <= 4e-16 * m:
                m = mn
                break
            m = mn
        return m

    @_numba.njit(cache=True, fastmath=True)
    def _eta_scalar_nb(c, Dvir):
        return _eta_root_nb(c, Dvir) / c

    @_numba.njit(cache=True, fastmath=True)
    def _eta_vec_nb(c, Dvir):
        n = c.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = _eta_root_nb(c[i], Dvir) / c[i]
        return out


if _HAVE_NUMBA:
    @_numba.njit(cache=True)
    def _n3_proj_nb(eta, C_, x0):
        """Line-of-sight projection of rho_NFW B for SubhaloModel.host (2026-10-06):
        the same 256 x 160 trapezoid as the numpy form, fused, with the bias
        exponent 2.5 as q^-2.5 = 1/(q^2 sqrt q) (no pow): 3.2x, n2 to 1.5e-14."""
        nR, nu = 256, 160
        Rh = np.empty(nR)
        n3 = np.empty(nR)
        du = 1.0 / (nu - 1)
        for i in range(nR):
            R = (i + 0.5) / 256 * eta
            Rh[i] = R
            um = _math.sqrt(max(eta * eta - R * R, 0.0))
            acc = 0.0
            prev = 0.0
            for j in range(nu):
                uu = j * du * um
                X = _math.sqrt(R * R + uu * uu)
                q = X / x0
                B = 1.0 / _math.sqrt(1.0 / (q * q * _math.sqrt(q)) + 1.0)
                d = 1.0 + C_ * X
                pj = B / (X * d * d)
                if j > 0:
                    acc += 0.5 * (pj + prev) * du
                prev = pj
            n3[i] = acc * um * 2.0
        return Rh, n3


def _eta(cos, c, z):
    if _HAVE_NUMBA:
        return _eta_scalar_nb(float(c), float(_Dvir(cos, z)))
    target = 200.0 * c**3 / _mu(c) / _Dvir(cos, z)
    lo, hi = c, 3 * c
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if m**3 / _mu(m) < target:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi) / c


def _eta_vec(cos, c, z):
    """`_eta` for an array of concentrations at one z (same bisection)."""
    if _HAVE_NUMBA:
        return _eta_vec_nb(np.ascontiguousarray(c, dtype=float), float(_Dvir(cos, z)))
    c = np.asarray(c, float)
    target = 200.0 * c**3 / _mu(c) / _Dvir(cos, z)
    lo, hi = c.copy(), 3 * c
    for _ in range(80):
        m = 0.5 * (lo + hi)
        below = m**3 / _mu(m) < target
        lo = np.where(below, m, lo)
        hi = np.where(below, hi, m)
    return 0.5 * (lo + hi) / c


def _uinc(s, x):          # upper incomplete Gamma, unnormalised (gsl_sf_gamma_inc)
    return gammaincc(s, x) * Gfn(s)


def _m200_from_mvir(cos, mvir, z, tol=1e-13, itmax=50):
    """M200 of an NFW whose virial mass (inside r_vir = eta r200) is mvir, with
    c = c(M200, z): fixed point M200 = mvir mu(c)/mu(eta c) (2026-10-05).
    v_r = mu(eta c)/mu(c) varies weakly with mass, so it converges in a few
    iterations to tol (relative)."""
    # (2026-10-06) secant in x = ln M200 on g(x) = x + ln v_r(x) - ln mvir,
    # started from the first two fixed-point iterates: ~3-4 evaluations of
    # c(M200) and eta instead of ~7, same root (to ~1e-15 relative).
    mvir = np.asarray(mvir, float)
    lmv = np.log(mvir)
    zd = cos_zs_dummy(cos, z)

    def g(x):
        M = np.exp(x)
        C = np.asarray(cos.nfw_params(M, z, zd)[0], float)
        return x + np.log(_mu(_eta_vec(cos, C, z) * C) / _mu(C)) - lmv

    x0 = lmv.copy()
    g0 = g(x0)
    x1 = x0 - g0                      # = one fixed-point step
    for _ in range(itmax):
        g1 = g(x1)
        if np.all(np.abs(g1) < tol):
            break
        den = g1 - g0
        safe = np.abs(den) > 0
        x2 = np.where(safe, x1 - g1 * (x1 - x0) / np.where(safe, den, 1.0), x1 - g1)
        x0, g0, x1 = x1, g1, x2
    return np.exp(x1)


def cos_zs_dummy(cos, z):
    """A source redshift behind z for nfw_params, whose concentration and r_s do
    not depend on z_s (only ks does, and it is not used here)."""
    return 2.0 * z + 1.0


class SubhaloModel:
    def __init__(self, cos, zs, kthr, Mhost_min=1e9, npsi=28, nd=90, nth=48,
                 split=10.0, ntheta=4, nr=40, prune=0.0, pair_angle="uniform",
                 sector="pairs", exact_kw=None, clump_edge="kthr", clump_mass="vir"):
        self.cos, self.zs, self.kthr = cos, zs, kthr
        # Clump mass definition (2026-10-05, user: the consistent choice), used
        # with clump_edge="rvir":
        #   "vir": m = psi M_vir is the clump's VIRIAL mass: its NFW is built at
        #       the M200 with M200 * v_r,c(c(M200)) = m (v_r,c = mu(eta c)/mu(c),
        #       as for the host), so the profile truncated at the clump's r_vir
        #       encloses exactly m -- the mass the host loses (S = sum m / v_r);
        #   "m200": m used as the clump's M200 (pre-2026-10-05): the truncated
        #       clump enclosed m v_r,c = 1.07-1.15 m, more than the host lost.
        if clump_mass not in ("vir", "m200"):
            raise ValueError(f"clump_mass must be 'vir' or 'm200', not {clump_mass!r}")
        self.clump_mass = clump_mass
        # Clump edge (2026-10-02):
        #   "kthr": the engine's reach, kappa_c(D) = 0.1 kthr, untruncated NFW;
        #   "rvir": each clump is an NFW truncated at its own virial radius
        #       (sgl_full.nfw_trunc_kg), rendered when the ray passes inside it
        #       (D = r_vir of the clump): no kthr anywhere in the clump law.
        #       Exact sector only (the pairs sector also needs ksplit).
        if clump_edge not in ("kthr", "rvir"):
            raise ValueError(f"clump_edge must be 'kthr' or 'rvir', not {clump_edge!r}")
        if clump_edge == "rvir" and sector != "exact":
            raise ValueError('clump_edge="rvir" needs sector="exact"')
        # kthr (2026-10-02) enters only through ksplit (pairs sector) and the
        # clump reach ksub (clump_edge "kthr"); the default exact + "rvir"
        # model takes kthr=None and has no threshold anywhere
        if kthr is None and clump_edge != "rvir":
            raise ValueError('kthr=None needs sector="exact", clump_edge="rvir"')
        self.clump_edge = clump_edge
        if pair_angle not in ("uniform", "engine"):
            raise ValueError(f"pair_angle must be 'uniform' or 'engine', not {pair_angle!r}")
        # Subhalo sector (2026-10-01):
        #   "pairs": the validated 2026-09-24 weak/strong representation below
        #       (bit-identical to the pre-option code);
        #   "exact": the engine's decorated-host law evaluated without sampling
        #       (analytic/subhalos_exact.py): exact clump multiplicity, weak and
        #       strong together, host REBUILT at M - S (kappa, gamma, eps),
        #       engine placement angle, exact conditional shear.
        if sector not in ("pairs", "exact"):
            raise ValueError(f"sector must be 'pairs' or 'exact', not {sector!r}")
        self.sector = sector
        self.exact_kw = dict(exact_kw or {})
        # exact constructor args, so a worker process can rebuild an identical
        # model (recovering split as ksplit/kthr need not round-trip exactly)
        self._init_kw = dict(Mhost_min=Mhost_min, npsi=npsi, nd=nd, nth=nth,
                             split=split, ntheta=ntheta, nr=nr, prune=prune,
                             pair_angle=pair_angle, sector=sector,
                             exact_kw=dict(exact_kw or {}), clump_edge=clump_edge,
                             clump_mass=clump_mass)
        # pair-sector resolution (defaults = the validated 2026-09-24 setting)
        self.ntheta, self.nr = ntheta, nr
        # Relative angle theta between the host->ray and clump->ray directions
        # of a host+strong-clump pair (2026-09-27).
        #   "uniform": theta uniform on the ntheta nodes (the validated
        #       2026-09-24 setting; bit-identical to the pre-option code).
        #   "engine": node weights prop. to n(|r e_x - d e^{i theta}|), the
        #       host's projected clump density at the clump's host radius --
        #       the engine's placement density (cpp/subhalo.cpp:223 r_vec =
        #       r (cos phi, sin phi); both branches put the clump at host radius
        #       |r_vec - d_vec|, d_vec = (dx, dy): 253-254 and 266), with
        #       |gamma|^2 = gh^2 + gc^2 + 2 gh gc cos(theta) in the engine's
        #       spin-1 convention (host (cos phi, sin phi) gh, lensing.cpp:1118;
        #       clump (dx, dy)/d gc, subhalo.cpp:275-284).  Clumps are favoured
        #       on the host-centre side, where the two shears ALIGN.  The node
        #       weights at fixed (r, d) sum to the uniform total, so nu_c(r)
        #       and all marginal weights are unchanged.  The density is peaked
        #       in theta when d >~ r: use ntheta >= 16 (see analytic/README.md,
        #       2026-09-27, for the ntheta-doubling check).
        self.pair_angle = pair_angle
        # Drop pair groups whose VARIANCE contribution w*kappa^2 is below this
        # (absolute).  0 = off (the validated setting).  The analytic analogue
        # of the MC not rendering clumps that do not matter: at z_s=1, cutting
        # at 1e-16.9 drops 54% of pair samples for 9e-7 of the pair variance.
        self.prune = prune
        # encounters with kappa_c < ksplit are "weak": mean + conditional
        # variance ride on the host cell; stronger ones are Poisson cells
        self.ksplit = None if kthr is None else split * kthr
        self.ksub = None if kthr is None else KTHR_FACTOR * kthr
        self.Mhost_min = Mhost_min
        self.npsi, self.nd, self.nth = npsi, nd, nth
        self.zl, self.Ml = F.engine_grids()
        s = (1 + ALPHA) / OMEGA
        af = 0.815 * exp(-0.25) / 0.5**0.707
        self.wf = sqrt(2 * log(af + 1))
        self.gden = _uinc(s, BETA * PSI_RES**OMEGA) - _uinc(s, BETA)
        self.s = s
        self._cache = {}
        self.stats = dict(fsub_mass=[], hosts=0)

    @property
    def exact(self):
        """The ExactSector sharing this model's host tables (sector="exact")."""
        if getattr(self, "_exact", None) is None:
            import subhalos_exact
            self._exact = subhalos_exact.ExactSector(self, **self.exact_kw)
        return self._exact

    # ----------------------------------------------------------- per host --
    def fs(self, M, z):
        """JvdB14 f_s with z_f from delta_c(z_f) = delta_c(z) + w_f dsigma."""
        cos = self.cos
        dcz = F.DC0 / F.Dg(cos, z)
        ds2 = cos.sigmaM(0.5 * M, 0.0)**2 - cos.sigmaM(M, 0.0)**2
        if ds2 <= 0:
            return 0.0
        rhs = dcz + self.wf * sqrt(ds2)
        if F.DC0 / F.Dg(cos, 30.0) < rhs:
            return 0.0
        lo, hi = z, 30.0
        for _ in range(60):
            zm = 0.5 * (lo + hi)
            if F.DC0 / F.Dg(cos, zm) < rhs:
                lo = zm
            else:
                hi = zm
        zf = 0.5 * (lo + hi)
        zz = z + (np.arange(200) + 0.5) * (zf - z) / 200
        d = _Om_z(cos, zz) - 1
        Ntau = np.sum(6.006 * sqrt((18 * pi**2 + 82 * d - 39 * d * d) / 178.0)
                      / (1 + zz)) * (zf - z) / 200
        if Ntau <= 0:
            return 0.0
        f = 0.3563 / Ntau**0.6 - 0.075
        return f if 0.0 < f < 0.95 else 0.0

    def host(self, jz, jM):
        key = (jz, jM)
        if key in self._cache:
            return self._cache[key]
        cos, z, M = self.cos, self.zl[jz], self.Ml[jM]
        out = None
        f = self.fs(M, z) if M >= self.Mhost_min else 0.0
        if f > 0:
            C_, rs, ks, fC = cos.nfw_params(M, z, self.zs)
            r200 = rs * C_
            eta = _eta(cos, C_, z)
            Mvir = M * _mu(eta * C_) / _mu(C_)
            g = OMEGA * BETA**self.s / self.gden * f
            psi_min = M_FLOOR / Mvir
            if psi_min < PSI_MAX:
                # clumps: psi grid (dN per bin, thinned), reach D, NFW params
                le = np.linspace(log(psi_min), 0.0, self.npsi + 1)
                lc = 0.5 * (le[1:] + le[:-1])
                psi = exp(lc)
                dN = g * psi**ALPHA * exp(-BETA * psi**OMEGA) * np.diff(le)
                m = psi * Mvir
                # one vectorised call (bit-identical to the per-mass loop, 2026-10-01)
                m200c = m
                if self.clump_edge == "rvir" and self.clump_mass == "vir":
                    m200c = _m200_from_mvir(cos, m, z)
                Cc, rsc, ksc, _ = cos.nfw_params(m200c, z, self.zs)
                Cc, rsc, ksc = np.asarray(Cc, float), np.asarray(rsc, float), np.asarray(ksc, float)
                if self.clump_edge == "rvir":
                    xtc = _eta_vec(cos, Cc, z) * Cc                # clump r_vir / r_s
                    D = xtc * rsc
                else:
                    xtc = None
                    D = F._xmax_for_vec(ksc, self.ksub) * rsc   # bit-equal to the per-k loop
                # projected, normalised clump number density per area n(R)
                x0 = X0_RVIR * eta
                if _HAVE_NUMBA and BIAS_EXP == 2.5:
                    Rh, n3 = _n3_proj_nb(float(eta), float(C_), float(x0))
                else:
                    Rh = (np.arange(256) + 0.5) / 256 * eta
                    u = np.linspace(0, 1, 160)
                    umax = sqrt(np.maximum(eta**2 - Rh**2, 0))
                    X = sqrt(Rh[:, None]**2 + (u[None, :] * umax[:, None])**2)
                    B = 1 / sqrt((X / x0)**-BIAS_EXP + 1)
                    p3 = X / (1 + C_ * X)**2 * B / X**2           # rho_NFW B (per vol)
                    n3 = np.trapezoid(p3, u, axis=1) * umax * 2  # line-of-sight integral
                n2 = n3 / np.trapezoid(n3 * 2 * pi * Rh, Rh)  # per unit Rhat^2
                out = dict(M=M, z=z, r200=r200, eta=eta, vr=Mvir / M, f=f,
                           dN=dN, m=m, rsc=rsc, ksc=ksc, D=D, xtc=xtc,
                           Rn=Rh * r200, n2=n2 / r200**2, rs=rs, ks=ks)
                self.stats["fsub_mass"].append(float(np.sum(dN * m) / Mvir))
        self._cache[key] = out
        return out

    def _ring(self, h, r, d):
        """n_ring(r, d) = angle-average of n(|r - d|): (Nr, Nd).

        Numba-fused when available (2026-09-25): the numpy form materialises a
        (Nr, Nd, nth) array per host purely to average it over theta; the
        profiler put ~215 s (np.interp 111 s + _ring self 104 s) of an 893 s
        build_population(+sub) here.  The kernel is gated against
        `_ring_numpy` (kept as the reference and the fallback)."""
        if not _HAVE_NUMBA:
            return self._ring_numpy(h, r, d)
        th = (np.arange(self.nth) + 0.5) * pi / self.nth
        return _ring_kernel(np.ascontiguousarray(r, dtype=float),
                            np.ascontiguousarray(d, dtype=float),
                            np.cos(th),
                            np.ascontiguousarray(h["Rn"], dtype=float),
                            np.ascontiguousarray(h["n2"], dtype=float))

    def _ring_numpy(self, h, r, d):
        """Reference implementation of `_ring` (the pre-2026-09-25 code)."""
        th = (np.arange(self.nth) + 0.5) * pi / self.nth
        R = sqrt(r[:, None, None]**2 + d[None, :, None]**2
                 - 2 * r[:, None, None] * d[None, :, None] * np.cos(th))
        return np.interp(R, h["Rn"], h["n2"], right=0.0).mean(axis=2)

    def _dgrid(self, h):
        Dm = h["D"].max()
        de = np.concatenate([[0.0], np.geomspace(1e-5 * Dm, Dm, self.nd)])
        dc = np.where(de[:-1] > 0, sqrt(de[:-1] * de[1:]), 0.5 * de[1])
        return de, dc, pi * np.diff(de**2)

    # -------------------------------------------------- interface to F ---
    def host_profiles(self, jz, jM, r):
        """(dk, s_k, s_g) at host-ray offsets r: mean kappa of WEAK clump
        encounters minus the mean carve (all rendered clumps), and the weak
        encounters' conditional (Poisson-given-host) kappa^2 and gamma^2."""
        h = self.host(jz, jM)
        z0 = np.zeros_like(r)
        if h is None or h["D"].max() <= 0:
            return z0, z0, z0
        de, dc, dA = self._dgrid(h)
        inside = dc[None, :] < h["D"][:, None]                       # (npsi, nd)
        kg = [sgl.kappa_gamma(dc / rs_, k_) for rs_, k_ in zip(h["rsc"], h["ksc"])]
        kc = np.array([a for a, b in kg])
        gc = np.array([b for a, b in kg])
        weak = inside & (kc < self.ksplit)
        dN = h["dN"][:, None]
        ring = self._ring(h, r, dc)                                  # (Nr, nd)
        kbar = ring @ (np.sum(dN * kc * weak, axis=0) * dA)
        sk = ring @ (np.sum(dN * kc**2 * weak, axis=0) * dA)
        sg = ring @ (np.sum(dN * gc**2 * weak, axis=0) * dA)
        dM = ring @ (np.sum(dN * h["m"][:, None] * weak, axis=0) * dA) / h["vr"]   # strong ones carve per pair
        Mh = h["M"]
        dkdM = (self._host_kappa(Mh * 1.01, h["z"], r)
                - self._host_kappa(Mh * 0.99, h["z"], r)) / (0.02 * Mh)
        h["_ring"], h["_r"] = ring, r
        return kbar - dM * dkdM, sk, sg

    def _host_kappa(self, M, z, r):
        C_, rs, ks, fC = self.cos.nfw_params(M, z, self.zs)
        return sgl.kappa_gamma(np.ascontiguousarray(r / rs), ks)[0]

    def add_pairs(self, cells, jz, jM, shell, beta, xc, W, ntheta=None, nr=None):
        """STRONG clump encounters as exact first-order Neyman-Scott pairs.

        A host hit at offset r carries 0 strong clumps with prob e^{-nu(r)} and
        exactly one (cell c) with prob e^{-nu} nu_c(r).  The pair jump is
        kappa_h(r) + dk_weak(r) - (m_c/vr) dkappa_h/dM + kappa_c, and
        |gamma_h (+) gamma_c| with the relative angle integrated by Neumann's
        addition theorem (J0(a)J0(b) = 1/pi int_0^pi J0(|a + b e^{i th}|)).
        pair_angle="engine" replaces the uniform theta weights by the engine's
        placement density n(|r - d e^{i th}|) (normalised per (r, d), so the
        weights still sum to nu_c); see __init__.
        nu_c is evaluated on the HOST'S OWN fine grid (xc, W) so the weight the
        host loses is exactly the weight the pairs carry; fine cells are then
        aggregated into nr radial groups only to bound the pair-cell count.
        Returns nu(r) on xc (the caller multiplies host-alone weights by e^-nu)."""
        ntheta = self.ntheta if ntheta is None else ntheta
        nr = self.nr if nr is None else nr
        h = self.host(jz, jM)
        if h is None or h["D"].max() <= 0:
            return None
        de, dc, dA = self._dgrid(h)
        rs, ks = h["rs"], h["ks"]
        rr = xc * rs
        ring = h["_ring"] if h.get("_r") is not None and h["_r"].shape == rr.shape \
            and np.allclose(h["_r"], rr) else self._ring(h, rr, dc)
        kh, gh = sgl.kappa_gamma(np.ascontiguousarray(xc), ks)
        Mh = h["M"]
        dkdM = (self._host_kappa(Mh * 1.01, h["z"], rr)
                - self._host_kappa(Mh * 0.99, h["z"], rr)) / (0.02 * Mh)
        dkh_weak = h.get("_dk_weak_fn", lambda x: 0.0)(xc)
        grp = np.minimum((np.log(np.maximum(xc, 1e-12) / xc.max()) / log(1e-5) * -nr
                          + nr).astype(int), nr - 1)
        grp = np.clip(grp, 0, nr - 1)
        strong = []
        for i in range(len(h["m"])):
            sel = dc < h["D"][i]
            if not sel.any():
                continue
            k_, g_ = sgl.kappa_gamma(np.ascontiguousarray(dc[sel] / h["rsc"][i]), h["ksc"][i])
            st = k_ >= self.ksplit
            if st.any():
                idx = np.nonzero(sel)[0][st]
                strong.append((i, idx, k_[st], g_[st], h["dN"][i] * ring[:, idx] * dA[idx]))
        if not strong:
            return None
        nu = sum(t[4].sum(axis=1) for t in strong)                   # (nfine,)
        # probability-conserving multiplicity: ">= 1 strong clump" (prob
        # 1-e^-nu) is represented by one clump drawn with prob nu_c/nu plus the
        # MEAN of the extra ones, (E[n|n>=1]-1) kbar_strong -- exact in
        # probability and in mean clump kappa (first order dropped the n>=2
        # mass, ~nu^2/2, which sits in cores where kappa_h^2 is largest)
        nus = np.maximum(nu, 1e-300)
        p1 = -np.expm1(-nu)
        wf = W * p1 / nus
        kbar_st = sum(t[4] @ t[2] for t in strong) / nus
        extra = (nus / np.maximum(p1, 1e-300) - 1.0) * kbar_st
        # Group aggregation of the host side (weights, weighted kappa/gamma).
        #
        # BATCHED OVER j (2026-09-25).  This used to be a per-(i,j) loop calling
        # a 2-bincount closure `agg`; a cProfile of one build_population(+sub)
        # at z_s=1 caught it at 12,232,716 calls and 201 s of SELF time -- ~23%
        # of an 893 s build.  That time was numpy per-call overhead on (nfine,)
        # arrays (~16 us for the multiply + 2 bincounts + where + divide), not
        # arithmetic, so the fix is to do every j at once rather than to JIT
        # the loop.
        #
        # S is the one-hot group operator: S @ v reproduces
        # np.bincount(grp, weights=v, minlength=nr) exactly, but takes a whole
        # MATRIX of j columns.  The old code also recomputed the denominator
        # three times per (i,j) (once as `wg`, once inside each `agg`); here it
        # is formed once.
        #
        # Output ORDER is preserved: the masks below are transposed to (nj, nr)
        # so the C-order flatten is j-major / r-minor, exactly the order the
        # old `Ks.append` per j produced.
        th = (np.arange(ntheta) + 0.5) * pi / ntheta
        cth = np.cos(th)
        engine = self.pair_angle == "engine"
        if engine:
            # angle-node weights: conditional theta distribution at every
            # (fine r, strong-clump d cell) -- see _pair_angle_table
            kcol_all = np.concatenate([t[1] for t in strong])
            ku = np.unique(kcol_all)
            kmap = np.searchsorted(ku, kcol_all)
            if _HAVE_NUMBA:
                Pang = _pair_angle_table(np.ascontiguousarray(rr, dtype=float),
                                         np.ascontiguousarray(dc[ku], dtype=float), cth,
                                         np.ascontiguousarray(h["Rn"], dtype=float),
                                         np.ascontiguousarray(h["n2"], dtype=float))
            else:
                Rq = sqrt(np.maximum(rr[:, None, None]**2 + dc[ku][None, :, None]**2
                                     - 2 * rr[:, None, None] * dc[ku][None, :, None] * cth, 0.0))
                v = np.interp(Rq, h["Rn"], h["n2"], right=0.0)
                sv = v.sum(axis=2, keepdims=True)
                Pang = np.where(sv > 0, v / np.where(sv > 0, sv, 1.0), 1.0 / ntheta)
        if _HAVE_NUMBA:
            # all strong bins of this host in ONE segment-sum call
            nuc_all = np.ascontiguousarray(np.concatenate([t[4] for t in strong], axis=1))
            cm_all = np.concatenate([np.full(t[1].size, h["m"][t[0]] / h["vr"]) for t in strong])
            kh_dkw = np.ascontiguousarray(kh + np.broadcast_to(dkh_weak, kh.shape), dtype=float)
            den_all, nk_all, ng_all = _pair_group_sums(
                np.ascontiguousarray(grp, dtype=np.int64), nr,
                np.ascontiguousarray(wf, dtype=float), nuc_all, kh_dkw,
                np.ascontiguousarray(np.broadcast_to(extra, kh.shape), dtype=float),
                np.ascontiguousarray(dkdM, dtype=float), cm_all,
                np.ascontiguousarray(gh, dtype=float))
            if engine:
                wt_all = _pair_angle_sums(np.ascontiguousarray(grp, dtype=np.int64), nr,
                                          np.ascontiguousarray(wf, dtype=float), nuc_all,
                                          np.ascontiguousarray(kmap, dtype=np.int64), Pang)
        else:
            S = np.zeros((nr, xc.size))
            S[grp, np.arange(xc.size)] = 1.0
        Ks, Gs, Ws = [], [], []
        col = 0
        st = self.stats
        for i, idx, kc, gc, nuc in strong:
            if _HAVE_NUMBA:
                sl = slice(col, col + idx.size)
                col += idx.size
                den = den_all[:, sl]
                ds = np.where(den > 0, den, 1.0)
                kgm = nk_all[:, sl] / ds + kc[None, :]
                ggm = ng_all[:, sl] / ds
                if engine:
                    wang = wt_all[:, sl, :]                          # (nr, nj, ntheta)
            else:
                carve = (h["m"][i] / h["vr"]) * dkdM
                khost = kh + dkh_weak - carve + extra
                wt = wf[:, None] * nuc                               # (nfine,nj)
                den = S @ wt                                         # (nr, nj)
                ds = np.where(den > 0, den, 1.0)
                kgm = (S @ (wt * khost[:, None])) / ds + kc[None, :]
                ggm = (S @ (wt * gh[:, None])) / ds
                if engine:
                    sl = slice(col, col + idx.size)
                    col += idx.size
                    wang = np.einsum("gf,fjt->gjt", S,
                                     wt[:, :, None] * Pang[:, kmap[sl], :])
            ok = (den > 0).T                                         # (nj, nr)
            if self.prune > 0:
                # variance carried by each (j, r) group: sum over theta of
                # (wg/ntheta) * kg^2 = wg * kg^2 (K is theta-independent)
                ok &= (den.T * kgm.T**2) >= self.prune
            if not ok.any():
                continue
            kg_ = kgm.T[ok]
            gg_ = ggm.T[ok]
            wg_ = den.T[ok]
            gcb = np.broadcast_to(gc[:, None], ok.shape)[ok]
            G2 = (gg_[:, None]**2 + gcb[:, None]**2
                  + 2 * gg_[:, None] * gcb[:, None] * cth)
            Ks.append(np.repeat(kg_, ntheta))
            Gs.append(sqrt(G2).ravel())
            if engine:
                wn = wang.transpose(1, 0, 2)[ok]                     # (n_ok, ntheta)
                Ws.append(wn.ravel())
            else:
                wn = np.broadcast_to((wg_ / ntheta)[:, None], G2.shape)
                Ws.append(np.repeat(wg_, ntheta) / ntheta)
            # diagnostics only (never feed back into the cells)
            ggc = gg_ * gcb
            st["pair_W"] = st.get("pair_W", 0.0) + float(wn.sum())
            st["pair_Wcos"] = st.get("pair_Wcos", 0.0) + float((wn * cth).sum())
            st["pair_Wgg"] = st.get("pair_Wgg", 0.0) + float((wn.sum(axis=1) * ggc).sum())
            st["pair_Wggcos"] = st.get("pair_Wggcos", 0.0) + float(((wn * cth).sum(axis=1) * ggc).sum())
            st["pair_WG2"] = st.get("pair_WG2", 0.0) + float((wn * G2).sum())
            st["pair_WG2u"] = st.get("pair_WG2u", 0.0) + float(
                (wn.sum(axis=1) * (gg_**2 + gcb**2)).sum())
        if Ks:
            cells.add(np.concatenate(Ks), np.concatenate(Gs), np.concatenate(Ws), shell, beta)
        self.stats["hosts"] += 1
        return xc, nu

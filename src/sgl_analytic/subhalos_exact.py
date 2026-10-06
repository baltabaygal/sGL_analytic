#!/usr/bin/env python
"""Exact decorated-host subhalo sector (2026-10-01).

Replaces SubhaloModel's weak/strong split (mean + Gaussian blur on host-alone
cells; strong clumps as first-order pairs with a mean-extra multiplicity and a
linear, kappa-only carve) by the engine's own law, evaluated without sampling.

Engine law (cpp/subhalo.cpp addClumpsRestricted, cpp/lensing.cpp add_host):
for a host hit at offset r the RENDERED clumps are an unconditioned Poisson
process (Poisson proposals, independent thinnings) on cells c = (mass bin i,
ray-distance cell d) with rates nu_c(r) = dN_i n_ring(r, d) dA_d [d < D_i],
marks kappa_c, spin-1 shear gamma_c e_theta (theta ~ the host's projected clump
density at the clump's host radius), and mass m_c = m_i / vr.  The host is
REBUILT at M_eff = max(M - S, Mmin), S = sum m_c, for kappa, gamma AND eps.

The law needed is the JOINT law of (K, S), K = sum kappa_c.  Its CF is exact,
    phi(k, u) = exp[ sum_i e^{i u m_i} sum_d nu_id e^{i k kappa_id} - nu ],
because m depends on the mass bin only; it is sampled on an FFT lattice and
inverted (Gaussian smoothing of one lattice step; the n = 0 atom e^{-nu} is
taken out exactly).  The shear enters through the exact conditional mean of
|gamma|^2 given (K, S) (Palm / Campbell identities, products in Fourier space):
    |gamma|^2 | K = gh^2 + 2 gh E1(K) + E2(K),
    E1 = E[sum gc cos theta | K],  E2 = E[|sum gc e_theta|^2 | K],
with gh = gh(M - S; r, phi) -- the only shear approximation is replacing |gamma|
in a (K, S) bin by its conditional rms.  Marks with cumulative rate <= eps_core
from the top ("core hits", kappa_c >~ 0.1-1) are added at first order on the
bulk mean (error O(eps_core^2)).

Per host the fine r cells are split into ADAPTIVE radial groups (a group closes
when the clump-law summaries change by > group_tol); one lattice per group; every
fine cell keeps its own exact rebuilt host kappa_h(M - S; r, phi) AND is shifted
by its exact mean offsets E[K|r] - E[K|group], E[S|r] - E[S|group] (all rates
share the ring factor, so these are exact and linear): first moments are exact
per cell for any grouping; only the shape varies within a group.  The lattice
is compressed to K bins (equal-probability body, geometric tails), each carrying its
weight, mean K, an NSQ-node Gauss rule for the conditional S law (Stieltjes on
the lattice row, smoothing variance removed; exact to degree 2 NSQ - 1 in S --
the rebuilt host kappa_h(M - S) is far from cubic, 2 nodes lost 1-10% of Var),
and E1, E2.

Validated per host against an engine-law MC (tmp/sub_exact_test.py): mean
<= 3e-4, Var 0.97-1.02, kappa_3 within MC error, <gamma^2> <= 5e-4, for the
top hosts and a small one, x = r/rs = 0.02-1.
"""
from __future__ import annotations

import math as _math
import numpy as np
from numpy import sqrt

from . import sgl
from . import sgl_full as F
from . import subhalos

try:
    import numba as _numba
    _HAVE_NUMBA = True
except ImportError:                    # pragma: no cover
    _HAVE_NUMBA = False


NSQ = 3                                   # Gauss nodes in S per K bin
# _gauss3_tau acceptance tolerances (2026-10-03).  A valid moment sequence on
# [lo, hi] has a 3-node rule with nodes inside and weights > 0, so a test that
# fails only by roundoff must not switch rules: at the outermost coarse r node
# the clump rate is ~1e-8 (law = atom at 0 + one rare jump), the third weight
# is ~nu^2/2 ~ 1e-16 and the low node sits ~1e-12 from lo -- both pure noise,
# and a 2-ulp change of the NFW kernel flipped 3 <-> 2 nodes there.  Node noise
# is ~eps * max|t| (Cardano from tau_5 ~ 1e12): nodes within G3_TOL_T (1 +
# max|t|) of a bound are accepted and clamped, weights > -G3_TOL_W clamped to 0.
G3_TOL_T = 1e-10
G3_TOL_W = 1e-12
MT_WFLOOR = 1e-12                         # min point weight that sets the host-mass table range
KS_RIDGE = 1e-10                          # relative ridge on the S-perp step of the shear regression
GAUSS_FALLBACK = [0]                      # rows that fell back to one node (diagnostic)


def _cut_for(nu, k, eps):
    """Smallest kappa_cut with total rate of marks above it <= eps."""
    o = np.argsort(-k)
    cr = np.cumsum(nu[o])
    j = int(np.searchsorted(cr, eps, side="right"))
    return float(k[o[j]]) if j < k.size else 0.0


if _HAVE_NUMBA:
    @_numba.njit(cache=True, parallel=True)
    def _bin_sums(NK, dk, kb, inv, nbin, c0, c1, c2):
        """A_j[n, b] = sum_{cells in bin b} c_j e^{i k_n kappa}, k_n = n dk for
        n < NK/2 and (n - NK) dk above (np.fft.fftfreq order); the positive half
        by a phase recurrence, the negative half by Hermitian symmetry."""
        A0 = np.zeros((NK, nbin), np.complex128)
        A1 = np.zeros((NK, nbin), np.complex128)
        A2 = np.zeros((NK, nbin), np.complex128)
        half = NK // 2
        nc = kb.shape[0]
        for b in _numba.prange(nbin):
            for c in range(nc):
                if inv[c] != b:
                    continue
                st = np.exp(1j * dk * kb[c])
                z = 1.0 + 0j
                for n in range(half + 1):
                    A0[n, b] += c0[c] * z
                    A1[n, b] += c1[c] * z
                    A2[n, b] += c2[c] * z
                    z *= st
                    if n % 64 == 63:                     # renormalise the recurrence
                        z = np.exp(1j * (n + 1) * dk * kb[c])
            for n in range(half + 1, NK):
                A0[n, b] = np.conj(A0[NK - n, b])
                A1[n, b] = np.conj(A1[NK - n, b])
                A2[n, b] = np.conj(A2[NK - n, b])
        return A0, A1, A2


def lattice_law(nu, k, m, g, ec, ibin, smooth=1.0, smoothS=2.0, eps_core=1e-4,
                resK=10.0, resS=8.0, nmax=2**21):
    """Joint (K, S) law of one clump population (see module docstring).
    Returns a dict with the lattice pmf p (NK, NS), density-weighted shear sums
    e1, e2, grids, the n = 0 atom p0, and the core list."""
    kc = _cut_for(nu, k, eps_core)
    core = k > kc
    b = ~core
    nub, kb, mb, gb, eb, ib = nu[b], k[b], m[b], g[b], ec[b], ibin[b]
    if not nub.size or nub.sum() <= 0:
        z1 = np.zeros((1, 1))
        return dict(p=z1, e1=z1, e2=z1, K=np.zeros(1), S=np.zeros(1), dK=1.0, dS=1.0,
                    sK=0.0, mK=0.0, mS=0.0, p0=1.0, sS2=0.0, E1b=0.0, E2b=0.0,
                    core=dict(nu=nu[core], k=k[core], m=m[core], g=g[core], ec=ec[core]),
                    nu_core=float(nu[core].sum()))
    mK, vK = float(np.sum(nub * kb)), float(np.sum(nub * kb**2))
    mS, vS = float(np.sum(nub * mb)), float(np.sum(nub * mb**2))
    sK, sS = sqrt(vK) + 1e-300, sqrt(vS) + 1e-300
    LK = mK + 14 * sK + (kb.max() if kb.size else 0.0)
    LS = mS + 14 * sS + (mb.max() if mb.size else 0.0)
    # padding below 0 / above the support: >= 8 smoothing widths, else the
    # kernel's tail around the dominant S ~ 0 mass WRAPS to the top of the
    # periodic lattice (2% of it at S ~ 2M with 4 cells and smoothS = 2:
    # Var S doubled for a small host)
    padK, padS = int(np.ceil(8 * smooth)) + 2, int(np.ceil(8 * smoothS)) + 2
    p2 = lambda n: int(2 ** np.ceil(np.log2(max(n, 16))))
    NK, NS = p2(resK * LK / sK + 2 * padK), p2(resS * LS / sS + 2 * padS)
    while NK * NS > nmax:
        if NK / resK >= NS / resS:
            NK //= 2
        else:
            NS //= 2
    dK, dS = LK / (NK - 2 * padK), LS / (NS - 2 * padS)
    K0, S0 = -padK * dK, -padS * dS
    kq = 2 * np.pi * np.fft.fftfreq(NK, d=dK)
    # every lattice array is Hermitian (real marks): only the u >= 0 half is
    # formed and inverted with irfft2 (identical result, half the work)
    uq = 2 * np.pi * np.arange(NS // 2 + 1) / (NS * dS)
    # E = sum_i A_i(k) e^{i u m_i}: per-bin sums as ONE (NK x ncell) @ (ncell x nbin)
    # product, then ONE (NK x nbin) @ (nbin x NS) product (same algebra as the
    # per-bin outer-product loop; reordered sums)
    ub, inv = np.unique(ib, return_inverse=True)
    mbin = np.array([mb[inv == j][0] for j in range(ub.size)])
    if _HAVE_NUMBA:
        A0, A1, A2 = _bin_sums(NK, 2 * np.pi / (NK * dK), kb, inv.astype(np.int64), ub.size,
                               nub, nub * gb * eb, nub * gb**2)
    else:                                                          # pragma: no cover
        oh = np.zeros((kb.size, ub.size))
        oh[np.arange(kb.size), inv] = 1.0
        ph = np.exp(1j * np.outer(kq, kb))
        A0 = ph @ (oh * nub[:, None])
        A1 = ph @ (oh * (nub * gb * eb)[:, None])
        A2 = ph @ (oh * (nub * gb**2)[:, None])
    EM = np.exp(1j * np.outer(mbin, uq))                         # (nbin, NS)
    E = A0 @ EM
    F1 = A1 @ EM
    F2 = A2 @ EM
    E -= nub.sum()
    p0 = float(np.exp(-nub.sum()))
    phi = np.exp(E) - p0
    # S is a sum of a few discrete clump masses for small hosts (atomic): two
    # lattice steps of smoothing in S kill its ringing (Nyquist damping 3e-9);
    # the added variance (smoothS dS)^2 is removed per K bin in compress().
    sh = np.outer(np.exp(-0.5 * (smooth * dK * kq)**2 - 1j * kq * K0),
                  np.exp(-0.5 * (smoothS * dS * uq)**2 - 1j * uq * S0))
    # real(fft2(X)) / N for Hermitian X  ==  irfft2(conj(X_half), s)
    inv = lambda X: np.fft.irfft2(np.conj(X), s=(NK, NS))
    p = inv(phi * sh)
    e1 = inv(phi * F1 * sh)
    e2 = inv(phi * (F2 + F1**2) * sh)
    E1b = float(np.sum(nub * gb * eb))
    return dict(p=p, e1=e1, e2=e2, K=K0 + dK * np.arange(NK), S=S0 + dS * np.arange(NS),
                dK=dK, dS=dS, sK=sK, mK=mK, mS=mS, p0=p0, sS2=(smoothS * dS)**2,
                E1b=E1b, E2b=float(np.sum(nub * gb**2)) + E1b**2,
                core=dict(nu=nu[core], k=k[core], m=m[core], g=g[core], ec=ec[core]),
                nu_core=float(nu[core].sum()))


def _gauss_rows(Wm, x, n):
    """n-point Gauss rule for every row of a non-negative discrete measure
    Wm (nb, nx) on nodes x (Stieltjes procedure + Golub-Welsch); rows with
    fewer than n support points get repeated nodes with zero weight."""
    nb = Wm.shape[0]
    tot = Wm.sum(axis=1)
    sc = np.where(tot > 0, tot, 1.0)
    w = Wm / sc[:, None]
    xm = np.sum(w * x[None, :], axis=1)
    xs = sqrt(np.maximum(np.sum(w * (x[None, :] - xm[:, None])**2, axis=1), 1e-300))
    t = (x[None, :] - xm[:, None]) / xs[:, None]                # standardised, per row
    a = np.zeros((nb, n)); b = np.zeros((nb, n))
    pm1 = np.zeros_like(t); p0 = np.ones_like(t)
    nrm0 = np.ones(nb)
    for j in range(n):
        nrm = np.sum(w * p0**2, axis=1)
        a[:, j] = np.sum(w * t * p0**2, axis=1) / np.maximum(nrm, 1e-300)
        if j > 0:
            b[:, j] = nrm / np.maximum(nrm0, 1e-300)
        p1 = (t - a[:, j][:, None]) * p0 - (b[:, j][:, None] * pm1 if j > 0 else 0.0)
        pm1, p0, nrm0 = p0, p1, nrm
    J = np.zeros((nb, n, n))
    idx = np.arange(n)
    J[:, idx, idx] = a
    off = sqrt(np.maximum(b[:, 1:], 0.0))
    J[:, idx[:-1], idx[1:]] = off
    J[:, idx[1:], idx[:-1]] = off
    # degenerate rows (empty, or a measure with < n support points that makes
    # the recurrence non-finite) fall back to ONE node at the row mean carrying
    # the whole weight -- exact in mean; such rows carry negligible weight
    good = (tot > 0) & np.all(np.isfinite(J), axis=(1, 2)) & np.isfinite(xm) & np.isfinite(xs)
    nodes = np.repeat(np.where(np.isfinite(xm), xm, 0.0)[:, None], n, axis=1)
    wts = np.zeros((nb, n)); wts[:, 0] = tot
    if good.any():
        try:
            ev, V = np.linalg.eigh(J[good])
            nodes[good] = xm[good][:, None] + xs[good][:, None] * ev
            wts[good] = tot[good][:, None] * V[:, 0, :]**2
        except np.linalg.LinAlgError:                   # row by row, then fall back
            for r_ in np.nonzero(good)[0]:
                try:
                    ev, V = np.linalg.eigh(J[r_])
                    nodes[r_] = xm[r_] + xs[r_] * ev
                    wts[r_] = tot[r_] * V[0, :]**2
                except np.linalg.LinAlgError:
                    GAUSS_FALLBACK[0] += 1
    GAUSS_FALLBACK[0] += int((~good).sum())
    return nodes, wts


def compress(L, nq=48, q_tail=1e-3, tail_ratio=1.25, wfloor=1e-16):
    """Lattice -> points (K, S_node, w, E1, E2).  K bins: EQUAL-PROBABILITY in
    the body (nq bins between the q_tail and 1 - q_tail quantiles of the K
    marginal -- the K spread is set by rare strong marks, so sd-based widths
    put the narrow body into a few bins and lost 8-18% of the variance), and
    geometric beyond.  Two S nodes per bin at the conditional mean +- sd
    (lattice smoothing variance removed).  Plus the n = 0 atom and the
    (log-binned) first-order core hits."""
    p, K, S = L["p"], L["K"], L["S"]
    pK = p.sum(axis=1)
    a = np.abs(pK)
    if a.max() <= 1e-300 or a.sum() < 1e-14:            # no bulk clump law: n = 0 (+ cores)
        return _points_tail(L, [], [], [], [], [])
    nz = np.nonzero(a > wfloor * a.max())[0]
    kmin, kmax = K[nz[0]] - 1e-12, K[nz[-1]] + 1e-12
    cum = np.cumsum(a) / a.sum()
    qs = np.linspace(q_tail, 1 - q_tail, nq + 1)
    body = np.unique(np.interp(qs, cum, K))
    hb = max(float(np.median(np.diff(body))) if body.size > 1 else L["dK"], L["dK"])
    up = [body[-1]]
    while up[-1] < kmax:
        up.append(body[-1] + (up[-1] - body[-1] + hb) * tail_ratio)
    dn = [body[0]]
    while dn[-1] > kmin:
        dn.append(body[0] - (body[0] - dn[-1] + hb) * tail_ratio)
    edges = np.unique(np.concatenate([dn[::-1], body, up]))
    bi = np.clip(np.searchsorted(edges, K, side="right") - 1, 0, edges.size - 2)
    nb = edges.size - 1
    w = np.bincount(bi, pK, nb)
    wK = np.bincount(bi, pK * K, nb)
    wK2 = np.bincount(bi, pK * K * K, nb)
    # conditional S law per K bin (lattice smoothing removed by shrinking
    # each row about its mean: var -> var - sS2, exact for the Gaussian kernel)
    # bi is non-decreasing in K (sorted lattice): contiguous row blocks
    starts = np.searchsorted(bi, np.arange(nb))
    nonempty = np.zeros(nb, bool); nonempty[np.unique(bi)] = True
    PS = np.zeros((nb, S.size))
    PS[nonempty] = np.add.reduceat(p, starts[nonempty], axis=0)
    w1 = np.bincount(bi, L["e1"].sum(axis=1), nb)
    w2 = np.bincount(bi, L["e2"].sum(axis=1), nb)
    ok = np.abs(w) > wfloor
    w, wK, wK2, w1, w2, PS = w[ok], wK[ok], wK2[ok], w1[ok], w2[ok], PS[ok]
    ws = np.where(np.abs(w) > 1e-300, w, 1e-300)
    Kb = wK / ws
    pos = np.maximum(PS, 0.0)                                   # ringing lobes out of the rule
    tot = pos.sum(axis=1)
    mS_ = np.sum(pos * S[None, :], axis=1) / np.where(tot > 0, tot, 1.0)
    vS_ = np.sum(pos * (S[None, :] - mS_[:, None])**2, axis=1) / np.where(tot > 0, tot, 1.0)
    shrink = sqrt(np.maximum(vS_ - L["sS2"], 0.0) / np.where(vS_ > 0, vS_, 1.0))
    Ssh = mS_[:, None] + shrink[:, None] * (S[None, :] - mS_[:, None])
    nodes = np.empty((Kb.size, NSQ)); wts = np.empty((Kb.size, NSQ))
    # a shared S grid per row after shrinking: apply the rule to t = (S - m)/1
    # via per-row affine map (Gauss rules are affine-covariant)
    nd0, wt0 = _gauss_rows(pos, S, NSQ)
    nodes = mS_[:, None] + shrink[:, None] * (nd0 - mS_[:, None])
    wts = wt0 * (w / np.where(tot > 0, tot, 1.0))[:, None]       # keep the signed bin weight
    E1, E2 = w1 / ws, w2 / ws
    # two K nodes per bin at the bin mean +- its within-bin sd (half weight
    # each): every bin's K second moment is exact whatever the binning (48
    # equal-probability bins alone lost 3-12% of Var in the wide heavy-tail
    # bins).  Lattice smoothing variance in K (one step) is removed here too.
    vK = np.where(w > 0, wK2 / ws - Kb**2 - (L["dK"])**2, 0.0)
    sKb = sqrt(np.maximum(vK, 0.0))
    scale = np.exp(-L["nu_core"])
    Kn = np.stack([Kb - sKb, Kb + sKb], axis=1)                  # (nb, 2)
    Kq = np.repeat(Kn, NSQ, axis=1)                              # (nb, 2*NSQ)
    Sq = np.tile(np.maximum(nodes, 0.0), (1, 2))
    wq = np.tile(wts * scale / 2, (1, 2))
    return _points_tail(L, [Kq.ravel()], [Sq.ravel()], [wq.ravel()],
                        [np.repeat(E1, 2 * NSQ)], [np.repeat(E2, 2 * NSQ)])


def _points_tail(L, Kp, Sp, wp, e1p, e2p):
    """Append the n = 0 atom and the first-order core hits to bulk points."""
    nc = L["nu_core"]
    scale = np.exp(-nc)
    Kp = list(Kp) + [[0.0]]
    Sp = list(Sp) + [[0.0]]
    wp = list(wp) + [[L["p0"] * scale]]
    e1p = list(e1p) + [[0.0]]
    e2p = list(e2p) + [[0.0]]
    c = L["core"]
    if c["nu"].size:
        Kc = L["mK"] + c["k"]
        ge = c["g"] * c["ec"]
        e1c = L["E1b"] + ge
        e2c = L["E2b"] + c["g"]**2 + 2 * L["E1b"] * ge
        wc = c["nu"] * (-np.expm1(-nc) / nc)
        Sc = L["mS"] + c["m"]
        # log bins in K for the core hits (weight-averaged S, E1, E2)
        ce = np.geomspace(Kc.min() * (1 - 1e-12), Kc.max() * (1 + 1e-12), 40)
        cb = np.clip(np.searchsorted(ce, Kc, side="right") - 1, 0, ce.size - 2)
        n = ce.size - 1
        cw = np.bincount(cb, wc, n)
        g = cw > 0
        cws = np.where(g, cw, 1.0)
        for arr, v in ((Kp, Kc), (Sp, Sc), (e1p, e1c), (e2p, e2c)):
            arr.append((np.bincount(cb, wc * v, n) / cws)[g])
        wp.append(cw[g])
    return (np.concatenate(Kp), np.concatenate(Sp), np.concatenate(wp),
            np.concatenate(e1p), np.concatenate(e2p))


# =================================================================== 1D ===
# S enters only through the host carve kappa_h(M - S), and every K bin only
# needs a few CONDITIONAL MOMENTS of S.  Those are u-derivatives of the joint
# CF at u = 0, and the exponent is a sum over mass bins, so
#     (-i)^j d^j E / du^j |_0 = D_j(k) = sum_i m_i^j A_i(k),
#     E[S^j e^{ikK}] = phi0(k) B_j(D_1, ..., D_j)        (Bell polynomials),
# i.e. ~10 1D transforms of length NK replace the NK x NS lattice -- exactly,
# with no S lattice, no S smoothing and no S wrap-around (2026-10-01).
if _HAVE_NUMBA:
    @_numba.njit(cache=True, parallel=True)
    def _bin_sums_half(nh, dk, kb, inv, nbin, c0, c1, c2):
        """A_j[n, b] = sum_{cells in bin b} c_j e^{i n dk kappa}, n = 0..nh-1."""
        A0 = np.zeros((nh, nbin), np.complex128)
        A1 = np.zeros((nh, nbin), np.complex128)
        A2 = np.zeros((nh, nbin), np.complex128)
        nc = kb.shape[0]
        for b in _numba.prange(nbin):
            for c in range(nc):
                if inv[c] != b:
                    continue
                st = np.exp(1j * dk * kb[c])
                z = 1.0 + 0j
                for n in range(nh):
                    A0[n, b] += c0[c] * z
                    A1[n, b] += c1[c] * z
                    A2[n, b] += c2[c] * z
                    z *= st
                    if n % 64 == 63:
                        z = np.exp(1j * (n + 1) * dk * kb[c])
        return A0, A1, A2


def _bell(D):
    """Complete Bell polynomials B_1..B_5 of D[1..5] (D[0] unused)."""
    D1, D2, D3, D4, D5 = D[1], D[2], D[3], D[4], D[5]
    return [None, D1, D1**2 + D2, D1**3 + 3 * D1 * D2 + D3,
            D1**4 + 6 * D1**2 * D2 + 4 * D1 * D3 + 3 * D2**2 + D4,
            D1**5 + 10 * D1**3 * D2 + 10 * D1**2 * D3 + 15 * D1 * D2**2
            + 5 * D1 * D4 + 10 * D2 * D3 + D5]


def lattice_law_1d(nu, k, m, g, ec, ibin, smooth=1.0, eps_core=1e-4, resK=10.0,
                   nmax=2**18):
    """K lattice + exact conditional S moments (j <= 5) per K, see above."""
    kc = _cut_for(nu, k, eps_core)
    core = k > kc
    b = ~core
    nub, kb, mb, gb, eb, ib = nu[b], k[b], m[b], g[b], ec[b], ibin[b]
    cored = dict(nu=nu[core], k=k[core], m=m[core], g=g[core], ec=ec[core])
    if not nub.size or nub.sum() <= 0:
        return dict(p=np.zeros(1), sm=[None] + [np.zeros(1)] * 5, ms=1.0, Scap=0.0, e1=np.zeros(1),
                    e2=np.zeros(1), K=np.zeros(1), dK=1.0, sK=0.0, mK=0.0, mS=0.0, p0=1.0,
                    sS2=0.0, E1b=0.0, E2b=0.0, core=cored, nu_core=float(nu[core].sum()))
    mK, vK = float(np.sum(nub * kb)), float(np.sum(nub * kb**2))
    mS, vS = float(np.sum(nub * mb)), float(np.sum(nub * mb**2))
    sK = sqrt(vK) + 1e-300
    LK = mK + 14 * sK + kb.max()
    padK = int(np.ceil(8 * smooth)) + 2
    NK = int(2 ** np.ceil(np.log2(max(resK * LK / sK + 2 * padK, 16))))
    NK = min(NK, nmax)
    dK = LK / (NK - 2 * padK)
    K0 = -padK * dK
    nh = NK // 2 + 1
    kq = 2 * np.pi * np.arange(nh) / (NK * dK)
    ub, inv = np.unique(ib, return_inverse=True)
    mbin = np.array([mb[inv == j][0] for j in range(ub.size)])
    ms = mS + sqrt(vS) + 1e-300                                 # moment scale
    if _HAVE_NUMBA:
        A0, A1, A2 = _bin_sums_half(nh, 2 * np.pi / (NK * dK), kb, inv.astype(np.int64),
                                    ub.size, nub, nub * gb * eb, nub * gb**2)
    else:                                                          # pragma: no cover
        oh = np.zeros((kb.size, ub.size)); oh[np.arange(kb.size), inv] = 1.0
        ph = np.exp(1j * np.outer(kq, kb))
        A0, A1, A2 = (ph @ (oh * c[:, None]) for c in (nub, nub * gb * eb, nub * gb**2))
    mh = mbin / ms
    D = [None] + [A0 @ mh**j for j in range(1, 6)]
    phi0 = np.exp(A0.sum(axis=1) - nub.sum())
    p0 = float(np.exp(-nub.sum()))
    sh = np.exp(-0.5 * (smooth * dK * kq)**2 - 1j * kq * K0)
    invf = lambda X: np.fft.irfft(np.conj(X * sh), n=NK)
    B = _bell(D)
    F1, F2 = A1.sum(axis=1), A2.sum(axis=1)
    E1b = float(np.sum(nub * gb * eb))
    return dict(p=invf(phi0 - p0), sm=[None] + [invf(phi0 * B[j]) for j in range(1, 6)],
                ms=ms, e1=invf(phi0 * F1), e2=invf(phi0 * (F2 + F1**2)),
                K=K0 + dK * np.arange(NK), dK=dK, sK=sK, mK=mK, mS=mS, p0=p0, sS2=0.0,
                smooth=smooth, E1b=E1b, E2b=float(np.sum(nub * gb**2)) + E1b**2,
                core=cored, nu_core=float(nu[core].sum()), NK=NK,
                Scap=(mS + 14 * sqrt(vS) + mb.max()) / ms)


def _gauss3_from_moments(mu, cap=np.inf):
    """3-node Gauss rule from raw moments mu[:, 0..5] (rows normalised to
    mu0 = 1), in standardised variables; rows that are not a valid moment
    sequence -- or whose nodes leave the physical support [0, cap] (a far
    tiny-weight node at S ~ 34 M once stretched the host table and biased the
    mean kappa by 0.8%) -- fall back to the 2-node rule, then to the mean.
    Returns nodes (nb, 3) in the raw variable and probabilities (nb, 3)."""
    nb = mu.shape[0]
    a = mu[:, 1]
    var = mu[:, 2] - a**2
    sd = sqrt(np.maximum(var, 0.0))
    from math import comb
    cen = np.zeros((nb, 6))
    for j in range(6):
        cen[:, j] = sum(comb(j, i) * mu[:, i] * (-a)**(j - i) for i in range(j + 1))
    ss = np.where(sd > 0, sd, 1.0)
    tau = cen / ss[:, None]**np.arange(6)[None, :]
    nodes = np.repeat(a[:, None], 3, axis=1)
    prob = np.zeros((nb, 3)); prob[:, 0] = 1.0
    ok = sd > 0
    # 3-node: monic p3 = t^3 + al t^2 + be t + ga orthogonal to 1, t, t^2
    Hm = np.stack([np.stack([tau[:, i + 2], tau[:, i + 1], tau[:, i]], axis=1) for i in range(3)], axis=1)
    rhs = -np.stack([tau[:, 3], tau[:, 4], tau[:, 5]], axis=1)
    good3 = np.zeros(nb, bool)
    idx = np.nonzero(ok)[0]
    if idx.size:
        with np.errstate(all="ignore"):
            det = np.linalg.det(Hm[idx])
            sel = idx[np.abs(det) > 1e-12]
            if sel.size:
                co = np.linalg.solve(Hm[sel], rhs[sel][:, :, None])[:, :, 0]
                Cm = np.zeros((sel.size, 3, 3))
                Cm[:, 0, :] = -co; Cm[:, 1, 0] = 1.0; Cm[:, 2, 1] = 1.0
                ev = np.linalg.eigvals(Cm)
                real = np.all(np.abs(ev.imag) < 1e-8 * (1 + np.abs(ev.real)), axis=1)
                t = np.sort(ev.real, axis=1)
                tmin = (0.0 - a[sel]) / sd[sel]                              # S >= 0
                tmax = (cap - a[sel]) / sd[sel]
                dist = np.min(np.diff(t, axis=1), axis=1) > 1e-8 * (1 + np.abs(t).max(axis=1))
                pw = np.full((sel.size, 3), -1.0)
                if dist.any():
                    V = np.stack([np.ones_like(t[dist]), t[dist], t[dist]**2], axis=1)
                    pw[dist] = np.linalg.solve(V, np.tile(np.array([1.0, 0.0, 1.0]),
                                                          (int(dist.sum()), 1))[:, :, None])[:, :, 0]
                fine = (real & dist & np.all(pw > 0, axis=1)
                        & np.all(t >= tmin[:, None] - 1e-9, axis=1)
                        & np.all(t <= tmax[:, None] + 1e-9, axis=1))
                s3 = sel[fine]
                nodes[s3] = a[s3, None] + sd[s3, None] * t[fine]
                prob[s3] = pw[fine]
                good3[s3] = True
    # 2-node fallback (exact to the 3rd moment)
    r2 = ok & ~good3
    if r2.any():
        gsk = tau[r2, 3]
        rt = sqrt(gsk**2 + 4.0)
        t1, t2 = (gsk + rt) / 2, (gsk - rt) / 2
        q1, q2 = -t2 / (t1 - t2), t1 / (t1 - t2)
        n1, n2 = a[r2] + sd[r2] * t1, a[r2] + sd[r2] * t2
        in2 = (n1 <= cap) & (n2 >= 0) & np.isfinite(n1) & np.isfinite(n2)
        rr = np.nonzero(r2)[0]
        g2 = rr[in2]
        nodes[g2, 0], nodes[g2, 1], nodes[g2, 2] = n1[in2], n2[in2], a[g2]
        prob[g2, 0], prob[g2, 1], prob[g2, 2] = q1[in2], q2[in2], 0.0
        GAUSS_FALLBACK[0] += int(r2.sum())          # rows left at the mean keep prob (1, 0, 0)
    return nodes, prob


def compress_1d(L, nq=48, q_tail=1e-3, tail_ratio=1.25, wfloor=1e-16):
    """compress() for the 1D law: same K bins / K nodes; the S rule per bin
    from the EXACT conditional moments (no smoothing to remove in S)."""
    pK, K = L["p"], L["K"]
    a = np.abs(pK)
    if a.max() <= 1e-300 or a.sum() < 1e-14:
        return _points_tail(L, [], [], [], [], [])
    nz = np.nonzero(a > wfloor * a.max())[0]
    kmin, kmax = K[nz[0]] - 1e-12, K[nz[-1]] + 1e-12
    cum = np.cumsum(a) / a.sum()
    body = np.unique(np.interp(np.linspace(q_tail, 1 - q_tail, nq + 1), cum, K))
    hb = max(float(np.median(np.diff(body))) if body.size > 1 else L["dK"], L["dK"])
    up = [body[-1]]
    while up[-1] < kmax:
        up.append(body[-1] + (up[-1] - body[-1] + hb) * tail_ratio)
    dn = [body[0]]
    while dn[-1] > kmin:
        dn.append(body[0] - (body[0] - dn[-1] + hb) * tail_ratio)
    edges = np.unique(np.concatenate([dn[::-1], body, up]))
    bi = np.clip(np.searchsorted(edges, K, side="right") - 1, 0, edges.size - 2)
    nb = edges.size - 1
    w = np.bincount(bi, pK, nb)
    wK = np.bincount(bi, pK * K, nb)
    wK2 = np.bincount(bi, pK * K * K, nb)
    Sj = np.stack([w] + [np.bincount(bi, L["sm"][j], nb) for j in range(1, 6)], axis=1)
    w1 = np.bincount(bi, L["e1"], nb)
    w2 = np.bincount(bi, L["e2"], nb)
    ok = np.abs(w) > wfloor
    w, wK, wK2, w1, w2, Sj = w[ok], wK[ok], wK2[ok], w1[ok], w2[ok], Sj[ok]
    ws = np.where(np.abs(w) > 1e-300, w, 1e-300)
    Kb = wK / ws
    mu = Sj / ws[:, None]
    mu[:, 0] = 1.0
    nodes, prob = _gauss3_from_moments(mu, cap=L["Scap"])
    nodes = np.clip(nodes, 0.0, L["Scap"]) * L["ms"]
    wts = prob * w[:, None]
    E1, E2 = w1 / ws, w2 / ws
    vK = np.where(w > 0, wK2 / ws - Kb**2 - (L["smooth"] * L["dK"])**2, 0.0)
    sKb = sqrt(np.maximum(vK, 0.0))
    scale = np.exp(-L["nu_core"])
    Kn = np.stack([Kb - sKb, Kb + sKb], axis=1)
    Kq = np.repeat(Kn, NSQ, axis=1)
    Sq = np.tile(nodes, (1, 2))
    wq = np.tile(wts * scale / 2, (1, 2))
    return _points_tail(L, [Kq.ravel()], [Sq.ravel()], [wq.ravel()],
                        [np.repeat(E1, 2 * NSQ)], [np.repeat(E2, 2 * NSQ)])


# ======================================================= moment quadrature ===
# Fast path (2026-10-01, sector "exact", method "moments"): no lattice, no
# radial groups.  Per fine host cell the clump law is represented by
#   bulk (kappa_c <= kappa*, the host's cut with max_r rate(kappa_c > kappa*) <=
#   eps_rare): 3-node Gauss rule in K from its exact cumulants Sum nu kappa^a
#   (a <= 5), and for each K node a 3-node rule in S around the regression
#   E[S | K] = s1 + (c11 / c2)(K - c1) with the residual variance and the
#   marginal S shape (exact mean, Var, Cov(K, S));
#   rare strong clumps (kappa_c > kappa*): first-order Poisson in log bins of
#   kappa_c, each with its own mass and shear (error O(eps_rare^2)).
# All cumulants are exact and linear in the rates, which share the ring factor:
# rate_c(r) = pref_c n_ring(r, d_c), so they are ONE product ring @ Q on a coarse
# r grid, interpolated to the fine cells.  The host is rebuilt at
# max(M - S, Mmin) at every node from interpolated (rs, ks, eps)(M).


def _std_moments_from_cumulants(k1, k2, k3, k4, k5):
    """Standardised raw moments tau_0..tau_5 of a law with cumulants k1..k5."""
    sd = sqrt(np.maximum(k2, 0.0))
    ss = np.where(sd > 0, sd, 1.0)
    g3, g4, g5 = k3 / ss**3, k4 / ss**4, k5 / ss**5
    tau = np.zeros(k1.shape + (6,))
    tau[..., 0] = 1.0
    tau[..., 2] = 1.0
    tau[..., 3] = g3
    tau[..., 4] = g4 + 3.0
    tau[..., 5] = g5 + 10.0 * g3
    return sd, tau


if _HAVE_NUMBA:
    @_numba.njit(cache=True)
    def _gauss3_tau_kernel(tau, lo, hi):
        n = tau.shape[0]
        t = np.zeros((n, 3))
        pr = np.zeros((n, 3))
        for i in range(n):
            pr[i, 0] = 1.0
            t3 = tau[i, 3]
            t4 = tau[i, 4]
            t5 = tau[i, 5]
            D = t4 - t3 * t3 - 1.0
            
            ok = False
            if abs(D) > 1e-10 and not _math.isnan(D):
                al = (t3 * t4 + t3 - t5) / D
                be = -t4 - t3 * al
                ga = -t3 - al
                
                pp = be - al * al / 3.0
                qq = 2.0 * al * al * al / 27.0 - al * be / 3.0 + ga
                
                if pp < 0.0:
                    rp = _math.sqrt(-pp / 3.0)
                    arg = 3.0 * qq / (2.0 * pp * rp)
                    if abs(arg) <= 1.0 + 1e-12:
                        if arg > 1.0: arg = 1.0
                        elif arg < -1.0: arg = -1.0
                        ph = _math.acos(arg) / 3.0
                        al3 = al / 3.0
                        
                        r0 = 2.0 * rp * _math.cos(ph) - al3
                        r1 = 2.0 * rp * _math.cos(ph - 2.0 * _math.pi / 3.0) - al3
                        r2 = 2.0 * rp * _math.cos(ph - 4.0 * _math.pi / 3.0) - al3
                        
                        if r0 > r1: r0, r1 = r1, r0
                        if r1 > r2: r1, r2 = r2, r1
                        if r0 > r1: r0, r1 = r1, r0
                        
                        if (r1 - r0 > 1e-7) and (r2 - r1 > 1e-7):
                            w0 = (1.0 + r1 * r2) / ((r0 - r1) * (r0 - r2))
                            w1 = (1.0 + r0 * r2) / ((r1 - r0) * (r1 - r2))
                            w2 = (1.0 + r0 * r1) / ((r2 - r0) * (r2 - r1))
                            # roundoff-robust acceptance (2026-10-03): see
                            # G3_TOL_T / G3_TOL_W; within tolerance, clamp
                            tt = G3_TOL_T * (1.0 + max(abs(r0), abs(r2)))
                            lo_i = lo[i] - tt
                            hi_i = hi[i] + tt
                            if (w0 > -G3_TOL_W and w1 > -G3_TOL_W and w2 > -G3_TOL_W and
                                r0 >= lo_i and r2 <= hi_i):
                                t[i, 0] = min(max(r0, lo[i]), hi[i])
                                t[i, 1] = min(max(r1, lo[i]), hi[i])
                                t[i, 2] = min(max(r2, lo[i]), hi[i])
                                pr[i, 0] = max(w0, 0.0)
                                pr[i, 1] = max(w1, 0.0)
                                pr[i, 2] = max(w2, 0.0)
                                ok = True
            if not ok:
                # 2-node rule in the 3-node LAYOUT (low, high, empty slot at
                # the high node): the coarse-r rows are interpolated column by
                # column, so every row must order its nodes the same way
                g = tau[i, 3]
                rt = _math.sqrt(g * g + 4.0)
                t1 = (g + rt) * 0.5
                t2 = (g - rt) * 0.5
                tt = G3_TOL_T * (1.0 + max(abs(t1), abs(t2)))
                if t2 >= lo[i] - tt and t1 <= hi[i] + tt:
                    t1c = min(max(t1, lo[i]), hi[i])
                    t[i, 0] = min(max(t2, lo[i]), hi[i])
                    t[i, 1] = t1c
                    t[i, 2] = t1c
                    pr[i, 0] = t1 / (t1 - t2)
                    pr[i, 1] = -t2 / (t1 - t2)
                    pr[i, 2] = 0.0
                    
        return t, pr


def _gauss3_tau(tau, lo, hi):
    """3-node Gauss rule for a standardised law (rows of tau_0..tau_5: mean 0,
    sd 1) with nodes inside [lo, hi]; else the 2-node rule (exact to the 3rd
    moment) if inside; else one node at 0.  Returns t (n, 3), prob (n, 3)."""
    if _HAVE_NUMBA:
        return _gauss3_tau_kernel(np.ascontiguousarray(tau, dtype=float),
                                  np.ascontiguousarray(lo, dtype=float),
                                  np.ascontiguousarray(hi, dtype=float))
    n = tau.shape[0]
    t = np.zeros((n, 3)); pr = np.zeros((n, 3)); pr[:, 0] = 1.0
    # monic p3(t) = t^3 + al t^2 + be t + ga orthogonal to 1, t, t^2
    # Exact closed-form for standardized Hankel 3x3 system (tau_0=1, tau_1=0, tau_2=1)
    t3, t4, t5 = tau[:, 3], tau[:, 4], tau[:, 5]
    D = t4 - t3**2 - 1.0
    good = np.isfinite(D) & (np.abs(D) > 1e-10)
    ok3 = np.zeros(n, bool)
    if good.any():
        gi = np.nonzero(good)[0]
        al = (t3[gi] * t4[gi] + t3[gi] - t5[gi]) / D[gi]
        be = -t4[gi] - t3[gi] * al
        ga = -t3[gi] - al
        # roots of t^3 + al t^2 + be t + ga in closed form (trigonometric
        # Cardano; three real roots iff the depressed discriminant allows)
        pp = be - al**2 / 3.0
        qq = 2.0 * al**3 / 27.0 - al * be / 3.0 + ga
        real = pp < 0
        rp = sqrt(np.maximum(-pp / 3.0, 1e-300))
        arg = np.clip(3.0 * qq / (2.0 * pp + np.where(real, 0.0, 1.0)) / rp * np.where(real, 1.0, 0.0), -1.0, 1.0)
        real &= np.abs(3.0 * qq / np.where(real, 2.0 * pp, 1.0) / rp) <= 1.0 + 1e-12
        ph = np.arccos(arg) / 3.0
        tt = np.sort(np.stack([2 * rp * np.cos(ph - 2 * np.pi * j / 3) for j in range(3)], axis=1)
                     - (al / 3.0)[:, None], axis=1)
        dist = np.min(np.diff(tt, axis=1), axis=1) > 1e-7
        sel = real & dist
        pw = np.full((gi.size, 3), -1.0)
        if sel.any():
            tt_sel = tt[sel]
            t0, t1, t2 = tt_sel[:, 0], tt_sel[:, 1], tt_sel[:, 2]
            w0 = (1.0 + t1 * t2) / ((t0 - t1) * (t0 - t2))
            w1 = (1.0 + t0 * t2) / ((t1 - t0) * (t1 - t2))
            w2 = (1.0 + t0 * t1) / ((t2 - t0) * (t2 - t1))
            pw[sel] = np.column_stack([w0, w1, w2])
        tol = G3_TOL_T * (1.0 + np.max(np.abs(tt), axis=1))
        fine = (sel & np.all(pw > -G3_TOL_W, axis=1) & np.all(tt >= lo[gi, None] - tol[:, None], axis=1)
                & np.all(tt <= hi[gi, None] + tol[:, None], axis=1))
        t[gi[fine]] = np.clip(tt[fine], lo[gi[fine], None], hi[gi[fine], None])
        pr[gi[fine]] = np.maximum(pw[fine], 0.0)
        ok3[gi[fine]] = True
    r2 = ~ok3
    if r2.any():
        # 2-node rule in the 3-node layout (low, high, empty slot at high)
        g = tau[r2, 3]
        rt = sqrt(g**2 + 4.0)
        t1, t2 = (g + rt) / 2, (g - rt) / 2
        tol = G3_TOL_T * (1.0 + np.maximum(np.abs(t1), np.abs(t2)))
        ok2 = (t2 >= lo[r2] - tol) & (t1 <= hi[r2] + tol)
        idx = np.nonzero(r2)[0][ok2]
        t1c = np.clip(t1[ok2], lo[idx], hi[idx])
        t[idx, 0], t[idx, 1], t[idx, 2] = np.clip(t2[ok2], lo[idx], hi[idx]), t1c, t1c
        pr[idx, 0], pr[idx, 1], pr[idx, 2] = t1[ok2] / (t1[ok2] - t2[ok2]), -t2[ok2] / (t1[ok2] - t2[ok2]), 0.0
        GAUSS_FALLBACK[0] += int(r2.sum())
    return t, pr


def _kg_unit(x, xt=None):
    """kappa, gamma of an NFW with ks = 1 at x (both are linear in ks);
    xt (array, same shape as x): truncated at x_t (F.nfw_trunc_unit)."""
    if xt is None:
        return sgl.kappa_gamma(np.ascontiguousarray(np.ravel(x), dtype=float), 1.0)
    return F.nfw_trunc_unit(np.ravel(x), np.ravel(xt))


def _kg_eps_vec(eps, ks, x, phi, xt=None):
    """F.kappagamma_eps with per-point eps and ks (same formulas); xt as in
    F.kappagamma_eps (kappa = 0, circular shear beyond the truncation)."""
    x1 = sqrt(1 - eps) * np.cos(phi) * x
    x2 = sqrt(1 + eps) * np.sin(phi) * x
    xe = np.maximum(sqrt(x1**2 + x2**2), 1e-12)
    c2 = (x1**2 - x2**2) / np.maximum(xe**2, 1e-300)
    k1, g1 = _kg_unit(xe, None if xt is None else np.broadcast_to(xt, xe.shape))
    k0, g0 = k1.reshape(xe.shape) * ks, g1.reshape(xe.shape) * ks
    g0 = np.where(xe < 1e-4, ks, g0)
    k = k0 + eps * c2 * g0
    g2 = g0**2 + 2 * eps * c2 * g0 * k0 + eps**2 * (k0**2 - (c2 * g0)**2)
    if xt is not None:          # outside: monopole shear of the virial mass at x
        out = xe >= xt
        gpm = ks * 4 * (np.log1p(xt) - xt / (1 + xt)) / np.maximum(x, 1e-300)**2
        k = np.where(out, 0.0, k)
        g2 = np.where(out, gpm**2, g2)
    return k, sqrt(np.maximum(g2, 0.0))


if _HAVE_NUMBA:
    @_numba.njit(cache=True, parallel=True)
    def _emit(cg, Wc, dKc, dSc, tk, tg, St, goff, Kp, Sp, wp, e1p, e2p, coff, Ko, Go, Wo):
        """For every host cell f (group cg[f]) and every point of its group:
        kappa = kappa_h(M - S) + K, |gamma| = sqrt(gh^2 + 2 gh E1 + E2), from the
        cell's host table (tk, tg on St, linear in S).  dKc, dSc: the cell's
        exact mean offsets from its group's law (E[K|r_f] - E[K|group], same
        for S), so every cell's first moments are exact whatever the grouping."""
        nt = St.shape[0]
        dS = St[1] - St[0]
        for f in _numba.prange(cg.shape[0]):
            g = cg[f]
            o = coff[f]
            for q in range(goff[g], goff[g + 1]):
                s = Sp[q] + dSc[f]
                if s < 0.0:
                    s = 0.0
                x = s / dS
                j = int(x)
                if j > nt - 2:
                    j = nt - 2
                t = x - j
                if t > 1.0:
                    t = 1.0
                kh = tk[f, j] + t * (tk[f, j + 1] - tk[f, j])
                gh = tg[f, j] + t * (tg[f, j + 1] - tg[f, j])
                Ko[o] = kh + Kp[q] + dKc[f]
                g2 = gh * gh + 2.0 * gh * e1p[q] + e2p[q]
                Go[o] = np.sqrt(g2) if g2 > 0.0 else 0.0
                Wo[o] = Wc[f] * wp[q]
                o += 1

    @_numba.njit(cache=True, inline="always")
    def _kg_unit_scalar(xi):
        # 2026-10-03: shared precision-safe kernel (sgl.nfw_Fh); the inline
        # copy of the old formulas lost up to 8e-6 near x = 1
        F_val, h = sgl.nfw_Fh(xi)
        k = 2.0 * F_val
        kbar = 4.0 * h / (xi * xi)
        return k, kbar - k

    @_numba.njit(cache=True, inline="always")
    def _tc_scalar(x, xt, gx, gw):
        # 2026-10-03: closed-form truncated NFW (sgl_full._tc_point); was a
        # 24-node Gauss-Legendre loop, off by up to 0.8 in kappa near x_t
        return F._tc_point(x, xt, gx, gw)

    @_numba.njit(cache=True, parallel=True)
    def _moments_fused_general(xq, phq, W, rs, xt, Kp_c, Sp_c, wp_c, A1_c, A2_c,
                               xg, lxg, Mt, lMt, log_rs, log_ks, ept, M, Mmin, gx, gw):
        n_cell = xq.shape[0]
        nq = Sp_c.shape[1]
        n_tot = n_cell * nq
        n_g = lxg.shape[0]
        m_t = lMt.shape[0]
        
        Ko = np.empty(n_tot)
        Go = np.empty(n_tot)
        Wo = np.empty(n_tot)
        
        for i in _numba.prange(n_cell):
            x_val = xq[i]
            phi_val = phq[i]
            w_cell = W[i]
            
            lx_val = _math.log(x_val) if x_val > xg[0] else lxg[0]
            if lx_val > lxg[n_g - 1]:
                lx_val = lxg[n_g - 1]
                
            lo_g = 0
            hi_g = n_g - 1
            while lo_g < hi_g - 1:
                mid = (lo_g + hi_g) >> 1
                if lxg[mid] <= lx_val:
                    lo_g = mid
                else:
                    hi_g = mid
            jj = lo_g
            tw = (lx_val - lxg[jj]) / (lxg[jj + 1] - lxg[jj])
            om_tw = 1.0 - tw
            
            c_phi = _math.cos(phi_val)
            s_phi = _math.sin(phi_val)
            
            idx_base = i * nq
            for q in range(nq):
                idx = idx_base + q
                kp = om_tw * Kp_c[jj, q] + tw * Kp_c[jj + 1, q]
                sp = om_tw * Sp_c[jj, q] + tw * Sp_c[jj + 1, q]
                wp = om_tw * wp_c[jj, q] + tw * wp_c[jj + 1, q]
                a1 = om_tw * A1_c[jj, q] + tw * A1_c[jj + 1, q]
                a2 = om_tw * A2_c[jj, q] + tw * A2_c[jj + 1, q]
                
                Me = M - sp
                if Me < Mmin:
                    Me = Mmin
                lMe = _math.log(Me)
                
                if lMe <= lMt[0]:
                    rse = _math.exp(log_rs[0])
                    kse = _math.exp(log_ks[0])
                    epe = ept[0]
                elif lMe >= lMt[m_t - 1]:
                    rse = _math.exp(log_rs[m_t - 1])
                    kse = _math.exp(log_ks[m_t - 1])
                    epe = ept[m_t - 1]
                else:
                    lo_t = 0
                    hi_t = m_t - 1
                    while lo_t < hi_t - 1:
                        mid = (lo_t + hi_t) >> 1
                        if lMt[mid] <= lMe:
                            lo_t = mid
                        else:
                            hi_t = mid
                    w_t = (lMe - lMt[lo_t]) / (lMt[lo_t + 1] - lMt[lo_t])
                    rse = _math.exp(log_rs[lo_t] + w_t * (log_rs[lo_t + 1] - log_rs[lo_t]))
                    kse = _math.exp(log_ks[lo_t] + w_t * (log_ks[lo_t + 1] - log_ks[lo_t]))
                    epe = ept[lo_t] + w_t * (ept[lo_t + 1] - ept[lo_t])
                    
                xe = (x_val * rs) / rse
                xte = (xt * rs) / rse
                
                x1 = _math.sqrt(max(1.0 - epe, 0.0)) * c_phi * xe
                x2 = _math.sqrt(1.0 + epe) * s_phi * xe
                xe2 = x1 * x1 + x2 * x2
                x_ell = _math.sqrt(xe2)
                if x_ell < 1e-12:
                    x_ell = 1e-12
                c2 = (x1 * x1 - x2 * x2) / (xe2 if xe2 > 1e-300 else 1e-300)
                
                ku, gu = _kg_unit_scalar(x_ell)
                I1, I2 = _tc_scalar(x_ell, xte, gx, gw)
                
                inside = x_ell < xte
                mut = _math.log(1.0 + xte) - xte / (1.0 + xte)
                denom = x_ell * x_ell if x_ell > 1e-150 else 1e-300
                
                if inside:
                    k1 = ku - 2.0 * I1
                    kb = ku + gu - 4.0 * I2
                else:
                    k1 = 0.0
                    kb = 4.0 * mut / denom
                g1 = kb - k1
                
                ks = kse
                k0 = k1 * ks
                g0 = ks if x_ell < 1e-4 else g1 * ks
                
                cg = c2 * g0
                k = k0 + epe * cg
                g2 = g0 * g0 + 2.0 * epe * cg * k0 + epe * epe * (k0 * k0 - cg * cg)
                
                if not inside:
                    den_orig = xe * xe if xe > 1e-150 else 1e-300
                    gpm = ks * 4.0 * mut / den_orig
                    kh = 0.0
                    gh2 = gpm * gpm
                else:
                    kh = k
                    gh2 = g2
                    
                gh = _math.sqrt(gh2) if gh2 > 0.0 else 0.0
                Ko[idx] = kh + kp
                
                g2_tot = gh * gh + 2.0 * gh * a1 + a2
                Go[idx] = _math.sqrt(g2_tot) if g2_tot > 0.0 else 0.0
                Wo[idx] = w_cell * wp
                
        return Ko, Go, Wo

    @_numba.njit(cache=True, parallel=True)
    def _moments_fused_indexed(xu, inv, phq, W, rs, xt, Kp_c, Sp_c, wp_c, A1_c, A2_c,
                               xg, lxg, Mt, lMt, log_rs, log_ks, ept, M, Mmin, gx, gw):
        """_moments_fused_general with the (radius, node) work done once per
        UNIQUE radius (2026-10-03): cells (xu[inv[i]], phq[i]).  The per-phi
        refined grids share most radii, so the interpolation of the coarse
        moment tables and the rebuilt-host lookups (log, 2 binary searches,
        3 exp per point) are no longer repeated for every phi.  Same
        arithmetic in the same order as _moments_fused_general, per point:
        bit-identical output."""
        nr = xu.shape[0]
        n_cell = inv.shape[0]
        nq = Sp_c.shape[1]
        n_tot = n_cell * nq

        # 1. table over the UNIQUE radii xu
        xe_r = np.empty((nr, nq))
        xte_r = np.empty((nr, nq))
        kse_r = np.empty((nr, nq))
        epe_r = np.empty((nr, nq))
        Kp_r = np.empty((nr, nq))
        wp_r = np.empty((nr, nq))
        A1_r = np.empty((nr, nq))
        A2_r = np.empty((nr, nq))
        
        n_g = lxg.shape[0]
        m_t = lMt.shape[0]
        
        for i in _numba.prange(nr):
            x_val = xu[i]
            lx_val = _math.log(x_val) if x_val > xg[0] else lxg[0]
            if lx_val > lxg[n_g - 1]:
                lx_val = lxg[n_g - 1]
            lo_g = 0
            hi_g = n_g - 1
            while lo_g < hi_g - 1:
                mid = (lo_g + hi_g) >> 1
                if lxg[mid] <= lx_val:
                    lo_g = mid
                else:
                    hi_g = mid
            jj = lo_g
            tw = (lx_val - lxg[jj]) / (lxg[jj + 1] - lxg[jj])
            om_tw = 1.0 - tw
            
            for q in range(nq):
                kp = om_tw * Kp_c[jj, q] + tw * Kp_c[jj + 1, q]
                sp = om_tw * Sp_c[jj, q] + tw * Sp_c[jj + 1, q]
                wp = om_tw * wp_c[jj, q] + tw * wp_c[jj + 1, q]
                a1 = om_tw * A1_c[jj, q] + tw * A1_c[jj + 1, q]
                a2 = om_tw * A2_c[jj, q] + tw * A2_c[jj + 1, q]
                
                Kp_r[i, q] = kp
                wp_r[i, q] = wp
                A1_r[i, q] = a1
                A2_r[i, q] = a2
                
                Me = M - sp
                if Me < Mmin:
                    Me = Mmin
                lMe = _math.log(Me)
                if lMe <= lMt[0]:
                    rse = _math.exp(log_rs[0])
                    kse = _math.exp(log_ks[0])
                    epe = ept[0]
                elif lMe >= lMt[m_t - 1]:
                    rse = _math.exp(log_rs[m_t - 1])
                    kse = _math.exp(log_ks[m_t - 1])
                    epe = ept[m_t - 1]
                else:
                    lo_t = 0
                    hi_t = m_t - 1
                    while lo_t < hi_t - 1:
                        mid = (lo_t + hi_t) >> 1
                        if lMt[mid] <= lMe:
                            lo_t = mid
                        else:
                            hi_t = mid
                    w_t = (lMe - lMt[lo_t]) / (lMt[lo_t + 1] - lMt[lo_t])
                    rse = _math.exp(log_rs[lo_t] + w_t * (log_rs[lo_t + 1] - log_rs[lo_t]))
                    kse = _math.exp(log_ks[lo_t] + w_t * (log_ks[lo_t + 1] - log_ks[lo_t]))
                    epe = ept[lo_t] + w_t * (ept[lo_t + 1] - ept[lo_t])
                kse_r[i, q] = kse
                epe_r[i, q] = epe
                xe_r[i, q] = (x_val * rs) / rse
                xte_r[i, q] = (xt * rs) / rse

        Ko = np.empty(n_tot)
        Go = np.empty(n_tot)
        Wo = np.empty(n_tot)

        # 2. cells: only the phi-dependent part
        for i in _numba.prange(n_cell):
            j = inv[i]
            phi_val = phq[i]
            w_cell = W[i]
            c_phi = _math.cos(phi_val)
            s_phi = _math.sin(phi_val)
            idx_base = i * nq
            for q in range(nq):
                idx = idx_base + q
                kp = Kp_r[j, q]
                wp = wp_r[j, q]
                a1 = A1_r[j, q]
                a2 = A2_r[j, q]
                xe = xe_r[j, q]
                xte = xte_r[j, q]
                epe = epe_r[j, q]
                kse = kse_r[j, q]

                x1 = _math.sqrt(max(1.0 - epe, 0.0)) * c_phi * xe
                x2 = _math.sqrt(1.0 + epe) * s_phi * xe
                xe2 = x1 * x1 + x2 * x2
                x_ell = _math.sqrt(xe2)
                if x_ell < 1e-12:
                    x_ell = 1e-12
                c2 = (x1 * x1 - x2 * x2) / (xe2 if xe2 > 1e-300 else 1e-300)

                ku, gu = _kg_unit_scalar(x_ell)
                I1, I2 = _tc_scalar(x_ell, xte, gx, gw)

                inside = x_ell < xte
                mut = _math.log(1.0 + xte) - xte / (1.0 + xte)
                denom = x_ell * x_ell if x_ell > 1e-150 else 1e-300

                if inside:
                    k1 = ku - 2.0 * I1
                    kb = ku + gu - 4.0 * I2
                else:
                    k1 = 0.0
                    kb = 4.0 * mut / denom
                g1 = kb - k1

                ks = kse
                k0 = k1 * ks
                g0 = ks if x_ell < 1e-4 else g1 * ks

                cg = c2 * g0
                k = k0 + epe * cg
                g2 = g0 * g0 + 2.0 * epe * cg * k0 + epe * epe * (k0 * k0 - cg * cg)

                if not inside:
                    den_orig = xe * xe if xe > 1e-150 else 1e-300
                    gpm = ks * 4.0 * mut / den_orig
                    kh = 0.0
                    gh2 = gpm * gpm
                else:
                    kh = k
                    gh2 = g2

                gh = _math.sqrt(gh2) if gh2 > 0.0 else 0.0
                Ko[idx] = kh + kp

                g2_tot = gh * gh + 2.0 * gh * a1 + a2
                Go[idx] = _math.sqrt(g2_tot) if g2_tot > 0.0 else 0.0
                Wo[idx] = w_cell * wp

        return Ko, Go, Wo

    @_numba.njit(cache=True, parallel=True)
    def _moments_fused_uniform(xc0, W0, phis, rs, xt, Kp_c, Sp_c, wp_c, A1_c, A2_c,
                               xg, lxg, Mt, lMt, log_rs, log_ks, ept, M, Mmin, gx, gw):
        nr = xc0.shape[0]
        nq = Sp_c.shape[1]
        nph = phis.shape[0]
        n_tot = nph * nr * nq
        
        # 1. Precalculate radial table on xc0
        xe_r = np.empty((nr, nq))
        xte_r = np.empty((nr, nq))
        kse_r = np.empty((nr, nq))
        epe_r = np.empty((nr, nq))
        Kp_r = np.empty((nr, nq))
        wp_r = np.empty((nr, nq))
        A1_r = np.empty((nr, nq))
        A2_r = np.empty((nr, nq))
        
        n_g = lxg.shape[0]
        m_t = lMt.shape[0]
        
        for i in range(nr):
            x_val = xc0[i]
            lx_val = _math.log(x_val) if x_val > xg[0] else lxg[0]
            if lx_val > lxg[n_g - 1]:
                lx_val = lxg[n_g - 1]
            lo_g = 0
            hi_g = n_g - 1
            while lo_g < hi_g - 1:
                mid = (lo_g + hi_g) >> 1
                if lxg[mid] <= lx_val:
                    lo_g = mid
                else:
                    hi_g = mid
            jj = lo_g
            tw = (lx_val - lxg[jj]) / (lxg[jj + 1] - lxg[jj])
            om_tw = 1.0 - tw
            
            for q in range(nq):
                kp = om_tw * Kp_c[jj, q] + tw * Kp_c[jj + 1, q]
                sp = om_tw * Sp_c[jj, q] + tw * Sp_c[jj + 1, q]
                wp = om_tw * wp_c[jj, q] + tw * wp_c[jj + 1, q]
                a1 = om_tw * A1_c[jj, q] + tw * A1_c[jj + 1, q]
                a2 = om_tw * A2_c[jj, q] + tw * A2_c[jj + 1, q]
                
                Kp_r[i, q] = kp
                wp_r[i, q] = wp
                A1_r[i, q] = a1
                A2_r[i, q] = a2
                
                Me = M - sp
                if Me < Mmin:
                    Me = Mmin
                lMe = _math.log(Me)
                if lMe <= lMt[0]:
                    rse = _math.exp(log_rs[0])
                    kse = _math.exp(log_ks[0])
                    epe = ept[0]
                elif lMe >= lMt[m_t - 1]:
                    rse = _math.exp(log_rs[m_t - 1])
                    kse = _math.exp(log_ks[m_t - 1])
                    epe = ept[m_t - 1]
                else:
                    lo_t = 0
                    hi_t = m_t - 1
                    while lo_t < hi_t - 1:
                        mid = (lo_t + hi_t) >> 1
                        if lMt[mid] <= lMe:
                            lo_t = mid
                        else:
                            hi_t = mid
                    w_t = (lMe - lMt[lo_t]) / (lMt[lo_t + 1] - lMt[lo_t])
                    rse = _math.exp(log_rs[lo_t] + w_t * (log_rs[lo_t + 1] - log_rs[lo_t]))
                    kse = _math.exp(log_ks[lo_t] + w_t * (log_ks[lo_t + 1] - log_ks[lo_t]))
                    epe = ept[lo_t] + w_t * (ept[lo_t + 1] - ept[lo_t])
                kse_r[i, q] = kse
                epe_r[i, q] = epe
                xe_r[i, q] = (x_val * rs) / rse
                xte_r[i, q] = (xt * rs) / rse

        Ko = np.empty(n_tot)
        Go = np.empty(n_tot)
        Wo = np.empty(n_tot)
        
        # 2. Parallel loop over phi and r
        for p in _numba.prange(nph):
            phi_val = phis[p]
            c_phi = _math.cos(phi_val)
            s_phi = _math.sin(phi_val)
            base_out = p * nr * nq
            for i in range(nr):
                w_cell = W0[i]
                r_out = base_out + i * nq
                for q in range(nq):
                    idx = r_out + q
                    xe = xe_r[i, q]
                    xte = xte_r[i, q]
                    epe = epe_r[i, q]
                    
                    x1 = _math.sqrt(max(1.0 - epe, 0.0)) * c_phi * xe
                    x2 = _math.sqrt(1.0 + epe) * s_phi * xe
                    xe2 = x1 * x1 + x2 * x2
                    x_ell = _math.sqrt(xe2)
                    if x_ell < 1e-12:
                        x_ell = 1e-12
                    c2 = (x1 * x1 - x2 * x2) / (xe2 if xe2 > 1e-300 else 1e-300)
                    
                    ku, gu = _kg_unit_scalar(x_ell)
                    I1, I2 = _tc_scalar(x_ell, xte, gx, gw)
                    
                    inside = x_ell < xte
                    mut = _math.log(1.0 + xte) - xte / (1.0 + xte)
                    denom = x_ell * x_ell if x_ell > 1e-150 else 1e-300
                    if inside:
                        k1 = ku - 2.0 * I1
                        kb = ku + gu - 4.0 * I2
                    else:
                        k1 = 0.0
                        kb = 4.0 * mut / denom
                    g1 = kb - k1
                    
                    ks = kse_r[i, q]
                    k0 = k1 * ks
                    g0 = ks if x_ell < 1e-4 else g1 * ks
                    cg = c2 * g0
                    k = k0 + epe * cg
                    g2 = g0 * g0 + 2.0 * epe * cg * k0 + epe * epe * (k0 * k0 - cg * cg)
                    
                    if not inside:
                        den_orig = xe * xe if xe > 1e-150 else 1e-300
                        gpm = ks * 4.0 * mut / den_orig
                        kh = 0.0
                        gh2 = gpm * gpm
                    else:
                        kh = k
                        gh2 = g2
                    gh = _math.sqrt(gh2) if gh2 > 0.0 else 0.0
                    Ko[idx] = kh + Kp_r[i, q]
                    
                    A1_val = A1_r[i, q]
                    A2_val = A2_r[i, q]
                    g2_tot = gh * gh + 2.0 * gh * A1_val + A2_val
                    Go[idx] = _math.sqrt(g2_tot) if g2_tot > 0.0 else 0.0
                    Wo[idx] = w_cell * wp_r[i, q]
                    
        return Ko, Go, Wo


    @_numba.njit(cache=True)
    def _clump_cells_kernel(dc, dA, D, rsc, ksc, xtc, m, dN, vr, gx, gw):
        nm = D.shape[0]
        nd = dc.shape[0]
        has_xtc = xtc is not None and xtc.shape[0] > 0
        
        tot = 0
        for i in range(nm):
            di = D[i]
            for d in range(nd):
                if dc[d] < di:
                    tot += 1
                else:
                    break
                    
        if tot == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
            
        I_arr = np.empty(tot, dtype=np.int64)
        Dd_arr = np.empty(tot, dtype=np.int64)
        Kc_arr = np.empty(tot)
        Gc_arr = np.empty(tot)
        m_arr = np.empty(tot)
        pref_arr = np.empty(tot)
        
        o = 0
        for i in range(nm):
            di = D[i]
            rsi = rsc[i]
            ksi = ksc[i]
            mi = m[i] / vr
            dNi = dN[i]
            
            if has_xtc:
                xti = xtc[i]
                mut = _math.log(1.0 + xti) - xti / (1.0 + xti)
                for d in range(nd):
                    dcd = dc[d]
                    if dcd >= di:
                        break
                    x = dcd / rsi
                    ku, gu = _kg_unit_scalar(x)
                    I1, I2 = _tc_scalar(x, xti, gx, gw)
                    inside = x < xti
                    k = ku - 2.0 * I1 if inside else 0.0
                    denom = x * x if x > 1e-150 else 1e-300
                    kb = ku + gu - 4.0 * I2 if inside else 4.0 * mut / denom
                    
                    I_arr[o] = i
                    Dd_arr[o] = d
                    Kc_arr[o] = ksi * k
                    Gc_arr[o] = ksi * (kb - k)
                    m_arr[o] = mi
                    pref_arr[o] = dNi * dA[d]
                    o += 1
            else:
                for d in range(nd):
                    dcd = dc[d]
                    if dcd >= di:
                        break
                    x = dcd / rsi
                    ku, gu = _kg_unit_scalar(x)
                    I_arr[o] = i
                    Dd_arr[o] = d
                    Kc_arr[o] = ksi * ku
                    Gc_arr[o] = ksi * gu
                    m_arr[o] = mi
                    pref_arr[o] = dNi * dA[d]
                    o += 1
                
        return I_arr, Dd_arr, Kc_arr, Gc_arr, m_arr, pref_arr

    @_numba.njit(cache=True)
    def _bulk_moments_accum(d, bulk, pref, k, mh, g, nd):
        Q = np.zeros((nd, 16))
        nc = d.shape[0]
        for c in range(nc):
            if not bulk[c]:
                continue
            di = d[c]
            p = pref[c]
            kc = k[c]
            mhc = mh[c]
            gc = g[c]
            
            pk = p * kc
            pk2 = pk * kc
            pk3 = pk2 * kc
            pk4 = pk3 * kc
            pk5 = pk4 * kc
            
            pm = p * mhc
            pm2 = pm * mhc
            pm3 = pm2 * mhc
            pm4 = pm3 * mhc
            pm5 = pm4 * mhc
            
            Q[di, 0] += pk
            Q[di, 1] += pk2
            Q[di, 2] += pk3
            Q[di, 3] += pk4
            Q[di, 4] += pk5
            
            Q[di, 5] += pm
            Q[di, 6] += pm2
            Q[di, 7] += pm3
            Q[di, 8] += pm4
            Q[di, 9] += pm5
            
            Q[di, 10] += pk * mhc
            Q[di, 11] += p * (gc * gc)
            Q[di, 12] += pk * (gc * gc)
            Q[di, 13] += p * gc
            Q[di, 14] += pk * gc
            Q[di, 15] += pm * gc
            
        return Q

    @_numba.njit(cache=True)
    def _rare_bins_accum(ring, ecd, dr, bi, p, kr, mr, gr, nbr):
        ng, nd = ring.shape
        nc = dr.shape[0]
        out = np.zeros((6, ng, nbr))
        
        p_kr = p * kr
        p_mr = p * mr
        p_gr2 = p * (gr * gr)
        p_kr2 = p_kr * kr
        p_gr = p * gr
        
        for c in range(nc):
            d_idx = dr[c]
            b_idx = bi[c]
            v0 = p[c]
            v1 = p_kr[c]
            v2 = p_mr[c]
            v3 = p_gr2[c]
            v4 = p_kr2[c]
            v5 = p_gr[c]
            
            for i in range(ng):
                r_val = ring[i, d_idx]
                re_val = r_val * ecd[i, d_idx]
                out[0, i, b_idx] += r_val * v0
                out[1, i, b_idx] += r_val * v1
                out[2, i, b_idx] += r_val * v2
                out[3, i, b_idx] += r_val * v3
                out[4, i, b_idx] += r_val * v4
                out[5, i, b_idx] += re_val * v5
                
        return out

    @_numba.njit(cache=True)
    def _rare_bins_var(ring, dr, bi, p, kr, kbg):
        """Within-bin kappa variance numerator sum ring p (kappa - kbar_bin)^2 per
        (coarse row, rare bin), with kbar_bin = kbg (2nd pass): non-negative term
        by term.  E[k^2] - E[k]^2 cancelled to sqrt(eps) kappa noise in the sd --
        a one-cell bin (sd exactly 0) came out 1e-10 or 0 by roundoff."""
        ng = ring.shape[0]
        nbr = kbg.shape[1]
        out = np.zeros((ng, nbr))
        for c in range(dr.shape[0]):
            d_idx = dr[c]
            b_idx = bi[c]
            pc = p[c]
            kc = kr[c]
            for i in range(ng):
                dk = kc - kbg[i, b_idx]
                out[i, b_idx] += ring[i, d_idx] * pc * dk * dk
        return out


def _det_KS_numpy(ring, d, ib, pref, k, mbin, nbin):
    nd = ring.shape[1]
    Pn = np.zeros((nd, nbin)); np.add.at(Pn, (d, ib), pref)
    Pa = np.zeros((nd, nbin)); np.add.at(Pa, (d, ib), pref * k)
    n_, a_ = ring @ Pn, ring @ Pa
    ab = a_ / np.where(n_ > 0, n_, 1.0)
    W = (ring[:, d] * pref[None, :] * (k[None, :] - ab[:, ib])**2).sum(axis=1)
    X = mbin[None, None, :] * ab[:, :, None] - mbin[None, :, None] * ab[:, None, :]
    cross = 0.5 * np.einsum("ri,rj,rij->r", n_, n_, X**2)
    return cross, W


if _HAVE_NUMBA:
    @_numba.njit(cache=True)
    def _det_KS(ring, d, ib, pref, k, mbin, nbin):
        """Cancellation-free det = c2 sS2 - c11^2 of the bulk (K, S) law per
        coarse row (2026-10-03).  The clump mass depends on the mass bin i only,
        so with per-bin n_i = sum nu, a_i = sum nu kappa, abar_i = a_i / n_i,
            det = 1/2 sum_ij n_i n_j (m_j abar_i - m_i abar_j)^2 + sS2 sum_i W_i,
            W_i = sum_{c in i} nu_c (kappa_c - abar_i)^2   (two-pass),
        every term >= 0.  Returns (cross, sum_i W_i); det = cross + sS2 * W.
        The direct difference is pure noise when K and S are collinear (one
        clump cell in reach at the outer rows), and a 2-ulp NFW change flipped
        the shear regression there."""
        ng = ring.shape[0]
        nc = d.shape[0]
        n_ = np.zeros((ng, nbin))
        a_ = np.zeros((ng, nbin))
        for c in range(nc):
            for r in range(ng):
                v = ring[r, d[c]] * pref[c]
                n_[r, ib[c]] += v
                a_[r, ib[c]] += v * k[c]
        ab = np.zeros((ng, nbin))
        for r in range(ng):
            for i in range(nbin):
                if n_[r, i] > 0.0:
                    ab[r, i] = a_[r, i] / n_[r, i]
        W = np.zeros(ng)
        for c in range(nc):
            for r in range(ng):
                dk = k[c] - ab[r, ib[c]]
                W[r] += ring[r, d[c]] * pref[c] * dk * dk
        cross = np.zeros(ng)
        for r in range(ng):
            acc = 0.0
            for i in range(nbin):
                if n_[r, i] <= 0.0:
                    continue
                for j in range(i + 1, nbin):
                    if n_[r, j] <= 0.0:
                        continue
                    x = mbin[j] * ab[r, i] - mbin[i] * ab[r, j]
                    acc += n_[r, i] * n_[r, j] * x * x
            cross[r] = acc
        return cross, W


class ExactSector:
    """Exact decorated-host cells for one SubhaloModel (shares its tables)."""

    def __init__(self, sub: "subhalos.SubhaloModel", group_tol=0.1, resK=10.0, resS=8.0,
                 eps_core=1e-4, nq=48, ntheta=32, nS_tab=64, lattice="1d",
                 method="lattice", eps_rare=3e-3, nbr=3, nr_coarse=128, ntheta_m=16, nM_tab=17):
        if lattice not in ("1d", "2d"):
            raise ValueError(f"lattice must be '1d' or '2d', not {lattice!r}")
        if method not in ("lattice", "moments"):
            raise ValueError(f"method must be 'lattice' or 'moments', not {method!r}")
        self.lattice, self.method = lattice, method
        self.eps_rare, self.nbr, self.nr_coarse = eps_rare, nbr, nr_coarse
        self.ntheta_m, self.nM_tab = ntheta_m, nM_tab
        self.th_m = (np.arange(self.ntheta_m) + 0.5) * np.pi / self.ntheta_m
        self.cth_m = np.cos(self.th_m)
        self.sub = sub
        self.group_tol, self.resK, self.resS = group_tol, resK, resS
        self.eps_core, self.nq, self.ntheta, self.nS_tab = eps_core, nq, ntheta, nS_tab
        self.Mmin = F.engine_grids()[1][0]
        self.stats = dict(hosts=0, groups=0, points=0, lattice=0)

    # ----------------------------------------------------------------------
    def _clump_cells(self, h):
        if "_clump_cells" in h:
            return h["_clump_cells"]
        de, dc, dA = self.sub._dgrid(h)
        if _HAVE_NUMBA:
            gx, gw = np.array(F._GLX, dtype=float), np.array(F._GLW, dtype=float)
            xtc = np.ascontiguousarray(h["xtc"], dtype=float) if h.get("xtc") is not None else np.zeros(0, dtype=float)
            I, Dd, Kc, Gc, mc, pref = _clump_cells_kernel(
                np.ascontiguousarray(dc, dtype=float),
                np.ascontiguousarray(dA, dtype=float),
                np.ascontiguousarray(h["D"], dtype=float),
                np.ascontiguousarray(h["rsc"], dtype=float),
                np.ascontiguousarray(h["ksc"], dtype=float),
                xtc,
                np.ascontiguousarray(h["m"], dtype=float),
                np.ascontiguousarray(h["dN"], dtype=float),
                float(h["vr"]), gx, gw)
            if not I.size:
                h["_clump_cells"] = None
                return None
            res = dict(i=I, d=Dd, k=Kc, g=Gc, m=mc, pref=pref, dc=dc)
            h["_clump_cells"] = res
            return res
        I, Dd, Kc, Gc = [], [], [], []
        for i in range(len(h["m"])):
            sel = np.nonzero(dc < h["D"][i])[0]
            if not sel.size:
                continue
            if h.get("xtc") is None:
                k_, g_ = sgl.kappa_gamma(np.ascontiguousarray(dc[sel] / h["rsc"][i]), h["ksc"][i])
            else:   # clump truncated at its own r_vir (clump_edge="rvir")
                k_, g_ = F.nfw_trunc_kg(dc[sel] / h["rsc"][i], h["ksc"][i], h["xtc"][i])
            I.append(np.full(sel.size, i)); Dd.append(sel); Kc.append(k_); Gc.append(g_)
        if not I:
            h["_clump_cells"] = None
            return None
        I, Dd = np.concatenate(I), np.concatenate(Dd)
        res = dict(i=I, d=Dd, k=np.concatenate(Kc), g=np.concatenate(Gc),
                    m=h["m"][I] / h["vr"], pref=h["dN"][I] * dA[Dd], dc=dc)
        h["_clump_cells"] = res
        return res

    def _groups(self, ring_d, W):
        """Adaptive radial groups on the reference cells (sorted in r)."""
        s = ring_d                                                   # (nr, nsum) summaries
        groups, a = [], 0
        n = s.shape[0]
        for j in range(1, n + 1):
            if j == n:
                groups.append((a, j))
                break
            ref = np.maximum(np.abs(s[a]), 1e-300)
            if np.any(np.abs(s[j] - s[a]) > self.group_tol * ref):
                groups.append((a, j))
                a = j
        return groups

    def host_cells(self, jz, jM, xref, Wref, xcell, Wcell, phicell=None, eps=None, xt=None):
        """xt (2026-10-02): host truncated at x_t = r_t / r_s (fixed physical
        edge r_t, kept for the rebuilt host); moments method only."""
        if self.method == "moments":
            return self.host_cells_moments(jz, jM, xref, Wref, xcell, Wcell, phicell, xt)
        if xt is not None:
            raise NotImplementedError("truncated hosts need method='moments'")
        return self.host_cells_lattice(jz, jM, xref, Wref, xcell, Wcell, phicell)

    # ------------------------------------------------------------------
    def host_cells_moments(self, jz, jM, xref, Wref, xcell, Wcell, phicell=None, xt=None):
        """Moment-quadrature decorated host (see the section comment)."""
        sub, h = self.sub, self.sub.host(jz, jM)
        cc = self._clump_cells(h)
        if cc is None:
            return None
        cm = h.get("_coarse_moments")
        if cm is None:
            rs, M = h["rs"], h["M"]
            xpos = xref[xref > 0]
            xg = np.geomspace(xpos.min(), xref.max(), self.nr_coarse)
            lxg = np.log(xg)
            if _HAVE_NUMBA:
                th_nth = (np.arange(sub.nth) + 0.5) * np.pi / sub.nth
                cth = np.cos(th_nth)
                ring, ecd = subhalos._ring_and_pair_angle_mean_cos(
                    np.ascontiguousarray(xg * rs, dtype=float),
                    np.ascontiguousarray(cc["dc"], dtype=float), cth,
                    np.ascontiguousarray(h["Rn"], dtype=float),
                    np.ascontiguousarray(h["n2"], dtype=float))
            else:
                ring = sub._ring(h, xg * rs, cc["dc"])
                cth = self.cth_m
                ecd = subhalos._pair_angle_mean_cos(np.ascontiguousarray(xg * rs, dtype=float),
                                                    np.ascontiguousarray(cc["dc"], dtype=float), cth,
                                                    np.ascontiguousarray(h["Rn"], dtype=float),
                                                    np.ascontiguousarray(h["n2"], dtype=float))
            nd = cc["dc"].size
            k, m, g, pref, d = cc["k"], cc["m"], cc["g"], cc["pref"], cc["d"]
            # host cut kappa*: max_r rate(kappa_c > kappa*) <= eps_rare
            kstar = _cut_for(pref * ring.max(axis=0)[d], k, self.eps_rare)
            bulk = k <= kstar
            ms = max(float(np.sum(pref * ring.mean(axis=0)[d] * m)), 1e-300)   # mass scale
            mh = m / ms
            if _HAVE_NUMBA:
                Q = _bulk_moments_accum(d, bulk, pref, k, mh, g, nd)
            else:
                cols = [pref * k**a for a in range(1, 6)] + [pref * mh**b for b in range(1, 6)] \
                    + [pref * k * mh, pref * g**2, pref * k * g**2, pref * g, pref * k * g, pref * mh * g]
                Q = np.zeros((nd, len(cols)))
                for j, cvec in enumerate(cols):
                    Q[:, j] = np.bincount(d[bulk], cvec[bulk], nd)
            CQ = ring @ Q
            C = CQ[:, :-3]                                               # (ng, 13)
            RE = ring * ecd
            E1 = RE @ Q[:, -3]                                           # sum nu g <cos>
            KE1 = RE @ Q[:, -2]                                          # sum nu kappa g <cos>
            SE1 = RE @ Q[:, -1]                                          # sum nu m g <cos>
            # rare strong clumps: log bins in kappa_c
            rare = ~bulk
            nbr = self.nbr if rare.any() else 0
            if nbr:
                kr = k[rare]
                be = np.geomspace(kr.min() * (1 - 1e-12), kr.max() * (1 + 1e-12), nbr + 1)
                bi = np.clip(np.searchsorted(be, kr, side="right") - 1, 0, nbr - 1)
                dr = d[rare]
                if _HAVE_NUMBA:
                    rare_moms = _rare_bins_accum(ring, ecd, dr, bi, pref[rare], kr, m[rare], g[rare], nbr)
                    nub_g, k1b_g, m1b_g, g2b_g, k2b_g, e1b_g = (rare_moms[i] for i in range(6))
                else:
                    rq = []
                    for v in (pref[rare], pref[rare] * kr, pref[rare] * m[rare], pref[rare] * g[rare]**2,
                              pref[rare] * kr**2):
                        A = np.zeros((nd, nbr)); np.add.at(A, (dr, bi), v); rq.append(ring @ A)
                    Ag = np.zeros((nd, nbr)); np.add.at(Ag, (dr, bi), pref[rare] * g[rare])
                    rq.append((ring[:, :, None] * ecd[:, :, None] * Ag[None, :, :]).sum(axis=1))
                    nub_g, k1b_g, m1b_g, g2b_g, k2b_g, e1b_g = rq            # (ng, nbr)
                # within-bin kappa variance, two-pass (see _rare_bins_var)
                kbg = k1b_g / np.where(nub_g > 0, nub_g, 1.0)
                if _HAVE_NUMBA:
                    v2b_g = _rare_bins_var(ring, dr, bi, pref[rare], kr, kbg)
                else:
                    t2 = ring[:, dr] * pref[rare][None, :] * (kr[None, :] - kbg[:, bi])**2
                    oh = np.zeros((kr.size, nbr)); oh[np.arange(kr.size), bi] = 1.0
                    v2b_g = t2 @ oh
            # every rule / regression quantity is formed ON THE COARSE GRID (smooth
            # in r); only the final point arrays are interpolated to the cells
            itp = lambda a: a
            c = [None] + [itp(C[:, a]) for a in range(5)]               # K cumulants c1..c5
            sS = [None] + [itp(C[:, 5 + b]) for b in range(5)]           # S cumulants (/ms^b)
            c11, E2b, KE2, E1c, KE1c, SE1c = (itp(C[:, 10]), itp(C[:, 11]), itp(C[:, 12]),
                                              itp(E1), itp(KE1), itp(SE1))
            n = xg.size
            # K rule
            sdK, tauK = _std_moments_from_cumulants(*c[1:])
            tK, pK = _gauss3_tau(tauK, (0.0 - c[1]) / np.where(sdK > 0, sdK, 1.0), np.full(n, np.inf))
            Kn = c[1][:, None] + sdK[:, None] * tK                       # (n, 3)
            # S | K rule: regression + residual with the marginal shape
            sdS, tauS = _std_moments_from_cumulants(*sS[1:])
            tS, pS = _gauss3_tau(tauS, (0.0 - sS[1]) / np.where(sdS > 0, sdS, 1.0), np.full(n, np.inf))
            c2s = np.where(c[2] > 0, c[2], 1.0)
            bK = np.where(c[2] > 0, c11 / c2s, 0.0)
            # residual variance of S given K, var_perp = det / c2, cancellation-
            # free (_det_KS); sS2 - c11^2/c2 left sqrt(eps)-sized noise in sres
            ibk, mbin = cc["i"][bulk], h["m"] / h["vr"] / ms
            if _HAVE_NUMBA:
                cross, Wk = _det_KS(np.ascontiguousarray(ring), d[bulk], ibk, pref[bulk], k[bulk],
                                    mbin, mbin.size)
            else:
                cross, Wk = _det_KS_numpy(ring, d[bulk], ibk, pref[bulk], k[bulk], mbin, mbin.size)
            vperp = np.where(c[2] > 0, (cross + sS[2] * Wk) / c2s, sS[2])
            sres = sqrt(vperp)
            Sm = (sS[1][:, None, None] + bK[:, None, None] * (Kn[:, :, None] - c[1][:, None, None])
                  + sres[:, None, None] * tS[:, None, :])               # (n, 3, 3), units of ms
            Sm = np.maximum(Sm, 0.0) * ms
            pb = pK[:, :, None] * pS[:, None, :]                         # (n, 3, 3)
            # bulk shear CONDITIONAL on the (K, S) node by exact linear regression
            # of the projected clump shear (covariances sum nu kappa g <cos>, sum nu
            # m g <cos>): a clump near the ray raises kappa and gamma together.
            # Sequential (Gram-Schmidt) form, 2026-10-03: on K, then on S_perp =
            # S - (c11/c2) K with the exact var_perp and a KS_RIDGE ridge -- equal
            # to the joint regression when well posed, continuous when K and S
            # are collinear (the old det > 1e-30 c2 sS2 branch flipped by roundoff)
            dKn = np.repeat(Kn, 3, axis=1) - c[1][:, None]               # (n, 9)
            dSn = Sm.reshape(n, 9) / ms - sS[1][:, None]                 # (n, 9), units of ms
            b1 = np.where(c[2] > 0, KE1c / c2s, 0.0)
            cov = SE1c - bK * KE1c
            den = vperp + KS_RIDGE * sS[2]
            b2 = np.where(den > 0, cov / np.where(den > 0, den, 1.0), 0.0)
            E1n = E1c[:, None] + b1[:, None] * dKn + b2[:, None] * (dSn - bK[:, None] * dKn)
            expl = b1**2 * c[2] + b2**2 * vperp
            G2n = np.repeat(np.maximum(E2b - expl, 0.0)[:, None], 9, axis=1)
            E1b = E1c
            E2 = E2b + E1b**2
            # rare bins on cells
            if nbr:
                nub = np.stack([itp(nub_g[:, j]) for j in range(nbr)], axis=1)
                nur = nub.sum(axis=1)
                p0r = np.exp(-nur)
                sc = np.where(nur > 0, -np.expm1(-nur) / np.where(nur > 0, nur, 1.0), 0.0)
                nz = np.where(nub > 0, nub, 1.0)
                kb_ = np.stack([itp(k1b_g[:, j]) for j in range(nbr)], axis=1) / nz
                mb_ = np.stack([itp(m1b_g[:, j]) for j in range(nbr)], axis=1) / nz
                g2b = np.stack([itp(g2b_g[:, j]) for j in range(nbr)], axis=1) / nz
                e1b = np.stack([itp(e1b_g[:, j]) for j in range(nbr)], axis=1) / nz
                skb = sqrt(np.stack([itp(v2b_g[:, j]) for j in range(nbr)], axis=1) / nz)   # within-bin kappa sd
                kb_ = np.concatenate([kb_ - skb, kb_ + skb], axis=1)
                mb_, g2b, e1b = (np.tile(a, (1, 2)) for a in (mb_, g2b, e1b))
                nub = np.tile(nub / 2, (1, 2))
            else:
                p0r = np.ones(n)
            # all points per cell: 9 bulk + nbr rare
            Kp_c = np.concatenate([np.repeat(Kn, 3, axis=1)] + ([c[1][:, None] + kb_] if nbr else []), axis=1)
            Sp_c = np.concatenate([Sm.reshape(n, 9)] + ([(sS[1] * ms)[:, None] + mb_] if nbr else []), axis=1)
            wp_c = np.concatenate([pb.reshape(n, 9) * p0r[:, None]] + ([nub * sc[:, None]] if nbr else []), axis=1)
            A1_c = np.concatenate([E1n] + ([E1b[:, None] + e1b] if nbr else []), axis=1)
            A2_c = np.concatenate([G2n + E1n**2]
                                + ([E2[:, None] + g2b + 2 * E1b[:, None] * e1b] if nbr else []), axis=1)
            # host-mass table range from nodes that carry weight only: zero /
            # ~1e-16-weight nodes (degenerate Gauss rules at the outermost r)
            # have roundoff-noise S up to ~1e3 x the real ones, and Mlo sets the
            # WHOLE log-M interpolation grid, i.e. every node's rebuilt kappa
            # (2026-10-03: a 2-ulp NFW change moved z_s=1 bin centroids 6e-9)
            Mlo = max(M - float(np.max(np.where(wp_c > MT_WFLOOR, Sp_c, 0.0))), self.Mmin)
            Mt = np.unique(np.concatenate([np.geomspace(Mlo, M, self.nM_tab), [M]]))
            cos, z, zs = sub.cos, h["z"], sub.zs
            _, rs_t, ks_t, _ = cos.nfw_params(Mt, z, zs)
            par = np.column_stack([rs_t, ks_t])
            lMt = np.log(Mt)
            log_rs = np.log(par[:, 0])
            log_ks = np.log(par[:, 1])
            ept = F.epsilon_NFW_vec(cos, Mt, z)
            cm = dict(xg=xg, lxg=lxg, Kp_c=Kp_c, Sp_c=Sp_c, wp_c=wp_c, A1_c=A1_c, A2_c=A2_c,
                      rs=rs, M=M, Mt=Mt, lMt=lMt, log_rs=log_rs, log_ks=log_ks, ept=ept)
            h["_coarse_moments"] = cm

        xg, lxg = cm["xg"], cm["lxg"]
        Kp_c, Sp_c, wp_c, A1_c, A2_c = cm["Kp_c"], cm["Sp_c"], cm["wp_c"], cm["A1_c"], cm["A2_c"]
        rs, M = cm["rs"], cm["M"]
        Mt, lMt, log_rs, log_ks, ept = cm["Mt"], cm["lMt"], cm["log_rs"], cm["log_ks"], cm["ept"]

        # coarse grid -> emitted cells: one shared index / weight table
        if _HAVE_NUMBA and xt is not None:
            gx, gw = np.array(F._GLX, dtype=float), np.array(F._GLW, dtype=float)
            if phicell is None:
                ph_dummy = np.zeros(xcell.shape[0], dtype=float)
                ep_zero = np.zeros_like(ept)
                Ko, Go, Wo = _moments_fused_general(
                    np.ascontiguousarray(xcell, dtype=float),
                    ph_dummy,
                    np.ascontiguousarray(Wcell, dtype=float),
                    float(rs), float(xt),
                    np.ascontiguousarray(Kp_c, dtype=float),
                    np.ascontiguousarray(Sp_c, dtype=float),
                    np.ascontiguousarray(wp_c, dtype=float),
                    np.ascontiguousarray(A1_c, dtype=float),
                    np.ascontiguousarray(A2_c, dtype=float),
                    np.ascontiguousarray(xg, dtype=float),
                    np.ascontiguousarray(lxg, dtype=float),
                    np.ascontiguousarray(Mt, dtype=float),
                    np.ascontiguousarray(lMt, dtype=float),
                    np.ascontiguousarray(log_rs, dtype=float),
                    np.ascontiguousarray(log_ks, dtype=float),
                    ep_zero,
                    float(M), float(self.Mmin), gx, gw)
            else:
                nph = 12
                is_uniform = False
                if xcell.size % nph == 0:
                    nr = xcell.size // nph
                    if np.all(xcell.reshape(nph, nr) == xcell[:nr]) and np.all(Wcell.reshape(nph, nr) == Wcell[:nr]):
                        is_uniform = True
                if is_uniform:
                    nr = xcell.size // nph
                    xc0 = np.ascontiguousarray(xcell[:nr], dtype=float)
                    W0 = np.ascontiguousarray(Wcell[:nr], dtype=float)
                    phis_arr = np.ascontiguousarray(phicell[::nr], dtype=float)
                    Ko, Go, Wo = _moments_fused_uniform(
                        xc0, W0, phis_arr,
                        float(rs), float(xt),
                        np.ascontiguousarray(Kp_c, dtype=float),
                        np.ascontiguousarray(Sp_c, dtype=float),
                        np.ascontiguousarray(wp_c, dtype=float),
                        np.ascontiguousarray(A1_c, dtype=float),
                        np.ascontiguousarray(A2_c, dtype=float),
                        np.ascontiguousarray(xg, dtype=float),
                        np.ascontiguousarray(lxg, dtype=float),
                        np.ascontiguousarray(Mt, dtype=float),
                        np.ascontiguousarray(lMt, dtype=float),
                        np.ascontiguousarray(log_rs, dtype=float),
                        np.ascontiguousarray(log_ks, dtype=float),
                        np.ascontiguousarray(ept, dtype=float),
                        float(M), float(self.Mmin), gx, gw)
                else:
                    # per-phi refined grids: radii repeat across phi -> do the
                    # (radius, node) work once per unique radius (2026-10-03,
                    # bit-identical to _moments_fused_general)
                    xu, inv = np.unique(np.asarray(xcell, dtype=float), return_inverse=True)
                    Ko, Go, Wo = _moments_fused_indexed(
                        np.ascontiguousarray(xu), np.ascontiguousarray(inv.ravel(), dtype=np.int64),
                        np.ascontiguousarray(phicell, dtype=float),
                        np.ascontiguousarray(Wcell, dtype=float),
                        float(rs), float(xt),
                        np.ascontiguousarray(Kp_c, dtype=float),
                        np.ascontiguousarray(Sp_c, dtype=float),
                        np.ascontiguousarray(wp_c, dtype=float),
                        np.ascontiguousarray(A1_c, dtype=float),
                        np.ascontiguousarray(A2_c, dtype=float),
                        np.ascontiguousarray(xg, dtype=float),
                        np.ascontiguousarray(lxg, dtype=float),
                        np.ascontiguousarray(Mt, dtype=float),
                        np.ascontiguousarray(lMt, dtype=float),
                        np.ascontiguousarray(log_rs, dtype=float),
                        np.ascontiguousarray(log_ks, dtype=float),
                        np.ascontiguousarray(ept, dtype=float),
                        float(M), float(self.Mmin), gx, gw)
            self.stats["hosts"] += 1
            self.stats["points"] += int(Ko.size)
            return Ko, Go, Wo

        lx = np.clip(np.log(np.maximum(xcell, xg[0])), lxg[0], lxg[-1])
        jj = np.clip(np.searchsorted(lxg, lx) - 1, 0, lxg.size - 2)
        tw = ((lx - lxg[jj]) / (lxg[jj + 1] - lxg[jj]))[:, None]
        lerp = lambda A: (1.0 - tw) * A[jj] + tw * A[jj + 1]
        Kp, Sp, wp, A1, A2 = (lerp(A) for A in (Kp_c, Sp_c, wp_c, A1_c, A2_c))
        # rebuilt host at max(M - S, Mmin): interpolated (rs, ks, eps)(log M)
        Mlo = max(M - float(np.max(np.where(wp > MT_WFLOOR, Sp, 0.0))), self.Mmin)
        Mt = np.unique(np.concatenate([np.geomspace(Mlo, M, self.nM_tab), [M]]))
        cos, z, zs = sub.cos, h["z"], sub.zs
        _, rs_t, ks_t, _ = cos.nfw_params(Mt, z, zs)
        par = np.column_stack([rs_t, ks_t])                                 # rs, ks
        lMt = np.log(Mt)
        lMe = np.log(np.maximum(M - Sp, self.Mmin))
        rse = np.exp(np.interp(lMe, lMt, np.log(par[:, 0])))
        kse = np.exp(np.interp(lMe, lMt, np.log(par[:, 1])))
        xe = (xcell * rs)[:, None] / rse
        xte = None if xt is None else np.broadcast_to(xt * rs / rse, xe.shape)   # same physical edge
        if phicell is None:
            k1, g1 = _kg_unit(xe, xte)
            kh, gh = k1.reshape(xe.shape) * kse, g1.reshape(xe.shape) * kse
        else:
            ept = F.epsilon_NFW_vec(cos, Mt, z)     # scalar-only helper
            epe = np.interp(lMe, lMt, ept)
            kh, gh = _kg_eps_vec(epe, kse, xe, np.broadcast_to(phicell[:, None], xe.shape), xte)
        Ko = (kh + Kp).ravel()
        self._dbg = dict(Kp=Kp, Sp=Sp, wp=wp, kh=kh)
        g2 = gh**2 + 2 * gh * A1 + A2
        Go = sqrt(np.maximum(g2, 0.0)).ravel()
        Wo = (np.asarray(Wcell)[:, None] * wp).ravel()
        self.stats["hosts"] += 1
        self.stats["points"] += int(Ko.size)
        return Ko, Go, Wo

    def host_cells_lattice(self, jz, jM, xref, Wref, xcell, Wcell, phicell=None):
        """Cells (K, G, W) of the decorated host.
        xref, Wref: the circular reference grid (r/rs, weights) defining the
        groups and the group-averaged clump law; xcell, Wcell[, phicell]: the
        host cells to emit (circular: = xref; elliptical: every (r, phi) cell)."""
        sub, h = self.sub, self.sub.host(jz, jM)
        cc = self._clump_cells(h)
        if cc is None:
            return None
        rs = h["rs"]
        rr = xref * rs
        ring = sub._ring(h, rr, cc["dc"])                            # (nref, nd)
        rate = ring[:, cc["d"]] * cc["pref"][None, :]                # (nref, ncell)
        mKr, mSr = rate @ cc["k"], rate @ cc["m"]                    # E[K | r], E[S | r] (exact, linear)
        # groups on the law's SHAPE only (rate and kappa variance): the means
        # are restored exactly per cell below
        groups = self._groups(np.column_stack([rate.sum(axis=1), rate @ cc["k"]**2]), Wref)
        # <cos theta>(r, d): engine placement density, half-circle nodes
        th = (np.arange(self.ntheta) + 0.5) * np.pi / self.ntheta
        cth = np.cos(th)
        ecd = subhalos._pair_angle_mean_cos(np.ascontiguousarray(rr, dtype=float),
                                            np.ascontiguousarray(cc["dc"], dtype=float), cth,
                                            np.ascontiguousarray(h["Rn"], dtype=float),
                                            np.ascontiguousarray(h["n2"], dtype=float))
        ecc = ecd[:, cc["d"]]                                        # (nref, ncell)
        # group laws -> compressed points
        Kp, Sp, wp, e1p, e2p, goff = [], [], [], [], [], [0]
        gid_ref = np.empty(xref.size, np.int64)
        mKg, mSg = np.empty(len(groups)), np.empty(len(groups))
        for gi, (a, b) in enumerate(groups):
            gid_ref[a:b] = gi
            w = Wref[a:b] / Wref[a:b].sum()
            nu = w @ rate[a:b]
            mKg[gi], mSg[gi] = float(nu @ cc["k"]), float(nu @ cc["m"])
            ec = np.where(nu > 0, (w @ (rate[a:b] * ecc[a:b])) / np.where(nu > 0, nu, 1), 0.0)
            ok = nu > 0
            args = (nu[ok], cc["k"][ok], cc["m"][ok], cc["g"][ok], ec[ok], cc["i"][ok])
            if self.lattice == "1d":
                L = lattice_law_1d(*args, eps_core=self.eps_core, resK=self.resK)
                pts = compress_1d(L, nq=self.nq)
            else:
                L = lattice_law(*args, eps_core=self.eps_core, resK=self.resK, resS=self.resS)
                pts = compress(L, nq=self.nq)
            self.stats["lattice"] += L["p"].size
            for arr, v in zip((Kp, Sp, wp, e1p, e2p), pts):
                arr.append(v)
            goff.append(goff[-1] + pts[0].size)
        Kp, Sp, wp, e1p, e2p = (np.concatenate(a) for a in (Kp, Sp, wp, e1p, e2p))
        goff = np.array(goff, np.int64)
        # host tables on an S grid: rebuilt host at max(M - S, Mmin)
        Shi = max(float(Sp.max()) + max(float(np.max(mSr) - np.min(mSg)), 0.0), 1e-30) * (1 + 1e-9)
        St = np.linspace(0.0, Shi, self.nS_tab)
        cos, z, zs = sub.cos, h["z"], sub.zs
        rcell = xcell * rs
        tk = np.empty((xcell.size, St.size)); tg = np.empty_like(tk)
        for t, s in enumerate(St):
            Me = max(h["M"] - s, self.Mmin)
            C_, rs_t, ks_t, fC = cos.nfw_params(Me, z, zs)
            if phicell is None:
                k_, g_ = sgl.kappa_gamma(np.ascontiguousarray(rcell / rs_t), ks_t)
            else:
                k_, g_ = F.kappagamma_eps(F.epsilon_NFW(cos, Me, z), ks_t, rcell / rs_t, phicell)
            tk[:, t], tg[:, t] = k_, g_
        # cell -> group by r (elliptical cells: nearest reference r)
        cg = gid_ref[np.clip(np.searchsorted(xref, xcell), 0, xref.size - 1)]
        dKc = np.interp(xcell, xref, mKr) - mKg[cg]
        dSc = np.interp(xcell, xref, mSr) - mSg[cg]
        npt = np.diff(goff)[cg]
        coff = np.concatenate([[0], np.cumsum(npt)]).astype(np.int64)
        Ko = np.empty(coff[-1]); Go = np.empty(coff[-1]); Wo = np.empty(coff[-1])
        _emit(cg.astype(np.int64), np.ascontiguousarray(Wcell, dtype=float),
              np.ascontiguousarray(dKc, dtype=float), np.ascontiguousarray(dSc, dtype=float),
              tk, tg, St, goff,
              Kp, Sp, wp, e1p, e2p, coff[:-1].copy(), Ko, Go, Wo)
        self.stats["hosts"] += 1
        self.stats["groups"] += len(groups)
        self.stats["points"] += int(coff[-1])
        return Ko, Go, Wo

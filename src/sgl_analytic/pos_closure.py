"""Positive Cox clustering closures for the xi_lin chain (2026-10-17 lognormal port; 2026-10-19 'vp').

Replaces the Gaussian (Gauss-Poisson) pair term Psi_2 of Lam._Qxi by the characteristic function of a
POSITIVE random lens intensity on the SAME cells as Psi_1 (experiments/psi2_positivity/positive_closure,
ported from its hook-based prototype; experiments/lognormal_closure/README.md, experiments/lowz_vp/):

  Phi = exp(Psi_1) * E_U[ exp( sum_s sum_c W_sc (lam_sc(U_s) - 1) F_c(k) ) ] * exp(-shear_var k_g^2 / 4)

  U_s       latent N(0,1) state per lens shell s, a stationary Markov chain along the line of sight
            (48 Gauss-Hermite states; reversible Gaussian-copula kernel with correlation rho_s between
            neighbouring shells, from the leading m=0 k_perp profile and the top-hat l.o.s. correlation),
  v_sc      = eta f_s phi_sc^2, the cell's relative rate VARIANCE; phi = the cell's coarse-parent loading on
            the single m=0 mode that carries all of the k -> 0 kappa clustering variance ('kappa' mode),
            f_s = (1 - rho_bar)/(1 + rho_bar) (chain row-sum rule),
  lam_sc(u) has E_w lam = 1 and Var_w lam = v_sc EXACTLY on the 48 nodes (mean rates preserved), with
    kind 'vp' (default):  lam = (s u - t)^+ / E_w[(s u - t)^+],  s = sign(phi), t solved from Var = v.
                          The Gaussian closure's own intensity 1 + s sqrt(v) u, rectified ("variance-
                          preserving clamp"): t = -1/sqrt(v) (exactly linear) while that is positive on every
                          node; t < 0 = the clamp max(1 + b u, 0) with the slope b re-fitted to keep Var = v;
                          t > 0 continues past the plain clamp's cap Var <= pi - 1.
    kind 'lognormal':     lam = exp(a u) / E_w[exp(a U)],  a = s sqrt(ln(1 + v)).
  eta       ONE number, fitted so that the model's kappa clustering variance equals the xi_lin m=0
            target sum_r (rows_r . kappa_c + res_r)^2 (= what Lam.sigma() adds),
  shear_var the xi_lin m=1 shear variance, as an independent isotropic Gaussian; m=2 dropped.

Positive rates + a stochastic kernel => Phi is a characteristic function for every cosmology and z_s
(|Phi| <= 1, no revival). This is a DIFFERENT higher-order model from the Gaussian closure (same mean
rates and kappa / shear clustering variances), not an algebraic fix of it. For a positive rate with
E = 1 and Var = v the third central moment is >= v^2 - v (Cauchy-Schwarz), so where the Gaussian
closure's v > 1 (low z_s, high sigma8) NO positive closure agrees with it beyond second order.

The inner expectation is a forward transfer recursion over shells (log-scaled); cells of a shell with
equal loading share lam(u), so  sum_c W_c (lam_c(u) - 1) F_c = sum_g (lam_g(u) - 1) Y_g,
Y_g = sum_{c in g} W_c F_c  (exact; numba pass + one (n x G_s)(G_s x 48) GEMM per shell).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.special import j0, logsumexp

from . import sgl_full as F

NSTATE = 48
H_A = 0.01          # loading-grid spacing of the interpolated rates: |d lnR| <= 1.2e-9 at Planck z_s=1 (h^4)

try:
    import numba

    @numba.njit(parallel=True, cache=True)
    def _group_sums(Fr, Fi, uidx, gidx, W, G):
        n = Fr.shape[0]
        Yr = np.zeros((n, G))
        Yi = np.zeros((n, G))
        for p in numba.prange(n):
            for c in range(uidx.size):
                u = uidx[c]
                g = gidx[c]
                Yr[p, g] += W[c] * Fr[p, u]
                Yi[p, g] += W[c] * Fi[p, u]
        return Yr, Yi

    @numba.njit(parallel=True, cache=True)
    def _group_sums_T(Fr, Fi, cu, cw, gstart, Yr, Yi):
        """Cell-major group sums: Y[g, :] = sum_{c in g} w_c F[u_c, :] (cells sorted by group,
        gstart = group offsets); contiguous AXPYs over the point axis."""
        n = Fr.shape[1]
        for g in numba.prange(gstart.size - 1):
            for p in range(n):
                Yr[g, p] = 0.0
                Yi[g, p] = 0.0
            for c in range(gstart[g], gstart[g + 1]):
                u = cu[c]
                w = cw[c]
                for p in range(n):
                    Yr[g, p] += w * Fr[u, p]
                    Yi[g, p] += w * Fi[u, p]

    @numba.njit(parallel=True, cache=True)
    def _node_agg(Yr, Yi, glo, ghi, j0, Lw, Zr, Zi):
        """Z[j, :] = sum_g L_{j - j0_g}(a_g) Y[g, :] over the groups whose 4-point stencil holds node j
        (groups sorted by a: glo[j]..ghi[j] is contiguous)."""
        n = Yr.shape[1]
        for j in numba.prange(Zr.shape[0]):
            for p in range(n):
                Zr[j, p] = 0.0
                Zi[j, p] = 0.0
            for g in range(glo[j], ghi[j]):
                m = j - j0[g]
                if m < 0 or m > 3:
                    continue
                w = Lw[g, m]
                for p in range(n):
                    Zr[j, p] += w * Yr[g, p]
                    Zi[j, p] += w * Yi[g, p]

    @numba.njit(parallel=True, cache=True)
    def _group_sums_krow(er, ei, J, cu, cw, gstart, Yr, Yi):
        """Fused F + group sums for one k_kappa row: F[u, p] = e_u J[u, p] + J[u, p] - 1 with
        e_u = e^{i k kappa_u} - 1 (scalar per cell) and J[u, :] = J0(q_p gamma_u) (row of the q grid)."""
        n = J.shape[1]
        for g in numba.prange(gstart.size - 1):
            for p in range(n):
                Yr[g, p] = 0.0
                Yi[g, p] = 0.0
            for c in range(gstart[g], gstart[g + 1]):
                u = cu[c]
                w = cw[c]
                a = w * er[u]
                b = w * ei[u]
                for p in range(n):
                    jv = J[u, p]
                    Yr[g, p] += a * jv + w * (jv - 1.0)
                    Yi[g, p] += b * jv

    @numba.njit(parallel=True, cache=True)
    def _group_sums_qrow(Er, Ei, jq, cu, cw, gstart, Yr, Yi):
        """Fused F + group sums for one k_gamma column: F[u, p] = E[u, p] j_u + j_u - 1."""
        n = Er.shape[1]
        for g in numba.prange(gstart.size - 1):
            for p in range(n):
                Yr[g, p] = 0.0
                Yi[g, p] = 0.0
            for c in range(gstart[g], gstart[g + 1]):
                u = cu[c]
                w = cw[c]
                jv = jq[u]
                d = w * (jv - 1.0)
                wj = w * jv
                for p in range(n):
                    Yr[g, p] += wj * Er[u, p] + d
                    Yi[g, p] += wj * Ei[u, p]

    @numba.njit(parallel=True, cache=True)
    def _block_nodes(er, ei, J, Jm1, cu, cw, gstart, sh_g, j0, Lw, sh_node, Zr, Zi):
        """Register-blocked group sums on a (k block) x (q block) of points, scattered onto the rate
        nodes:  Z[node, a, b] = sum_g L_{node}(a_g) sum_{c in g} w_c (e_u(k_a) J_u(q_b) + J_u(q_b) - 1),
        e = expm1(i k kappa), J = J0(q gamma) (prototype numerics). Parallel over shells (each shell
        owns its node rows). Exact groups: j0 = local group index, Lw = [1, 0, 0, 0]."""
        nka, nqb = er.shape[1], J.shape[1]
        for s in numba.prange(sh_g.size - 1):
            accr = np.empty((nka, nqb))
            acci = np.empty((nka, nqb))
            S = np.empty(nqb)
            wr = np.empty(nka)
            wi = np.empty(nka)
            n0, n1 = sh_node[s], sh_node[s + 1]
            for nd in range(n0, n1):
                for a in range(nka):
                    for b in range(nqb):
                        Zr[nd, a, b] = 0.0
                        Zi[nd, a, b] = 0.0
            for g in range(sh_g[s], sh_g[s + 1]):
                accr[:, :] = 0.0
                acci[:, :] = 0.0
                S[:] = 0.0
                for c in range(gstart[g], gstart[g + 1]):
                    u = cu[c]
                    w = cw[c]
                    for b in range(nqb):
                        S[b] += w * Jm1[u, b]
                    for a in range(nka):
                        wr[a] = w * er[u, a]
                        wi[a] = w * ei[u, a]
                    for a in range(nka):
                        x = wr[a]
                        y = wi[a]
                        for b in range(nqb):
                            jv = J[u, b]
                            accr[a, b] += x * jv
                            acci[a, b] += y * jv
                for m in range(4):
                    L = Lw[g, m]
                    if L == 0.0:
                        continue
                    nd = n0 + j0[g] + m
                    for a in range(nka):
                        for b in range(nqb):
                            Zr[nd, a, b] += L * (accr[a, b] + S[b])
                            Zi[nd, a, b] += L * acci[a, b]

    @numba.njit(parallel=True, cache=True)
    def _chain_recursion(zr, zi, K, w):
        """ln E exp(sum_s z_s(U_s)) for the stationary chain, per point (log-scaled forward pass).
        zr, zi: (nshell, nstate, npts); K: (nshell - 1, nstate, nstate)."""
        nsh, ns, P = zr.shape
        out_r = np.empty(P)
        out_i = np.empty(P)
        for p in numba.prange(P):
            ar = np.empty(ns)
            ai = np.empty(ns)
            br = np.empty(ns)
            bi = np.empty(ns)
            scale = 0.0
            ph = 0.0
            for s in range(nsh):
                if s == 0:
                    for j in range(ns):
                        br[j] = w[j]
                        bi[j] = 0.0
                else:
                    for j in range(ns):
                        tr = 0.0
                        ti = 0.0
                        for i in range(ns):
                            kv = K[s - 1, i, j]
                            tr += kv * ar[i]
                            ti += kv * ai[i]
                        br[j] = tr
                        bi[j] = ti
                mx = zr[s, 0, p]
                for j in range(1, ns):
                    if zr[s, j, p] > mx:
                        mx = zr[s, j, p]
                nrm = 0.0
                for j in range(ns):
                    m = np.exp(zr[s, j, p] - mx)
                    c = np.cos(zi[s, j, p])
                    sn = np.sin(zi[s, j, p])
                    er_ = m * c
                    ei_ = m * sn
                    ar[j] = br[j] * er_ - bi[j] * ei_
                    ai[j] = br[j] * ei_ + bi[j] * er_
                    a2 = ar[j] * ar[j] + ai[j] * ai[j]
                    if a2 > nrm:
                        nrm = a2
                nrm = np.sqrt(nrm)
                for j in range(ns):
                    ar[j] /= nrm
                    ai[j] /= nrm
                scale += mx + np.log(nrm)
            sr = 0.0
            si = 0.0
            for j in range(ns):
                sr += ar[j]
                si += ai[j]
            out_r[p] = scale + 0.5 * np.log(sr * sr + si * si)
            out_i[p] = np.arctan2(si, sr)
        return out_r, out_i
except ImportError:                                          # pragma: no cover
    def _group_sums(Fr, Fi, uidx, gidx, W, G):
        Yr = np.zeros((Fr.shape[0], G)); Yi = np.zeros((Fr.shape[0], G))
        for c in range(uidx.size):
            Yr[:, gidx[c]] += W[c] * Fr[:, uidx[c]]
            Yi[:, gidx[c]] += W[c] * Fi[:, uidx[c]]
        return Yr, Yi


# ----------------------------------------------------------------------------- latent chain
def transition(u, w, rho):
    """Reversible Gaussian-copula transition on the GH nodes with EXACTLY stationary weights w
    (symmetric matrix balancing; positive_closure/model.py)."""
    x, y = u[:, None], u[None, :]
    logratio = (2 * rho * x * y - rho * rho * (x * x + y * y)) / (2 * (1 - rho * rho))
    J = w[:, None] * w[None, :] * np.exp(logratio)
    v = np.ones(len(u))
    for _ in range(10000):
        marg = v * (J @ v)
        if np.max(abs(marg / w - 1)) < 3e-14:
            break
        v *= np.sqrt(w / marg)
    else:
        raise RuntimeError("stationary transition balancing did not converge")
    K = (v[:, None] * J * v[None, :]) / w[:, None]
    assert np.max(abs(w @ K - w)) < 1e-13 and np.max(abs(K.sum(axis=1) - 1)) < 1e-12
    return K


KINDS = ("vp", "lognormal")


def vp_rates(v, sgn, u, w):
    """Variance-preserving rectified rates (G, nstate): lam = (s u - t)^+ / E_w[(s u - t)^+] with
    E_w lam = 1, Var_w lam = v (per group), s = sgn. u must be the symmetric ascending GH nodes.
    Between consecutive nodes the positive set {u > t} is fixed, so with A, B, C = sum_{u > t} w u, w, w u^2:
    E1 = A - t B, E2 = C - 2 t A + t^2 B, and Var = v  <=>  E2 = (1 + v) E1^2, a quadratic in t; the
    relative variance increases monotonically in t, so the node segment is found by bisection on the
    node values. Segment 'all nodes' (A = 0, B = C = 1 exactly on GH) gives t = -1/sqrt(v): exactly the
    linear rate 1 + s sqrt(v) u."""
    v = np.asarray(v, float)
    n = u.size
    A = np.cumsum((w * u)[::-1])[::-1]; B = np.cumsum(w[::-1])[::-1]; C = np.cumsum((w * u * u)[::-1])[::-1]
    # segment j = 0..n-1: positive set {i >= j}, t in [u[j-1], u[j]) (u[-1] = -inf)
    tb = u[:-1]                                                     # left end of segments 1..n-1
    E1b = A[1:] - tb * B[1:]; E2b = C[1:] - 2 * tb * A[1:] + tb * tb * B[1:]
    rvb = E2b / E1b**2 - 1.0                                        # rel. variance at t = u[j-1], j = 1..n-1
    assert np.all(np.diff(rvb) > 0)
    if np.any(v >= rvb[-1]):
        raise ValueError(f"vp closure: relative variance {v.max():.3g} beyond the {n}-state cap {rvb[-1]:.3g}")
    j = np.searchsorted(rvb, v, side="right")                       # 0 -> all nodes positive
    a, b, c = A[j], B[j], C[j]
    q2 = b - (1 + v) * b * b; q1 = 2 * a * ((1 + v) * b - 1); q0 = c - (1 + v) * a * a
    lo = np.where(j > 0, u[np.maximum(j - 1, 0)], -np.inf); hi = u[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = np.sqrt(np.maximum(q1 * q1 - 4 * q2 * q0, 0.0))
        r1 = np.where(q2 != 0, (-q1 - disc) / (2 * q2), -q0 / q1)
        r2 = np.where(q2 != 0, (-q1 + disc) / (2 * q2), -q0 / q1)
    t = np.where((r1 >= lo) & (r1 < hi), r1, r2)
    lin = j == 0
    t = np.where(lin, u[0] - 1.0, t)                 # placeholder: the linear branch is set exactly below
    if not np.all((t >= lo - 1e-12 * (1 + np.abs(lo))) & (t <= hi)):
        raise RuntimeError("vp closure: threshold root outside its node segment")
    us = np.where(np.asarray(sgn)[:, None] >= 0, u[None, :], -u[None, :])
    x = np.maximum(us - t[:, None], 0.0)
    lam = x / (x @ w)[:, None]
    lam[lin] = 1.0 + np.sqrt(v[lin])[:, None] * us[lin]                  # exact linear branch
    return lam


def group_rates(ph, eta, f, u, w, kind="vp"):
    """lam(u) per group, (G, nstate), E_w lam = 1 and Var_w lam = eta f ph^2, for loadings ph."""
    if kind == "vp":
        return vp_rates(eta * f * ph**2, np.sign(ph), u, w)
    a = np.sign(ph) * np.sqrt(np.log1p(eta * f * ph**2))
    ll = np.outer(a, u)
    ll -= logsumexp(ll + np.log(w)[None, :], axis=1)[:, None]
    return np.exp(ll)


def node_rates(a, u, w):
    """lognormal lam(u, a) for an array of loadings a (no sign/sqrt map): (len(a), nstate), E_w lam = 1."""
    ll = np.outer(a, u)
    ll -= logsumexp(ll + np.log(w)[None, :], axis=1)[:, None]
    return np.exp(ll)


def loading(ph, eta, f):
    """lognormal loading a (the interpolated-rate path h_a > 0 is lognormal only)."""
    return np.sign(ph) * np.sqrt(np.log1p(eta * f * ph**2))


def lagrange_stencil(a, h):
    """Uniform a-grid (spacing h) covering a with a 4-point stencil: nodes, j0 (first stencil node),
    cubic Lagrange weights Lw (len(a), 4); a must be sorted."""
    lo = float(a.min()) - h
    J = int(np.ceil((float(a.max()) - lo) / h)) + 3
    nodes = lo + h * np.arange(J)
    x = (a - lo) / h
    j0 = np.clip(np.floor(x).astype(np.int64) - 1, 0, J - 4)
    t = x - j0
    Lw = np.empty((a.size, 4))
    for m in range(4):
        v = np.ones_like(t)
        for k in range(4):
            if k != m:
                v *= (t - k) / (m - k)
        Lw[:, m] = v
    return nodes, j0, Lw


def chain_variance(ms, w, K):
    """Var of sum_s m_s(U_s) for the stationary chain (m_s: nstate-vectors)."""
    h = None
    v = 0.0
    for s, m in enumerate(ms):
        mu = m - float(w @ m)
        v += float(w @ (mu * mu))
        if s == 0:
            h = w * mu
        else:
            hp = K[s - 1].T @ h
            v += 2 * float(mu @ hp)
            h = hp + w * mu
    return float(v)


# ----------------------------------------------------------------------------- l.o.s. correlation
_PCACHE = {}


def los_rho(cos, kp, chi1, d1, chi2, d2, kpar_max=3.0, n=30001):
    """Correlation of top-hat shell averages (widths d1, d2, centres chi1, chi2) of the 3D linear field
    at transverse wavenumbers kp (cox_chain.los_rho, vectorised over kp; P(k) table cached per cos)."""
    kpar = np.linspace(0.0, kpar_max, n)
    key = (id(cos), kp.tobytes(), kpar_max, n)
    P = _PCACHE.get(key)
    if P is None:
        _PCACHE.clear()
        P = _PCACHE[key] = F.P0(cos, np.sqrt(kp[:, None]**2 + kpar[None, :]**2))
    s1, s2 = np.sinc(kpar * d1 / (2 * np.pi)), np.sinc(kpar * d2 / (2 * np.pi))
    c12 = np.trapezoid(P * (s1 * s2 * np.cos(kpar * (chi2 - chi1)))[None, :], kpar, axis=1)
    return c12 / np.sqrt(np.trapezoid(P * (s1 * s1)[None, :], kpar, axis=1)
                         * np.trapezoid(P * (s2 * s2)[None, :], kpar, axis=1))


# ----------------------------------------------------------------------------- population
def population(lam, cos):
    """Per-shell cells (W, kap, gam, phi), rho between consecutive shells, fvar, and the xi_lin targets,
    from a build made with xi['pos'] = True. Mirrors positive_closure/run.py snapshot() +
    cox_chain.Model(mode='kappa')."""
    c = lam.c
    pieces = getattr(c, "pos_pieces", None)
    if not pieces:
        raise ValueError("positive closure needs a build with xi['pos'] = True (run(..., xi_closure='lognormal'))")
    ce0, ce1 = c.ce
    ngc = ce1.size - 1
    g0 = {}
    for g in c.xi_groups:
        if g["m"] == 0:
            if g["shell"] in g0:
                raise RuntimeError("positive closure assumes one m=0 Limber node per shell (zint midpoint)")
            g0[g["shell"]] = g
    shells = sorted(set(pieces) & set(g0))
    flatW = c.H0.ravel()
    flatk = np.divide(c.H0k.ravel(), flatW, out=np.zeros_like(flatW), where=flatW > 0)
    flatg = np.divide(c.H0g.ravel(), flatW, out=np.zeros_like(flatW), where=flatW > 0)
    kmc = F._centers(ce0)
    cells = []
    for s in shells:
        p, g = pieces[s], g0[s]
        W = p["h0"]
        # coarse parent from the cell's OWN (per-shell) centroid, as the prototype
        a = np.searchsorted(ce0, p["h0k"] / W, side="right") - 1
        b = np.searchsorted(ce1, p["h0g"] / W, side="right") - 1
        ok = (a >= 0) & (a < ce0.size - 1) & (b >= 0) & (b < ngc)
        cidx = np.where(ok, a * ngc + b, -1)
        # 'kappa' mode: e ~ m_r = rows_r . kappa_c + res_r carries all the k -> 0 kappa variance
        mr = g["rows"][:, :-1] @ kmc[g["idx"] // ngc] + g["rows"][:, -1]
        r0 = (mr / np.linalg.norm(mr)) @ g["rows"][:, :-1]
        sgn = 1.0 if r0 @ g["W"] >= 0 else -1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            phic = np.where(g["W"] > 0, sgn * r0 / g["W"], 0.0)
        lut = np.zeros(ce0.size * ngc + 1)
        lut[g["idx"]] = phic
        phi = np.where(cidx >= 0, lut[np.where(cidx >= 0, cidx, ce0.size * ngc)], 0.0)
        # Psi_1's pooled (all-shell) centroid: Psi_1 + the closure share the cells exactly
        cells.append(dict(W=W, kap=flatk[p["idx"]], gam=flatg[p["idx"]], phi=phi))
    kp = c.xi["kp"]
    chi = np.array([pieces[s]["node"][1] for s in shells])
    dchi = np.array([pieces[s]["node"][2] for s in shells])
    v0 = [np.abs(g0[s]["v0"]) for s in shells]
    rho = np.zeros(max(len(shells) - 1, 0))
    for i in range(rho.size):
        rj = los_rho(cos, kp, chi[i], dchi[i], chi[i + 1], dchi[i + 1])
        v1, v2 = v0[i][:kp.size], v0[i + 1][:kp.size]
        rho[i] = float(np.sum(v1 * v2 * rj) / np.sqrt(np.sum(v1**2) * np.sum(v2**2)))
    rbar = np.array([np.mean([rho[j] for j in (s - 1, s) if 0 <= j < rho.size]) if rho.size else 0.0
                     for s in range(len(cells))])
    fvar = (1 - rbar) / (1 + rbar)
    vk = vg = 0.0
    for g in lam._xg:                        # the pruned groups Lam evaluates (= Lam.sigma's m=0 term)
        if g["m"] == 0:
            a_ = g["rows"] @ g["kp"][g["pinv"]] + g["res"]
            vk += float(a_ @ a_)
        if g["m"] == 1:
            a_ = g["rows"] @ g["gq"][g["qinv"]] + g["res"]
            vg += float(a_ @ a_)
    return dict(cells=cells, shells=shells, rho=rho, fvar=fvar, target_kappa=vk, target_gamma=vg)


# ----------------------------------------------------------------------------- closure
class Closure:
    def __init__(self, pop, nstate=NSTATE, eta=None, h_a=None, kind="vp"):
        """h_a: None = exact per-group rates; > 0 = rates cubic-Lagrange interpolated in the loading a
        on a uniform per-shell grid of spacing h_a (the GEMM runs over the a-nodes, not the groups)."""
        self.pop = pop
        if kind not in KINDS:
            raise ValueError(f"closure kind must be one of {KINDS}, got {kind!r}")
        if h_a and kind != "lognormal":
            raise ValueError("interpolated rates (h_a > 0) are lognormal only; the vp rates have kinks in a")
        self.kind = kind
        self.h_a = h_a
        u, w = np.polynomial.hermite_e.hermegauss(nstate)
        self.u, self.w = u, w / w.sum()
        self.K = [transition(self.u, self.w, float(r)) for r in pop["rho"]]
        self.shear_var = float(pop["target_gamma"])
        cells = pop["cells"]
        self._grp = []
        for s, cl in enumerate(cells):
            ph, gidx = np.unique(cl["phi"], return_inverse=True)
            self._grp.append((ph, gidx.astype(np.int64), np.bincount(gidx, cl["W"] * cl["kap"], minlength=ph.size)))
        self.eta = self.calibrate(pop["target_kappa"]) if eta is None else float(eta)
        kap = np.concatenate([cl["kap"] for cl in cells])
        gam = np.concatenate([cl["gam"] for cl in cells])
        uu, inv = np.unique(np.stack([kap, gam], 1), axis=0, return_inverse=True)
        self.ku, self.gu = uu[:, 0].copy(), uu[:, 1].copy()
        off = np.r_[0, np.cumsum([len(cl["W"]) for cl in cells])]
        self.sh = []
        self.shT = []
        for s, cl in enumerate(cells):
            ph, gidx, _ = self._grp[s]
            Dg = group_rates(ph, self.eta, pop["fvar"][s], self.u, self.w, self.kind) - 1.0
            uidx = inv[off[s]:off[s + 1]].astype(np.int64)
            self.sh.append((np.ascontiguousarray(uidx), gidx,
                            np.ascontiguousarray(cl["W"]), np.ascontiguousarray(Dg), ph.size))
            o = np.argsort(gidx, kind="stable")
            gstart = np.searchsorted(gidx[o], np.arange(ph.size + 1)).astype(np.int64)
            if h_a:
                a = loading(ph, self.eta, pop["fvar"][s])            # sorted (ph sorted, map monotone)
                nodes, j0, Lw = lagrange_stencil(a, h_a)
                Dn = node_rates(nodes, self.u, self.w) - 1.0
                glo = np.searchsorted(j0, np.arange(nodes.size) - 3).astype(np.int64)
                ghi = np.searchsorted(j0, np.arange(nodes.size), side="right").astype(np.int64)
                self.shT.append((np.ascontiguousarray(uidx[o]), np.ascontiguousarray(cl["W"][o]), gstart,
                                 np.ascontiguousarray(Dn.T), (glo, ghi, j0, Lw)))
            else:
                self.shT.append((np.ascontiguousarray(uidx[o]), np.ascontiguousarray(cl["W"][o]), gstart,
                                 np.ascontiguousarray(Dg.T), None))
        self._blocked_setup()
        self.seconds = 0.0

    def _blocked_setup(self):
        """Concatenated cell / group / node arrays for _block_nodes (exact groups when h_a is None)."""
        cu, cw, gst, shg, j0s, Lws, shn, Dts = [], [], [np.zeros(1, np.int64)], [0], [], [], [0], []
        c_off = 0
        for (cu_s, cw_s, gstart, DgT, itp) in self.shT:
            cu.append(cu_s); cw.append(cw_s)
            gst.append(gstart[1:] + c_off); c_off += cu_s.size
            G = gstart.size - 1
            shg.append(shg[-1] + G)
            if itp is None:
                j0s.append(np.arange(G, dtype=np.int64))
                L = np.zeros((G, 4)); L[:, 0] = 1.0; Lws.append(L)
                nn = G
            else:
                j0s.append(itp[2]); Lws.append(itp[3]); nn = itp[0].size
            shn.append(shn[-1] + nn)
            Dts.append(DgT)
        self._B = dict(cu=np.concatenate(cu), cw=np.concatenate(cw), gstart=np.concatenate(gst).astype(np.int64),
                       sh_g=np.array(shg, np.int64), j0=np.concatenate(j0s).astype(np.int64),
                       Lw=np.ascontiguousarray(np.concatenate(Lws)), sh_node=np.array(shn, np.int64), DT=Dts)
        self._Karr = np.ascontiguousarray(np.array(self.K)) if self.K else np.zeros((0, len(self.w), len(self.w)))

    def lnR_blocked(self, kv, qv, nkb=8, nqb=32, max_bytes=256e6):
        """lnR on the tensor grid kv x qv by (nkb x nqb) point blocks: register-blocked group sums onto
        rate nodes (_block_nodes), one BLAS GEMM per shell (rates), numba chain recursion."""
        kv, qv = np.atleast_1d(np.asarray(kv, float)), np.atleast_1d(np.asarray(qv, float))
        B = self._B
        ntot = int(B["sh_node"][-1])
        while nkb * nqb > 1 and ntot * nkb * nqb * 16 > max_bytes:
            if nkb >= nqb:
                nkb = max(1, nkb // 2)
            else:
                nqb = max(1, nqb // 2)
        nsh, ns = len(B["DT"]), len(self.w)
        out = np.empty((kv.size, qv.size), complex)
        for i0 in range(0, kv.size, nkb):
            kb = kv[i0:i0 + nkb]
            e = np.expm1(1j * np.outer(self.ku, kb))
            er, ei = np.ascontiguousarray(e.real), np.ascontiguousarray(e.imag)
            for j0_ in range(0, qv.size, nqb):
                qb = qv[j0_:j0_ + nqb]
                Jb = np.ascontiguousarray(j0(np.outer(self.gu, qb)))
                Jm1 = Jb - 1.0
                Zr = np.empty((ntot, kb.size, qb.size)); Zi = np.empty_like(Zr)
                _block_nodes(er, ei, Jb, Jm1, B["cu"], B["cw"], B["gstart"], B["sh_g"], B["j0"], B["Lw"],
                             B["sh_node"], Zr, Zi)
                P = kb.size * qb.size
                zr = np.empty((nsh, ns, P)); zi = np.empty((nsh, ns, P))
                for s_ in range(nsh):
                    a, b = B["sh_node"][s_], B["sh_node"][s_ + 1]
                    D = B["DT"][s_]
                    zr[s_] = D @ Zr[a:b].reshape(b - a, P)
                    zi[s_] = D @ Zi[a:b].reshape(b - a, P)
                o_r, o_i = _chain_recursion(zr, zi, self._Karr, self.w)
                out[i0:i0 + kb.size, j0_:j0_ + qb.size] = (o_r + 1j * o_i).reshape(kb.size, qb.size)
        return out

    def variance(self, eta):
        """Model kappa clustering variance. With h_a: m_s(u) = sum_g wk_g lam(u, a_g) from the rates on
        the a-grid (cubic Lagrange, as lnR); exact per-group rates otherwise."""
        ms = []
        for (ph, _, wk), f in zip(self._grp, self.pop["fvar"]):
            if self.h_a:
                nodes, j0, Lw = lagrange_stencil(loading(ph, eta, f), self.h_a)
                cn = np.zeros(nodes.size)
                for m in range(4):
                    np.add.at(cn, j0 + m, Lw[:, m] * wk)
                ms.append(node_rates(nodes, self.u, self.w).T @ cn)
            else:
                ms.append(group_rates(ph, eta, f, self.u, self.w, self.kind).T @ wk)
        return chain_variance(ms, self.w, self.K)

    def calibrate(self, target, lo=1e-3, hi=1e4):
        """eta with model kappa clustering variance = target (brentq in ln eta; the bracket starts at
        [1/1.5, 1.5] around eta = 1 and widens x1.5 per side until it brackets, within [lo, hi])."""
        from scipy.optimize import brentq
        fun = lambda le: self.variance(np.exp(le)) - target
        a, b = -np.log(1.5), np.log(1.5)
        fa, fb = fun(a), fun(b)
        while fa * fb > 0:
            if fa > 0 and a > np.log(lo):          # variance increases with eta
                b, fb = a, fa
                a -= np.log(1.5); fa = fun(a)
            elif fb < 0 and b < np.log(hi):
                a, fa = b, fb
                b += np.log(1.5); fb = fun(b)
            else:
                raise ValueError(f"positive closure: kappa clustering target {target:.3e} outside [{lo}, {hi}]")
        return float(np.exp(brentq(fun, a, b, xtol=2e-12)))

    def _recur(self, Fr, Fi):
        n = Fr.shape[0]
        alpha = np.broadcast_to(self.w[:, None], (len(self.w), n)).astype(complex)
        scale = np.zeros(n)
        for s, (uidx, gidx, W, Dg, G) in enumerate(self.sh):
            Yr, Yi = _group_sums(Fr, Fi, uidx, gidx, W, G)
            z = (Yr @ Dg).T + 1j * (Yi @ Dg).T
            if s:
                alpha = self.K[s - 1].T @ alpha
            mx = z.real.max(axis=0)
            alpha = alpha * np.exp(z - mx)
            scale += mx
            nrm = np.max(np.abs(alpha), axis=0)
            alpha /= nrm
            scale += np.log(nrm)
        return scale + np.log(alpha.sum(axis=0))

    def _recurT(self, Fr, Fi, sums=None, n=None):
        """_recur on cell-major F (U, n): group sums by contiguous AXPYs, z = Dg^T Y (48, n).
        sums(s, Yr, Yi) overrides the group-sum kernel (fused row kernels)."""
        n = Fr.shape[1] if n is None else n
        alpha = np.broadcast_to(self.w[:, None], (len(self.w), n)).astype(complex)
        scale = np.zeros(n)
        Gmax = max(x[2].size - 1 for x in self.shT)
        Yr_b, Yi_b = np.empty((Gmax, n)), np.empty((Gmax, n))
        for s, (cu, cw, gstart, DgT, itp) in enumerate(self.shT):
            G = gstart.size - 1
            Yr, Yi = Yr_b[:G], Yi_b[:G]
            if sums is None:
                _group_sums_T(Fr, Fi, cu, cw, gstart, Yr, Yi)
            else:
                sums(cu, cw, gstart, Yr, Yi)
            if itp is not None:
                glo, ghi, j0, Lw = itp
                Zr, Zi = np.empty((glo.size, n)), np.empty((glo.size, n))
                _node_agg(Yr, Yi, glo, ghi, j0, Lw, Zr, Zi)
                Yr, Yi = Zr, Zi
            z = (DgT @ Yr) + 1j * (DgT @ Yi)
            if s:
                alpha = self.K[s - 1].T @ alpha
            mx = z.real.max(axis=0)
            alpha = alpha * np.exp(z - mx)
            scale += mx
            nrm = np.max(np.abs(alpha), axis=0)
            alpha /= nrm
            scale += np.log(nrm)
        return scale + np.log(alpha.sum(axis=0))

    def lnR(self, kv, qv):
        """ln E_U exp(sum (lam - 1) W F) on the tensor grid kv x qv: one fused pass per k_kappa row
        (all k_gamma at once), or per k_gamma column when the grid has fewer columns than rows."""
        kv, qv = np.atleast_1d(np.asarray(kv, float)), np.atleast_1d(np.asarray(qv, float))
        out = np.empty((kv.size, qv.size), complex)
        if qv.size >= kv.size:
            J = np.ascontiguousarray(j0(np.outer(self.gu, qv)))           # (U, nq)
            for i, k in enumerate(kv):
                e = np.expm1(1j * k * self.ku)
                er, ei = np.ascontiguousarray(e.real), np.ascontiguousarray(e.imag)
                sums = lambda cu, cw, gs, Yr, Yi: _group_sums_krow(er, ei, J, cu, cw, gs, Yr, Yi)
                out[i] = self._recurT(None, None, sums=sums, n=qv.size)
        else:
            E = np.expm1(1j * np.outer(self.ku, kv))                      # (U, nk)
            Er, Ei = np.ascontiguousarray(E.real), np.ascontiguousarray(E.imag)
            for j, q in enumerate(qv):
                jq = np.ascontiguousarray(j0(q * self.gu))
                sums = lambda cu, cw, gs, Yr, Yi: _group_sums_qrow(Er, Ei, jq, cu, cw, gs, Yr, Yi)
                out[:, j] = self._recurT(None, None, sums=sums, n=kv.size)
        return out

    def lnR_cellmajor(self, kv, qv, chunk=128):
        """ln E_U exp(sum (lam - 1) W F) on the tensor grid kv x qv (cell-major kernel, F materialised)."""
        kv, qv = np.atleast_1d(np.asarray(kv, float)), np.atleast_1d(np.asarray(qv, float))
        E = np.expm1(1j * np.outer(self.ku, kv))                 # (U, nk)
        J = j0(np.outer(self.gu, qv))                             # (U, nq)
        I, Jq = [a.ravel() for a in np.meshgrid(np.arange(kv.size), np.arange(qv.size), indexing="ij")]
        out = np.empty(I.size, complex)
        for lo in range(0, I.size, chunk):
            ii, jj = I[lo:lo + chunk], Jq[lo:lo + chunk]
            b = J[:, jj]
            f = E[:, ii] * b + (b - 1)
            out[lo:lo + ii.size] = self._recurT(np.ascontiguousarray(f.real), np.ascontiguousarray(f.imag))
        return out.reshape(kv.size, qv.size)

    def lnR_ref(self, kv, qv, chunk=128):
        """Reference (prototype GroupLnR) evaluation, point-major gather kernel."""
        kv, qv = np.atleast_1d(np.asarray(kv, float)), np.atleast_1d(np.asarray(qv, float))
        E = np.expm1(1j * np.outer(kv, self.ku))
        J = j0(np.outer(qv, self.gu))
        I, Jq = [a.ravel() for a in np.meshgrid(np.arange(kv.size), np.arange(qv.size), indexing="ij")]
        out = np.empty(I.size, complex)
        for lo in range(0, I.size, chunk):
            ii, jj = I[lo:lo + chunk], Jq[lo:lo + chunk]
            b = J[jj]
            f = E[ii] * b + (b - 1)
            out[lo:lo + ii.size] = self._recur(np.ascontiguousarray(f.real), np.ascontiguousarray(f.imag))
        return out.reshape(kv.size, qv.size)

    def qxi(self, kk, kg):
        """lnR - shear_var k_g^2 / 4 on the tensor grid kk x kg, with Im lnR UNWRAPPED (principal-branch
        jumps of 2 pi removed along k_kappa at the first k_gamma, then along k_gamma): Phi is unchanged
        at the nodes, but Lam's coarse-grid spline interpolates a continuous phase (2026-10-17: the
        principal branch wrapped ~once per k_gamma column on invert's grid, at |Phi| <~ 4e-7)."""
        t = time.time()
        kk, kg = np.atleast_1d(kk), np.atleast_1d(kg)
        v = self.lnR_blocked(kk, kg)
        if getattr(self, "unwrap", True):
            im = v.imag.copy()
            im[:, 0] = np.unwrap(im[:, 0])
            im = np.unwrap(im, axis=1)
            v = v.real + 1j * im
        v = v - 0.25 * self.shear_var * kg[None, :]**2
        self.seconds += time.time() - t
        return v


def install(lam, cos, kind="vp", nstate=NSTATE, h_a="auto"):
    """Build the closure from lam's population and route Lam._Qxi through it."""
    t = time.time()
    lam._cos = cos
    pop = population(lam, cos)
    if h_a == "auto":
        h_a = H_A if kind == "lognormal" else None
    lam.pos = Closure(pop, nstate=nstate, h_a=h_a, kind=kind)
    lam.pos.eval_kernel = "grouped"
    lam.pos.eval_kernel_setup_seconds = 0.0
    lam.pos.eval_kernel_fallback = None
    if kind == "vp" and not h_a:
        try:
            from .pos_closure_fused import FusedVP
            fused = FusedVP(lam.pos)
        except (ImportError, ValueError) as exc:
            # Retain the original evaluation if the optional kernel is unavailable
            # or the exact-group rates fail its reconstruction accuracy bound.
            lam.pos.eval_kernel_fallback = str(exc)
        else:
            lam.pos.lnR_blocked = fused.lnR_blocked
            lam.pos.eval_kernel = "vp_fused"
            lam.pos.eval_kernel_setup_seconds = fused.setup_seconds
    lam.xi_coarse_rule = True          # invert scales the clustering coarse grid with the box
    lam.pos.setup_seconds = time.time() - t
    return lam.pos

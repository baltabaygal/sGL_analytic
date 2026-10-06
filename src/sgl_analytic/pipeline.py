#!/usr/bin/env python
"""Semi-analytic magnification PDF of the default lens model.

`run(zs, "full", cosmo=dict(h=, Om=, s8=, Ok=, w0=, wa=))` returns (out, meta): `out["xi"]`,
`out["P_s"]` = source-plane dP/d ln mu on ln mu in [-1, 1], `out["xi_tail"]`,
`out["P_s_tail"]` on ln mu in (1, 9], plus moments, sigma_kappa and stage timings.
Config strings: "full" (default), "halo", "+ell", "+fil", "+sub", "+bias", "-<x>".
"""
from __future__ import annotations

import argparse
import inspect
import os
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from . import sgl
from . import sgl_full as F

P = dict(h=0.674, Om=0.315, s8=0.811, Ob=0.0493, ns=0.965, zeq=3402.0)
# Curvature and CPL dark energy (2026-10-06). Kept out of P on purpose: P is
# the Planck LCDM reference that config.COSMO and the stored PDFs refer to.
#   Ok: Omega_K (> 0 open, < 0 closed); Omega_DE = 1 - Om - Ok.
#   w0, wa: w(a) = w0 + wa (1 - a); (-1, 0) is a cosmological constant.
#   growth_mode: "auto" = the historical LCDM growth for any Lambda model
#     (exact integral for sigma(M), Carroll-Press-Turner in bias and M*) and
#     the GR growth ODE otherwise; "ode" = the ODE for every model, including
#     Lambda. "auto" therefore jumps at w = -1 (1.9e-4 in the z_s = 1 clipped
#     variance): compare a w model with an "ode" Lambda run.
EXTENSIONS = dict(Ok=0.0, w0=-1.0, wa=0.0, growth_mode="auto")


def make_cosmology(cosmo=None):
    """The chain's Cosmology: EH98 transfer, smooth-k sigma(M) window, top-hat
    sigma_8 anchor, Ludlow+16 concentrations.  `cosmo` may set h, Om, s8 (P
    otherwise), Ob, ns, zeq and the EXTENSIONS Ok, w0, wa, growth_mode.
    sigma8 is accepted as an alias for s8."""
    supplied = dict(cosmo or {})
    if "sigma8" in supplied:
        if "s8" in supplied:
            raise ValueError("supply sigma8 or s8, not both")
        supplied["s8"] = supplied.pop("sigma8")
    cp = {**P, **EXTENSIONS, **supplied}
    bad = set(cp) - set(P) - set(EXTENSIONS)
    if bad:
        raise ValueError(f"unknown cosmology parameter(s) {sorted(bad)}")
    for key in P:
        value = cp[key]
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    for key in ("Ok", "w0", "wa"):
        value = cp[key]
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    if cp["growth_mode"] not in ("auto", "ode", "legacy"):
        raise ValueError("growth_mode must be 'auto', 'ode' or 'legacy'")
    if not 0 < cp["Ob"] < cp["Om"] <= 1:
        raise ValueError("require 0 < Ob < Om <= 1")
    # sgl.Cosmology enforces the remaining domain: -2 <= w0 < -0.3, w0 + wa <
    # -0.3 with negligible early dark energy, E(z)^2 > 0, legacy only for Lambda
    return sgl.Cosmology(window="smoothk", transfer="eh98", anchor="tophat",
                         conc_model=16, h=cp["h"], Om=cp["Om"], s8=cp["s8"],
                         Ob=cp["Ob"], ns=cp["ns"], zeq=cp["zeq"],
                         Ok=cp["Ok"], w0=cp["w0"], wa=cp["wa"],
                         growth_mode=cp["growth_mode"])


INGREDIENTS = ("ell", "fil", "sub", "bias")


def on(cfg):
    """Lens ingredients switched on, on top of spherical NFW halos.
    cfg: "full", "halo", "+x" (halos + x), "-x" (full minus x), or any
    iterable of names from INGREDIENTS (e.g. {"ell", "sub"})."""
    ing = set(INGREDIENTS)
    if not isinstance(cfg, str):
        s = set(cfg)
        if s - ing:
            raise ValueError(f"unknown ingredient(s) {sorted(s - ing)}; choose from {INGREDIENTS}")
        return s
    if cfg == "halo":
        return set()
    if cfg == "full":
        return ing
    if len(cfg) < 2 or cfg[0] not in "+-" or cfg[1:] not in ing:
        raise ValueError(f"unknown configuration {cfg!r}; use full, halo, or +/-{INGREDIENTS}")
    return {cfg[1:]} if cfg[0] == "+" else ing - {cfg[1:]}


def cfg_name(cfg):
    """Readable label of a lens configuration (string or ingredient set)."""
    if isinstance(cfg, str):
        return cfg
    s = on(cfg)
    return "full" if s == set(INGREDIENTS) else ("halo" if not s else
                                                 "halo+" + "+".join(i for i in INGREDIENTS if i in s))


def to_image_plane(xi, P_s):
    """Image-plane dP_I/d ln mu from the source-plane PDF on the grid xi = ln mu:
    dP_I prop. to mu dP_S (Eq. PS inverted), normalised on the given grid."""
    xi, P_s = np.asarray(xi, float), np.asarray(P_s, float)
    P = np.exp(xi) * P_s
    return P / np.trapezoid(P, xi)


def run(zs, cfg, nord=4, grid_scale=1.0, xi_out=None, split=10.0,
        exact_s=True, k3=True, nproc=1, sub_kw=None, s_taper="cos2", cosmo=None,
        tail_box=True, xi_tail=None, s_is=True, s_seed=None, q_coarse=101,
        clustering="xilin", edge="rvir", kap_floor=1e-8, phi_tol=1e-8,
        lam_coarse=0, zint="midpoint", zint_n=2, xi_kw=None, keep=False):
    # keep=True: meta also carries the intermediate objects (cos, sub, xi, cells,
    # info, lam) so each stage can be inspected (the notebook uses this)
    # zint (2026-10-05): lens-redshift quadrature, sgl_full.z_shells. "engine" =
    # the MC's shells (upper-node backward difference; default until 2026-10-05,
    # ~3% less lensing weight); "midpoint" (DEFAULT since 2026-10-05, user) /
    # "gauss" = shells covering [0, z_s], 1 / zint_n Gauss-Legendre nodes each.
    # clustering (2026-10-02): "linear" = DEFAULT, the closed-form linear-bias
    # two-point term 1/2 int P |J|^2 (Lam nord=1, no k3, no exact-S): no
    # sampling anywhere in the chain.  "closure" = the older lognormal Hermite
    # closure + exact-S, controlled by nord / exact_s / k3 / s_* as before.
    # "limber" (2026-10-02) = the halo-model two-halo term: a Gaussian
    # large-scale-structure convergence and shear with the Limber variance of
    # the linear P (F.limber_lss_var), replacing the engine's 20 Mpc 1D field.
    # Meant with edge="rvir" (finite halo masses); matches HMcode-2020 sigma_kappa
    # to +-3% there (tmp/hmcode_check.py).
    # "xilin" (2026-10-05, DEFAULT, user/supervisor): the pair term with the 3D
    # linear correlation function, xi_hh = b1 b2 xi_lin(r12), in Limber form --
    # ONE line-of-sight integral, no z shells and no C_ij (F.xi_kperp_grid).
    # "linear" = the engine's 20 Mpc 1D field C_ij (the default 2026-10-02..05).
    if clustering in ("linear", "limber", "xilin"):
        nord, k3, exact_s = 1, False, False
    elif clustering != "closure":
        raise ValueError(f"clustering must be 'xilin', 'linear', 'limber' or 'closure', "
                         f"got {clustering!r}")
    if not np.isscalar(zs) or not np.isfinite(zs) or zs <= 0:
        raise ValueError("z_s must be a finite positive number")
    if isinstance(nproc, bool) or not isinstance(nproc, (int, np.integer)) or nproc < 1:
        raise ValueError("nproc must be a positive integer")
    if edge not in ("rvir", "kthr"):
        raise ValueError("edge must be 'rvir' or 'kthr'")
    s = on(cfg)
    cos = make_cosmology(cosmo)
    if (cosmo or {}).get("growth_mode", "auto") == "auto" and cos.growth_mode == "ode":
        warnings.warn("growth_mode='auto' resolved to 'ode' for non-Lambda dark energy, "
                      "while Lambda runs (and the stored references) use the historical "
                      "growth; compare against a growth_mode='ode' Lambda run.", stacklevel=2)
    t0 = time.time()
    # defaults (2026-10-02) = the CLI defaults: exact moment sector, clumps
    # truncated at their r_vir; explicit sub_kw entries override them
    skw = dict(sector="exact", clump_edge="rvir", exact_kw=dict(method="moments"))
    skw.update(sub_kw or {})
    # kthr only for the engine split (edge "kthr") or a kthr-based subhalo law
    # (pairs sector / clump_edge "kthr"); the default chain has none (2026-10-02)
    need_kthr = edge == "kthr" or ("sub" in s and skw["clump_edge"] != "rvir")
    kthr = F.kappa_threshold(cos, zs) if need_kthr else None
    sub = None
    if "sub" in s:
        from . import subhalos
        sub = subhalos.SubhaloModel(cos, zs, kthr, split=split, **skw)
    # edge (2026-10-02): "kthr" = the engine split (explicit halos to r_max,
    # Gaussian faint-lens arm beyond); "rvir" = every NFW halo truncated at its
    # virial radius and integrated exactly to the kap_floor of its shear tail
    # (no kthr / r_max / diffusion term; the whole truncated halo is the host).
    ekw = {} if edge == "kthr" else dict(kap_floor=kap_floor, trunc="rvir")
    use_xi = clustering == "xilin" and "bias" in s
    if use_xi and edge == "kthr":
        raise ValueError("clustering='xilin' needs edge='rvir'")
    xi = F.xi_kperp_grid(cos, **(xi_kw or {})) if use_xi else None
    cells, info = F.build_population(cos, zs, ell="ell" in s, fils="fil" in s,
                                     nord=nord, kthr=kthr, subhalo=sub, nproc=nproc,
                                     zint=zint, zint_n=zint_n, xi=xi, **ekw)
    t_build = time.time()
    limber = clustering == "limber" and "bias" in s
    C = (F.shell_covariance(cos, zs, zint=zint, zint_n=zint_n)
         if ("bias" in s and not limber and not use_xi) else None)
    lss_var = F.limber_lss_var(cos, zs) if limber else 0.0
    # s_seed=None -> Lam's own default seed (0): default runs are unchanged
    lam = F.Lam(cells, C=C, nord=nord, exactS=exact_s, k3=k3, s_taper=s_taper, s_is=s_is,
                q_coarse=q_coarse, lss_var=lss_var, lam_coarse=lam_coarse, xi=use_xi,
                **({} if s_seed is None else dict(seed=int(s_seed))))
    sig = lam.sigma()
    t_lam = time.time()
    # one inversion, one normalisation: the clipped grid and the tail grid are
    # evaluated together and split afterwards
    # xi_out=None (2026-10-05): the CLI grids, so run(zs, cfg) works without
    # arguments (it raised TypeError on len(None) before)
    if xi_out is None:
        xi_out = np.linspace(-1.0, 1.0, 1601)
        if xi_tail is None:
            xi_tail = np.linspace(1.005, 9.0, 1600)
    nx = len(xi_out)
    xall = xi_out if xi_tail is None else np.concatenate([xi_out, xi_tail])
    # phi_tol (2026-10-02): k-reach |Phi| <= phi_tol.  1e-8 (default) vs the
    # old 1e-10 at z_s=1: body max|dP| 5e-8 of peak, var 2e-8, tail (ln mu 1-9)
    # <= 7e-4 relative, 36% fewer k-points (tmp/reach_test.py, log alongside)
    # inversion stays in-process (2026-10-03): its Lambda GEMMs are already
    # BLAS-threaded, and a process pool on top (nproc > 1) oversubscribed the
    # cores -- z_s=3 invert 278 s vs 8 s serial, PDF identical.  nproc is for
    # the build only.
    xa, Pa, meta = F.invert(lam, sig, xi_out=xall, grid_scale=grid_scale,
                            tail_box=tail_box, phi_tol=phi_tol, nproc=1)
    xo, Ps = xa[:nx], Pa[:nx]
    t_inv = time.time()
    out = dict(t_stages=dict(build=t_build - t0, lam=t_lam - t_build, invert=t_inv - t_lam),zs=zs, cfg=cfg_name(cfg),
               zint=zint, zint_n=(zint_n if zint == "gauss" else None), clustering=clustering,
               xi_kperp=({k: xi[k] for k in ("kmin", "kmax", "per_decade", "mmax")} if xi else None), edge=edge, lss_var=lss_var,
               kap_floor=(kap_floor if edge != "kthr" else None), kbar=info["kbar"], h=cos.h, Om=cos.Om, s8=cos.s8, Ob=cos.Ob, ns=cos.ns, zeq=cos.zeq, sigma8=cos.s8,
               Ok=cos.Ok, w0=cos.w0, wa=cos.wa, growth_mode=cos.growth_mode, nord=nord, grid_scale=grid_scale, split=split,
               nproc=nproc, sub_kw=(sub._init_kw if sub is not None else None),
               pair_angle=(sub.pair_angle if sub is not None else None),
               k3=lam.k3, exactS=bool(lam.exactS), s_taper=lam.s_taper,
               sigma_kappa=sig, kthr=kthr, N_expl=info["N_expl"],
               N_fil=info["N_fil"], V_weak=info["V_weak"], mean=meta["mean"],
               var=meta["var"], skew=meta["skew"], sigmaDL=meta["sigmaDL"],
               grid={k: v for k, v in meta["grid"].items()
                     if isinstance(v, (int, float))},
               tail_box=bool(tail_box), fine=dict(F.FINE), s_is=bool(s_is),
               s_seed=(int(s_seed) if s_seed is not None else None),
               s_seed_effective=(int(s_seed) if s_seed is not None
                                 else int(inspect.signature(F.Lam.__init__).parameters["seed"].default)),
               s_ess=getattr(lam, "S_ess", None),
               is_design=getattr(lam, "is_design", None) if s_is else None,
               seconds=time.time() - t0, xi=xo.tolist(), P_s=Ps.tolist())
    if xi_tail is not None:
        out.update(xi_tail=xa[nx:].tolist(), P_s_tail=Pa[nx:].tolist())
    if keep:
        meta.update(cos=cos, sub=sub, xi=xi, cells=cells, info=info, lam=lam)
    return out, meta


def clipped_moments(xi, P, clip=1.0):
    m = np.abs(xi) <= clip
    x, p = xi[m], P[m]
    p = p / np.trapezoid(p, x)
    mu = np.trapezoid(x * p, x)
    v = np.trapezoid((x - mu)**2 * p, x)
    return dict(mean=float(mu), var=float(v),
                skew=float(np.trapezoid((x - mu)**3 * p, x) / v**1.5))

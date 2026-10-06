"""Convenience API using the same cosmological parameter names as FLUMEN,
plus keyword-only curvature (Ok) and CPL dark energy (w0, wa)."""
import warnings

import numpy as np

from .pipeline import run


def _grid(values, lo, hi, count, name):
    if values is None:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 2:
            raise ValueError(f"n_{name} must be an integer >= 2")
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"{name}_min and {name}_max must be finite and increasing")
        values = np.linspace(lo, hi, count)
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite one-dimensional grid with >= 2 points")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def generate_pdf_lnmu(z_s, h=0.674, Om=0.315, sigma8=0.811, Ob=0.0493,
                       ns=0.965, zeq=3402.0, lnmu=None, *, Ok=0.0, w0=-1.0,
                       wa=0.0, growth_mode="auto", lnmu_min=-1.0,
                       lnmu_max=9.0, n_lnmu=3201, config="full", nproc=1,
                       return_info=False):
    """Return (lnmu, dP_S/dlnmu) as NumPy arrays.

    z_s is required. h=H0/100; Om and Ob are density fractions; sigma8 is
    the real-space top-hat amplitude at 8/h Mpc; ns is the scalar spectral
    index; zeq is the matter-radiation equality redshift (unscaled).

    Keyword-only background extensions (defaults: flat LCDM). Ok is the
    curvature density (positive open, negative closed); dark energy has
    Omega_DE = 1 - Om - Ok and the CPL equation of state w(a) = w0 + wa(1-a).
    growth_mode="auto" keeps the historical LCDM growth for any Lambda model
    and uses the GR growth ODE otherwise, which leaves a small jump at w = -1;
    "ode" uses the ODE for every model. Compare a dark-energy model with a
    Lambda run made with growth_mode="ode".

    Supply a finite, strictly increasing lnmu grid, or use the grid bounds
    and count. config accepts full, halo, +ell/+fil/+sub/+bias and the minus
    variants. nproc controls population workers; the default works in
    notebooks without process spawning. Scripts using nproc>1 need a
    __main__ guard.

    The density retains the solver's normalization and numerical residuals;
    it is not renormalized to a custom grid. Small negative inversion
    residuals emit a warning and remain visible. return_info=True adds a
    third return value containing JSON-compatible model metadata,
    restricted-interval moments, timings and grid diagnostics.
    """
    x = _grid(lnmu, lnmu_min, lnmu_max, n_lnmu, "lnmu")
    out, _ = run(z_s, config, cosmo=dict(h=h, Om=Om, sigma8=sigma8,
                                       Ob=Ob, ns=ns, zeq=zeq, Ok=Ok, w0=w0,
                                       wa=wa, growth_mode=growth_mode),
                 xi_out=x, nproc=nproc)
    p = np.asarray(out.pop("P_s"))
    out.pop("xi")
    if not np.all(np.isfinite(p)):
        raise RuntimeError("PDF inversion returned non-finite values")
    negative_mass = float(np.trapezoid(np.maximum(-p, 0), x))
    out["diagnostics"] = dict(grid_mass=float(np.trapezoid(p, x)),
                              negative_mass=negative_mass,
                              minimum_density=float(p.min()))
    out["moment_interval"] = [out["grid"]["xi_min"], out["grid"]["xi_max"]]
    if negative_mass > 0:
        warnings.warn(f"PDF inversion has negative residuals (integrated mass "
                      f"{negative_mass:.3g}); inspect diagnostics before using tail probabilities.",
                      RuntimeWarning, stacklevel=2)
    return (x, p, out) if return_info else (x, p)


def generate_pdf(z_s, h=0.674, Om=0.315, sigma8=0.811, Ob=0.0493,
                 ns=0.965, zeq=3402.0, mu=None, *, Ok=0.0, w0=-1.0, wa=0.0,
                 growth_mode="auto", mu_min=0.3, mu_max=30.0,
                 n_mu=2000, config="full", nproc=1, return_info=False):
    """Return (mu, p_S(mu)); cosmological inputs match generate_pdf_lnmu.

    The default mu grid is logarithmically spaced. Density is with respect
    to dmu, using p(mu) = (dP/dlnmu)/mu. Finite grids can omit probability
    mass; this function does not renormalize to the requested interval.
    return_info=True adds solver metadata and diagnostics in lnmu.
    """
    if mu is None:
        if mu_min <= 0 or mu_max <= 0:
            raise ValueError("mu bounds must be positive")
        x = _grid(None, np.log(mu_min), np.log(mu_max), n_mu, "mu")
        mu = np.exp(x)
    else:
        mu = _grid(mu, 0, 1, n_mu, "mu")
        if np.any(mu <= 0):
            raise ValueError("mu must be positive")
        x = np.log(mu)
    result = generate_pdf_lnmu(z_s, h, Om, sigma8, Ob, ns, zeq, x, Ok=Ok, w0=w0,
                               wa=wa, growth_mode=growth_mode, config=config,
                               nproc=nproc, return_info=return_info)
    return (mu, result[1] / mu, result[2]) if return_info else (mu, result[1] / mu)

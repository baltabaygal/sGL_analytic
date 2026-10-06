# sGL analytic

Sampling-free stochastic weak-lensing magnification PDFs for a source redshift
and a LCDM cosmology, optionally with spatial curvature and time-varying (CPL)
dark energy. The solver builds a lens population, evaluates its
compensated compound-Poisson characteristic function with a two-point clustering
correction, and inverts it to a source-plane PDF. A calculation takes roughly
10–30 seconds on an M-series laptop; this is a numerical solver, not an emulator.

## Install from this repository

Requires Python 3.12 or newer. In a virtual environment, from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

For development use `python -m pip install -e '.[test,notebooks]'`.
The distribution is named `sgl-analytic`; the Python import is `sgl_analytic`.
This repository has not been published to PyPI. For the exact numerical reference
environment, use `conda env create -f environment.yml`, activate `sgl_analytic`,
then install the package. Reference versions: NumPy 2.4.3, SciPy 1.17.1,
Numba 0.65.1. The first calculation may take longer while Numba compiles kernels.

## Python quick start

```python
from sgl_analytic import generate_pdf, generate_pdf_lnmu

# p_S(mu): density with respect to dmu
mu, pdf = generate_pdf(z_s=1.0)

# dP_S/dlnmu, with all six cosmological inputs
lnmu, pdf_lnmu, info = generate_pdf_lnmu(
    z_s=1.0, h=0.674, Om=0.315, sigma8=0.811,
    Ob=0.0493, ns=0.965, zeq=3402.0,
    return_info=True,
)
print(info['sigma_kappa'], info['diagnostics'])

# open universe with evolving dark energy, w(a) = -1 + 0.3 (1 - a)
lnmu, pdf_lnmu = generate_pdf_lnmu(z_s=1.0, Ok=0.05, w0=-1.0, wa=0.3)
```

Both functions return two NumPy arrays. `return_info=True` adds a third value
with JSON-compatible cosmology, model settings, moments, timings and diagnostics.
A runnable export example is in `examples/quickstart.py`.

### Cosmological inputs

The public names and defaults match FLUMEN:

| Parameter | Meaning | Default |
|---|---|---:|
| `z_s` | Source redshift; required | — |
| `h` | Hubble constant divided by 100 km/s/Mpc | 0.674 |
| `Om` | Total matter density fraction, including baryons | 0.315 |
| `sigma8` | Real-space top-hat amplitude at 8/h Mpc | 0.811 |
| `Ob` | Baryon density fraction | 0.0493 |
| `ns` | Scalar spectral index | 0.965 |
| `zeq` | Matter–radiation equality redshift, without scaling | 3402.0 |

Keyword-only background extensions (not FLUMEN inputs):

| Parameter | Meaning | Default |
|---|---|---:|
| `Ok` | Curvature density Omega_K; positive is open, negative closed | 0.0 |
| `w0` | Dark-energy equation of state today | -1.0 |
| `wa` | CPL slope: w(a) = w0 + wa (1 - a) | 0.0 |
| `growth_mode` | Linear growth: `"auto"` or `"ode"` (see below) | `"auto"` |

The six FLUMEN inputs must be finite and positive, with 0 < Ob < Om <= 1; `Ok`,
`w0` and `wa` must be finite. The CMB temperature is held at 2.7255 K. There
are no neutrino-mass inputs. The transfer function is EH98, with a smooth-k
mass-variance window and a real-space top-hat sigma8 normalization;
concentrations use Ludlow+16. sigma8 is fixed at z = 0 for every background.

### Curvature and dark energy

The background has no radiation. With a = 1/(1+z) and Omega_DE = 1 − Om − Ok,

    E^2(a) = Om a^-3 + Ok a^-2 + Omega_DE a^(-3(1+w0+wa)) exp(3 wa (a − 1)),

so the defaults give flat LCDM. Transverse comoving distances are
f_K(chi) = (c/H0) sinh(sqrt(Ok) H0 chi/c)/sqrt(Ok) for an open universe (sin and
−Ok for a closed one), and the lens-source angular-diameter distance is
f_K(chi_s − chi_l)/(1 + z_s). These enter the critical
surface density, and H(z) enters the critical density and the lens rate per
unit area. The Limber clustering term stays local and flat-sky.

Linear growth is general-relativistic and scale independent, with smooth dark
energy (no dark-energy perturbations). `growth_mode="ode"` solves
D'' + (2 + dlnH/dlna) D' = (3/2) Omega_m(a) D in ln a, with the instantaneous
w(a) in dlnH/dlna. The default `"auto"` keeps the historical LCDM treatment for
any Lambda model, flat or curved: the exact growth integral for sigma(M), and
the Carroll–Press–Turner approximation for halo bias and M*. For any other
w0, wa it uses the ODE everywhere. The two treatments differ slightly, so
`"auto"` jumps at w = −1: the clipped variance of ln mu changes by a relative
2e-4 at z_s = 1.
Compare a dark-energy model with an LCDM run made with `growth_mode="ode"`;
the solver warns when `"auto"` switches to the ODE. The stored reference PDFs
use `"auto"`.

The supported domain is −2 <= w0 < −0.3. When wa ≠ 0 it also requires
w0 + wa < −0.3 and negligible dark energy at a = 1e-6, where growth starts in
the matter era. Cosmologies with E^2 <= 0 at some z <= 1e6 are rejected, as
are closed-universe distances that reach the antipode.

The lens population is not recalibrated for curvature or dark energy. The
Sheth–Tormen mass function, halo bias and Allgood ellipticities respond
through sigma(M, z), growth and H(z). The Ludlow+16 concentration fit
depends on sigma(M) at z = 0 and on z, so at fixed sigma8 it does not respond
to the growth history. The virial radius uses the Bryan–Norman LCDM fit,
evaluated at the instantaneous Omega_m(z). These are extrapolations of
LCDM-calibrated relations, not results for curved or CPL cosmologies.

Stored regression PDFs cover z_s = 0.5, 1, 3 and 10 at the default cosmology.
They establish numerical reproducibility, not accuracy over a parameter box.
FLUMEN's emulator training ranges are not validation ranges for this solver.

### Grids and density conventions

```python
import numpy as np
from sgl_analytic import generate_pdf

mu, p = generate_pdf(z_s=2.0, sigma8=0.9, mu=np.geomspace(0.5, 30, 1000))
```

Custom grids must be finite, one-dimensional and strictly increasing; `mu`
must also be positive. Default `generate_pdf` uses 2,000 logarithmically spaced
points from mu = 0.3 to 30. Default `generate_pdf_lnmu` uses 3,201 uniformly
spaced points from lnmu = −1 to 9. Bounds and counts can be changed with
`mu_min`, `mu_max`, `n_mu` or `lnmu_min`, `lnmu_max`, `n_lnmu`.

The conversion is p_S(mu) = (dP_S/dlnmu)/mu. A finite output grid can omit mass;
the API does not renormalize to that interval. `diagnostics.grid_mass` and
`diagnostics.negative_mass` are integrated in lnmu. Small negative Fourier
inversion residuals are retained and emit a RuntimeWarning; check them before
using sensitive tail statistics. No positivity correction is applied.

`mean`, `var`, `skew` and `sigmaDL` come from a separate, redshift-dependent
integration interval reported as `moment_interval`. They are not moments over
the entire exported grid. Compare intervals and numerical convergence when
using them across redshifts.

### Lens ingredients and multiprocessing

`config='full'` is the default. `config='halo'` selects spherical halos only.
`+ell`, `+fil`, `+sub`, `+bias` add one ingredient to halos; the corresponding
minus variants remove one ingredient from the full model.

The default `nproc=1` works directly in notebooks. To use `nproc>1` in a Python
script, put calls inside `if __name__ == '__main__':` because workers use spawn.
Only population construction uses processes; inversion stays in the main process.

## Command line

After installation, run from any directory:

```bash
sgl-analytic --z-s 1 --out pdf.json
sgl-analytic --z-s 3 --h 0.70 --Om 0.30 --sigma8 0.90 --Ob 0.05 --ns 0.97 --zeq 3450 --out pdf_zs3.json
python -m sgl_analytic --z-s 1 --config=-sub --nproc 4
sgl-analytic --z-s 1 --Ok 0.05 --out pdf_open.json
sgl-analytic --z-s 1 --w0 -1 --wa 0.3 --out pdf_cpl.json
sgl-analytic --z-s 1 --growth-mode ode --out pdf_lcdm_ode.json   # LCDM reference for w models
```

The command writes JSON with `xi` and `P_s` on lnmu ∈ [−1, 1], plus
`xi_tail` and `P_s_tail` on (1, 9], sharing the solver normalization. It also
includes moments, clipped moments on |lnmu| <= 1, `moment_interval`, diagnostics,
cosmological inputs and stage timings. Without `--out`, it writes to
`output/pdf_zs<z>.json` relative to the current directory.

The original `python run_pdf.py --zs 1 --s8 0.811` remains supported.
`--zs` and `--s8` are aliases for `--z-s` and `--sigma8`.

## Model scope

The default includes Sheth–Tormen field halos with NFW profiles truncated at
r_vir, pseudo-elliptical projections, full filament cross-sections, decorated
hosts with moment-quadrature subhalos, and linear-bias clustering with the 3D
linear correlation function in the Limber approximation. Lens-redshift
integration uses midpoint shells covering [0, z_s].

The Born approximation adds lens distortions, and the magnification map uses
the weak, positive-parity branch. Subhalo shear is represented by a conditional
rms and clustering is truncated at the two-point term. High-magnification
output does not constitute a complete strong-lensing calculation. This model
also differs from FLUMEN's simulation model in halo truncation and clustering;
matching cosmological inputs does not imply identical PDFs or flux calibration.

## Notebooks and advanced use

`notebooks/sgl_analytic_walkthrough.ipynb` explains the solver stages.
Edit `config.py` or `scripts/build_notebook.py`, then regenerate and execute it:

```bash
python scripts/build_notebook.py
```

`notebooks/sgl_pdf_explorer.ipynb` offers editable scenarios, plots and exports;
its generator is `scripts/build_explorer.py`. Install the `notebooks` extra
before running either. Exported `zeq` is included in the solver metadata.

Advanced users can import `sgl_analytic.pipeline.run` for the original
`(out, meta)` interface and numerical options. Its cosmology dictionary accepts
`sigma8` or the legacy `s8`, plus h, Om, Ob, ns, zeq, Ok, w0, wa and growth_mode. Compatibility modules
in `src/` keep existing notebooks working; installed code uses relative package
imports without modifying sys.path.

## Verification

With the package installed:

```bash
python -m unittest discover -s tests -p test_api.py
python -m pytest tests/test_cosmology_extensions.py
python tests/test_reference.py 0.5 1 3 10
```

The first command checks the public input and density contracts. The second
checks curvature and dark energy against independent references: distances from
the transverse Jacobi equation, the dark-energy density from energy
conservation, and growth from separate ODE solvers and the constant-w
hypergeometric solution. The third checks default PDFs against stored
references, with a tolerance of 1e-9 of the body peak. For pytest, install the `test` extra and run `python -m pytest`.
The expensive reference regression remains a separate explicit command.

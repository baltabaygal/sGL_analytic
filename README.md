# sGL analytic

Sampling-free stochastic weak-lensing magnification PDFs for a source redshift
and a LCDM cosmology, optionally with spatial curvature and time-varying (CPL)
dark energy. The solver builds a lens population, evaluates its
compensated compound-Poisson characteristic function with a two-point clustering
correction, and inverts it to a source-plane PDF. A calculation takes roughly
10–25 seconds on an M-series laptop (35–45 s with the positive clustering
closure); this is a numerical solver, not an emulator.

Version 0.2.0 (2026-10-21) carries the current default chain of the research
repository; see [Changes in 0.2.0](#changes-in-020) for what moved.

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

# low source redshift: the positive clustering closure
lnmu, pdf_lnmu = generate_pdf_lnmu(z_s=0.3, closure="vp")
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
| `growth_mode` | Linear growth: `"ode"` or `"auto"` (see below) | `"ode"` |
| `radiation` | `"engine"`: Omega_r = Om/(1 + zeq) in the background; `None`: none | `"engine"` |

Keyword-only model option:

| Parameter | Meaning | Default |
|---|---|---:|
| `closure` | Clustering beyond second order: `"gauss"` or `"vp"` (see [Model scope](#model-scope)) | `"gauss"` |

The six FLUMEN inputs must be finite and positive, with 0 < Ob < Om <= 1; `Ok`,
`w0` and `wa` must be finite. The CMB temperature is held at 2.7255 K. There
are no neutrino-mass inputs. The transfer function is EH98, with a smooth-k
mass-variance window and a real-space top-hat sigma8 normalization;
concentrations use Ludlow+16. sigma8 is fixed at z = 0 for every background.

### Curvature, dark energy and radiation

With a = 1/(1+z), Omega_r = Om/(1 + zeq) and Omega_DE = 1 − Om − Omega_r − Ok,

    E^2(a) = Om a^-3 + Omega_r a^-4 + Ok a^-2 + Omega_DE a^(-3(1+w0+wa)) exp(3 wa (a − 1)),

so the defaults give flat LCDM with radiation, as in the C++ Monte Carlo engine.
zeq is a free input, not derived from Om h^2. `radiation=None` sets Omega_r = 0
(the background of version 0.1). Transverse comoving distances are
f_K(chi) = (c/H0) sinh(sqrt(Ok) H0 chi/c)/sqrt(Ok) for an open universe (sin and
−Ok for a closed one), and the lens-source angular-diameter distance is
f_K(chi_s − chi_l)/(1 + z_s). These enter the critical
surface density, and H(z) enters the critical density and the lens rate per
unit area. The Limber clustering term stays local and flat-sky.

Linear growth is general-relativistic and scale independent, with smooth dark
energy (no dark-energy perturbations). The default `growth_mode="ode"` solves
D'' + (2 + dlnH/dlna) D' = (3/2) Omega_m(a) D in ln a, with the instantaneous
w(a) and the radiation term in dlnH/dlna, from the matter–radiation growing mode
D = a + (2/3) a_eq. The same D enters sigma(M), halo bias, M*, delta_c(z) and
the subhalo formation redshifts. `"auto"` keeps the historical LCDM treatment
for any Lambda model, flat or curved: the exact growth integral for sigma(M),
and the Carroll–Press–Turner approximation for halo bias and M*; for any other
w0, wa it uses the ODE everywhere. The two treatments differ slightly, so
`"auto"` jumps at w = −1 (a relative 2e-4 in the clipped variance of ln mu at
z_s = 1), and the solver warns when `"auto"` switches to the ODE. With the
default `"ode"`, Lambda and dark-energy runs are directly comparable.

The supported domain is −2 <= w0 <= 1 and w0 + wa < 0 (matter domination at
early times), with rho_DE/rho_m <= 1e-2 at z = 1100, so that the EH98 transfer
function and the matter-era start of the growth ODE stay valid. Cosmologies
with E^2 <= 0 at some z <= 1e6 are rejected, as are closed-universe distances
that reach the antipode.

The lens population is not recalibrated for curvature or dark energy. The
Sheth–Tormen mass function, halo bias and Allgood ellipticities respond
through sigma(M, z), growth and H(z). The Ludlow+16 concentration fit
depends on sigma(M) at z = 0 and on z, so at fixed sigma8 it does not respond
to the growth history. The Bryan–Norman virial overdensity, evaluated at the
instantaneous Omega_m(z), enters only the subhalo calibration (and the halo
edge with `edge="rvir"`). These are extrapolations of LCDM-calibrated
relations, not results for curved or CPL cosmologies.

Stored regression PDFs cover z_s = 0.5, 1, 3 and 10 at the default cosmology,
and the positive closure at z_s = 0.5 and 1. They establish numerical
reproducibility, not accuracy over a parameter box. FLUMEN's emulator training
ranges are not validation ranges for this solver.

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
sgl-analytic --z-s 0.3 --closure vp --out pdf_zs0.3_vp.json   # positive clustering closure
sgl-analytic --z-s 1 --radiation none --growth-mode auto --out pdf_v01_background.json
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
r200c (each halo encloses exactly its mass-function mass M200), pseudo-elliptical
projections, full filament cross-sections, decorated hosts with
moment-quadrature subhalos, and linear-bias clustering with the 3D linear
correlation function in the Limber approximation. Subhalos follow the same
convention: the virial-unit calibrated population is kept inside the host's
r200, and each clump is an NFW halo with M200 = m cut at its own r200.
`pipeline.run(..., edge="rvir")` truncates halos and clumps at the virial radius
instead (the version 0.1 default). Lens-redshift integration uses midpoint
shells covering [0, z_s].

Clustering has two-point accuracy. Beyond second order there are two closures:

- `closure="gauss"` (default): the Gauss–Poisson pair term, whose random lens
  intensity is Gaussian and can go negative. At low z_s and high sigma8 this can
  leave negative regions in the inverted PDF: at the Planck cosmology and
  z_s = 0.5 a lobe of 4e-5 of the peak at the lower edge (negative mass 1.6e-5).
- `closure="vp"`: a positive variance-preserving Cox closure on the same lens
  cells, with the same mean rates and the same convergence and shear clustering
  variances. It is a valid characteristic function at every z_s (negative mass
  1e-9 at z_s = 0.5) and differs from the Gaussian closure by 2.3e-3 / 1.1e-3 of
  the peak at z_s = 0.5 / 1. It costs about three times as much. Use it below
  z_s ~ 0.5, or wherever the diagnostics report negative mass.

The Born approximation adds lens distortions, and the magnification map uses
the weak, positive-parity branch. Subhalo shear is represented by a conditional
rms. High-magnification output does not constitute a complete strong-lensing
calculation. This model also differs from FLUMEN's simulation model in halo
truncation and clustering; matching cosmological inputs does not imply
identical PDFs or flux calibration. The difference grows toward low z_s, where
the engine's untruncated halo outskirts matter most: below z_s ~ 0.5 the PDF
depends strongly on the halo-edge model (for halos alone at z_s = 0.2, r200
truncation gives about twice the peak height of the engine's untruncated halos).

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
`(out, meta)` interface and numerical options (`edge`, `xi_closure`, `phi_tol`,
`grid_scale`, ...). Its cosmology dictionary accepts `sigma8` or the legacy `s8`,
plus h, Om, Ob, ns, zeq, Ok, w0, wa, growth_mode and radiation. Compatibility modules
in `src/` keep existing notebooks working; installed code uses relative package
imports without modifying sys.path.

## Verification

With the package installed:

```bash
python -m unittest discover -s tests -p test_api.py
python -m pytest tests/test_cosmology_extensions.py tests/test_radiation_and_edge.py
python tests/test_reference.py 0.5 1 3 10
python tests/test_reference.py --closure vp 0.5 1
```

The first command checks the public input and density contracts. The second
checks curvature, dark energy and radiation against independent references:
distances from the transverse Jacobi equation, the dark-energy density from
energy conservation, growth from separate ODE solvers and the constant-w
hypergeometric solution, the engine's radiation closure, and the r200c subhalo
convention. The last two check default and positive-closure PDFs against stored
references, with a tolerance of 1e-9 of the body peak. For pytest, install the
`test` extra and run `python -m pytest`. The expensive reference regressions
remain separate explicit commands.

## Provenance

The physics modules in `src/sgl_analytic/` (`sgl`, `sgl_full`, `subhalos`,
`subhalos_exact`, `filaments`, `pos_closure`, `pos_closure_fused`, `sgl_direct`)
are copies of the research repository's `analytic/` files with only the local
imports made relative. `scripts/sync_from_analytic.py` performs that copy and
writes `SOURCE.json` (source commit, branch, whether its `analytic/` had
uncommitted changes, and the SHA-256 of every source file); `--check` reports
modules that have drifted. The reference PDFs in `tests/reference/` were
produced by the research repository's own `run_full_chain.run`, not by this
package (`scripts/make_references.py`), and this package reproduces them bit for
bit (max|dP| = 0). After a sync, regenerate them, carry any new `run()` defaults
over to `pipeline.py` by hand, and rerun both reference regressions.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{Baltabay:2026nmz,
    author = "Baltabay, Galymzhan and Vaskonen, Ville",
    title = "{FLUMEN: Neural Emulator of an Advanced Stochastic Weak Lensing Model}",
    eprint = "2609.21806",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "9",
    year = "2026"
}

@misc{Baltabay2026cos,
      title={Cosmological lensing as a jump process},
      author={Galymzhan Baltabay and Ville Vaskonen},
      year={2026},
      eprint={2610.40341},
      archivePrefix={arXiv},
      primaryClass={astro-ph.CO},
      url={https://arxiv.org/abs/2610.40341},
}
```

## Changes in 0.2.0

The defaults follow the research repository's chain as of 2026-10-21:

- halos and subhalo clumps are truncated at r200c (was r_vir);
- the background includes radiation, Omega_r = Om/(1 + zeq) (was none);
- linear growth from the ODE for every model, also in bias, M*, delta_c(z) and
  subhalo formation redshifts (was `"auto"`);
- a 101 x 101 coarse grid for the clustering term (was 51 x 51), finer
  inversion grids (72/36 points per sigma in kappa/gamma, 2400 integration
  nodes), a kink-split angle rule for filaments, the inferred number of
  ellipticity angles in the subhalo sector, block-wise inversion of very large
  grids, and a finer shear-axis grid for narrow PDFs;
- the optional positive clustering closure (`closure="vp"`), and speed-ups of
  the direct cell sum and the kappa transform (identical results up to
  floating-point roundoff);
- the dark-energy domain is wider (see above).

Combined effect on the Planck reference PDFs (new vs 0.1.0), z_s = 0.5 / 1 / 3 / 10:
sigma_kappa −2.3 / −2.0 / −1.6 / −1.4%, clipped variance of ln mu (|ln mu| <= 1)
−3.9 / −3.9 / −3.3 / −2.8%, peak +5.3 / +3.8 / +2.4 / +1.9%, and
max|dP| = 11 / 5.4 / 2.9 / 2.3% of the peak. `edge="rvir"`, `radiation=None`
and `growth_mode="auto"` restore the old model choices but not the old
numerical resolution.

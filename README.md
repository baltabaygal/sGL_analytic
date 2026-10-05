# sGL_analytic

Sampling-free magnification PDF of stochastic weak lensing. For a given source
redshift and cosmology (h, Omega_m, sigma_8) the code computes the source-plane
distribution dP/d ln mu of the lens model of Vaskonen (2026) / FLUMEN by solving
the compensated Kolmogorov-Feller (compound-Poisson) master equation in Fourier
space and inverting its characteristic function. No lens configuration is sampled
and no random number is drawn. init

Default lens model (`config="full"`):
- field halos: Sheth-Tormen mass function, NFW profiles truncated at r_vir,
  integrated over the whole lens plane (shear tail down to kappa = 1e-8);
- pseudo-elliptical projections;
- filaments (full cylinder cross-section);
- subhalos: exact moment quadrature of the decorated-host law, clumps truncated
  at their own r_vir with virial mass m, host rebuilt at M - S;
- clustering: linear-bias two-point term with the 3D linear correlation
  function in the Limber approximation (`xilin`);
- lens-redshift integral: midpoint rule on shells covering [0, z_s].

## Usage
```bash
python run_pdf.py --zs 1
python run_pdf.py --zs 3 --s8 0.75 --Om 0.30 --out output/pdf_zs3.json
```
From Python:
```python
import sys; sys.path.insert(0, "src")
from pipeline import run
out, meta = run(1.0, "full", cosmo=dict(h=0.674, Om=0.315, s8=0.811))
xi, P = out["xi"], out["P_s"]          # ln mu on [-1, 1], dP_S/d ln mu
```
Output JSON: `xi`, `P_s` (body), `xi_tail`, `P_s_tail` (ln mu in (1, 9], same
normalisation), `sigma_kappa`, moments (`mean`, `var`, `skew`, `clipped`),
`sigmaDL`, stage timings `t_stages`. One PDF takes ~10-30 s on an M-series laptop
(population build, parallel over `--nproc` processes, + inversion).

## Walkthrough notebook
`notebooks/sgl_analytic_walkthrough.ipynb` walks the chain stage by stage
(cosmology, single lenses, master equation, clustering, population build,
exponent, inversion, z_s and sigma_8 dependence), with the equations, their
references and the implementing functions, and every number and figure produced
by the solver itself. It is GENERATED: edit `config.py` (z_s, cosmology, lens
config, comparison redshifts, sigma_8 scan) or `scripts/build_notebook.py`, then
```bash
python scripts/build_notebook.py     # executes it, ~2.5 min
```

## Layout
| file | role |
|---|---|
| `src/pipeline.py` | `run()`: builds the population, the exponent, and inverts it |
| `src/sgl.py` | cosmology (EH98 transfer, growth, sigma(M), mass function, concentration), NFW lensing |
| `src/sgl_full.py` | lens population on (kappa, gamma) cells, ellipticity, truncated NFW, Lambda(k), clustering term, inversion to P(mu) |
| `src/filaments.py` | filament population |
| `src/subhalos.py`, `src/subhalos_exact.py` | subhalo model and the exact (moment) decorated-host sector |
| `run_pdf.py` | command-line driver |
| `config.py` | inputs of the notebook |
| `scripts/build_notebook.py` | generates + executes the notebook |
| `tests/test_reference.py` | regression test against `tests/reference/pdf_zs{0.5,1,3,10}.json` |

The modules still carry non-default options from development (engine kthr split,
lognormal Hermite closure, pair subhalo sector, ...); `run()` exposes them as
keyword arguments. Only the defaults are covered by the reference test.

## Requirements
Python >= 3.12, numpy, scipy, numba (tested: numpy 2.4.3, scipy 1.17.1,
numba 0.65.1; see `environment.yml`). The notebook also needs matplotlib,
ipykernel, nbformat and nbclient.

## Test
```bash
python tests/test_reference.py            # z_s = 1
python tests/test_reference.py 0.5 1 3 10
```
Passes at max|dP| <= 1e-9 of the peak (bit-identical on the reference machine).

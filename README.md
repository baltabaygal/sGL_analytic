# sGL analytic

Sampling-free stochastic weak-lensing magnification PDFs for a source redshift
and a LCDM cosmology, optionally with spatial curvature and time-varying (CPL)
dark energy. The solver builds a lens population, evaluates its
compensated compound-Poisson characteristic function with a two-point clustering
correction, and inverts it to a source-plane PDF. A calculation takes roughly
10–25 seconds on an M-series laptop; this is a numerical solver, not an emulator.

Model scope, numerical conventions, verification and provenance are in
[docs/details.md](docs/details.md); release notes are in [CHANGELOG.md](CHANGELOG.md).

## Install

Requires Python 3.12 or newer. From the repository root:

```bash
python -m pip install .
```

For development use `python -m pip install -e '.[test,notebooks]'`. For the exact
numerical reference environment, use `conda env create -f environment.yml`.
The first calculation may take longer while Numba compiles kernels.

## Quick start

```python
from sgl_analytic import generate_pdf, generate_pdf_lnmu

# p_S(mu): density with respect to dmu
mu, pdf = generate_pdf(z_s=1.0)

# dP_S/dlnmu, with cosmology and run metadata
lnmu, pdf_lnmu, info = generate_pdf_lnmu(
    z_s=1.0, h=0.674, Om=0.315, sigma8=0.811,
    Ob=0.0493, ns=0.965, zeq=3402.0,
    return_info=True,
)

# open universe with evolving dark energy, w(a) = -1 + 0.3 (1 - a)
lnmu, pdf_lnmu = generate_pdf_lnmu(z_s=1.0, Ok=0.05, w0=-1.0, wa=0.3)

# low source redshift: the positive clustering closure
lnmu, pdf_lnmu = generate_pdf_lnmu(z_s=0.3, closure="vp")
```

A runnable export example is in `examples/quickstart.py`.

## Inputs

| Parameter | Meaning | Default |
|---|---|---:|
| `z_s` | Source redshift; required | — |
| `h` | Hubble constant divided by 100 km/s/Mpc | 0.674 |
| `Om` | Total matter density fraction, including baryons | 0.315 |
| `sigma8` | Real-space top-hat amplitude at 8/h Mpc | 0.811 |
| `Ob` | Baryon density fraction | 0.0493 |
| `ns` | Scalar spectral index | 0.965 |
| `zeq` | Matter–radiation equality redshift | 3402.0 |
| `Ok` | Curvature density; positive is open | 0.0 |
| `w0`, `wa` | CPL dark energy, w(a) = w0 + wa (1 - a) | -1.0, 0.0 |
| `closure` | Clustering closure: `"gauss"` or `"vp"` (positive) | `"gauss"` |

The first seven match FLUMEN. See [docs/details.md](docs/details.md) for the
remaining options (`growth_mode`, `radiation`, `config`, `nproc`, output grids)
and the supported cosmology domain.

## Command line

```bash
sgl-analytic --z-s 1 --out pdf.json
sgl-analytic --z-s 1 --Ok 0.05 --w0 -1 --wa 0.3 --out pdf_cpl.json
sgl-analytic --z-s 0.3 --closure vp --out pdf_zs0.3_vp.json
```

## Tests

```bash
python -m pytest
python tests/test_reference.py 0.5 1 3 10
```

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

@article{Baltabay:2026hka,
    author = "Baltabay, Galymzhan and Vaskonen, Ville",
    title = "{Cosmological lensing as a jump process}",
    eprint = "2610.40341",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "10",
    year = "2026"
}
```

## License

MIT; see [LICENSE](LICENSE).

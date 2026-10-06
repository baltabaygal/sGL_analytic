"""Tunable inputs of the walkthrough notebook (notebooks/sgl_analytic_walkthrough.ipynb).

Numbers and scenario choices only: no physics lives here.  Edit a value, rerun
    python scripts/build_notebook.py
and every number and figure in the notebook is recomputed from it.
The library default cosmology is pipeline.P (Planck 2018); this file chooses
what the notebook evaluates.
"""
import os

# Source redshift of the stage-by-stage walkthrough (one full PDF, ~10-20 s).
ZS = 1.0

# Cosmology (h, Omega_m, sigma_8, Omega_b, n_s, z_eq). The CMB temperature
# is held at the sgl.DEFAULTS value; sigma_8 is a real-space
# top-hat sigma_8 (the engine's anchor).  Changing sigma_8 moves sigma(M), the
# mass function, the Ludlow+16 concentrations and the linear P(k) together.
COSMO = dict(h=0.674, Om=0.315, s8=0.811, Ob=0.0493, ns=0.965, zeq=3402.0)

# Lens model: "full" = halos + ellipticity + filaments + subhalos + clustering.
# Single-ingredient arms: "halo" (spherical halos only), "+ell", "+fil",
# "+sub", "+bias"; leave-one-out arms: "-ell", "-fil", "-sub", "-bias".
CONFIG = "full"

# Processes for the population build (the result does not depend on it).
NPROC = os.cpu_count() or 1

# Source redshifts of the closing comparison figure (one PDF each).
ZS_LIST = (0.5, 1.0, 3.0, 10.0)

# sigma_8 values of the cosmology-response figure, evaluated at ZS.
S8_SCAN = (0.75, 0.811, 0.87)

# Example lens used to draw the single-lens profiles (Msun, lens redshift).
EXAMPLE_M = 1e14
EXAMPLE_ZL = 0.4

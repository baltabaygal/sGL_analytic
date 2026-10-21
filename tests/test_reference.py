"""Regression test: the default chain reproduces the stored reference PDFs.

  python tests/test_reference.py                     # z_s = 1 (about 15 s)
  python tests/test_reference.py 0.5 1 3 10          # all Gaussian-closure references
  python tests/test_reference.py --closure vp 1      # positive (vp) closure, about 35 s

References (tests/reference/pdf_zs<z>.json, pdf_zs<z>_vp.json) were produced on
2026-10-21 by stochastic_wl_analytic's own analytic/run_full_chain.run, not by this
package: midpoint z quadrature, xi_lin clustering with the 101^2 coarse grid,
r200c-truncated halos and clumps, exact moment subhalos, ODE growth, engine
radiation, Planck cosmology h=0.674, Om=0.315, s8=0.811. Each file records its
source commit and file hashes under "provenance". The port reproduced them
bit-for-bit (max|dP| = 0).
Tolerance: max|dP| <= 1e-9 of the peak (the build is bit-identical for any
nproc; the residual allowance covers BLAS thread-count roundoff).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from pipeline import run  # noqa: E402

TOL = 1e-9


def check(zs, closure="gauss"):
    suffix = "" if closure == "gauss" else f"_{closure}"
    ref = json.loads((HERE / "reference" / f"pdf_zs{zs:g}{suffix}.json").read_text())
    out, _ = run(zs, "full", xi_out=np.linspace(-1.0, 1.0, 1601),
                 xi_tail=np.linspace(1.005, 9.0, 1600), nproc=4, xi_closure=closure)
    P0, P1 = np.asarray(ref["P_s"]), np.asarray(out["P_s"])
    T0, T1 = np.asarray(ref["P_s_tail"]), np.asarray(out["P_s_tail"])
    d = np.max(np.abs(P1 - P0)) / P0.max()
    dt = np.max(np.abs(T1 - T0)) / P0.max()
    ok = d <= TOL and dt <= TOL
    print(f"zs={zs:g} {closure}: max|dP|/peak body {d:.2e}, tail {dt:.2e}, "
          f"sigma_kappa {out['sigma_kappa']:.8f} (ref {ref['sigma_kappa']:.8f}) "
          f"{'PASS' if ok else 'FAIL'}  [{out['seconds']:.1f} s]")
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    closure = "gauss"
    if "--closure" in args:
        i = args.index("--closure")
        closure = args[i + 1]
        del args[i:i + 2]
    zss = [float(z) for z in args] or [1.0]
    sys.exit(0 if all([check(z, closure) for z in zss]) else 1)

"""Regression test: the default chain reproduces the stored reference PDFs.

  python tests/test_reference.py            # z_s = 1 (about 10 s)
  python tests/test_reference.py 0.5 1 3 10 # all references

References (tests/reference/pdf_zs<z>.json) were produced on 2026-10-05 by the
default chain (midpoint z quadrature, xi_lin clustering, r_vir-truncated halos,
exact moment subhalos), Planck cosmology h=0.674, Om=0.315, s8=0.811.
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


def check(zs):
    ref = json.loads((HERE / "reference" / f"pdf_zs{zs:g}.json").read_text())
    out, _ = run(zs, "full", xi_out=np.linspace(-1.0, 1.0, 1601),
                 xi_tail=np.linspace(1.005, 9.0, 1600), nproc=4)
    P0, P1 = np.asarray(ref["P_s"]), np.asarray(out["P_s"])
    T0, T1 = np.asarray(ref["P_s_tail"]), np.asarray(out["P_s_tail"])
    d = np.max(np.abs(P1 - P0)) / P0.max()
    dt = np.max(np.abs(T1 - T0)) / P0.max()
    ok = d <= TOL and dt <= TOL
    print(f"zs={zs:g}: max|dP|/peak body {d:.2e}, tail {dt:.2e}, "
          f"sigma_kappa {out['sigma_kappa']:.8f} (ref {ref['sigma_kappa']:.8f}) "
          f"{'PASS' if ok else 'FAIL'}  [{out['seconds']:.1f} s]")
    return ok


if __name__ == "__main__":
    zss = [float(z) for z in sys.argv[1:]] or [1.0]
    sys.exit(0 if all([check(z) for z in zss]) else 1)

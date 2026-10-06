#!/usr/bin/env python
"""Magnification PDF of the default lens model for one source redshift and cosmology.

  python run_pdf.py --zs 1
  python run_pdf.py --zs 3 --s8 0.75 --Om 0.30 --out output/pdf_zs3_s8lo.json

Writes JSON with xi = ln mu on [-1, 1] and P_s = source-plane dP/d ln mu there,
xi_tail / P_s_tail on (1, 9] (same normalisation), plus moments and timings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import P, clipped_moments, run


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z-s", "--zs", dest="zs", type=float, required=True, help="source redshift")
    ap.add_argument("--h", type=float, default=P["h"])
    ap.add_argument("--Om", type=float, default=P["Om"])
    ap.add_argument("--sigma8", "--s8", dest="s8", type=float, default=P["s8"])
    for name in ("Ob", "ns", "zeq"):
        ap.add_argument(f"--{name}", type=float, default=P[name])
    ap.add_argument("--config", default="full",
                    help='lens model: "full" (default), "halo", "+ell", "-sub", ... '
                         '(pass minus arms as --config=-sub)')
    ap.add_argument("--nproc", type=int, default=1,
                    help="processes for the population build")
    ap.add_argument("--out", default=None, help="output JSON (default output/pdf_zs<zs>.json)")
    a = ap.parse_args(argv)

    try:
        out, _ = run(a.zs, a.config, nproc=a.nproc,
                     cosmo=dict(h=a.h, Om=a.Om, s8=a.s8, Ob=a.Ob, ns=a.ns, zeq=a.zeq),
                     xi_out=np.linspace(-1.0, 1.0, 1601),
                     xi_tail=np.linspace(1.005, 9.0, 1600))
    except ValueError as exc:
        ap.error(str(exc))
    x = np.array(out["xi"] + out["xi_tail"])
    p = np.array(out["P_s"] + out["P_s_tail"])
    out["moment_interval"] = [out["grid"]["xi_min"], out["grid"]["xi_max"]]
    out["diagnostics"] = dict(grid_mass=float(np.trapezoid(p, x)),
                              negative_mass=float(np.trapezoid(np.maximum(-p, 0), x)),
                              minimum_density=float(p.min()))
    out["clipped"] = clipped_moments(np.array(out["xi"]), np.array(out["P_s"]))
    dest = Path(a.out) if a.out else Path("output") / f"pdf_zs{a.zs:g}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as stream:
        json.dump(out, stream, allow_nan=False)
    c = out["clipped"]
    print(f"zs={a.zs:g} sigma_kappa={out['sigma_kappa']:.5f} clipped var={c['var']:.5e} "
          f"skew={c['skew']:.3f}  {out['seconds']:.1f} s -> {dest}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Regenerate tests/reference/ from the research repository's own chain.

  python scripts/make_references.py 0.5 1 3 10
  python scripts/make_references.py --closure vp 0.5 1
  python scripts/make_references.py --repo /path/to/stochastic_wl_analytic 1

The references are produced by stochastic_wl_analytic's analytic/run_full_chain.run,
NOT by this package, so that tests/test_reference.py checks the port against its
source. Each file records the source commit and the SHA-256 of the ported files.
Run one z_s at a time on a small machine: each PDF needs up to ~3 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from sync_from_analytic import MODULES  # noqa: E402


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zs", type=float, nargs="+")
    ap.add_argument("--closure", default="gauss", choices=("gauss", "vp"))
    ap.add_argument("--repo", type=Path, default=ROOT.parent / "stochastic_wl_analytic")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.repo / "analytic"))
    import run_full_chain as R

    src = a.repo / "analytic"
    provenance = dict(
        producer="stochastic_wl_analytic analytic/run_full_chain.run (not the sgl_analytic port)",
        source_commit=git(a.repo, "rev-parse", "HEAD"),
        source_branch=git(a.repo, "rev-parse", "--abbrev-ref", "HEAD"),
        source_analytic_dirty=bool(git(a.repo, "status", "--porcelain", "--", "analytic/")),
        sha256={f"{m}.py": hashlib.sha256((src / f"{m}.py").read_bytes()).hexdigest()
                for m in MODULES})
    suffix = "" if a.closure == "gauss" else f"_{a.closure}"
    for zs in a.zs:
        out, _ = R.run(zs, "full", xi_out=np.linspace(-1.0, 1.0, 1601),
                       xi_tail=np.linspace(1.005, 9.0, 1600), nproc=4, xi_closure=a.closure)
        out["provenance"] = dict(provenance, call=(
            f'run({zs:g}, "full", xi_out=linspace(-1, 1, 1601), '
            f'xi_tail=linspace(1.005, 9, 1600), nproc=4, xi_closure="{a.closure}")'))
        dest = ROOT / "tests" / "reference" / f"pdf_zs{zs:g}{suffix}.json"
        dest.write_text(json.dumps(jsonable(out), allow_nan=False))
        print(f"zs={zs:g} {a.closure}: sigma_kappa {out['sigma_kappa']:.8f}, "
              f"{out['seconds']:.1f} s -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

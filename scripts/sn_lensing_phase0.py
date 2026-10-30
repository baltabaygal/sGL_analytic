#!/usr/bin/env python
"""Phase-0 supernova-lensing model-shift test.

This script is deliberately a *pre-likelihood* diagnostic.  It asks whether the
lensing-PDF modelling difference between two sGL configurations is comparable to
what a supernova sample could see, before adding DES/Pantheon+ selection effects,
intrinsic non-Gaussianity, compact-object microlensing, or a full cosmology
likelihood.

Main outputs
------------
1. For each requested sGL configuration, build the source-plane lensing PDF at
   every redshift node.
2. Convert dP/dln(mu) to a magnitude-residual PDF,

       Delta m_lens = -2.5 log10(mu),

   optionally convolve with a Gaussian intrinsic SN scatter.
3. Form the redshift-mixture PDF for the supplied SN redshift weights.
4. Report moment shifts and PDF distances relative to a baseline configuration.
5. Optionally fit sigma8 by minimizing the expected negative log-likelihood of
   a candidate baseline model against a chosen "truth" configuration.

Important limitations
---------------------
- config="halo" is the current sGL spherical-halo baseline.  It is not yet an
  exact turboGL clone because the solver still uses the sGL magnification map
  and halo shear treatment.  Use this script first to quantify internal model
  shifts.  A stricter turboGL reproduction should add a kappa-only / no-shear
  baseline to the solver.
- Compact objects / PBHs are not implemented here.  Add them only after the
  baseline model-shift test is understood.

Example
-------
python scripts/sn_lensing_phase0.py \
  --z 0.2,0.35,0.5,0.65,0.8,0.95,1.1 \
  --configs halo,+ell,+fil,+sub,+bias,full \
  --baseline halo \
  --truth full \
  --sigma8-grid 0.70,0.75,0.80,0.811,0.85,0.90 \
  --out output/sn_phase0.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from sgl_analytic import generate_pdf_lnmu

MAG_PER_LNMU = 2.5 / np.log(10.0)


def _parse_floats(text: str | None) -> list[float] | None:
    if text is None or text == "":
        return None
    return [float(x) for x in text.replace(";", ",").split(",") if x.strip()]


def _normalised_weights(z: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    if w is None:
        w = np.ones_like(z, dtype=float)
    if w.shape != z.shape:
        raise ValueError("redshift weights must have the same length as redshifts")
    if np.any(~np.isfinite(w)) or np.any(w < 0.0) or np.sum(w) <= 0.0:
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    return w / np.sum(w)


def load_redshift_weights(path: str | None, z_text: str, w_text: str | None):
    """Return sorted unique redshift nodes and normalized weights.

    CSV mode accepts either columns named z,weight or two unnamed columns.  The
    weights can be counts; they are normalized here.
    """
    if path is None:
        z = np.asarray(_parse_floats(z_text), dtype=float)
        w = None if w_text is None else np.asarray(_parse_floats(w_text), dtype=float)
    else:
        rows = []
        with open(path, newline="") as stream:
            sample = stream.read(4096)
            stream.seek(0)
            has_header = csv.Sniffer().has_header(sample)
            if has_header:
                for row in csv.DictReader(stream):
                    key_z = "z" if "z" in row else "redshift"
                    key_w = "weight" if "weight" in row else ("count" if "count" in row else None)
                    rows.append((float(row[key_z]), 1.0 if key_w is None else float(row[key_w])))
            else:
                for row in csv.reader(stream):
                    if not row or row[0].strip().startswith("#"):
                        continue
                    rows.append((float(row[0]), 1.0 if len(row) == 1 else float(row[1])))
        z = np.asarray([r[0] for r in rows], dtype=float)
        w = np.asarray([r[1] for r in rows], dtype=float)
    if z.ndim != 1 or z.size == 0 or np.any(~np.isfinite(z)) or np.any(z <= 0.0):
        raise ValueError("redshifts must be finite positive numbers")
    w = _normalised_weights(z, w)
    order = np.argsort(z)
    z = z[order]
    w = w[order]
    # merge duplicate redshift nodes, useful when the CSV contains binned counts
    zu, inv = np.unique(z, return_inverse=True)
    wu = np.zeros_like(zu, dtype=float)
    np.add.at(wu, inv, w)
    return zu, _normalised_weights(zu, wu)


def normalize_pdf(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.maximum(p, 0.0)
    area = np.trapezoid(p, x)
    if area <= 0.0:
        raise RuntimeError("PDF has non-positive mass on the requested grid")
    return p / area


def convolve_intrinsic(dm_grid: np.ndarray, p_dm: np.ndarray, sigma_mag: float) -> np.ndarray:
    if sigma_mag <= 0.0:
        return normalize_pdf(dm_grid, p_dm)
    dx = float(dm_grid[1] - dm_grid[0])
    half = int(np.ceil(6.0 * sigma_mag / dx))
    x = np.arange(-half, half + 1, dtype=float) * dx
    g = np.exp(-0.5 * (x / sigma_mag) ** 2)
    g = normalize_pdf(x, g)
    out = fftconvolve(p_dm, g, mode="same") * dx
    return normalize_pdf(dm_grid, out)


def dm_pdf_from_lnmu(lnmu: np.ndarray, p_lnmu: np.ndarray, dm_grid: np.ndarray,
                     sigma_int_mag: float) -> np.ndarray:
    # Delta m = -2.5 log10(mu) = -(2.5/ln 10) ln(mu).
    dm = -MAG_PER_LNMU * np.asarray(lnmu, dtype=float)
    p_dm = np.asarray(p_lnmu, dtype=float) / MAG_PER_LNMU
    order = np.argsort(dm)
    p_on_grid = np.interp(dm_grid, dm[order], p_dm[order], left=0.0, right=0.0)
    p_on_grid = normalize_pdf(dm_grid, p_on_grid)
    return convolve_intrinsic(dm_grid, p_on_grid, sigma_int_mag)


def pdf_metrics(x: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = normalize_pdf(x, p)
    mean = float(np.trapezoid(x * p, x))
    var = float(np.trapezoid((x - mean) ** 2 * p, x))
    sig = float(np.sqrt(max(var, 0.0)))
    mu3 = float(np.trapezoid((x - mean) ** 3 * p, x))
    skew = mu3 / sig**3 if sig > 0.0 else float("nan")
    return {
        "mean_mag": mean,
        "sigma_mag": sig,
        "var_mag": var,
        "skew_mag": float(skew),
        "p_bright_0p1": float(np.trapezoid(p[x < -0.1], x[x < -0.1])) if np.any(x < -0.1) else 0.0,
        "p_demag_0p1": float(np.trapezoid(p[x > 0.1], x[x > 0.1])) if np.any(x > 0.1) else 0.0,
    }


def pdf_distances(x: np.ndarray, p: np.ndarray, q: np.ndarray, n_sne: int) -> dict[str, float]:
    p = normalize_pdf(x, p)
    q = normalize_pdf(x, q)
    floor = 1e-300
    kl = float(np.trapezoid(p * (np.log(p + floor) - np.log(q + floor)), x))
    hell = float(np.sqrt(0.5 * np.trapezoid((np.sqrt(p) - np.sqrt(q)) ** 2, x)))
    l1 = float(np.trapezoid(np.abs(p - q), x))
    return {
        "KL_p_true_q_model": kl,
        "expected_delta_nll_for_N": float(n_sne * kl),
        "hellinger": hell,
        "l1_distance": l1,
    }


def build_model(config: str, sigma8: float, z_nodes: np.ndarray, z_weights: np.ndarray,
                dm_grid: np.ndarray, args) -> dict:
    per_z = []
    mix = np.zeros_like(dm_grid)
    for z, wz in zip(z_nodes, z_weights):
        lnmu, p_lnmu, info = generate_pdf_lnmu(
            z_s=float(z),
            sigma8=float(sigma8),
            config=config,
            nproc=args.nproc,
            lnmu_min=args.lnmu_min,
            lnmu_max=args.lnmu_max,
            n_lnmu=args.n_lnmu,
            growth_mode=args.growth_mode,
            return_info=True,
        )
        p_dm = dm_pdf_from_lnmu(lnmu, p_lnmu, dm_grid, args.sigma_int_mag)
        mix += wz * p_dm
        per_z.append({
            "z": float(z),
            "weight": float(wz),
            "metrics": pdf_metrics(dm_grid, p_dm),
            "solver": {
                "sigma_kappa": float(info.get("sigma_kappa", np.nan)),
                "seconds": float(info.get("seconds", np.nan)),
                "diagnostics": info.get("diagnostics", {}),
            },
        })
    mix = normalize_pdf(dm_grid, mix)
    return {
        "config": config,
        "sigma8": float(sigma8),
        "per_z": per_z,
        "mixture_metrics": pdf_metrics(dm_grid, mix),
        "dm_grid": dm_grid.tolist(),
        "pdf_dm": mix.tolist(),
    }


def expected_fit_sigma8(truth: dict, candidate_config: str, sigma8_grid: list[float],
                        z_nodes: np.ndarray, z_weights: np.ndarray, dm_grid: np.ndarray,
                        args) -> dict:
    p_true = np.asarray(truth["pdf_dm"], dtype=float)
    trials = []
    best = None
    for s8 in sigma8_grid:
        model = build_model(candidate_config, s8, z_nodes, z_weights, dm_grid, args)
        dist = pdf_distances(dm_grid, p_true, np.asarray(model["pdf_dm"]), args.n_sne)
        row = {"sigma8": float(s8), **dist, "metrics": model["mixture_metrics"]}
        trials.append(row)
        if best is None or row["expected_delta_nll_for_N"] < best["expected_delta_nll_for_N"]:
            best = row
    return {"candidate_config": candidate_config, "best": best, "trials": trials}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--z", default="0.2,0.35,0.5,0.65,0.8,0.95,1.1",
                    help="comma-separated redshift nodes, ignored if --z-csv is supplied")
    ap.add_argument("--weights", default=None,
                    help="comma-separated redshift weights/counts matching --z")
    ap.add_argument("--z-csv", default=None,
                    help="CSV with columns z,weight or z,count; weights are normalized")
    ap.add_argument("--configs", default="halo,+ell,+fil,+sub,+bias,full",
                    help="comma-separated sGL configs to compare")
    ap.add_argument("--baseline", default="halo", help="configuration used as model-shift baseline")
    ap.add_argument("--truth", default="full", help="configuration treated as truth for sigma8 fit")
    ap.add_argument("--sigma8", type=float, default=0.811, help="fiducial sigma8")
    ap.add_argument("--sigma8-grid", default=None,
                    help="optional comma-separated sigma8 grid for expected fit bias")
    ap.add_argument("--fit-config", default=None,
                    help="candidate config for sigma8 fit; default is --baseline")
    ap.add_argument("--n-sne", type=int, default=1532,
                    help="sample size used to scale expected log-likelihood differences")
    ap.add_argument("--sigma-int-mag", type=float, default=0.10,
                    help="Gaussian intrinsic magnitude scatter convolved with lensing PDF")
    ap.add_argument("--dm-min", type=float, default=-1.5)
    ap.add_argument("--dm-max", type=float, default=0.8)
    ap.add_argument("--n-dm", type=int, default=1601)
    ap.add_argument("--lnmu-min", type=float, default=-1.0)
    ap.add_argument("--lnmu-max", type=float, default=9.0)
    ap.add_argument("--n-lnmu", type=int, default=1601)
    ap.add_argument("--growth-mode", default="auto", choices=("auto", "ode", "legacy"))
    ap.add_argument("--nproc", type=int, default=1)
    ap.add_argument("--out", default="output/sn_phase0.json")
    args = ap.parse_args(argv)

    z_nodes, z_weights = load_redshift_weights(args.z_csv, args.z, args.weights)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if args.baseline not in configs:
        configs.insert(0, args.baseline)
    if args.truth not in configs:
        configs.append(args.truth)
    dm_grid = np.linspace(args.dm_min, args.dm_max, args.n_dm)

    models = {}
    for cfg in configs:
        models[cfg] = build_model(cfg, args.sigma8, z_nodes, z_weights, dm_grid, args)

    base_pdf = np.asarray(models[args.baseline]["pdf_dm"], dtype=float)
    shifts = {}
    for cfg, model in models.items():
        p = np.asarray(model["pdf_dm"], dtype=float)
        dist = pdf_distances(dm_grid, p, base_pdf, args.n_sne)
        b = models[args.baseline]["mixture_metrics"]
        m = model["mixture_metrics"]
        moment_shift = {f"delta_{k}": float(m[k] - b[k]) for k in m}
        shifts[cfg] = {**dist, "moment_shift_vs_baseline": moment_shift}

    fit = None
    s8_grid = _parse_floats(args.sigma8_grid)
    if s8_grid:
        fit_cfg = args.fit_config or args.baseline
        fit = expected_fit_sigma8(models[args.truth], fit_cfg, s8_grid, z_nodes, z_weights, dm_grid, args)
        fit["truth_config"] = args.truth
        fit["truth_sigma8"] = float(args.sigma8)
        fit["sigma8_bias"] = float(fit["best"]["sigma8"] - args.sigma8)

    # Store compact summaries plus the full mixed PDFs.  Per-z PDFs are not stored
    # to keep the JSON manageable.
    result = {
        "purpose": "Phase-0 SN lensing model-shift diagnostic; not a DES likelihood",
        "limitations": [
            "The halo baseline is not yet a strict turboGL no-shear clone.",
            "No DES/Pantheon+ selection effects or intrinsic non-Gaussianity.",
            "No compact-object/PBH lensing term yet.",
        ],
        "redshift_nodes": z_nodes.tolist(),
        "redshift_weights": z_weights.tolist(),
        "n_sne": int(args.n_sne),
        "sigma_int_mag": float(args.sigma_int_mag),
        "fiducial_sigma8": float(args.sigma8),
        "baseline": args.baseline,
        "truth": args.truth,
        "models": models,
        "shifts_vs_baseline": shifts,
        "sigma8_fit": fit,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)

    print(f"wrote {out}")
    print("mixture metrics:")
    for cfg in configs:
        met = models[cfg]["mixture_metrics"]
        print(f"  {cfg:>6s}: sigma_mag={met['sigma_mag']:.5f} skew={met['skew_mag']:.3f} "
              f"P(dm<-0.1)={met['p_bright_0p1']:.4g}")
    if fit is not None:
        print(f"sigma8 fit with {fit['candidate_config']}: best={fit['best']['sigma8']:.4f}, "
              f"bias={fit['sigma8_bias']:+.4f}")


if __name__ == "__main__":
    main()

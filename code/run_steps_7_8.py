#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7 and step 8 at the scheduled inactive-cell weights, resumable.

Runs, for every (seed, rate) in the results file:
    step 6  re-solved with the hinge at the scheduled theta_V (algorithm3_refit)
    step 7  Algorithm 2' at the fixed observed fares, 40 iterations
    step 8  the support check, reported SEPARATELY on two sets
              R        the reachable inactive cells, imposed in step 6:
                       a violation here means theta_V is too small
              V \\ R    the rest of the zero set, never imposed:
                       this is the genuine specification check
and records G(tau0) at the new estimate, which run_step_9.py consumes.

Resumable: --resume appends to --out and skips cases already present, so the
run can be stopped and restarted at any point without losing work.

    python run_steps_7_8.py --resume --out io2_support_lamv.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import lirr_bench as R
import algorithm3 as O

COLUMNS = [
    "seed", "rate", "lam_v", "lambda_sigma", "weight", "sigma_cap",
    "fix_drive", "admissible", "rank_theta",
    "n_eq", "n_ineq", "n_inactive", "reach",
    "viol_before", "viol_after", "maxv_after",
    "eq_resid_before", "eq_resid_after", "dtheta_rel",
    *R.FEATURE_NAMES, "sigma_hat_max",
    "n_R", "viol_R", "max_psi_R", "n_VmR", "viol_VmR", "max_psi_VmR",
    "n_A0", "viol_A0", "max_psi_A0",
    "sigma_consistency_gap", "sigma_corr", "sigma_relerr",
    "GS_tau0_hat", "GS_tau0_relerr",
    "r2_all", "wmape_all", "wmape_held", "wmape_val",
    "fp_residual", "fp_iters", "fp_converged", "secs",
]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path,
                   default=Path("io2_oneshot_unweighted.csv"),
                   help="the reported results file; supplies the case list")
    p.add_argument("--out", type=Path, default=Path("io2_support_lamv.csv"))
    p.add_argument("--lam-v", type=float, default=None,
                   help="override the schedule with one weight for all rates")
    p.add_argument("--lambda-sigma", type=float, default=None,
                   help="override the ridge on sigma (default: read per case)")
    p.add_argument("--sigma-cap", type=float, default=None,
                   help="kappa in Assumption 6; 0 removes the bound entirely "
                        "(default: 1.0, as reported)")
    p.add_argument("--rates", type=float, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--step6-max-iter", type=int, default=40,
                   help="Algorithm 2' iteration cap; 40 matches the paper")
    p.add_argument("--inner-tol", type=float, default=1e-4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args()

    inp, out_path = args.inp.resolve(), args.out.resolve()
    data_dir = R.locate_data_dir(args.data_dir)
    os.chdir(data_dir)
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    O._load_package()
    import algorithm3_refit as LR

    print(f"  data dir : {data_dir}")
    print(f"  cases    : {inp}")
    print(f"  output   : {out_path}")
    print(f"  theta_V  : " + ("override " + str(args.lam_v) if args.lam_v
                              is not None else str(LR.SCHEDULE)))
    print(f"  theta_sig: " + ("override " + str(args.lambda_sigma)
                              if args.lambda_sigma is not None
                              else "per case (1e-3 as reported)"))
    print(f"  sigma cap: " + ("REMOVED (Assumption 6 bound off)"
                              if args.sigma_cap == 0 else
                              f"kappa = {1.0 if args.sigma_cap is None else args.sigma_cap:g}"))
    print(f"  step 7 cap {args.step6_max_iter}, inner tol {args.inner_tol:g}",
          flush=True)

    prev = pd.read_csv(inp)
    n0 = len(prev)
    prev = prev.dropna(subset=["seed", "rate"])
    if len(prev) < n0:
        print(f"  NOTE     : skipped {n0 - len(prev)} rows of {inp.name} with "
              f"no seed/rate -- the file has extra rows below the data "
              f"(an Excel edit will do this). The {len(prev)} real cases are "
              f"unaffected.", flush=True)
    if args.rates:
        prev = prev[prev.rate.isin(args.rates)]
    if args.seeds:
        prev = prev[prev.seed.isin(args.seeds)]

    done = set()
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        done = {(int(t.seed), round(float(t.rate), 4))
                for t in old.itertuples()}
        print(f"  resuming : {len(done)} of {len(prev)} cases already done",
              flush=True)
    fh = out_path.open("a" if done else "w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    if not done:
        w.writeheader()
    fh.flush()

    bench = R.Bench(verbose=True)
    inst = bench.inst
    avail = inst.avail

    zero_link = np.zeros(inst.n_links, bool)
    zero_link[bench.own] = bench.f_link_true <= 0.0
    print(f"  owned links with f0_a = 0 : "
          f"{int(zero_link[bench.own].sum())} of {inst.n_own}\n", flush=True)

    for _, r in prev.sort_values(["rate", "seed"]).iterrows():
        seed, rate = int(r.seed), float(r.rate)
        if (seed, round(rate, 4)) in done:
            continue
        t0 = time.perf_counter()
        lam_v = LR.weight_for(rate, args.lam_v)
        lam_s = (float(r.lambda_sigma) if args.lambda_sigma is None
                 else float(args.lambda_sigma))
        cap = 1.0 if args.sigma_cap is None else float(args.sigma_cap)
        a = argparse.Namespace(
            val_multiple=1.2, design_alpha=0.5, rho_c=100.0, rho_pi=1.0,
            bind_tol=1e-5, weight=str(r.weight), sigma_cap=cap,
            fix_drive=str(r.fix_drive), lambda_theta=0.0)
        E = O.estimate_case(bench, seed, rate, lam_s, a)
        F = LR.refit(bench, E, lam_v, lam_s)
        data = E["data"]

        fp = O.solve_p2_fixed_price(F["est"], data.tau0, tol=args.inner_tol,
                                    max_iter=args.step6_max_iter)
        u = fp.u
        price_all = np.zeros(inst.n_links)
        price_all[bench.own] = data.tau0 + F["sigma"]
        psi = F["phi"] - price_all[None, :] - (u[:, inst.tail] - u[:, inst.head])

        A0 = F["A_zero"]
        Rm = LR.reachable_mask(inst, F["A_pos"], A0)
        V = A0 | (avail & zero_link[None, :])
        VmR = V & ~Rm

        def stats(mask):
            if not mask.any():
                return 0, float("nan"), float("nan")
            v = psi[mask]
            return int(mask.sum()), float((v > 0).mean()), float(v.max())

        nR, vR, mR = stats(Rm)
        nV, vV, mV = stats(VmR)
        n0, v0, m0 = stats(A0)

        # GS_tau0_hat comes from trace_metrics, which holds capacity at the
        # OBSERVED y0 -- the same convention as the published dwl pass.
        fit = bench.trace_metrics(fp.f_seg, F["phi"], E["c_hat"], E["keep"],
                                  E["vkeep"])
        GS_tau0 = float(fit["GS_tau0_hat"])
        gap = float(np.abs(F["sigma"] - fp.sigma).max()
                    / max(data.tau0.max(), 1e-12))
        st = bench.sigma_true

        row = dict(
            seed=seed, rate=rate, lam_v=lam_v, lambda_sigma=lam_s,
            weight=str(r.weight), sigma_cap=cap,
            fix_drive=str(r.fix_drive), admissible=int(r.admissible),
            rank_theta=int(r.rank_theta),
            n_eq=F["n_eq"], n_ineq=F["n_ineq"], n_inactive=F["n_inactive"],
            reach=F["reach"], viol_before=F["viol_before"],
            viol_after=F["viol_after"], maxv_after=F["maxv_after"],
            eq_resid_before=F["eq_resid_before"],
            eq_resid_after=F["eq_resid_after"], dtheta_rel=F["dtheta_rel"],
            **dict(zip(R.FEATURE_NAMES, map(float, F["theta"]))),
            sigma_hat_max=float(F["sigma"].max()),
            n_R=nR, viol_R=vR, max_psi_R=mR,
            n_VmR=nV, viol_VmR=vV, max_psi_VmR=mV,
            n_A0=n0, viol_A0=v0, max_psi_A0=m0,
            sigma_consistency_gap=gap,
            sigma_corr=float(np.corrcoef(F["sigma"], st)[0, 1]),
            sigma_relerr=float(np.linalg.norm(F["sigma"] - st)
                               / np.linalg.norm(st)),
            **fit,
            fp_residual=float(fp.residual), fp_iters=int(fp.n_iter),
            fp_converged=int(bool(fp.converged)),
            secs=round(time.perf_counter() - t0, 1))
        w.writerow(row); fh.flush()
        print(f"  seed {seed} rate {rate:.0%} tV {lam_v:<5g} "
              f"R {vR:6.2%} (n={nR:5d})  V\\R {vV:6.2%} (n={nV:6d})  "
              f"r2 {fit['r2_all']:.4f}  G(tau0) {GS_tau0:>12,.0f}  "
              f"{row['secs']:.0f}s", flush=True)
    fh.close()
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

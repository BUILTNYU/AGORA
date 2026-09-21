#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 9 and the deadweight loss at the scheduled inactive-cell weights.

Reads G(tau0) at the new estimate from the step-7 pass, so run
run_steps_7_8.py FIRST. Both sides of the difference use the same formula,

    G(tau) = sum_ia phi_ia f_ia - sum_ma c_ma y_ma ,

so the deadweight loss is clean; the tau0 side holds capacity at the observed
y0 while the clearing side lets operators redeploy, which is the intended
comparison but mixes a price effect with a capacity effect. Both components
(phi_f and c_y) are written out so the decomposition can be made at plot time.

The truth needs one clearing solve, independent of seed and rate, so it is
computed once and reused.

Resumable: --resume appends to --out and skips cases already present.

    python run_step_9.py --resume --in io2_support_lamv.csv --out io2_dwl_lamv.csv
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
    "seed", "rate", "lam_v", "lambda_sigma", "sigma_cap",
    "admissible", "rank_theta",
    "GS_tau0_hat", "GS_taustar_hat", "DWL_hat", "DWL_share_hat",
    "GS_ratio_hat", "taustar_phi_f", "taustar_c_y",
    "GS_tau0_true", "GS_taustar_true", "DWL_true", "DWL_share_true",
    "GS_ratio_true", "DWL_relerr", "DWL_share_err",
    "tau_star_mean", "wedge_mean", "wedge_absmean", "wedge_corr",
    "wedge_absmean_true", "wedge_absmean_err",
    "clear_iters", "clear_residual", "clear_converged", "secs",
]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path,
                   default=Path("io2_support_lamv.csv"),
                   help="output of run_steps_7_8.py; supplies G(tau0)")
    p.add_argument("--out", type=Path, default=Path("io2_dwl_lamv.csv"))
    p.add_argument("--lam-v", type=float, default=None,
                   help="override the schedule with one weight for all rates")
    p.add_argument("--lambda-sigma", type=float, default=None,
                   help="override the ridge on sigma (default: read per case)")
    p.add_argument("--sigma-cap", type=float, default=None,
                   help="kappa in Assumption 6; 0 removes the bound entirely")
    p.add_argument("--rates", type=float, nargs="+", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--clearing-max-iter", type=int, default=20000)
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

    if not inp.exists():
        print(f"  ERROR: {inp} not found.\n"
              f"  Run run_support.bat (run_steps_7_8.py) first: it "
              f"produces G(tau0) at the new estimate.")
        return 2

    print(f"  data dir : {data_dir}\n  cases    : {inp}\n"
          f"  output   : {out_path}", flush=True)
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

    bench = R.Bench(verbose=True)
    inst = bench.inst

    # ---- the truth: one clearing solve, reused for every row
    t0 = time.perf_counter()
    th_true = np.array(R.THETA_TRUE, float)
    phi_true = np.where(inst.avail, bench.Z @ th_true, 0.0)
    est_true = O.P2Instance(
        tail=inst.tail, head=inst.head, link_owner=inst.link_owner,
        n_nodes=inst.n_nodes, D=inst.D, origin=inst.origin, sink=inst.sink,
        F=inst.F, Cmax=inst.Cmax, c=bench.c_true, ell=inst.ell, phi=phi_true,
        alpha_B=1.0, alpha_S=R.ALPHA_S_TRUE, avail=inst.avail,
        c_idle=np.array(R.C_IDLE_TRUE), link_names=inst.link_names,
        segment_names=inst.segment_names, operator_names=inst.operator_names,
        node_names=inst.node_names, outside_link=inst.outside_link,
        b_supply=inst.b_supply)
    print("  clearing solve at the TRUE primitives ...", flush=True)
    clr_t = O.solve_p2(est_true, tol=args.inner_tol,
                       max_iter=args.clearing_max_iter)
    GS_taustar_true = bench.gross_surplus(phi_true, clr_t.f_seg, bench.c_true,
                                          clr_t.y)
    GS_tau0_true = bench.GS_true
    DWL_true = GS_taustar_true - GS_tau0_true
    DWL_share_true = DWL_true / abs(GS_taustar_true)
    GS_ratio_true = GS_tau0_true / GS_taustar_true
    wedge_true = bench.tau0 - clr_t.tau
    wedge_absmean_true = float(np.abs(wedge_true).mean())
    print(f"    G(tau0) true {GS_tau0_true:,.0f}   "
          f"G(tau*) true {GS_taustar_true:,.0f}   "
          f"DWL {DWL_true:,.0f} ({DWL_share_true:.2%})   "
          f"mean |wedge| {wedge_absmean_true:.4f}   "
          f"{time.perf_counter() - t0:.0f}s\n", flush=True)

    fh = out_path.open("a" if done else "w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    if not done:
        w.writeheader()
    fh.flush()

    for _, r in prev.sort_values(["rate", "seed"]).iterrows():
        seed, rate = int(r.seed), float(r.rate)
        if (seed, round(rate, 4)) in done:
            continue
        t1 = time.perf_counter()
        lam_v = LR.weight_for(rate, args.lam_v)
        lam_s = (float(r.lambda_sigma) if args.lambda_sigma is None
                 else float(args.lambda_sigma))
        cap = (float(getattr(r, "sigma_cap", 1.0)) if args.sigma_cap is None
               else float(args.sigma_cap))
        a = argparse.Namespace(
            val_multiple=1.2, design_alpha=0.5, rho_c=100.0, rho_pi=1.0,
            bind_tol=1e-5, weight=str(r.weight), sigma_cap=cap,
            fix_drive=str(r.fix_drive), lambda_theta=0.0)
        E = O.estimate_case(bench, seed, rate, lam_s, a)
        F = LR.refit(bench, E, lam_v, lam_s)

        clr = O.solve_p2(F["est"], tol=args.inner_tol,
                         max_iter=args.clearing_max_iter)
        phi_f = float((F["phi"] * clr.f_seg).sum())
        c_y = float(np.asarray(E["c_hat"]) @ np.asarray(clr.y))
        GS_taustar = phi_f - c_y
        GS_tau0 = float(r.GS_tau0_hat)
        DWL = GS_taustar - GS_tau0
        wedge = E["data"].tau0 - clr.tau

        row = dict(
            seed=seed, rate=rate, lam_v=lam_v, lambda_sigma=lam_s,
            sigma_cap=cap,
            admissible=int(r.admissible), rank_theta=int(r.rank_theta),
            GS_tau0_hat=GS_tau0, GS_taustar_hat=GS_taustar, DWL_hat=DWL,
            DWL_share_hat=DWL / abs(GS_taustar),
            GS_ratio_hat=GS_tau0 / GS_taustar,
            taustar_phi_f=phi_f, taustar_c_y=c_y,
            GS_tau0_true=GS_tau0_true, GS_taustar_true=GS_taustar_true,
            DWL_true=DWL_true, DWL_share_true=DWL_share_true,
            GS_ratio_true=GS_ratio_true,
            DWL_relerr=(DWL - DWL_true) / abs(DWL_true),
            DWL_share_err=DWL / abs(GS_taustar) - DWL_share_true,
            tau_star_mean=float(clr.tau.mean()),
            wedge_mean=float(wedge.mean()),
            wedge_absmean=float(np.abs(wedge).mean()),
            wedge_corr=float(np.corrcoef(wedge, wedge_true)[0, 1]),
            wedge_absmean_true=wedge_absmean_true,
            wedge_absmean_err=float(np.abs(wedge).mean()) - wedge_absmean_true,
            clear_iters=int(clr.n_iter), clear_residual=float(clr.residual),
            clear_converged=int(bool(clr.converged)),
            secs=round(time.perf_counter() - t1, 1))
        w.writerow(row); fh.flush()
        print(f"  seed {seed} rate {rate:.0%} tV {lam_v:<5g} "
              f"G(tau0) {GS_tau0:>12,.0f}  G(tau*) {GS_taustar:>12,.0f}  "
              f"DWL share {row['DWL_share_hat']:>6.2%} "
              f"(true {DWL_share_true:.2%})  "
              f"|wedge| {row['wedge_absmean']:.3f} "
              f"(true {wedge_absmean_true:.3f})  {row['secs']:.0f}s",
              flush=True)
    fh.close()
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

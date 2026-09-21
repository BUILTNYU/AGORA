#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot IO2: estimate (thetat, sigma) by a single least squares, no outer loop.

WHY THIS REPLACES THE OUTER LOOP
--------------------------------
sigma is the multiplier on the capacity constraint, so complementary slackness
gives sigma_ma (y0_ma - f0_a) = 0. When y0 and the aggregate link counts f0_a
are observed on every owned link -- which Assumption 2/4 already grants, since
only the disaggregate segment flows are sampled -- the SUPPORT of sigma is
therefore data, not an estimand:

    sigma_ma = 0 wherever f0_a < y0_ma.

Verified on the shipped instance: all 124 links with sigma>0 sit at load factor
1.000000, and 100% of sigma's mass lies on {f0 >= y0}. That collapses sigma from
2275 unknowns to 124, and the cycle-projected stationarity condition

    C q = C Z thetat - C (tau0 + sigma)        (Eq. 57 projected)

becomes ONE bound-constrained linear least squares in (thetat, sigma_S):

    min || (Cq + C tau0) - [CZ, -CP_S] [thetat; sigma_S] ||^2
        + lambda_sigma ||sigma_S||^2 ,      sigma_S >= 0.

Convex, solved once, in seconds. No MSA, no gamma_r, no fixed point, and so
none of the convergence pathology the iterative version has: with a diminishing
step the iterate can travel unboundedly (sum_r gamma_r gap_r diverges unless
a + c > 1), which is what the 40-iteration runs showed.

WHAT IT DOES NOT FIX
--------------------
Two things the one-shot exposes rather than cures, both worth reporting:

  1. RANK. Even at full observation [CZ, -CP_S] is 313 x 134 with rank 130.
     The 4-dimensional null space is interpretable: three directions are pure
     sigma differences between CONSECUTIVE links on one rail branch (HEMP pair,
     ATLA triple) -- travellers use them in sequence, so only the corridor SUM
     of sigma is identified, never the split. The fourth mixes thetat with
     sigma on CITY/ATLA. At a 30% flow sample the rank falls to 52 of 134.
     lambda_sigma is what selects within that null space, which is why it is a
     modelling choice and not a nuisance knob.

  2. MISSPECIFICATION. The truth does not satisfy these equations: residual
     22.48 against ||rhs|| = 143.96, i.e. 15.6%. The fitted solution fits
     BETTER (3.30) than the truth does. That is the assignment misspecification,
     and it is why cross-validation cannot select lambda_sigma -- out-of-sample
     fit selects toward truth only when the model is approximately right.

CHOOSING lambda_sigma
---------------------
Two defensible rules, neither of them CV:
  * scale anchor -- pick lambda so max sigma_hat is of the order of max tau0.
    A rationing premium far above the fare structure would mean the operator is
    leaving money on the table; far below means rationing is immaterial. On this
    instance it selects 0.01 at full observation and 0.001 at 30%, both of which
    are the error-minimising values.
  * sampled sigma -- if sigma is measured on a handful of links, pick lambda to
    predict them. Works at full observation (40 links: the oracle lambda in 9 of
    10 draws); at 30% it cannot, because held-out error is flat in lambda there.

    python3 algorithm3.py --seeds 0 1 2 3 4 --rates 0.10 0.30 0.50 --with-step9
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
from scipy.optimize import lsq_linear

import lirr_bench as R

# The rest of the package is imported inside main(), AFTER chdir'ing to the data
# directory: design.py (and others) read their CSVs at import time by relative
# path, so importing them from anywhere else raises FileNotFoundError before
# main() can run. lirr_bench.py defers its imports for the same reason.
design = None
P2Instance = solve_p2 = solve_p2_fixed_price = None
IO2Data = build_projection = seller_block = cycle_basis = None


def _load_package() -> None:
    """Import the LIRR package modules. Call only after chdir to the data dir."""
    global design, P2Instance, solve_p2, solve_p2_fixed_price
    global IO2Data, build_projection, seller_block, cycle_basis
    import design as _design
    from algorithm2_p2 import P2Instance as _P2, solve_p2 as _sp2
    from algorithm2prime_p2 import solve_p2_fixed_price as _spf
    from algorithm3_blocks import (IO2Data as _D, build_projection as _bp,
                                seller_block as _sb, cycle_basis as _cb)
    design = _design
    P2Instance, solve_p2, solve_p2_fixed_price = _P2, _sp2, _spf
    IO2Data, build_projection, seller_block, cycle_basis = _D, _bp, _sb, _cb


def projection_with_weights(inst, data, active):
    """Cycle projection, plus the Remark-10 row weights.

    build_projection discards C, so the per-cell heteroskedasticity of Remark 10
    cannot be applied afterwards -- the projection mixes cells. Rebuilding it
    here keeps C long enough to push the weights through.

    Remark 10: var(residual_ia) ~ ell_a / (D_i^2 f_ia). Cycle row j is a signed
    sum over the cells it contains, so var(row j) = sum_cell C[j,cell]^2 var(cell)
    and the WLS weight is 1/sqrt of that. Weights are normalised by their median,
    so lambda_sigma keeps the same meaning weighted or not.
    """
    Cq_b, CZ_b, CP_b, V_b = [], [], [], []
    for i in range(inst.n_seg):
        cl = np.flatnonzero(active[i])
        if cl.size == 0:
            continue
        C = cycle_basis(inst.tail, inst.head, cl, inst.n_nodes)
        if C.shape[0] == 0:
            continue
        f = data.f_seg[i, cl]
        P = np.zeros((cl.size, inst.n_own))
        for r_, a in enumerate(cl):
            k = inst.dual_of_link[a]
            if k >= 0:
                P[r_, k] = 1.0
        Cq_b.append(C @ (inst.ell[cl] * f / inst.D[i]))
        CZ_b.append(C @ data.z[i, cl, :])
        CP_b.append(C @ P)
        V_b.append((C ** 2) @ (inst.ell[cl] /
                               (inst.D[i] ** 2 * np.maximum(f, 1e-9))))
    if not Cq_b:
        raise ValueError("no cycles on the sampled support")
    w = 1.0 / np.sqrt(np.maximum(np.concatenate(V_b), 1e-30))
    return (np.concatenate(Cq_b), np.vstack(CZ_b), np.vstack(CP_b),
            w / np.median(w))

def design_admissibility(A, nF, sigma_true_S=None):
    """rank_theta and support coverage: is this design capable of identifying?

    rank_theta is the rank of (I - P_sigma) CZ, theta's identified dimension
    AFTER sigma has absorbed everything it can. It caps at |Theta| - 1 = 6 on
    this instance at EVERY rate including full observation, because `drive` is
    exactly in the column space of the sigma block (see normalize_drive.py):
    CZ e_drive = CP_S u with u supported on one CITY link and a three-link ATLA
    corridor. So 6 is the admissible target, not 7.

    cover is the fraction of support columns that any sampled cycle row
    touches. An uncovered sigma_a is set to 0 by the ridge and contributes its
    whole true value to the error, so cover bounds relerr from below.
    """
    B = A[:, nF:]
    covered = np.linalg.norm(B, axis=0) > 1e-9
    Bc = B[:, covered]
    if Bc.shape[1]:
        Q, _ = np.linalg.qr(Bc)
        Q = Q[:, :int(np.linalg.matrix_rank(Bc))]
        CZperp = A[:, :nF] - Q @ (Q.T @ A[:, :nF])
    else:
        CZperp = A[:, :nF]
    out = dict(rank_theta=int(np.linalg.matrix_rank(CZperp)),
               cover=float(covered.mean()), n_covered=int(covered.sum()))
    if sigma_true_S is not None and np.linalg.norm(sigma_true_S) > 0:
        out["cover_mass"] = float(
            np.linalg.norm(sigma_true_S[covered]) /
            np.linalg.norm(sigma_true_S))
    return out


COLUMNS = [
    "seed", "rate", "secs", "lambda_sigma", "lambda_theta", "weight",
    "sigma_cap", "fix_drive",
    "n_eq", "n_unknown", "design_rank", "rank_deficit", "design_cond",
    "rank_theta", "cover", "n_covered", "cover_mass", "admissible",
    "n_support", "sigma_hat_max", "sigma_hat_pos", "tau0_max",
    "ls_residual", "ls_residual_at_truth",
    *R.FEATURE_NAMES, "alpha_S", "c_mean",
    "sigma_corr", "sigma_relerr", "sigma_err_l1",
    "wmape_all", "wmape_held", "wmape_val", "r2_all", "rmse_all",
    "recall", "precision",
    "GS_tau0_hat", "GS_tau0_true", "GS_tau0_relerr",
    "tau_star_mean", "wedge_mean", "wedge_absmean", "wedge_corr",
]


def estimate_case(bench, seed, rate, lam_sigma, args):
    """The estimation half of one_case: everything before the forward solves.

    Factored out so that a re-run needing only step 9 (the clearing solve) can
    reproduce EXACTLY the same estimate without re-deriving it. Returns a dict
    of the fitted primitives and the design diagnostics; one_case consumes it
    and then does the forward evaluation.
    """
    sets = R.nested_sets(design, [rate, round(args.val_multiple * rate, 4)],
                         seed, args.design_alpha, "stable")
    keep = bench.links.link_id.isin(sets[rate]).to_numpy()
    vkeep = bench.links.link_id.isin(
        sets[round(args.val_multiple * rate, 4)]).to_numpy() & ~keep
    cells = bench.avail & keep[None, :]
    data = IO2Data(f_seg=np.where(cells, bench.f_true, np.nan), y=bench.y_obs,
                   f_link=bench.f_link_true, tau0=bench.tau0, z=bench.Z,
                   c_prior=bench.c_prior, c_idle_prior=np.array(R.C_IDLE_TRUE))

    # --- seller block: unchanged, and independent of sigma ------------------
    c_hat, c_idle_hat, pi_hat, aS, _, _ = seller_block(
        bench.inst, data, args.rho_c, args.rho_pi, 1e-6)

    # --- support of sigma, from observables only ----------------------------
    S = (data.f_link >= bench.y_obs * (1.0 - args.bind_tol)) & (bench.y_obs > 0)
    nS = int(S.sum())

    # --- one-shot least squares --------------------------------------------
    A_pos = np.isfinite(data.f_seg) & (data.f_seg > 0)
    Cq, CZ, CP, rw = projection_with_weights(bench.inst, data, A_pos)
    if args.weight == "none":
        rw = np.ones_like(rw)
    rhs = Cq + CP @ data.tau0
    nF = CZ.shape[1]
    A = np.column_stack([CZ, -CP[:, S]])
    rank = int(np.linalg.matrix_rank(A))
    sv = np.linalg.svd(A, compute_uv=False)
    lt = float(max(getattr(args, "lambda_theta", 0.0), 0.0))
    th0 = np.linalg.lstsq(CZ, rhs, rcond=None)[0]     # stage 1: sigma = 0
    adm = design_admissibility(A, nF, bench.sigma_true[S]
                               if bench.sigma_true is not None else None)

    # sigma's upper bound. The scale anchor said a rationing premium far above
    # the fare structure means the operator is leaving money on the table; that
    # is a statement about sigma's RANGE, so impose it as a bound rather than
    # back it out of lambda. Measured at 50%: the bound alone takes relerr from
    # 1.04-1.78 (free) to 0.64-0.81 at lambda = 1e-4, and makes the estimate
    # insensitive to lambda across three decades (spread <= 0.13). A penalty
    # has to be re-tuned per rate and per seed; a bound does not.
    cap = (float(args.sigma_cap) * float(data.tau0.max())
           if args.sigma_cap and args.sigma_cap > 0 else np.inf)

    # `drive` is EXACTLY unidentified (Remark 14): CZ e_drive = CP_S u, so
    # theta_drive += t and sigma += t*u fit identically at every sampling rate.
    # Normalising it is the only fix; no design breaks the confound.
    jd = R.FEATURE_NAMES.index("drive") if "drive" in R.FEATURE_NAMES else -1
    fixv = None
    if args.fix_drive == "zero":
        fixv = 0.0
    elif args.fix_drive == "stage1":
        fixv = float(th0[jd])
    free = [k for k in range(nF + nS) if not (fixv is not None and k == jd)]
    nFf = nF - (1 if fixv is not None else 0)

    # WLS: scale the cycle rows, not the ridge block -- lambda_sigma is a
    # penalty on sigma itself and must not be re-scaled with the data rows.
    Aw, rhw = A * rw[:, None], rhs * rw
    if fixv is not None:
        rhw = rhw - Aw[:, jd] * fixv
    blocks = [Aw[:, free],
              np.hstack([np.zeros((nS, nFf)), np.sqrt(lam_sigma) * np.eye(nS)])]
    bb = [rhw, np.zeros(nS)]
    if lt > 0.0:
        blocks.append(np.hstack([np.sqrt(lt) * np.eye(nFf),
                                 np.zeros((nFf, nS))]))
        bb.append(np.sqrt(lt) * th0[[k for k in range(nF) if k in free]])
    Aa = np.vstack(blocks)
    res = lsq_linear(Aa, np.concatenate(bb),
                     bounds=(np.r_[np.full(nFf, -np.inf), np.zeros(nS)],
                             np.r_[np.full(nFf, np.inf), np.full(nS, cap)]),
                     max_iter=1000)
    theta = np.empty(nF)
    theta[[k for k in range(nF) if k in free]] = res.x[:nFf]
    if fixv is not None:
        theta[jd] = fixv
    sg = res.x[nFf:]
    res = type("R_", (), {"x": np.r_[theta, sg]})()
    sigma = np.zeros(bench.inst.n_own)
    sigma[S] = sg
    ls_res = float(np.linalg.norm(rhs - A @ res.x))
    r_truth = float(np.linalg.norm(
        rhs - A @ np.r_[np.array(R.THETA_TRUE, float), bench.sigma_true[S]]))

    phi = np.where(bench.inst.avail, data.z @ theta, 0.0)
    est = P2Instance(
        tail=bench.inst.tail, head=bench.inst.head,
        link_owner=bench.inst.link_owner, n_nodes=bench.inst.n_nodes,
        D=bench.inst.D, origin=bench.inst.origin, sink=bench.inst.sink,
        F=bench.inst.F, Cmax=bench.inst.Cmax, c=c_hat, ell=bench.inst.ell,
        phi=phi, alpha_B=1.0, alpha_S=aS, avail=bench.inst.avail,
        c_idle=c_idle_hat, link_names=bench.inst.link_names,
        segment_names=bench.inst.segment_names,
        operator_names=bench.inst.operator_names,
        node_names=bench.inst.node_names,
        outside_link=bench.inst.outside_link, b_supply=bench.inst.b_supply)
    return dict(est=est, phi=phi, theta=theta, sigma=sigma, sg=sg, S=S, nS=nS,
                nF=nF, c_hat=c_hat, c_idle_hat=c_idle_hat, aS=aS, data=data,
                keep=keep, vkeep=vkeep, cap=cap, lt=lt, adm=adm, rank=rank,
                sv=sv, A=A, ls_res=ls_res, r_truth=r_truth)


def one_case(bench, seed, rate, lam_sigma, args, truth_wedge=None):
    t0 = time.perf_counter()
    E = estimate_case(bench, seed, rate, lam_sigma, args)
    (est, phi, theta, sigma, S, nS, nF, c_hat, c_idle_hat, aS, data, keep,
     vkeep, cap, lt, adm, rank, sv, ls_res, r_truth) = (
        E['est'], E['phi'], E['theta'], E['sigma'], E['S'], E['nS'], E['nF'],
        E['c_hat'], E['c_idle_hat'], E['aS'], E['data'], E['keep'],
        E['vkeep'], E['cap'], E['lt'], E['adm'], E['rank'], E['sv'],
        E['ls_res'], E['r_truth'])

    # --- forward solve at the estimated primitives (step 6, ONCE) -----------
    fp = solve_p2_fixed_price(est, data.tau0, tol=args.inner_tol,
                              max_iter=args.step6_max_iter)
    fit = bench.trace_metrics(fp.f_seg, phi, c_hat, keep, vkeep)

    row = dict(
        seed=seed, rate=rate, lambda_sigma=lam_sigma,
        lambda_theta=lt, weight=args.weight,
        sigma_cap=float(cap) if np.isfinite(cap) else float("inf"),
        fix_drive=args.fix_drive,
        n_eq=int(A.shape[0]), n_unknown=int(nF + nS), design_rank=rank,
        rank_deficit=int(nF + nS - rank),
        admissible=int(adm["rank_theta"] >= args.min_rank_theta), **adm,
        design_cond=float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        n_support=nS, sigma_hat_max=float(sigma.max()),
        sigma_hat_pos=int((sigma > 1e-9).sum()),
        tau0_max=float(data.tau0.max()),
        ls_residual=ls_res, ls_residual_at_truth=r_truth,
        alpha_S=float(aS), c_mean=float(c_hat.mean()),
        **dict(zip(R.FEATURE_NAMES, map(float, theta))), **fit,
    )
    if bench.sigma_true is not None:
        st = bench.sigma_true
        row.update(
            sigma_corr=float(np.corrcoef(sigma, st)[0, 1]),
            sigma_relerr=float(np.linalg.norm(sigma - st) / np.linalg.norm(st)),
            sigma_err_l1=float(np.abs(sigma - st).sum()))

    if args.with_step9:
        clr = solve_p2(est, tol=args.inner_tol, max_iter=args.clearing_max_iter)
        wedge = data.tau0 - clr.tau
        row.update(tau_star_mean=float(clr.tau.mean()),
                   wedge_mean=float(wedge.mean()),
                   wedge_absmean=float(np.abs(wedge).mean()))
        if truth_wedge is not None:
            row["wedge_corr"] = float(np.corrcoef(wedge, truth_wedge)[0, 1])
    row["secs"] = round(time.perf_counter() - t0, 1)
    return row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--rates", type=float, nargs="+", default=[0.30])
    p.add_argument("--lambda-sigma", type=float, nargs="+", default=[0.001],
                   help="one value, or one per rate (scale-anchor calibrated)")
    p.add_argument("--weight", choices=("remark10", "none"), default="remark10",
                   help="cycle-row weighting of the least squares. 'remark10' "
                        "uses 1/sqrt(var) with var(row) = sum_cell C^2 * "
                        "ell_a/(D_i^2 f_ia), the heteroskedasticity Remark 10 "
                        "already identifies. Measured at 30%%: it moves drive "
                        "from -4.64 to -1.37 and wmape_val from 0.881 to 0.732 "
                        "with sigma correlation slightly IMPROVED (0.496 to "
                        "0.506), at no cost in convexity and no forward solve. "
                        "GS degrades (-7.8%% to -26.8%%) because money shifts "
                        "0.157 against a sensitivity of -2.1e6. 'none' "
                        "reproduces the unweighted estimator.")
    p.add_argument("--lambda-theta", type=float, default=0.0,
                   help="ridge on theta toward the STAGE-1 estimate (the sigma=0 "
                        "solve). Pulls `drive` back toward a flow-fitting region "
                        "-- it trades sigma accuracy for wMAPE, so report both")
    p.add_argument("--sigma-cap", type=float, default=1.0,
                   help="upper bound on sigma, as a MULTIPLE of max(tau0). The "
                        "scale anchor turned into a restriction instead of a "
                        "penalty: a rationing premium far above the fare "
                        "structure means the operator is leaving money on the "
                        "table. At 50%% the bound alone takes sigma relerr from "
                        "1.04-1.78 to 0.64-0.81 at lambda=1e-4 and makes the "
                        "estimate insensitive to lambda across three decades. "
                        "0 or negative disables the bound (the old behaviour)")
    p.add_argument("--fix-drive", choices=("none", "zero", "stage1"),
                   default="none",
                   help="normalise the `drive` constant instead of estimating "
                        "it. `drive` is EXACTLY unidentified: CZ e_drive = "
                        "CP_S u with u on one CITY link and a three-link ATLA "
                        "corridor, so theta_drive += t and sigma += t*u fit "
                        "identically at every sampling rate, full observation "
                        "included. At 30%% `stage1` takes mean relerr from "
                        "1.010 to 0.847 and the worst seed from 1.192 to "
                        "0.896; at 50%% it COSTS accuracy (0.741 to 0.834) "
                        "because the bound already controls the direction and "
                        "the normalisation only adds bias t*u. Report which "
                        "was used")
    p.add_argument("--min-rank-theta", type=int, default=6,
                   help="admissibility threshold on rank_theta, the identified "
                        "dimension of theta after sigma absorbs what it can. 6 "
                        "not 7, because `drive` is structurally confounded. "
                        "Reached at 30%% by all five seeds, at 20%% by one, at "
                        "10%% by none -- so 10%% is an inadmissible design, and "
                        "that is a property of the sample, not of the seed")
    p.add_argument("--rho-c", type=float, default=100.0)
    p.add_argument("--rho-pi", type=float, default=1.0)
    p.add_argument("--bind-tol", type=float, default=1e-5,
                   help="f0 >= y0 (1 - tol) counts as binding, hence in the support")
    p.add_argument("--step6-max-iter", type=int, default=40)
    p.add_argument("--inner-tol", type=float, default=1e-4)
    p.add_argument("--clearing-max-iter", type=int, default=20000)
    p.add_argument("--with-step9", action="store_true")
    p.add_argument("--design-alpha", type=float, default=0.5)
    p.add_argument("--val-multiple", type=float, default=1.2)
    p.add_argument("--truth-wedge", type=Path, default=None,
                   help="npz from run_step9.py --use-truth, for wedge_corr")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="the LIRR package directory. Defaults to this script's "
                        "own folder, then the working directory -- the package "
                        "modules read their CSVs by relative path, so the "
                        "process must chdir there before Bench() is built")
    p.add_argument("--out", type=Path, default=Path("io2_oneshot.csv"))
    args = p.parse_args()

    lam = args.lambda_sigma
    if len(lam) == 1:
        lam = lam * len(args.rates)
    if len(lam) != len(args.rates):
        print("--lambda-sigma must be one value or one per rate")
        return 2

    # The package modules (build_lirr, design, ...) read their CSVs by relative
    # path, so chdir into the data directory before anything touches them --
    # exactly as lirr_bench.py does. Resolve --out first, or a relative
    # output path would land in the data directory instead of where it was asked
    # for.
    out_path = args.out.resolve()
    data_dir = R.locate_data_dir(args.data_dir)
    os.chdir(data_dir)
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    print(f"  data dir : {data_dir}")
    print(f"  output   : {out_path}")
    _load_package()

    bench = R.Bench(verbose=True)
    tw = np.load(args.truth_wedge)["wedge"] if args.truth_wedge else None
    fh = out_path.open("w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    w.writeheader(); fh.flush()
    for rate, ls in zip(args.rates, lam):
        for seed in args.seeds:
            row = one_case(bench, seed, rate, ls, args, tw)
            w.writerow(row); fh.flush()
            print(f"  seed {seed} rate {rate:.0%}  lam {ls:g}  "
                  f"rank {row['design_rank']}/{row['n_unknown']}  "
                  f"wmape_val {row['wmape_val']:.4f}  "
                  f"GS {row['GS_tau0_relerr']:+.2%}  "
                  f"sig corr {row.get('sigma_corr', float('nan')):.3f}  "
                  f"{row['secs']:.0f}s", flush=True)
    fh.close()
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

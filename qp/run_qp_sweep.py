#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Algorithm 3 as a single buyer QP (node potentials kept), LIRR sweep.

For each (seed, rate) this runs the same pipeline as run_steps_7_8.py +
run_step_9.py, with the buyer block of the earlier cycle-projection estimator
(cycle projection, inactive-cell rows, hinge refit) replaced by ONE convex QP:

    min  sum_{observed active ia} (q_ia - g_ia)^2  +  lambda_sigma ||sigma||^2
    s.t. g_ia <= 0                  observed zero cells (hard)
         0 <= sigma_a <= kappa      a in S = {f0_a >= y0_a (1 - 1e-5)}
         sigma_a = 0                a not in S
         mu_{i, sink_i} = 0
    g_ia = z_ia' theta - tau0_a - sigma_a - mu_{i,tail(a)} + mu_{i,head(a)}
    q_ia = ell_a f_ia / D_i                                  (alpha_B = 1)

Unobserved cells simply do not enter. Everything else is unchanged: the same
sampling design (lirr_bench.nested_sets), the same seller block
(algorithm3.estimate_case), the same forward solve (Algorithm 2', 40
iterations), the same clearing solve for tau*, and the same metrics.

Welfare truth is reported TWO ways:
  *_paper  phi_true = Z theta_true (7 features), as lirr_bench / run_step_9 do
  *_gen    phi_true = the generator's phi (inst.phi, with the 97 station ASCs)
The generator's phi is the one that produced the flows, so *_gen is the
correct benchmark; *_paper is kept so the numbers can be compared with the
current draft.

The ablation arms are flags (run_ablation.bat runs them):
  --zero-cells none        drop the zero-cell inequalities
  --zero-set R             impose only the zero cells Algorithm 3 can reach
  --sigma-cap 0            remove the bound on sigma
  --no-sigma               sigma == 0
  --zero-cells hinge --rho-v X   squared hinge instead of hard constraints

Resumable: --resume skips (seed, rate) cases already in --out.

    python run_qp_sweep.py --release .. --out qp_sweep.csv --resume     (from qp/)
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
import scipy.sparse as sp

COLUMNS = [
    "seed", "rate", "method", "zero_cells", "zero_set", "rho_v", "lambda_sigma",
    "sigma_cap", "no_sigma", "qp_status", "qp_secs",
    "n_obs_links", "n_act_obs", "n_zero_obs", "n_zero_imposed", "n_R", "n_support",
    "eq_resid", "rhs_norm", "zero_viol_frac_insample",
    "t_RAIL", "t_DRIVE", "t_TRAN", "t_EGR", "t_OUT", "money", "drive",
    "alpha_S", "alpha_S_LIRR", "alpha_S_LOCAL", "alpha_S_PARK", "c_mean",
    "sigma_hat_max", "sigma_hat_pos", "sigma_corr", "sigma_relerr",
    # step 7-8: forward fit and support check at the Algorithm 2' solution
    "r2_all", "wmape_all", "wmape_held", "wmape_val", "rmse_all",
    "recall", "precision",
    "viol_A0", "n_A0", "viol_R", "viol_VmR", "n_VmR",
    "viol_notimposed", "n_notimposed",
    "fp_residual", "fp_iters", "fp_converged",
    # step 9: clearing solve, welfare
    "GS_tau0_hat", "GS_taustar_hat", "DWL_hat", "DWL_share_hat",
    "GS_tau0_true_paper", "GS_taustar_true_paper", "DWL_true_paper",
    "GS_tau0_relerr_paper", "DWL_relerr_paper", "wedge_absmean_err_paper", "wedge_corr_paper",
    "GS_tau0_true_gen", "GS_taustar_true_gen", "DWL_true_gen",
    "GS_tau0_relerr_gen", "DWL_relerr_gen", "wedge_absmean_err_gen", "wedge_corr_gen",
    "tau_star_mean", "wedge_mean", "wedge_absmean",
    "clear_iters", "clear_residual", "clear_converged", "secs",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release", type=Path, required=True,
                   help="the release root (contains code/ and data/lirr/); '..' from qp/")
    p.add_argument("--out", type=Path, default=Path("qp_sweep.csv"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--rates", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument("--zero-cells", choices=("hard", "hinge", "none"), default="hard")
    p.add_argument("--zero-set", choices=("all", "R"), default="all",
                   help="impose every observed zero cell, or only those in R")
    p.add_argument("--rho-v", type=float, default=1.0, help="hinge weight (--zero-cells hinge)")
    p.add_argument("--lambda-sigma", type=float, default=1e-3)
    p.add_argument("--sigma-cap", type=float, default=1.0,
                   help="kappa as a multiple of max tau0; 0 removes the bound")
    p.add_argument("--no-sigma", action="store_true")
    p.add_argument("--step7-max-iter", type=int, default=40)
    p.add_argument("--inner-tol", type=float, default=1e-4)
    p.add_argument("--clearing-max-iter", type=int, default=20000)
    p.add_argument("--skip-step9", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log", type=Path, default=None, help="also append console output here")
    return p.parse_args()


class _Tee:
    def __init__(self, stream, path):
        self.stream, self.fh = stream, open(path, "a", encoding="utf-8")
    def write(self, s):
        self.stream.write(s); self.fh.write(s); self.fh.flush()
    def flush(self):
        self.stream.flush(); self.fh.flush()


# ---------------------------------------------------------------------------
# Buyer QP
# ---------------------------------------------------------------------------

def build_design(inst, Z, tau0_link, S):
    """Sparse G with g = G x over ALL available cells.
    x = [theta (K), sigma_S (nS), mu (one per (segment, node) touched, sink excluded)]."""
    q_i, a_i = np.nonzero(inst.avail)
    n = q_i.size
    K = Z.shape[2]
    nS = int(S.sum())
    sidx = -np.ones(inst.n_own, int)
    sidx[np.flatnonzero(S)] = np.arange(nS)
    t, h = inst.tail[a_i], inst.head[a_i]
    sink = inst.sink[q_i]
    # potential index for (segment, node)
    key_t = q_i * inst.n_nodes + t
    key_h = q_i * inst.n_nodes + h
    keys = np.unique(np.r_[key_t[t != sink], key_h[h != sink]])
    pos = {int(k): j for j, k in enumerate(keys)}
    off_s, off_mu = K, K + nS
    Zc = Z[q_i, a_i, :]
    rr, cc = np.nonzero(Zc)
    rows, cols, vals = [rr], [cc], [Zc[rr, cc]]
    d = inst.dual_of_link[a_i]
    m = (d >= 0)
    m2 = m.copy()
    m2[m] = sidx[d[m]] >= 0
    r_s = np.flatnonzero(m2)
    rows.append(r_s); cols.append(off_s + sidx[d[r_s]]); vals.append(-np.ones(r_s.size))
    r_t = np.flatnonzero(t != sink)
    rows.append(r_t); cols.append(off_mu + np.array([pos[int(k)] for k in key_t[r_t]], int))
    vals.append(-np.ones(r_t.size))
    r_h = np.flatnonzero(h != sink)
    rows.append(r_h); cols.append(off_mu + np.array([pos[int(k)] for k in key_h[r_h]], int))
    vals.append(np.ones(r_h.size))
    G = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, off_mu + keys.size))
    rhs_tau = -tau0_link[a_i]                      # alpha_B = 1: g = G x - tau0
    return G, rhs_tau, q_i, a_i, dict(K=K, nS=nS, off_s=off_s, off_mu=off_mu)


def buyer_qp(inst, Z, f_obs, tau0_own, S, args, zero_mask_R=None):
    import cvxpy as cp
    tau0_link = np.zeros(inst.n_links)
    tau0_link[inst.owned_idx] = tau0_own
    if args.no_sigma:
        S = np.zeros_like(S)
    G, g0, q_i, a_i, o = build_design(inst, Z, tau0_link, S)
    fo = f_obs[q_i, a_i]
    seen = np.isfinite(fo)
    act = seen & (fo > 0)
    zero = seen & (fo <= 0)
    if args.zero_set == "R" and zero_mask_R is not None:
        zero_imp = zero & zero_mask_R[q_i, a_i]
    else:
        zero_imp = zero
    q = inst.ell[a_i] * np.where(act, fo, 0.0) / inst.D[q_i]
    x = cp.Variable(G.shape[1])
    K, nS = o["K"], o["nS"]
    sig = x[o["off_s"]:o["off_s"] + nS]
    GA = G[act]
    r = q[act] - (GA @ x + g0[act])
    obj = cp.sum_squares(r)
    cons = []
    if nS:
        obj = obj + args.lambda_sigma * cp.sum_squares(sig)
        cons.append(sig >= 0)
        if args.sigma_cap > 0:
            cons.append(sig <= args.sigma_cap * float(tau0_own.max()))
    GZ = G[zero_imp]
    if args.zero_cells == "hard" and GZ.shape[0]:
        cons.append(GZ @ x + g0[zero_imp] <= 0)
    elif args.zero_cells == "hinge" and GZ.shape[0]:
        obj = obj + args.rho_v * cp.sum_squares(cp.pos(GZ @ x + g0[zero_imp]))
    prob = cp.Problem(cp.Minimize(obj), cons)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        prob.solve(solver=cp.SCS, eps=1e-8, max_iters=200000)
    secs = time.perf_counter() - t0
    xv = x.value
    theta = xv[:K]
    sigma = np.zeros(inst.n_own)
    sigma[S] = xv[o["off_s"]:o["off_s"] + nS] if nS else 0.0
    gz = G[zero] @ xv + g0[zero]
    return dict(theta=theta, sigma=sigma, status=prob.status, secs=secs,
                eq_resid=float(np.linalg.norm(q[act] - (GA @ xv + g0[act]))),
                rhs_norm=float(np.linalg.norm(q[act])),
                n_act=int(act.sum()), n_zero=int(zero.sum()), n_zero_imp=int(zero_imp.sum()) if args.zero_cells != "none" else 0,
                viol_in=float(np.mean(gz > 1e-6)) if gz.size else float("nan"), nS=nS)


# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.log:
        sys.stdout = _Tee(sys.stdout, args.log)
        sys.stderr = _Tee(sys.stderr, args.log)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    release = args.release.resolve()
    code_dir, data_dir = release / "code", release / "data" / "lirr"
    if not (data_dir / "links_p2.csv").exists():
        sys.exit(f"cannot find {data_dir / 'links_p2.csv'}; pass --release <release root>")
    out_path = args.out.resolve()
    cache = out_path.with_name("qp_sweep_truth_cache.npz")
    os.chdir(data_dir)
    for pth in (str(code_dir), str(data_dir)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    import lirr_bench as R
    import algorithm3 as O
    O._load_package()
    import algorithm3_refit as LR

    method = ("QP" + ("" if args.zero_cells == "hard" else
                      (f"-hinge{args.rho_v:g}" if args.zero_cells == "hinge" else "-nozero"))
              + ("-Ronly" if args.zero_set == "R" else "")
              + ("-nocap" if args.sigma_cap <= 0 else "")
              + ("-nosigma" if args.no_sigma else ""))
    print(f"  release  : {release}\n  output   : {out_path}\n  method   : {method}\n"
          f"  seeds    : {args.seeds}\n  rates    : {args.rates}", flush=True)

    bench = R.Bench(verbose=True)
    inst = bench.inst
    avail = inst.avail
    st = bench.sigma_true
    keep_all = np.ones(inst.n_links, bool)
    zero_link = np.zeros(inst.n_links, bool)
    zero_link[bench.own] = bench.f_link_true <= 0.0

    import dataclasses

    def make_est(phi, c, aS, c_idle):
        # keep the instance's services H, track caps, operators and scale groups
        return dataclasses.replace(inst, c=c, phi=phi, alpha_B=1.0, alpha_S=aS,
                                   c_idle=np.asarray(c_idle, float))

    # ---- truths: one clearing solve each, cached ------------------------------
    truth = {}
    if not args.skip_step9:
        if cache.exists():
            z = np.load(cache)
            truth = {k: z[k] for k in z.files}
            print("  truth    : loaded from cache", flush=True)
        else:
            c_true = bench.c_true
            phi_paper = np.where(avail, bench.Z @ np.array(R.THETA_TRUE, float), 0.0)
            phi_gen = np.where(avail, inst.phi, 0.0)
            for tag, phi in (("paper", phi_paper), ("gen", phi_gen)):
                t0 = time.perf_counter()
                print(f"  truth    : clearing solve at the true primitives ({tag}) ...", flush=True)
                clr = O.solve_p2(make_est(phi, c_true, R.ALPHA_S_TRUE, np.array(R.C_IDLE_TRUE)),
                                 tol=args.inner_tol, max_iter=args.clearing_max_iter)
                gs0 = float((phi * bench.f_true).sum() - c_true @ bench.y_obs)
                gss = float((phi * clr.f_seg).sum() - c_true @ clr.y)
                truth[f"GS0_{tag}"] = np.array(gs0)
                truth[f"GSs_{tag}"] = np.array(gss)
                truth[f"wedge_{tag}"] = bench.tau0 - clr.tau
                truth[f"tau_{tag}"] = clr.tau
                print(f"             G(tau0) {gs0:,.0f}  G(tau*) {gss:,.0f}  DWL {gss - gs0:,.0f}"
                      f"  converged {bool(clr.converged)}  {time.perf_counter() - t0:.0f}s", flush=True)
            np.savez(cache, **truth)

    done = set()
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        done = {(int(t.seed), round(float(t.rate), 4)) for t in old.itertuples()}
        print(f"  resuming : {len(done)} cases already done", flush=True)
    fh = out_path.open("a" if done else "w", newline="")
    w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    if not done:
        w.writeheader()
    fh.flush()

    ea = argparse.Namespace(val_multiple=1.2, design_alpha=0.5, rho_c=100.0, rho_pi=1.0,
                            bind_tol=1e-5, weight="none", sigma_cap=1.0, fix_drive="none",
                            lambda_theta=0.0)
    for rate in sorted(args.rates):
        for seed in args.seeds:
            if (seed, round(rate, 4)) in done:
                continue
            t_case = time.perf_counter()
            # sampling design, seller block, support S: unchanged from Algorithm 3
            E = O.estimate_case(bench, seed, rate, args.lambda_sigma, ea)
            data, S = E["data"], E["S"]
            A_pos = np.isfinite(data.f_seg) & (data.f_seg > 0)
            A_zero = np.isfinite(data.f_seg) & (data.f_seg == 0) & avail
            Rm = LR.reachable_mask(inst, A_pos, A_zero)

            B = buyer_qp(inst, bench.Z, data.f_seg, data.tau0, S, args, zero_mask_R=Rm)
            theta, sigma = B["theta"], B["sigma"]
            phi = np.where(avail, data.z @ theta, 0.0)
            est = make_est(phi, E["c_hat"], E["aS"], E["c_idle_hat"])

            # step 7: forward solve at the fixed fares
            fp = O.solve_p2_fixed_price(est, data.tau0, tol=args.inner_tol,
                                        max_iter=args.step7_max_iter)
            fit = bench.trace_metrics(fp.f_seg, phi, E["c_hat"], E["keep"], E["vkeep"])
            # step 8: support check
            price_all = np.zeros(inst.n_links)
            price_all[bench.own] = data.tau0 + sigma
            u = fp.u
            psi = phi - price_all[None, :] - (u[:, inst.tail] - u[:, inst.head])
            V = A_zero | (avail & zero_link[None, :])
            imposed = A_zero if args.zero_cells != "none" else np.zeros_like(A_zero)
            if args.zero_set == "R":
                imposed = imposed & Rm

            def frac(mask):
                return (float((psi[mask] > 0).mean()) if mask.any() else float("nan"),
                        int(mask.sum()))
            vA0, nA0 = frac(A_zero)
            vR, _ = frac(Rm)
            vVR, nVR = frac(V & ~Rm)
            vNI, nNI = frac(V & ~imposed)

            row = dict(
                seed=seed, rate=rate, method=method, zero_cells=args.zero_cells,
                zero_set=args.zero_set, rho_v=args.rho_v if args.zero_cells == "hinge" else "",
                lambda_sigma=args.lambda_sigma, sigma_cap=args.sigma_cap,
                no_sigma=int(args.no_sigma), qp_status=B["status"], qp_secs=round(B["secs"], 2),
                n_obs_links=int(E["keep"].sum()), n_act_obs=B["n_act"], n_zero_obs=B["n_zero"],
                n_zero_imposed=B["n_zero_imp"], n_R=int(Rm.sum()), n_support=int(S.sum()),
                eq_resid=B["eq_resid"], rhs_norm=B["rhs_norm"],
                zero_viol_frac_insample=B["viol_in"],
                **dict(zip(R.FEATURE_NAMES, map(float, theta))),
                **O._alpha_cols(E["aS"], inst.operator_names), c_mean=float(np.mean(E["c_hat"])),
                sigma_hat_max=float(sigma.max()), sigma_hat_pos=int((sigma > 1e-9).sum()),
                sigma_corr=float(np.corrcoef(sigma, st)[0, 1]) if sigma.std() > 0 else float("nan"),
                sigma_relerr=float(np.linalg.norm(sigma - st) / np.linalg.norm(st)),
                r2_all=fit["r2_all"], wmape_all=fit["wmape_all"], wmape_held=fit["wmape_held"],
                wmape_val=fit["wmape_val"], rmse_all=fit["rmse_all"],
                recall=fit["recall"], precision=fit["precision"],
                viol_A0=vA0, n_A0=nA0, viol_R=vR, viol_VmR=vVR, n_VmR=nVR,
                viol_notimposed=vNI, n_notimposed=nNI,
                fp_residual=float(fp.residual), fp_iters=int(fp.n_iter),
                fp_converged=int(bool(fp.converged)),
                GS_tau0_hat=float(fit["GS_tau0_hat"]))

            # step 9: clearing solve at the estimate
            if not args.skip_step9:
                clr = O.solve_p2(est, tol=args.inner_tol, max_iter=args.clearing_max_iter)
                gss = float((phi * clr.f_seg).sum() - np.asarray(E["c_hat"]) @ np.asarray(clr.y))
                gs0 = row["GS_tau0_hat"]
                dwl = gss - gs0
                wedge = data.tau0 - clr.tau
                row.update(GS_taustar_hat=gss, DWL_hat=dwl, DWL_share_hat=dwl / abs(gss),
                           tau_star_mean=float(clr.tau.mean()), wedge_mean=float(wedge.mean()),
                           wedge_absmean=float(np.abs(wedge).mean()),
                           clear_iters=int(clr.n_iter), clear_residual=float(clr.residual),
                           clear_converged=int(bool(clr.converged)))
                for tag in ("paper", "gen"):
                    g0t, gst = float(truth[f"GS0_{tag}"]), float(truth[f"GSs_{tag}"])
                    wt = truth[f"wedge_{tag}"]
                    row.update({f"GS_tau0_true_{tag}": g0t, f"GS_taustar_true_{tag}": gst,
                                f"DWL_true_{tag}": gst - g0t,
                                f"GS_tau0_relerr_{tag}": (gs0 - g0t) / abs(g0t),
                                f"DWL_relerr_{tag}": (dwl - (gst - g0t)) / abs(gst - g0t),
                                f"wedge_absmean_err_{tag}": float(np.abs(wedge).mean() - np.abs(wt).mean()),
                                f"wedge_corr_{tag}": float(np.corrcoef(wedge, wt)[0, 1])})
            row["secs"] = round(time.perf_counter() - t_case, 1)
            w.writerow(row); fh.flush()
            np.savez(out_path.with_name(f"qp_est_{method}_s{seed}_r{int(round(rate * 100)):03d}.npz"),
                     theta=theta, sigma=sigma, tau_star=(clr.tau if not args.skip_step9 else np.nan))
            print(f"  seed {seed} rate {rate:.0%}  QP {B['status']} {B['secs']:.1f}s  "
                  f"t_RAIL {theta[0]:+.3f} money {theta[5]:+.3f}  sigma relerr {row['sigma_relerr']:.3f}  "
                  f"r2 {row['r2_all']:.4f}  "
                  + (f"DWL err (gen) {row['DWL_relerr_gen']:+.1%} (paper) {row['DWL_relerr_paper']:+.1%}  "
                     if not args.skip_step9 else "")
                  + f"{row['secs']:.0f}s", flush=True)
    fh.close()
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

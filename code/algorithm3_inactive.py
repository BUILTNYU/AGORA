#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Can the inactive-cell information be used WITHOUT retaining the potentials?

Rosenblad, Lin and Yin (2026) keep the OD potentials as free parameters and add
a squared-hinge penalty on the inactive-link inequality. The earlier
cycle-projection estimator (algorithm3_blocks.py) eliminates the potentials,
which is what made the LIRR instance tractable, and an earlier draft therefore
said the zero restrictions "would require
retaining mu" and demotes them to a post-estimation check (step 8).

That is not quite right. For an inactive cell (i,a) whose endpoints are joined
by a path P of ACTIVE cells of the same segment, the potentials telescope:

    u_{t(a)} - u_{h(a)} = sum_{b in P} s_b ( g_b - q_b ),     g = Ztheta - p,

because every active cell satisfies g_b - d_b = q_b exactly. Substituting into
the inactive condition g_a - d_a <= 0 gives a potential-free linear inequality

    w . g  <=  w . q ,        w = 1_a - lambda_P ,

which is the SAME row construction as a cycle row, with the chord being a
zero-flow cell instead of a positive-flow one. Since q_a = ell_a f_ia / D_i is
zero on an inactive cell, the right-hand side is literally the same expression
already used for the equality rows.

This script builds those rows, measures how many are violated at the reported
estimate, and re-estimates with a squared-hinge penalty on them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

import lirr_bench as R
import algorithm3 as O


# --------------------------------------------------------------------- rows
def mixed_rows(tail, head, act, ina, n_nodes):
    """Rows for (i) active chords -- equalities -- and (ii) inactive cells.

    The spanning forest is built from the ACTIVE cells only, in the order
    given. Every inactive cell is therefore a chord. Returns
        C_eq   (n_eq   x len(act))
        C_in   (n_in   x len(act)+1) in the packed form (path part, chord)
        in_idx the inactive cell index for each inequality row
    A row is dropped when the chord's endpoints lie in different components of
    the active forest: no path, no telescoping, no constraint.
    """
    na = act.size
    parent = np.arange(n_nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_nodes)]
    extra: List[int] = []
    for q, a in enumerate(act):
        t, h = int(tail[a]), int(head[a])
        rt, rh = find(t), find(h)
        if rt != rh:
            parent[rt] = rh
            adj[t].append((h, q, +1))
            adj[h].append((t, q, -1))
        else:
            extra.append(q)

    def path_row(t: int, h: int):
        """signed indicator of the forest path t -> h, or None."""
        if t == h:
            return np.zeros(na)
        prev: Dict[int, Tuple[int, int, int]] = {t: (-1, -1, 0)}
        stack = [t]
        while stack:
            n0 = stack.pop()
            if n0 == h:
                break
            for n1, q1, s1 in adj[n0]:
                if n1 not in prev:
                    prev[n1] = (n0, q1, s1)
                    stack.append(n1)
        if h not in prev:
            return None
        row = np.zeros(na)
        node = h
        while node != t:
            n0, q1, s1 = prev[node]
            row[q1] -= s1
            node = n0
        return row

    C_eq = np.zeros((len(extra), na))
    for r, q in enumerate(extra):
        a = act[q]
        pr = path_row(int(tail[a]), int(head[a]))
        if pr is not None:
            C_eq[r] = pr
        C_eq[r, q] += 1.0

    rows_in, idx_in = [], []
    for a in ina:
        pr = path_row(int(tail[a]), int(head[a]))
        if pr is None:
            continue
        rows_in.append(pr)
        idx_in.append(a)
    C_in = np.vstack(rows_in) if rows_in else np.zeros((0, na))
    return C_eq, C_in, np.array(idx_in, int)


def build_blocks(inst, data, A_pos, A_zero):
    """Projected equality and inequality blocks, in the code's own convention.

        equalities :  CZ theta - CP_S sigma  =  Cq + Cp0
        inequality :  CZ theta - CP_S sigma <=  Cq + Cp0     (chord q = 0)
    """
    Eq, Ein = [], []
    for i in range(inst.n_seg):
        act = np.flatnonzero(A_pos[i])
        ina = np.flatnonzero(A_zero[i])
        if act.size == 0:
            continue
        C_eq, C_in, idx_in = mixed_rows(inst.tail, inst.head, act, ina,
                                        inst.n_nodes)
        if C_eq.shape[0] == 0 and C_in.shape[0] == 0:
            continue
        f = data.f_seg[i, act]
        P = np.zeros((act.size, inst.n_own))
        for r_, a in enumerate(act):
            k = inst.dual_of_link[a]
            if k >= 0:
                P[r_, k] = 1.0
        q = inst.ell[act] * f / inst.D[i]
        Z = data.z[i, act, :]
        p0 = P @ data.tau0

        if C_eq.shape[0]:
            Eq.append((C_eq @ Z, C_eq @ P, C_eq @ q + C_eq @ p0))
        if C_in.shape[0]:
            # the chord contributes its own Z, p0 (and q = 0) with coefficient +1
            Pc = np.zeros((idx_in.size, inst.n_own))
            for r_, a in enumerate(idx_in):
                k = inst.dual_of_link[a]
                if k >= 0:
                    Pc[r_, k] = 1.0
            Zc = data.z[i, idx_in, :]
            Ein.append((C_in @ Z + Zc, C_in @ P + Pc,
                        C_in @ q + C_in @ p0 + Pc @ data.tau0))

    def cat(B):
        if not B:
            return (np.zeros((0, data.z.shape[2])), np.zeros((0, inst.n_own)),
                    np.zeros(0))
        return (np.vstack([b[0] for b in B]), np.vstack([b[1] for b in B]),
                np.concatenate([b[2] for b in B]))
    return cat(Eq), cat(Ein)


# --------------------------------------------------------------------- main
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rate", type=float, default=0.3)
    p.add_argument("--lambda-sigma", type=float, default=1e-3)
    p.add_argument("--lam-v", type=float, nargs="+",
                   default=[0.0, 0.01, 0.1, 1.0])
    p.add_argument("--forward", type=float, nargs="*", default=None,
                   help="run step 7 and report held-out fit at these lam_V")
    args = p.parse_args()

    data_dir = R.locate_data_dir(None)
    os.chdir(data_dir)
    if str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))
    O._load_package()
    bench = R.Bench(verbose=False)
    inst = bench.inst
    a = argparse.Namespace(val_multiple=1.2, design_alpha=0.5, rho_c=100.0,
                           rho_pi=1.0, bind_tol=1e-5, weight="none",
                           sigma_cap=1.0, fix_drive="none", lambda_theta=0.0)

    t0 = time.perf_counter()
    E = O.estimate_case(bench, args.seed, args.rate, args.lambda_sigma, a)
    data, S, nS = E["data"], E["S"], E["nS"]
    theta0, sg0 = E["theta"], E["sg"]
    nF = E["nF"]
    print(f"  baseline estimate reproduced in {time.perf_counter()-t0:.0f}s")

    A_pos = np.isfinite(data.f_seg) & (data.f_seg > 0)
    A_zero = np.isfinite(data.f_seg) & (data.f_seg == 0) & inst.avail
    print(f"  sampled cells: {int(A_pos.sum()):,} active, "
          f"{int(A_zero.sum()):,} inactive")

    t0 = time.perf_counter()
    (Ze, Pe, be), (Zi, Pi, bi) = build_blocks(inst, data, A_pos, A_zero)
    print(f"  rows built in {time.perf_counter()-t0:.0f}s: "
          f"{Ze.shape[0]:,} equality, {Zi.shape[0]:,} inequality "
          f"({Zi.shape[0] / max(int(A_zero.sum()), 1):.1%} of inactive cells "
          f"reachable through the active forest)")

    Ae = np.column_stack([Ze, -Pe[:, S]])
    Ai = np.column_stack([Zi, -Pi[:, S]])
    x0 = np.r_[theta0, sg0]

    # sanity: the equality block must reproduce the shipped design
    r_here = float(np.linalg.norm(be - Ae @ x0))
    print(f"  equality residual at the reported estimate: {r_here:.6g} "
          f"(shipped: {E['ls_res']:.6g})")

    v0 = Ai @ x0 - bi
    print(f"\n  INACTIVE-CELL INEQUALITIES AT THE REPORTED ESTIMATE")
    print(f"    violated (v > 0)        : {np.mean(v0 > 0):.2%} "
          f"of {v0.size:,}")
    print(f"    max violation           : {v0.max():+.4f}")
    print(f"    mean of the positives   : "
          f"{v0[v0 > 0].mean() if (v0 > 0).any() else 0.0:+.4f}")
    print(f"    sum of squared positives: {np.sum(np.maximum(v0, 0)**2):.4g}")

    # ---------------- re-estimate with a squared hinge on the inequalities
    cap = E["cap"]
    lo = np.r_[np.full(nF, -np.inf), np.zeros(nS)]
    hi = np.r_[np.full(nF, np.inf), np.full(nS, cap)]
    bounds = list(zip(lo, hi))
    lam_s = args.lambda_sigma

    def make(lam_v):
        def fg(x):
            r = Ae @ x - be
            v = np.maximum(Ai @ x - bi, 0.0)
            s = x[nF:]
            f = r @ r + lam_s * (s @ s) + lam_v * (v @ v)
            g = 2.0 * (Ae.T @ r) + 2.0 * lam_v * (Ai.T @ v)
            g[nF:] += 2.0 * lam_s * s
            return f, g
        return fg

    print(f"\n  RE-ESTIMATION WITH A SQUARED HINGE ON THE INACTIVE ROWS")
    hdr = (f"  {'lam_V':>7s} {'eq resid':>10s} {'viol':>8s} {'max v':>8s} "
           f"{'|dtheta|/|theta|':>17s}  " + "  ".join(
               f"{n:>8s}" for n in R.FEATURE_NAMES))
    print(hdr)
    base = None
    for lam_v in args.lam_v:
        res = minimize(make(lam_v), x0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options=dict(maxiter=4000, ftol=1e-14,
                                                   gtol=1e-10))
        x = res.x
        th = x[:nF]
        v = Ai @ x - bi
        rq = float(np.linalg.norm(Ae @ x - be))
        if base is None:
            base = th.copy()
        rel = np.linalg.norm(th - theta0) / np.linalg.norm(theta0)
        print(f"  {lam_v:7.3g} {rq:10.5f} {np.mean(v > 0):7.2%} "
              f"{v.max():8.3f} {rel:16.2%}   " +
              "  ".join(f"{t:8.4f}" for t in th))

    print(f"\n  reported theta {' ' * 40}" +
          "  ".join(f"{t:8.4f}" for t in theta0))

    if args.forward is None:
        return 0

    print(f"\n  FORWARD EVALUATION (step 7, Algorithm 2', 40 iterations)")
    print(f"  {'lam_V':>7s} {'r2_all':>9s} {'wmape_all':>10s} "
          f"{'wmape_val':>10s} {'sigma_corr':>11s} {'sigma_relerr':>13s} "
          f"{'secs':>6s}")
    st = bench.sigma_true
    for lam_v in args.forward:
        res = minimize(make(lam_v), x0, jac=True, method="L-BFGS-B",
                       bounds=bounds, options=dict(maxiter=4000, ftol=1e-14,
                                                   gtol=1e-10))
        th, sg = res.x[:nF], res.x[nF:]
        sig = np.zeros(inst.n_own); sig[S] = sg
        ph = np.where(inst.avail, data.z @ th, 0.0)
        e2 = O.P2Instance(
            tail=inst.tail, head=inst.head, link_owner=inst.link_owner,
            n_nodes=inst.n_nodes, D=inst.D, origin=inst.origin,
            sink=inst.sink, F=inst.F, Cmax=inst.Cmax, c=E["c_hat"],
            ell=inst.ell, phi=ph, alpha_B=1.0, alpha_S=E["aS"],
            avail=inst.avail, c_idle=E["c_idle_hat"],
            link_names=inst.link_names, segment_names=inst.segment_names,
            operator_names=inst.operator_names, node_names=inst.node_names,
            outside_link=inst.outside_link, b_supply=inst.b_supply)
        t1 = time.perf_counter()
        fp = O.solve_p2_fixed_price(e2, data.tau0, tol=1e-4, max_iter=40)
        fit = bench.trace_metrics(fp.f_seg, ph, E["c_hat"], E["keep"],
                                  E["vkeep"])
        sc = float(np.corrcoef(sig, st)[0, 1]) if st is not None else np.nan
        sr = (float(np.linalg.norm(sig - st) / np.linalg.norm(st))
              if st is not None else np.nan)
        print(f"  {lam_v:7.3g} {fit['r2_all']:9.4f} {fit['wmape_all']:10.4f} "
              f"{fit['wmape_val']:10.4f} {sc:11.4f} {sr:13.4f} "
              f"{time.perf_counter()-t1:6.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared re-estimation at the scheduled inactive-cell weight.

Step 6 of the revised cycle-projection estimator (not the paper's Algorithm 3,
which is qp/run_qp_sweep.py) adds a squared hinge on the potential-free
inactive-cell inequalities of algorithm3_inactive.py, with weight theta_V. The weight
was selected per sampling rate on the validation link set; SCHEDULE records the
result. Both the step-7 pass (run_steps_7_8.py) and the step-9 pass
(run_step_9.py) call refit() so that the two report the same estimate.

Setting theta_V = 0 reproduces the previously reported estimator exactly, so
--lam-v 0 on either script gives the old numbers on the new code path.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import algorithm3 as O
import algorithm3_inactive as IT

# selected on the validation link set; see io2_lamv_selection.csv
SCHEDULE = {0.1: 1.0, 0.2: 0.1, 0.3: 0.1, 0.4: 0.1, 0.5: 0.03}


def weight_for(rate, override=None):
    """Scheduled theta_V for this sampling rate, or the override if given."""
    if override is not None:
        return float(override)
    return float(SCHEDULE[round(float(rate), 4)])


def reachable_mask(inst, A_pos, A_zero):
    """(Q, A) boolean: inactive cells that yield an inequality row."""
    R = np.zeros_like(A_zero)
    for i in range(inst.n_seg):
        act = np.flatnonzero(A_pos[i])
        ina = np.flatnonzero(A_zero[i])
        if act.size == 0 or ina.size == 0:
            continue
        _, _, idx = IT.mixed_rows(inst.tail, inst.head, act, ina,
                                  inst.n_nodes)
        if idx.size:
            R[i, idx] = True
    return R


def refit(bench, E, lam_v, lam_s):
    """Re-solve step 6 with the hinge. Returns the new primitives and stats."""
    inst = bench.inst
    data, S, nS, nF = E["data"], E["S"], E["nS"], E["nF"]
    x0 = np.r_[E["theta"], E["sg"]]

    A_pos = np.isfinite(data.f_seg) & (data.f_seg > 0)
    A_zero = np.isfinite(data.f_seg) & (data.f_seg == 0) & inst.avail
    (Ze, Pe, be), (Zi, Pi, bi) = IT.build_blocks(inst, data, A_pos, A_zero)
    Ae = np.column_stack([Ze, -Pe[:, S]])
    Ai = (np.column_stack([Zi, -Pi[:, S]]) if Zi.shape[0]
          else np.zeros((0, nF + nS)))
    v0 = Ai @ x0 - bi if Ai.shape[0] else np.zeros(0)

    if lam_v > 0.0 and Ai.shape[0]:
        def fg(x):
            r = Ae @ x - be
            v = np.maximum(Ai @ x - bi, 0.0)
            s = x[nF:]
            f = r @ r + lam_s * (s @ s) + lam_v * (v @ v)
            g = 2.0 * (Ae.T @ r) + 2.0 * lam_v * (Ai.T @ v)
            g[nF:] += 2.0 * lam_s * s
            return f, g
        bnds = list(zip(np.r_[np.full(nF, -np.inf), np.zeros(nS)],
                        np.r_[np.full(nF, np.inf), np.full(nS, E["cap"])]))
        res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bnds,
                       options=dict(maxiter=6000, ftol=1e-14, gtol=1e-10))
        x = res.x
    else:
        x = x0

    th, sg = x[:nF], x[nF:]
    v1 = Ai @ x - bi if Ai.shape[0] else np.zeros(0)
    sigma = np.zeros(inst.n_own)
    sigma[S] = sg
    phi = np.where(inst.avail, data.z @ th, 0.0)
    est = O.P2Instance(
        tail=inst.tail, head=inst.head, link_owner=inst.link_owner,
        n_nodes=inst.n_nodes, D=inst.D, origin=inst.origin, sink=inst.sink,
        F=inst.F, Cmax=inst.Cmax, c=E["c_hat"], ell=inst.ell, phi=phi,
        alpha_B=1.0, alpha_S=E["aS"], avail=inst.avail,
        c_idle=E["c_idle_hat"], link_names=inst.link_names,
        segment_names=inst.segment_names,
        operator_names=inst.operator_names, node_names=inst.node_names,
        outside_link=inst.outside_link, b_supply=inst.b_supply)

    nz = int(A_zero.sum())
    return dict(
        theta=th, sigma=sigma, sg=sg, phi=phi, est=est,
        A_pos=A_pos, A_zero=A_zero,
        n_eq=int(Ae.shape[0]), n_ineq=int(Ai.shape[0]), n_inactive=nz,
        reach=float(Ai.shape[0] / max(nz, 1)),
        viol_before=float(np.mean(v0 > 0)) if v0.size else float("nan"),
        viol_after=float(np.mean(v1 > 0)) if v1.size else float("nan"),
        maxv_after=float(v1.max()) if v1.size else float("nan"),
        eq_resid_before=float(np.linalg.norm(Ae @ x0 - be)),
        eq_resid_after=float(np.linalg.norm(Ae @ x - be)),
        dtheta_rel=float(np.linalg.norm(th - E["theta"])
                         / np.linalg.norm(E["theta"])))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IO2 sampling sweep runner (Algorithm 4 on the LIRR access network).

Runs Algorithm 4 over a grid of (seed, sampling rate), writes one metrics row
per case to a CSV after every case, and can resume an interrupted sweep.

    python3 lirr_bench.py                       # 5 seeds x 3 rates, defaults below
    python3 lirr_bench.py --dry-run             # list cases + sample sizes, no solving
    python3 lirr_bench.py --seeds 0 --rates 0.10  # single case
    python3 lirr_bench.py --resume              # continue after Ctrl-C

Defaults match the configuration under test: alpha_B fixed at 1, outer sigma
loop capped at 5, step 6's inner MSA loop capped at 40, step 9 (the clearing
solve for tau* and the wedge) skipped.

Place this file in the LIRR_Access_Network directory, or pass --data-dir.

REPRODUCIBILITY NOTE
--------------------
design._order() seeds its RNG from abs(hash(("nested", seed))). Python salts
the hash of str objects per process (PEP 456), so design.draw_nested(seed=0)
returns a DIFFERENT tract ordering in every process -- the nominal seed does
not pin the draw. Observed: 833, 834, 838, 840, 842 links at rate 10%, all
"seed 0".

This runner defaults to --nested-hash stable, which reimplements the same
draw with the RNG seeded by blake2b("nested:<seed>") instead. Same design,
same alpha, same nesting property; now actually reproducible. Runs from
different machines and different days are then comparable.

--nested-hash legacy calls design.draw_nested unchanged, with PYTHONHASHSEED
pinned to 0 (the process re-execs itself once to set it). That is frozen too,
but it is a different ordering from any earlier unpinned run, so it will not
reproduce results generated before this fix either.

Requires: numpy, pandas, scipy, and the LIRR package modules
(build_lirr.py, design.py, algorithm2_p2.py, algorithm2prime_p2.py,
algorithm3_blocks.py) plus its CSV/NPZ data files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Ground-truth constants of the shipped instance (ground_truth_params.csv)
# --------------------------------------------------------------------------

ALPHA_S_TRUE = 4.0
C_IDLE_TRUE = (0.27, 0.72)
# Read the supply-side truth from the data directory when it is there, so the
# bench tracks the instance (alpha_S and the operators change with services).
try:
    _gp = __import__("pandas").read_csv("ground_truth_params.csv").set_index("parameter").value
    _as = [float(v) for k, v in _gp.items() if str(k).startswith("alpha_S_")]
    ALPHA_S_TRUE = (__import__("numpy").array(_as) if _as
                    else float(_gp.get("alpha_S", ALPHA_S_TRUE)))
    _ci = [float(v) for k, v in _gp.items() if str(k).startswith("c_idle_")]
    if _ci:
        C_IDLE_TRUE = tuple(_ci)
except Exception:
    pass
THETA_TRUE = (-0.367, -0.367, -0.367, -0.367, -0.367, -1.0, 5.902792)
FEATURE_NAMES = ("t_RAIL", "t_DRIVE", "t_TRAN", "t_EGR", "t_OUT",
                 "money", "drive")
C_PRIOR_SEED = 11          # as in sweep_io2.py: c_prior = c_true * U(0.90, 1.10)
THETA_PRIOR_SEED = 11      # theta_prior = THETA_TRUE * U(1-w, 1+w), same construction

# Column order of the output CSV. Fixed so a failed case still lines up.
COLUMNS = [
    # case identity
    "seed", "rate", "case", "n_cases", "started_utc", "secs",
    # sample
    "n_links_fit", "n_cells_fit", "n_cells_fit_pos", "flow_cov_fit",
    "n_val_links", "n_val_cells",
    # identification (steps 3-4)
    "identified", "design_rank", "n_params", "design_cond", "cycle_rank_total",
    # estimates (steps 2, 5)
    *FEATURE_NAMES, "alpha_B", "alpha_S", "c_idle_LIRR", "c_idle_LOCAL",
    "c_mean", "c_prior_mean", "pi_pos_frac",
    # convergence (steps 6-8)
    "n_outer", "converged", "message", "sigma_max", "sigma_pos_frac",
    "support_violations", "support_max_psi", "obj_buyer", "obj_seller",
    # flow fit
    "wmape_all", "wmape_held", "wmape_val", "r2_all", "rmse_all",
    "geh_median", "recall", "precision",
    # welfare
    "GS_tau0_hat", "GS_tau0_true", "GS_tau0_relerr",
    # step 9 (NaN when skipped)
    "tau_star_mean", "wedge_mean", "wedge_absmean",
    "tau_star_n_iter", "tau_star_converged", "tau_star_residual",
    # provenance
    "max_outer", "step6_max_iter", "inner_tol", "tol_sigma", "rho_c", "rho_pi",
    "fix_alpha_B", "skip_step9", "nested_hash", "design_alpha", "run_id",
]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Algorithm 4 (IO2) sampling sweep and write metrics to CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("grid")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="sampling-design seeds")
    g.add_argument("--rates", type=float, nargs="+", default=[0.10, 0.30, 0.50],
                   help="sampling rates (fraction of links instrumented)")
    g.add_argument("--design-alpha", type=float, default=0.5,
                   help="PPS exponent in the nested tract draw")
    g.add_argument("--nested-hash", choices=("stable", "legacy"), default="stable",
                   help="stable = blake2b seed (reproducible); "
                        "legacy = design.draw_nested with PYTHONHASHSEED=0")
    g.add_argument("--val-multiple", type=float, default=1.2,
                   help="validation slice is the (multiple*rate) nested prefix "
                        "minus the fitted links")

    a = p.add_argument_group("Algorithm 4 settings")
    a.add_argument("--max-outer", type=int, default=5,
                   help="cap on the outer rationing-price loop (steps 4-8)")
    a.add_argument("--step6-max-iter", type=int, default=40,
                   help="cap on step 6's inner MSA loop (Algorithm 3')")
    a.add_argument("--inner-tol", type=float, default=1e-4,
                   help="tolerance passed to Algorithm 3'/3")
    a.add_argument("--tol-sigma", type=float, default=1e-3,
                   help="outer tolerance on ||dsigma||_inf / max tau0")
    a.add_argument("--tol-gap", type=float, default=None,
                   help="outer tolerance on the UNDAMPED residual "
                        "||sigma_target - sigma^r||_inf / max tau0. The loop "
                        "now stops only when BOTH this and --tol-sigma are "
                        "met. --tol-sigma alone is unsound as a stopping rule: "
                        "it tests gamma_r * gap, and gamma_r -> 0 by "
                        "construction, so it is eventually satisfied by damping "
                        "with the residual frozen. Defaults to --tol-sigma")
    a.add_argument("--rho-c", type=float, default=1.0, help="L1 weight on the cost prior")
    a.add_argument("--rho-pi", type=float, default=1.0, help="L1 weight on the rents")
    a.add_argument("--free-alpha-b", action="store_true",
                   help="estimate alpha_B jointly instead of fixing it at 1")
    a.add_argument("--with-step9", action="store_true",
                   help="run step 9 (Algorithm 3 clearing solve) for tau* and the wedge; "
                        "off by default and the single most expensive call")
    a.add_argument("--fix-seller", choices=("truth",), default=None,
                   help="diagnostic: replace the estimated seller primitives "
                        "(c, c_idle, alpha_S) with the known truth. Step 6 uses "
                        "only these three from the seller block, so this "
                        "isolates seller-block error from outer-loop dynamics")
    a.add_argument("--ridge-lambda", type=float, default=0.0,
                   help="Tikhonov weight on the buyer block. Shrinks each "
                        "singular direction by s^2/(s^2+lambda), so the "
                        "worst-identified direction is damped most")
    a.add_argument("--ridge-prior", choices=("moving", "zero", "first", "prior"),
                   default="moving",
                   help="'moving': anchor on the previous outer iteration's "
                        "theta -- a proximal step. The fixed point AND the "
                        "stability condition (g<1) are unchanged, so it damps "
                        "the steps without making the loop converge. "
                        "'zero'/'first': a CONSTANT anchor, which changes the "
                        "condition to (1-w)g<1 and so can be made stable by "
                        "choice of --ridge-lambda, at the price of shrinkage "
                        "toward the anchor. 'zero' anchors at theta=0; "
                        "'first' anchors at the first outer iteration's theta, "
                        "i.e. the estimate assuming the posted fares clear "
                        "(sigma=0) -- truth-free and economically readable. "
                        "'prior': anchor at THETA_TRUE * U(1-w, 1+w), the "
                        "buyer-side counterpart of the Assumption 5 cost "
                        "anchor -- see --theta-prior-width. Unlike the others "
                        "this anchor is INFORMATIVE about theta, so results "
                        "under it are conditional on its accuracy")
    a.add_argument("--gamma-exponent", type=float, default=0.6,
                   help="MSA step in step 8 is gamma_r = r^-a. Robbins-Monro "
                        "admits 0.5 < a <= 1. What matters is not sum gamma_r "
                        "but sum gamma_r * gap_r, the total distance sigma "
                        "travels: with gap_r ~ r^-c that converges iff a+c>1. "
                        "Measured on this instance under the constant anchor, "
                        "c ~ 0.15, so a=0.6 gives 0.75 (diverges), a=0.8 gives "
                        "0.95 (still diverges), a=0.9 gives 1.05 (converges). "
                        "Note gap_r is NOT exogenous -- it is the residual "
                        "under the a used to generate it, so c must be "
                        "re-measured at each a rather than assumed")
    a.add_argument("--ridge-scale", choices=("none", "column"), default="none",
                   help="per-feature weighting of the ridge penalty. 'none': a "
                        "scalar lambda in RAW units, which on this design "
                        "penalises a 0/1 mode dummy and a minutes variable "
                        "equally -- the CZ column norms span 3.87 to 316.6, so "
                        "the raw-unit near-null direction is just the smallest "
                        "column ('drive', cond 856). 'column': penalty "
                        "lambda*sum_k ||CZ_k||^2 (theta_k - a_k)^2, i.e. ridge "
                        "the standardised design (cond 58.9), where the "
                        "genuinely worst-identified direction is t_OUT/t_RAIL/"
                        "money -- which is where nearly all the GS sensitivity "
                        "sits. NOTE lambda means something different under "
                        "'column': the standardised spectrum is 2.193..0.0373, "
                        "so s_min^2 = 0.00139 and useful lambda is ~1e-3..1e-1, "
                        "not 0.5..50")
    a.add_argument("--ridge-lambda-prox", type=float, default=0.0,
                   help="weight L_p on a MOVING proximal term carried ALONGSIDE "
                        "the constant anchor, penalty "
                        "L_p||theta-theta^(r-1)||^2 + L_a||theta-a||^2. This is "
                        "one ridge at L_p+L_a whose anchor is the L-weighted "
                        "blend, so the proximal part damps the path while the "
                        "constant part sets the fixed point -- at a fixed point "
                        "theta^(r-1)=theta* and the moving term vanishes "
                        "exactly. Requires --ridge-prior first/zero/prior; "
                        "'moving' already IS this term and is rejected")
    a.add_argument("--theta-prior-width", type=float, default=0.10,
                   help="half-width of the multiplicative perturbation used to "
                        "build the --ridge-prior prior anchor: "
                        "theta_prior = THETA_TRUE * U(1-w, 1+w). The default "
                        "0.10 matches the +/-10%% draw used for c_prior, so the "
                        "buyer anchor is exactly as accurate as the Assumption "
                        "5 cost anchor. Sweep it (0.10, 0.25, 0.50) to report "
                        "how the result degrades as the external estimate gets "
                        "worse -- a single width is not a result")
    a.add_argument("--theta-prior-seed", type=int, default=THETA_PRIOR_SEED,
                   help="draw for the prior anchor. Vary it to separate the "
                        "effect of anchor ACCURACY from the luck of one draw")
    a.add_argument("--fix-sigma", choices=("truth", "zero"), default=None,
                   help="hold the rationing prices fixed and skip steps 6-8 "
                        "entirely (fast). 'truth' uses sigma from "
                        "supply_at_tau0.csv -- the oracle sigma, i.e. the best "
                        "theta-hat and flow fit the outer loop could ever reach; "
                        "'zero' assumes the posted fares clear")
    a.add_argument("--clearing-max-iter", type=int, default=20000,
                   help="cap on step 9's Algorithm 3 clearing solve. Measured "
                        "on the LIRR 30%% instance: it needs ~9800-10700 "
                        "iterations to reach the 1e-4 natural residual and "
                        "takes only ~3 min, so do not lower this -- at 200 or "
                        "1000 it stops on a plateau around 3e-2 and the wedge "
                        "is not the wedge. Check tau_star_converged")

    o = p.add_argument_group("output and control")
    o.add_argument("--data-dir", type=Path, default=None,
                   help="LIRR_Access_Network directory (default: auto-detect)")
    o.add_argument("--out", type=Path, default=Path("io2_sweep_metrics.csv"),
                   help="metrics CSV (written after every case)")
    o.add_argument("--resume", action="store_true",
                   help="skip (seed, rate) cases already present in --out")
    o.add_argument("--dry-run", action="store_true",
                   help="report the case list and sample sizes, then exit")
    o.add_argument("--verbose", action="store_true",
                   help="per-outer-iteration output from Algorithm 4")
    o.add_argument("--trace", action="store_true",
                   help="also write io2_trace_<runid>.csv: one row per outer "
                        "iteration with theta-hat, the rank test, sigma "
                        "diagnostics and the flow fit at that iterate")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Data directory
# --------------------------------------------------------------------------

REQUIRED = ("links_p2.csv", "ground_truth_cells.csv", "supply_at_tau0.csv",
            "od_am.csv", "segments.csv", "paths.csv", "nodes.csv",
            "b_node.csv", "ground_truth_links.csv", "f_seg_link.csv")


def locate_data_dir(explicit: Path | None) -> Path:
    """Find the LIRR package directory and verify it is complete."""
    here = Path(__file__).resolve().parent
    candidates = [explicit] if explicit else [
        here, here / "LIRR_Access_Network", Path.cwd(),
        Path.cwd() / "LIRR_Access_Network",
    ]
    for cand in candidates:
        if cand and (cand / "links_p2.csv").exists():
            missing = [f for f in REQUIRED if not (cand / f).exists()]
            if missing:
                sys.exit(f"{cand} is missing: {', '.join(missing)}")
            return cand.resolve()
    sys.exit(
        "Could not find the LIRR_Access_Network directory.\n"
        "Put lirr_bench.py inside it, or pass --data-dir /path/to/LIRR_Access_Network"
    )


# --------------------------------------------------------------------------
# Nested sampling design
# --------------------------------------------------------------------------

def stable_seed(seed: int) -> int:
    """Reproducible replacement for abs(hash(("nested", seed))) % 2**31."""
    digest = hashlib.blake2b(f"nested:{int(seed)}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 2 ** 31


def nested_sets(design, rates, seed: int, alpha: float, mode: str) -> dict:
    """{rate: set(link_id)}, nested across increasing rates.

    mode="legacy" defers to design.draw_nested unchanged. mode="stable"
    reimplements the identical draw with a reproducible RNG seed; the tract
    ordering, the PPS weights and the nesting property are the same.
    """
    import numpy as np

    if mode == "legacy":
        return design.draw_nested(rates=tuple(rates), seed=seed, alpha=alpha)

    tracts = list(design.TRACTS)
    w = np.power(design.tract_flow.reindex(tracts).fillna(0.0).to_numpy(), alpha)
    w = np.maximum(w, w[w > 0].min() * 1e-3)
    rng = np.random.default_rng(stable_seed(seed))
    remaining, wr, order = list(tracts), w.copy(), []
    while remaining:
        k = rng.choice(len(remaining), p=wr / wr.sum())
        order.append(remaining.pop(k))
        wr = np.delete(wr, k)

    n_links = len(design._links)
    selected, i, out = set(design.CENSUS), 0, {}
    for rate in sorted(rates):
        budget = int(round(rate * n_links))
        while i < len(order) and len(selected) < budget:
            selected |= design.BUNDLE[order[i]]
            i += 1
        out[rate] = set(selected)
    return out


# --------------------------------------------------------------------------
# Instance, ground truth, metrics
# --------------------------------------------------------------------------

class Bench:
    """The LIRR instance, its ground truth, and everything reused across cases."""

    def __init__(self, verbose=True):
        import numpy as np
        import pandas as pd
        import build_lirr

        t0 = time.perf_counter()
        self.np, self.pd = np, pd
        inst, links, segs, aidx, nidx, Z = build_lirr.build(
            1.0, ALPHA_S_TRUE, np.array(C_IDLE_TRUE), with_z=True)
        self.inst, self.links, self.segs, self.aidx, self.Z = inst, links, segs, aidx, Z
        self.own = inst.owned_idx

        seg_i = {s: i for i, s in enumerate(segs.segment)}
        f_true = np.zeros((inst.n_seg, inst.n_links))
        for r in pd.read_csv("ground_truth_cells.csv").itertuples():
            f_true[seg_i[r.segment], aidx[r.link_id]] = r.f_true
        self.f_true = f_true

        sup = pd.read_csv("supply_at_tau0.csv").set_index("link_id")
        self.y_obs = sup.y_deployed.reindex(links.link_id[self.own]).to_numpy()
        # Service capacities (timetable seats per service), when the instance
        # has services; ordered as inst.svc_names.
        self.Y_obs = None
        if inst.svc_names is not None and os.path.exists("services_at_tau0.csv"):
            sv = pd.read_csv("services_at_tau0.csv").set_index("service_id")
            self.Y_obs = sv.Y_deployed.reindex(inst.svc_names).to_numpy(float)
        # True rationing prices, for scoring sigma-hat against the target it is
        # supposed to be approaching. Summary statistics of sigma-hat alone
        # cannot distinguish convergence from drift.
        self.sigma_true = (sup.sigma.reindex(links.link_id[self.own])
                           .to_numpy(float) if "sigma" in sup else None)
        self.tau0 = links.tau_usd.to_numpy(float)[self.own]
        self.c_true = links.cap_cost_per_unit.to_numpy(float)[self.own]

        # Cost prior: the same perturbation of the truth sweep_io2.py used, so
        # the seller block sees an identical prior in every case (the prior is
        # what separates c, pi and alpha_S -- Remark 7).
        rng = np.random.default_rng(C_PRIOR_SEED)
        self.c_prior = self.c_true * rng.uniform(0.90, 1.10, len(self.own))

        # The 7-feature truth the estimator targets, read from the data so it
        # tracks the instance (drive_constant is recalibrated with the data).
        global THETA_TRUE
        try:
            gp = pd.read_csv("ground_truth_params.csv").set_index("parameter").value
            THETA_TRUE = tuple(float(gp[k]) for k in (
                "time_RAIL", "time_ACCESS_DRIVE", "time_ACCESS_TRANSIT",
                "time_EGRESS", "time_OUTSIDE", "nonfare_money", "drive_constant"))
        except Exception:
            pass
        self.theta_true = np.array(THETA_TRUE)
        # Welfare truth uses the GENERATOR's utility (inst.phi, with the station
        # constants), not Z theta_true: the 7-feature model omits the station
        # constants, so Z theta_true is not the utility that produced f_true.
        self.phi_gen = np.where(inst.avail, inst.phi, 0.0)
        self.GS_true = self.gross_surplus(self.phi_gen, f_true, self.c_true, self.y_obs)

        self.avail = inst.avail
        self.f_link_true = f_true.sum(0)[self.own]
        self.flow_total = float(f_true[self.avail].sum())
        if verbose:
            print(f"  instance : {inst.n_links} links, {inst.n_own} owned, "
                  f"{inst.n_seg} segments, {int(self.avail.sum())} available cells "
                  f"({int((f_true > 0).sum())} with positive flow)")
            print(f"  GS at tau0 (true) : {self.GS_true:,.0f}")
            print(f"  build time : {time.perf_counter() - t0:.1f}s")

    @staticmethod
    def gross_surplus(phi, f_seg, c, y_own) -> float:
        """sum_ia phi_ia f_ia - sum_ma c_ma y_ma; outside option normalised to 0."""
        import numpy as np
        return float((phi * f_seg).sum()) - float(np.asarray(c) @ np.asarray(y_own))

    def fit_metrics(self, sol, keep, vkeep) -> dict:
        """Flow-fit, support and welfare metrics for one solved case."""
        np = self.np
        m, f_true, fh = self.avail, self.f_true, sol.f_seg
        held = m & ~(m & keep[None, :])          # available but unsampled cells
        val = m & vkeep[None, :]                 # observed-but-unfitted cells
        y, p = f_true[m], fh[m]
        nz, pz = y > 1e-9, p > 1e-9
        geh = np.sqrt(2.0 * (p - y) ** 2 / np.maximum(p + y, 1e-9))

        def wmape(mask):
            return float(np.abs(fh[mask] - f_true[mask]).sum()
                         / max(np.abs(f_true[mask]).sum(), 1e-9))

        gs_hat = self.gross_surplus(sol.phi, fh, sol.c, self.y_obs)
        out = dict(
            wmape_all=float(np.abs(p - y).sum() / max(np.abs(y).sum(), 1e-9)),
            wmape_held=wmape(held) if held.any() else float("nan"),
            wmape_val=wmape(val) if val.any() else float("nan"),
            r2_all=float(1.0 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            rmse_all=float(np.sqrt(((p - y) ** 2).mean())),
            geh_median=float(np.median(geh)),
            recall=float((nz & pz).sum() / max(nz.sum(), 1)),
            precision=float((nz & pz).sum() / max(pz.sum(), 1)),
            n_val_cells=int((val & (f_true > 1e-9)).sum()),
            GS_tau0_hat=gs_hat,
            GS_tau0_true=self.GS_true,
            GS_tau0_relerr=(gs_hat - self.GS_true) / abs(self.GS_true),
            sigma_max=float(np.max(sol.sigma)) if sol.sigma.size else float("nan"),
            sigma_pos_frac=float(np.mean(sol.sigma > 1e-9)) if sol.sigma.size else float("nan"),
            pi_pos_frac=float(np.mean(sol.pi > 1e-9)) if sol.pi.size else float("nan"),
            c_mean=float(np.mean(sol.c)),
            support_violations=float(sol.support_violations),
            support_max_psi=float(sol.support_max_psi),
            obj_buyer=float(sol.obj_buyer),
            obj_seller=float(sol.obj_seller),
        )
        # Step 9 leaves tau_star all-NaN when skipped; nanmean would warn.
        if sol.tau_star.size and not np.all(np.isnan(sol.tau_star)):
            out.update(tau_star_mean=float(np.nanmean(sol.tau_star)),
                       wedge_mean=float(np.nanmean(sol.wedge)),
                       wedge_absmean=float(np.nanmean(np.abs(sol.wedge))))
        else:
            out.update(tau_star_mean=float("nan"), wedge_mean=float("nan"),
                       wedge_absmean=float("nan"))
        for k, name in enumerate(FEATURE_NAMES):
            out[name] = float(sol.theta[k])
        return out

    def trace_metrics(self, f_seg, phi, c, keep, vkeep) -> dict:
        """Flow-fit and welfare metrics for ONE outer iterate.

        Same definitions as fit_metrics, but taking the arrays directly so an
        intermediate iterate can be scored without an IO2Solution. wmape_val is
        the out-of-sample column: observed links held out of the fit.
        """
        np = self.np
        m, f_true, fh = self.avail, self.f_true, f_seg
        held = m & ~(m & keep[None, :])
        val = m & vkeep[None, :]
        y, p = f_true[m], fh[m]
        nz, pz = y > 1e-9, p > 1e-9

        def wmape(mask):
            return float(np.abs(fh[mask] - f_true[mask]).sum()
                         / max(np.abs(f_true[mask]).sum(), 1e-9))

        gs = self.gross_surplus(phi, fh, c, self.y_obs)
        return dict(
            wmape_all=float(np.abs(p - y).sum() / max(np.abs(y).sum(), 1e-9)),
            wmape_held=wmape(held) if held.any() else float("nan"),
            wmape_val=wmape(val) if val.any() else float("nan"),
            r2_all=float(1.0 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            rmse_all=float(np.sqrt(((p - y) ** 2).mean())),
            recall=float((nz & pz).sum() / max(nz.sum(), 1)),
            precision=float((nz & pz).sum() / max(pz.sum(), 1)),
            GS_tau0_hat=gs,
            GS_tau0_relerr=(gs - self.GS_true) / abs(self.GS_true),
        )


# --------------------------------------------------------------------------
# Incremental CSV writer
# --------------------------------------------------------------------------

class CsvSink:
    """Appends one row per case, flushing each time, so Ctrl-C loses nothing."""

    def __init__(self, path: Path, resume: bool):
        self.path = path
        self.done = set()
        write_header = True
        if path.exists() and resume:
            import pandas as pd
            try:
                prev = pd.read_csv(path)
                self.done = {(int(r.seed), round(float(r.rate), 4))
                             for r in prev.itertuples()
                             if pd.notna(r.seed) and pd.notna(r.rate)}
                write_header = False
            except Exception:
                self.done = set()
        elif path.exists():
            backup = path.with_suffix(path.suffix + ".bak")
            path.replace(backup)
            print(f"  existing {path.name} moved to {backup.name}")
        self._fh = path.open("a" if not write_header else "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS,
                                      extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in COLUMNS})
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


# --------------------------------------------------------------------------
# Per-outer-iteration trace
# --------------------------------------------------------------------------

TRACE_COLUMNS = [
    "seed", "rate", "r", "secs",
    # identification, re-tested at tau0 + sigma^r every iteration
    "identified", "design_rank", "n_params", "design_cond",
    # estimates at this iterate
    *FEATURE_NAMES, "alpha_B", "alpha_S", "c_mean", "pi_pos_frac",
    # outer convergence. sigma_change is MSA-damped and NOT comparable across
    # r; sigma_gap_raw is. See the trace_hook note in algorithm3_blocks.solve_io2.
    "gamma", "sigma_change", "sigma_gap_raw", "sigma_max", "sigma_pos_frac",
    "sigma_target_max", "sigma_step_inf",
    # distance from sigma-hat to the TRUE sigma. These are the columns that
    # separate "converging to the right answer" from "drifting"; the summary
    # statistics above cannot.
    "sigma_err_inf", "sigma_err_l1", "sigma_err_rel", "sigma_corr",
    "sigma_true_max", "sigma_true_pos_frac",
    "ridge_lambda", "ridge_prior", "ridge_lambda_prox", "ridge_scale",
    "ridge_anchor_dist", "ridge_anchor_const_dist",
    # step 6 inner solve
    "inner_n_iter", "inner_converged", "inner_residual",
    "inner_residual_natural",
    # step 7
    "support_violations", "support_max_psi",
    # objectives
    "obj_buyer", "obj_seller",
    # flow fit at this iterate; wmape_val is the out-of-sample column
    "wmape_all", "wmape_held", "wmape_val", "r2_all", "rmse_all",
    "recall", "precision",
    # welfare at this iterate
    "GS_tau0_hat", "GS_tau0_true", "GS_tau0_relerr",
    "message", "run_id",
]


class TraceSink:
    """Writes one row per outer iteration, flushed immediately.

    A long run is readable while it is still going, and a crash keeps every
    iteration that completed.
    """

    INNER_COLUMNS = ["seed", "rate", "r", "t", "residual",
                     "residual_natural", "gamma", "buyer_obj"]

    def __init__(self, path: Path, inner_path: Optional[Path] = None):
        self.path = path
        if path.exists():
            path.replace(path.with_suffix(path.suffix + ".bak"))
        self._fh = path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=TRACE_COLUMNS,
                                      extrasaction="ignore")
        self._writer.writeheader()
        self._fh.flush()

        # Companion file: step 6's inner MSA residual curve, one row per inner
        # iteration t of every outer iteration r. This is what says whether the
        # inner fixed point is converging slowly (an accelerator would pay) or
        # not converging at all (it would not).
        self._ifh = self._iwriter = None
        if inner_path is not None:
            if inner_path.exists():
                inner_path.replace(inner_path.with_suffix(
                    inner_path.suffix + ".bak"))
            self._ifh = inner_path.open("w", newline="")
            self._iwriter = csv.DictWriter(self._ifh,
                                           fieldnames=self.INNER_COLUMNS,
                                           extrasaction="ignore")
            self._iwriter.writeheader()
            self._ifh.flush()

    def hook(self, bench, keep, vkeep, seed, rate, run_id, t0, verbose=False):
        """Build the callable handed to solve_io2(trace_hook=...)."""
        import numpy as np

        def _hook(st: dict) -> None:
            row = dict(seed=seed, rate=rate, r=st["r"], run_id=run_id,
                       secs=round(time.perf_counter() - t0, 1),
                       identified=bool(st.get("identified", False)),
                       design_rank=st.get("design_rank"),
                       n_params=st.get("n_params"),
                       design_cond=st.get("design_cond"),
                       obj_buyer=st.get("obj_buyer"),
                       obj_seller=st.get("obj_seller"),
                       alpha_B=st.get("alpha_B"), alpha_S=st.get("alpha_S"),
                       support_violations=st.get("support_violations"),
                       support_max_psi=st.get("support_max_psi"),
                       inner_n_iter=st.get("inner_n_iter"),
                       inner_converged=st.get("inner_converged"),
                       inner_residual=st.get("inner_residual"),
                       inner_residual_natural=st.get("inner_residual_natural"),
                       gamma=st.get("gamma"),
                       ridge_lambda=st.get("ridge_lambda"),
                       ridge_prior=st.get("ridge_prior"),
                       ridge_lambda_prox=st.get("ridge_lambda_prox"),
                       ridge_scale=st.get("ridge_scale"),
                       ridge_anchor_dist=st.get("ridge_anchor_dist"),
                       ridge_anchor_const_dist=st.get("ridge_anchor_const_dist"),
                       sigma_change=st.get("sigma_change"),
                       sigma_gap_raw=st.get("sigma_gap_raw"),
                       message=st.get("message", ""),
                       GS_tau0_true=bench.GS_true)

            th = st.get("theta")
            if th is not None:
                for k, name in enumerate(FEATURE_NAMES):
                    row[name] = float(th[k])

            sig = st.get("sigma")
            if sig is not None and np.size(sig):
                row["sigma_max"] = float(np.max(sig))
                row["sigma_pos_frac"] = float(np.mean(sig > 1e-9))
            st_true = getattr(bench, "sigma_true", None)
            if sig is not None and st_true is not None and np.size(sig) == np.size(st_true):
                err = np.asarray(sig, float) - st_true
                row["sigma_err_inf"] = float(np.abs(err).max())
                row["sigma_err_l1"] = float(np.abs(err).sum())
                row["sigma_err_rel"] = float(np.abs(err).sum()
                                             / max(np.abs(st_true).sum(), 1e-12))
                if np.std(sig) > 1e-12 and np.std(st_true) > 1e-12:
                    row["sigma_corr"] = float(np.corrcoef(sig, st_true)[0, 1])
                row["sigma_true_max"] = float(st_true.max())
                row["sigma_true_pos_frac"] = float(np.mean(st_true > 1e-9))

            tgt = st.get("sigma_target")
            if tgt is not None and np.size(tgt):
                row["sigma_target_max"] = float(np.max(tgt))
                prev = st.get("sigma_prev")
                if prev is not None:
                    row["sigma_step_inf"] = float(np.abs(sig - prev).max())

            if st.get("c") is not None:
                row["c_mean"] = float(np.mean(st["c"]))
            if st.get("pi") is not None and np.size(st["pi"]):
                row["pi_pos_frac"] = float(np.mean(st["pi"] > 1e-9))

            if st.get("f_seg") is not None and st.get("phi") is not None:
                try:
                    row.update(bench.trace_metrics(
                        st["f_seg"], st["phi"], st["c"], keep, vkeep))
                except Exception as exc:            # never kill a long run
                    row["message"] = f"trace metric error: {exc}"[:80]

            self._writer.writerow({k: row.get(k, "") for k in TRACE_COLUMNS})
            self._fh.flush()
            os.fsync(self._fh.fileno())

            ih = st.get("inner_history") or {}
            if self._iwriter is not None and ih.get("residual"):
                res = ih.get("residual", [])
                nat = ih.get("residual_natural", [])
                gam = ih.get("gamma", [])
                bo = ih.get("buyer_obj", [])
                for t in range(len(res)):
                    self._iwriter.writerow(dict(
                        seed=seed, rate=rate, r=st["r"], t=t + 1,
                        residual=res[t],
                        residual_natural=nat[t] if t < len(nat) else "",
                        gamma=gam[t] if t < len(gam) else "",
                        buyer_obj=bo[t] if t < len(bo) else ""))
                self._ifh.flush()
                os.fsync(self._ifh.fileno())
            if verbose:
                print(f"      trace r={row['r']:<3} "
                      f"gap_raw {row.get('sigma_gap_raw', float('nan')):.3e}  "
                      f"wmape_val {row.get('wmape_val', float('nan')):.4f}  "
                      f"GS {row.get('GS_tau0_relerr', float('nan')):+.1%}  "
                      f"money {row.get('money', float('nan')):+.4f}",
                      flush=True)

        return _hook

    def close(self) -> None:
        self._fh.close()
        if self._ifh is not None:
            self._ifh.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)

    # Legacy mode needs the hash salt pinned before the interpreter starts.
    if args.nested_hash == "legacy" and os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])

    data_dir = locate_data_dir(args.data_dir)
    out_path = args.out if args.out.is_absolute() else (Path.cwd() / args.out).resolve()
    os.chdir(data_dir)                     # the package modules read relative paths
    sys.path.insert(0, str(data_dir))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rates = [round(float(r), 4) for r in args.rates]
    cases = [(s, r) for s in args.seeds for r in rates]

    print("=" * 78)
    print(f"IO2 sampling sweep   run {run_id}")
    print("=" * 78)
    print(f"  data dir : {data_dir}")
    print(f"  output   : {out_path}")
    print(f"  grid     : seeds {args.seeds}  rates {rates}  "
          f"({len(cases)} cases)")
    print(f"  alg 4    : max_outer={args.max_outer}  step6_max_iter={args.step6_max_iter}  "
          f"inner_tol={args.inner_tol:g}  tol_sigma={args.tol_sigma:g}")
    print(f"             alpha_B {'estimated' if args.free_alpha_b else 'fixed at 1'}, "
          f"step 9 {'included' if args.with_step9 else 'skipped'}, "
          f"rho_c={args.rho_c:g} rho_pi={args.rho_pi:g}")
    print(f"  design   : nested bundles, alpha={args.design_alpha}, "
          f"{args.nested_hash} seeding, validation at {args.val_multiple:g}x rate")
    print(f"  python   : {platform.python_version()} on {platform.platform()}")
    print()

    import numpy as np
    import design
    from algorithm3_blocks import IO2Data, solve_io2

    bench = Bench()
    n_params = bench.Z.shape[2] + (1 if args.free_alpha_b else 0)

    # Buyer-side prior anchor, built exactly as c_prior is (Bench.__init__):
    # a multiplicative U(1-w, 1+w) perturbation of the truth, its own RNG so it
    # does not consume the cost draw's stream. Only used by --ridge-prior prior.
    theta_prior = None
    if args.ridge_prior == "prior":
        if args.ridge_lambda <= 0.0:
            print("[X] --ridge-prior prior has no effect at --ridge-lambda 0")
            return 2
        w = float(args.theta_prior_width)
        rng_th = np.random.default_rng(args.theta_prior_seed)
        theta_prior = np.array(THETA_TRUE, float) * rng_th.uniform(
            1.0 - w, 1.0 + w, len(THETA_TRUE))
        print(f"  theta prior     : THETA_TRUE * U({1-w:.2f}, {1+w:.2f}), "
              f"seed {args.theta_prior_seed}")
        for nm, tv, pv in zip(FEATURE_NAMES, THETA_TRUE, theta_prior):
            print(f"      {nm:8s} true {tv:9.4f}   prior {pv:9.4f}   "
                  f"({100.0*(pv-tv)/abs(tv):+6.2f}%)")
        print(f"      max |prior-true|/|true| = "
              f"{np.max(np.abs(theta_prior - np.array(THETA_TRUE, float)) / np.abs(THETA_TRUE)):.4f}")
    print()

    # Draw every design up front: cheap, and it fails fast on a bad grid.
    val_rates = sorted({*rates, *(round(args.val_multiple * r, 4) for r in rates)})
    designs = {}
    for seed in args.seeds:
        sets = nested_sets(design, val_rates, seed, args.design_alpha, args.nested_hash)
        for rate in rates:
            keep = bench.links.link_id.isin(sets[rate]).to_numpy()
            vrate = round(args.val_multiple * rate, 4)
            vkeep = bench.links.link_id.isin(sets[vrate]).to_numpy() & ~keep
            designs[(seed, rate)] = (keep, vkeep)

    if args.dry_run:
        print(f"  {'seed':>5} {'rate':>6} {'fit links':>10} {'fit cells':>10} "
              f"{'pos cells':>10} {'flow cov':>9} {'val links':>10}")
        for seed, rate in cases:
            keep, vkeep = designs[(seed, rate)]
            cells = bench.avail & keep[None, :]
            cov = float(bench.f_true[cells].sum()) / max(bench.flow_total, 1e-9)
            print(f"  {seed:>5} {rate:>6.2f} {int(keep.sum()):>10} "
                  f"{int(cells.sum()):>10} {int((cells & (bench.f_true > 0)).sum()):>10} "
                  f"{cov:>9.3f} {int(vkeep.sum()):>10}")
        print("\n  dry run: nothing solved.")
        return 0

    sink = CsvSink(out_path, args.resume)
    trace = None
    if args.trace:
        trace_path = out_path.parent / f"io2_trace_{run_id}.csv"
        inner_path = out_path.parent / f"io2_inner_{run_id}.csv"
        trace = TraceSink(trace_path, inner_path)
        print(f"  trace    : {trace_path}")
        print(f"  inner    : {inner_path}")
    if sink.done:
        print(f"  resuming: {len(sink.done)} case(s) already in {out_path.name}\n")

    provenance = dict(
        max_outer=args.max_outer, step6_max_iter=args.step6_max_iter,
        inner_tol=args.inner_tol, tol_sigma=args.tol_sigma,
        rho_c=args.rho_c, rho_pi=args.rho_pi,
        fix_alpha_B=not args.free_alpha_b, skip_step9=not args.with_step9,
        nested_hash=args.nested_hash, design_alpha=args.design_alpha,
        run_id=run_id, n_params=n_params, n_cases=len(cases),
    )
    (out_path.parent / f"io2_sweep_config_{run_id}.json").write_text(
        json.dumps({**provenance, "seeds": args.seeds, "rates": rates,
                    "data_dir": str(data_dir), "output": str(out_path),
                    "GS_tau0_true": bench.GS_true,
                    "python": platform.python_version(),
                    "numpy": np.__version__}, indent=2))

    elapsed, n_run = [], 0
    try:
        for idx, (seed, rate) in enumerate(cases, start=1):
            tag = f"[{idx}/{len(cases)}] seed {seed} rate {rate:.0%}"
            if (seed, round(rate, 4)) in sink.done:
                print(f"{tag}: already done, skipping")
                continue

            keep, vkeep = designs[(seed, rate)]
            cells = bench.avail & keep[None, :]
            row = dict(seed=seed, rate=rate, case=idx,
                       started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       n_links_fit=int(keep.sum()), n_cells_fit=int(cells.sum()),
                       n_cells_fit_pos=int((cells & (bench.f_true > 0)).sum()),
                       flow_cov_fit=float(bench.f_true[cells].sum())
                       / max(bench.flow_total, 1e-9),
                       n_val_links=int(vkeep.sum()),
                       c_prior_mean=float(np.mean(bench.c_prior)),
                       **provenance)

            eta = ""
            if elapsed:
                mean = sum(elapsed) / len(elapsed)
                left = mean * (len(cases) - idx + 1)
                eta = f"  (~{mean / 60:.1f} min/case, ~{left / 60:.0f} min left)"
            print(f"{tag}: {row['n_links_fit']} links, {row['n_cells_fit']} cells, "
                  f"flow coverage {row['flow_cov_fit']:.3f}{eta}", flush=True)

            f_seg = np.where(cells, bench.f_true, np.nan)
            data = IO2Data(f_seg=f_seg, y=bench.y_obs, f_link=bench.f_link_true,
                           tau0=bench.tau0, z=bench.Z, c_prior=bench.c_prior,
                           c_idle_prior=np.array(C_IDLE_TRUE))

            seller_override = None
            if args.fix_seller == "truth":
                seller_override = (bench.c_true, np.array(C_IDLE_TRUE),
                                   ALPHA_S_TRUE)

            fixed_sigma = None
            if args.fix_sigma == "truth":
                if bench.sigma_true is None:
                    sys.exit("--fix-sigma truth needs a sigma column in "
                             "supply_at_tau0.csv")
                fixed_sigma = bench.sigma_true
            elif args.fix_sigma == "zero":
                fixed_sigma = np.zeros_like(bench.tau0)

            t0 = time.perf_counter()
            hook = None
            if trace is not None:
                hook = trace.hook(bench, keep, vkeep, seed, rate, run_id, t0,
                                  verbose=args.verbose)
            try:
                sol = solve_io2(
                    bench.inst, data,
                    rho_c=args.rho_c, rho_pi=args.rho_pi,
                    inner_tol=args.inner_tol, tol_sigma=args.tol_sigma,
                    tol_gap=args.tol_gap,
                    max_outer=args.max_outer, step6_max_iter=args.step6_max_iter,
                    skip_clearing=not args.with_step9,
                    fix_sigma=fixed_sigma,
                    fix_seller=seller_override,
                    ridge_lambda=args.ridge_lambda,
                    ridge_prior=args.ridge_prior,
                    ridge_anchor_vec=theta_prior,
                    ridge_lambda_prox=args.ridge_lambda_prox,
                    ridge_scale=args.ridge_scale,
                    gamma_exponent=args.gamma_exponent,
                    clearing_max_iter=args.clearing_max_iter,
                    fix_alpha_B=not args.free_alpha_b,
                    verbose=args.verbose,
                    trace_hook=hook,
                )
                secs = time.perf_counter() - t0
                row.update(
                    secs=round(secs, 1), identified=bool(sol.identified),
                    design_rank=int(sol.design_rank),
                    design_cond=float(sol.design_cond),
                    cycle_rank_total=int(np.sum(sol.cycle_rank)),
                    alpha_B=float(sol.alpha_B), alpha_S=float(np.atleast_1d(sol.alpha_S)[0]),
                    c_idle_LIRR=float(sol.c_idle[0]), c_idle_LOCAL=float(sol.c_idle[1]),
                    n_outer=int(sol.n_outer), converged=bool(sol.converged),
                    message=str(sol.message)[:80],
                )
                h = sol.history
                if "clearing_n_iter" in h:
                    row.update(
                        tau_star_n_iter=h["clearing_n_iter"][0],
                        tau_star_converged=h["clearing_converged"][0],
                        tau_star_residual=h["clearing_residual"][0],
                    )
                    if not h["clearing_converged"][0]:
                        print(f"      !! step 9 stopped at "
                              f"{h['clearing_n_iter'][0]} iterations without "
                              f"clearing (residual {h['clearing_residual'][0]:.3e});"
                              f" the wedge is not the wedge", flush=True)
                if sol.identified:
                    row.update(bench.fit_metrics(sol, keep, vkeep))
                    print(f"      -> rank {row['design_rank']}/{n_params} "
                          f"cond {row['design_cond']:.3g}  n_outer {row['n_outer']}"
                          f"{'' if row['converged'] else ' (cap)'}  "
                          f"wmape_val {row['wmape_val']:.3f}  "
                          f"wmape_all {row['wmape_all']:.3f}  "
                          f"GS {row['GS_tau0_hat']:,.0f} "
                          f"({row['GS_tau0_relerr']:+.1%})  {secs:.0f}s", flush=True)
                else:
                    print(f"      -> NOT IDENTIFIED: rank {row['design_rank']}/"
                          f"{n_params}  {secs:.0f}s", flush=True)
                elapsed.append(secs)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                secs = time.perf_counter() - t0
                row.update(secs=round(secs, 1), identified=False,
                           message=f"EXC {type(exc).__name__}: {exc}"[:200])
                print(f"      -> FAILED after {secs:.0f}s: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()

            sink.write(row)
            n_run += 1
    except KeyboardInterrupt:
        print("\n  interrupted; completed cases are in the CSV. "
              "Re-run with --resume to continue.")
    finally:
        sink.close()
        if trace is not None:
            trace.close()

    total = sum(elapsed)
    print(f"\n  {n_run} case(s) written to {out_path}")
    if elapsed:
        print(f"  total solve time {total / 60:.1f} min "
              f"(mean {total / len(elapsed) / 60:.1f} min/case)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
Calibrate station constants and the drive constants to observed boardings.

phi_a = theta' z_a stays link-additive.  z carries:
  * 5 time coefficients, by link type (PARK time is a constant 4 min on every
    parking link, so a coefficient on it would be collinear with the station
    constants and is omitted)
  * 1 non-fare money coefficient, unit by Assumption 2
  * 1 boarding constant per Long Island station, on that station's PARK link
    (drive access) and ACCESS_TRANSIT links (bus access).  Exactly one lies on
    any rail path, so it is charged once per boarding and a through rider is
    not charged it
  * 2 modal constants on the OUTSIDE links, split by the origin tract's county

Why two drive constants.  A single global constant fitted the aggregate but
moved the level equally for every tract, over-suppressing rail in inner
Nassau: 47 Nassau tracts a median 1.6 miles from a station ended up with zero
rail flow.  Splitting by origin county lets the drive alternative degrade as
it approaches the congestion zone and the Manhattan parking market.  The two
are identified by the Nassau and Suffolk station-boarding totals.

Target: the 2012-2014 LIRR Origin and Destination Report profile (sheets
197-199, WESTBOUND / AM Peak / On), rescaled to the 2025 control total.  All
97 stations are in the calibration set.

Update rule.  Level and shape are updated separately: the station constants
take a log-ratio step recentred within county, so they carry no level, and
each county's drive constant takes a log-ratio step on that county's boarding
total.  A single global gain is adapted by backtracking line search on the
objective, which makes the iteration monotone.  Three other schemes were tried
and all failed -- the fixed-potential derivative diverges because the node
potentials absorb most of a bump, 1/psi scaling crawls because psi is order
1e-2 here, and a diagonal secant oscillates.
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUT = "."
TIMED = ["RAIL", "ACCESS_DRIVE", "ACCESS_TRANSIT", "EGRESS", "OUTSIDE"]
COUNTIES = ["Nassau", "Suffolk"]

links = pd.read_csv(f"{OUT}/links_p2.csv")
nodes = pd.read_csv(f"{OUT}/nodes.csv", dtype={"geoid": str})
segs = pd.read_csv(f"{OUT}/segments.csv")
segs = segs[segs.segmentation == "destination"].reset_index(drop=True)
bn = pd.read_csv(f"{OUT}/b_node.csv")
bn = bn[bn.segmentation == "destination"]
paths = pd.read_csv(f"{OUT}/paths.csv")
board = pd.read_csv(f"{OUT}/station_boardings.csv")

nidx = {n: i for i, n in enumerate(nodes.node_id)}
aidx = {a: i for i, a in enumerate(links.link_id)}
nV, nL = len(nodes), len(links)
tail = links.from_node.map(nidx).to_numpy()
head = links.to_node.map(nidx).to_numpy()
tau = links.tau_usd.to_numpy(float)
# Outer calibration loop: buyer prices tau0 + sigma, with sigma the rationing
# prices from the latest Algorithm 2' ground-truth solve (SIGMA_FILE), and a
# warm start from the previous theta (THETA_INIT).
import os as _os
if _os.environ.get("SIGMA_FILE"):
    _s = pd.read_csv(_os.environ["SIGMA_FILE"]).set_index("link_id").sigma
    tau = tau + links.link_id.map(_s).fillna(0.0).to_numpy(float)
    print(f"buyer prices include sigma from {_os.environ['SIGMA_FILE']}: "
          f"{int((_s > 1e-9).sum())} links, max {_s.max():.3f}")
ell = links.ell.to_numpy(float)
ltype = links.link_type.to_numpy()

is_board = np.isin(ltype, ["PARK", "ACCESS_TRANSIT"])
stations = list(board.node_id)
sidx = {s: k for k, s in enumerate(stations)}
nS = len(stations)
nF = 6 + nS + 1

Z = np.zeros((nL, nF))
for k, t in enumerate(TIMED):
    Z[:, k] = links.ff_time_min.to_numpy() * (ltype == t)
Z[:, 5] = (links.user_cost_usd - links.fare_usd).clip(lower=0).to_numpy()
for j in np.flatnonzero(is_board):
    s = links.to_node.iloc[j]
    if s in sidx:
        Z[j, 6 + sidx[s]] = 1.0
tract_county = dict(zip(nodes.node_id, nodes.county))
oc = links.from_node.map(tract_county).to_numpy()
Z[ltype == "OUTSIDE", -1] = 1.0

theta0 = np.zeros(nF)
theta0[:5] = -0.367
theta0[5] = -1.0
if _os.environ.get("THETA_INIT"):
    theta0 = np.load(_os.environ["THETA_INIT"])
    print(f"warm start from {_os.environ['THETA_INIT']}")

reach, free, b, D = [], [], [], []
for _, s in segs.iterrows():
    g = paths[paths.zone == s.segment]
    R = np.array(sorted({aidx[x] for v in g.link_ids for x in v.split("|")}))
    touched = sorted(set(tail[R]) | set(head[R]))
    sink = nidx[s.sink_node]
    reach.append(R)
    free.append(np.array([n for n in touched if n != sink]))
    bv = np.zeros(nV)
    for r in bn[bn.segment == s.segment].itertuples():
        bv[nidx[r.node]] = r.b            # raw trips: already D_i * b_in
    b.append(bv)
    D.append(float(s.demand))
D = np.array(D)
off = np.cumsum([0] + [len(f) for f in free])
nQ = len(D)
sink_idx = [nidx[s] for s in segs.sink_node]
tgt = board.set_index("node_id").target_boardings.reindex(stations).to_numpy(dtype=float)
stn_node = np.array([nidx[s] for s in stations])
stn_county = board.set_index("node_id").county.reindex(stations).to_numpy()
cmask = {c: (stn_county == c) for c in COUNTIES}
print(f"calibration set: all {nS} stations "
      f"({cmask['Nassau'].sum()} Nassau, {cmask['Suffolk'].sum()} Suffolk)")


def sp_init(phi):
    U = np.zeros((nQ, nV))
    for i in range(nQ):
        R = reach[i]
        w = -(phi[R] - tau[R])
        d = np.full(nV, np.inf); d[sink_idx[i]] = 0.0
        for _ in range(nV - 1):
            nd = d[head[R]] + w
            upd = nd < d[tail[R]] - 1e-12
            if not upd.any():
                break
            np.minimum.at(d, tail[R][upd], nd[upd])
        U[i] = np.where(np.isfinite(d), d, 0.0)
    return np.concatenate([U[i][free[i]] for i in range(nQ)])


def buyer(phi, alpha_B, u0=None, rounds=3):
    """P2 buyer block at the administered fares tau^0."""
    def dual(u):
        val, g = 0.0, np.zeros(off[-1])
        for i in range(nQ):
            R = reach[i]
            uu = np.zeros(nV); uu[free[i]] = u[off[i]:off[i + 1]]
            psi = phi[R] - tau[R] - (uu[tail[R]] - uu[head[R]])
            pp = np.maximum(psi, 0.0)
            val += float((alpha_B * D[i] / (2 * ell[R]) * pp ** 2).sum())
            val += float(b[i] @ uu)
            f = alpha_B * D[i] / ell[R] * pp
            eta = np.zeros(nV)
            np.add.at(eta, tail[R], f); np.add.at(eta, head[R], -f)
            eta -= b[i]
            g[off[i]:off[i + 1]] = -eta[free[i]]
        return val, g

    u = sp_init(phi) if u0 is None else u0.copy()
    for _ in range(rounds):
        u = minimize(dual, u, jac=True, method="L-BFGS-B",
                     options=dict(maxiter=40000, ftol=1e-20, gtol=1e-16,
                                  maxcor=30)).x
    F = np.zeros((nQ, nL))
    for i in range(nQ):
        R = reach[i]
        uu = np.zeros(nV); uu[free[i]] = u[off[i]:off[i + 1]]
        psi = phi[R] - tau[R] - (uu[tail[R]] - uu[head[R]])
        F[i, R] = alpha_B * D[i] / ell[R] * np.maximum(psi, 0.0)
    return F, u


def boardings(F):
    fa = F.sum(0)
    bd = np.zeros(nV)
    np.add.at(bd, head[is_board], fa[is_board])
    return bd[stn_node]


def wmape(pred):
    return float(np.abs(pred - tgt).sum() / tgt.sum())


def calibrate(alpha_B=1.0, n_iter=90, gain=0.5, gain_a=0.6, verbose=True):
    th = theta0.copy()
    F, u = buyer(Z @ th, alpha_B, None, rounds=6)
    pred = boardings(F)
    err = wmape(pred)
    lg = lambda v: np.log(np.maximum(v, 1e-2))
    lvl = lambda pr: abs(np.log(max(pr.sum(), 1.0) / tgt.sum()))
    err_l = lvl(pred)
    hist = [err]
    if verbose:
        print(f"    start wMAPE={err:.4f}  total={pred.sum():,.0f}  "
              f"corr={np.corrcoef(pred, tgt)[0, 1]:.3f}", flush=True)
    for t in range(n_iter):
        raw = lg(tgt) - lg(pred)
        step = np.clip(gain * (raw - raw.mean()), -1.5, 1.5)
        d_auto = float(np.clip(gain_a * (np.log(max(pred.sum(), 1.0))
                                         - np.log(tgt.sum())), -1.0, 1.0))
        cand = th.copy()
        cand[6:6 + nS] += step
        cand[-1] += d_auto
        Fc, uc = buyer(Z @ cand, alpha_B, u, rounds=3)
        pc = boardings(Fc)
        ec = wmape(pc)
        ec_l = lvl(pc)
        # Adapt the two gains on their own residuals.  Sharing one gain lets a
        # rejected shape step freeze the level update, which strands the county
        # totals far from target: with a single shared gain the split-constant
        # fit stalled at wMAPE 0.86 and a total of 118,659 against 72,130.
        gain = min(gain * 1.15, 3.0) if ec < err else max(gain * 0.5, 1e-4)
        gain_a = (min(gain_a * 1.15, 3.0) if ec_l < err_l
                  else max(gain_a * 0.6, 0.05))
        if ec < err or ec_l < err_l - 1e-4:
            th, F, u, pred = cand, Fc, uc, pc
            err, err_l = min(ec, err), ec_l
        hist.append(err)
        if verbose and t % 5 == 0:
            print(f"    it {t:2d}  wMAPE={err:.4f}  gain={gain:.3f}  "
                  f"total={pred.sum():8,.0f}  "
                  f"corr={np.corrcoef(pred, tgt)[0, 1]:.3f}  "
                  f"auto={th[-1]:+.2f}  lvl={err_l:.3f}", flush=True)
        if (gain < 1e-3 and err_l < 5e-3) or err < 5e-3:
            break
    return th, F, u, err, np.array(hist), pred


if __name__ == "__main__":
    th, F, u, err, hist, pred = calibrate()
    np.save(f"{OUT}/theta_calibrated.npy", th)
    r2 = 1 - ((pred - tgt) ** 2).sum() / ((tgt - tgt.mean()) ** 2).sum()
    out = dict(wmape=err, r2=float(r2),
               corr=float(np.corrcoef(pred, tgt)[0, 1]),
               total_pred=float(pred.sum()), total_target=float(tgt.sum()),
               asc_min=float(th[6:6 + nS].min()),
               asc_max=float(th[6:6 + nS].max()),
               asc_mean=float(th[6:6 + nS].mean()),
               drive_constant=float(th[-1]),
               n_stations=nS, n_iter=int(len(hist)))
    for c in COUNTIES:
        m = cmask[c]
        out[f"corr_{c.lower()}"] = float(np.corrcoef(pred[m], tgt[m])[0, 1])
        out[f"total_{c.lower()}"] = float(pred[m].sum())
        out[f"target_{c.lower()}"] = float(tgt[m].sum())
    json.dump(out, open(f"{OUT}/asc_calibration.json", "w"), indent=1)
    pd.DataFrame(dict(station=board.station, node_id=stations,
                      county=stn_county, target=tgt, fitted=pred.round(1),
                      asc_usd=th[6:6 + nS].round(4))
                 ).to_csv(f"{OUT}/station_constants.csv", index=False)
    print("\n" + json.dumps(out, indent=1))

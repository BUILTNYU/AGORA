# -*- coding: utf-8 -*-
"""
Apply the calibrated theta to the P2 instance.

Recomputes phi_a = theta' z_a on every link from the vector fitted by
fit_station_ascs.py and writes it into links_p2.csv, then saves the feature
matrix so IO2 can be handed exactly the z it needs to recover theta from.

theta layout (105 parameters):
    0-4    time coefficient on RAIL, ACCESS_DRIVE, ACCESS_TRANSIT, EGRESS,
           OUTSIDE, $/min
    5      non-fare money coefficient (unit by Assumption 2)
    6-102  boarding constant per Long Island station, $
    103    modal constant on outside links from Nassau tracts, $
    104    modal constant on outside links from Suffolk tracts, $

PARK links carry no time coefficient.  Their free-flow time is a constant
4 min on every link, so a coefficient on it would be exactly collinear with
the station constants; the search-and-walk time is absorbed into those.

Run after build_io2_package.py and before run_p2_ground_truth.py.
"""
import numpy as np
import pandas as pd

OUT = "."
TIMED = ["RAIL", "ACCESS_DRIVE", "ACCESS_TRANSIT", "EGRESS", "OUTSIDE"]

links = pd.read_csv(f"{OUT}/links_p2.csv")
board = pd.read_csv(f"{OUT}/station_boardings.csv")
theta = np.load(f"{OUT}/theta_calibrated.npy")

ltype = links.link_type.to_numpy()
nL = len(links)
stations = list(board.node_id)
nS = len(stations)
sidx = {s: k for k, s in enumerate(stations)}
nF = 6 + nS + 1
assert len(theta) == nF, f"theta has {len(theta)} entries, expected {nF}"

Z = np.zeros((nL, nF))
for k, t in enumerate(TIMED):
    Z[:, k] = links.ff_time_min.to_numpy() * (ltype == t)
Z[:, 5] = (links.user_cost_usd - links.fare_usd).clip(lower=0).to_numpy()
is_board = np.isin(ltype, ["PARK", "ACCESS_TRANSIT"])
for j in np.flatnonzero(is_board):
    s = links.to_node.iloc[j]
    if s in sidx:
        Z[j, 6 + sidx[s]] = 1.0
tract_county = dict(zip(pd.read_csv(f"{OUT}/nodes.csv", dtype={"geoid": str}).node_id,
                        pd.read_csv(f"{OUT}/nodes.csv", dtype={"geoid": str}).county))
oc = links.from_node.map(tract_county).to_numpy()
Z[ltype == "OUTSIDE", -1] = 1.0

phi_old = links.phi_usd.to_numpy().copy()
links["phi_usd"] = (Z @ theta).round(6)
links.to_csv(f"{OUT}/links_p2.csv", index=False)

names = ([f"time_{t}" for t in TIMED] + ["nonfare_money"]
         + [f"asc_{nm}" for nm in board.station]
         + ["drive_constant"])
np.savez_compressed(f"{OUT}/link_features.npz", Z=Z, theta=theta,
                    feature_names=np.array(names), link_id=links.link_id.to_numpy())
pd.DataFrame(dict(parameter=names, value=theta.round(6))).to_csv(
    f"{OUT}/params_true_p2.csv", index=False)

print(f"phi updated on {nL} links, nF = {nF}")
print(f"  phi range {links.phi_usd.min():.2f} to {links.phi_usd.max():.2f} "
      f"(was {phi_old.min():.2f} to {phi_old.max():.2f})")
print(f"  drive constant {theta[-1]:+.3f}   station constants "
      f"{theta[6:6+nS].min():.2f} to {theta[6:6+nS].max():.2f}")
chg = links.groupby("link_type").apply(
    lambda g: (g.phi_usd - phi_old[g.index]).mean(), include_groups=False)
print("\nmean change in phi by link type:")
print(chg.round(3).to_string())

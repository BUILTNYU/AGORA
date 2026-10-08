"""Branch seat fit of a service ground truth against the timetable. usage: branch_fit.py <dir>..."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/rebuild/v2026"); from data.geo import BRANCHES
brm = {}
for b, c in BRANCHES.items():
    for s in c["stops"]: brm.setdefault(s, b)
tt = pd.Series({b: c["peak_tph"] * c["cars"] * 110 * 3.2 for b, c in BRANCHES.items()
                if b not in ("City Terminal", "Atlantic", "GCM", "LIC")})
for D in sys.argv[1:]:
    sv = pd.read_csv(f"{D}/services_at_tau0.csv").merge(pd.read_csv(f"{D}/services.csv")[["service_id", "kind", "origin"]], on="service_id")
    rr = sv[sv.kind == "rail_route"]; m = rr.groupby(rr.origin.map(brm)).Y_deployed.sum().reindex(tt.index).fillna(0)
    print(f"{D}: corr {np.corrcoef(m, tt)[0,1]:.3f}  wMAPE {np.abs(m - tt).sum() / tt.sum():.3f}  "
          f"log-RMSE {np.sqrt(np.mean(np.log(m / tt) ** 2)):.3f}  total {m.sum():,.0f}")

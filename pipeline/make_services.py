"""Services for the LIRR instance: one per (origin station, NYC/Jamaica terminal),
running the rail path of paths.csv from the origin to the terminal. A seat on a
service is a seat on every link of its route, so link capacity y_a = sum_r H_ar Y_r
tapers outward along each line. Rail links on no route get a one-link service;
parking (LIRR) and feeder links (LOCAL) are their own services (H = I block).
Service cap = smallest link cap on the route.
usage: python make_services.py <data_dir>   -> writes <data_dir>/services.csv"""
import sys, numpy as np, pandas as pd
D = sys.argv[1]
L = pd.read_csv(f"{D}/links_p2.csv"); N = pd.read_csv(f"{D}/nodes.csv").drop_duplicates("name")
P = pd.read_csv(f"{D}/paths.csv", usecols=["kind", "station", "terminal", "link_ids"])
P = P[P.kind == "RAIL"]
cap = dict(zip(L.link_id, L.capacity_period.where(L.capacity_period > 0).fillna(1e7))); typ = dict(zip(L.link_id, L.link_type)); op = dict(zip(L.link_id, L.operator))
seq = {}
for st, te, ids in P[["station", "terminal", "link_ids"]].drop_duplicates(["station", "terminal"]).itertuples(index=False):
    r = [a for a in ids.split("|") if typ.get(a) == "RAIL"]
    seq[(st, te)] = r
# consistency: a (station, terminal) pair always takes the same rail route
chk = P.drop_duplicates(["station", "terminal", "link_ids"]).copy()
chk["r"] = chk.link_ids.map(lambda s: "|".join(a for a in s.split("|") if typ.get(a) == "RAIL"))
nroutes = chk.groupby(["station", "terminal"]).r.nunique()
assert nroutes.max() == 1, nroutes[nroutes > 1]
rows = []
for (st, te), r in sorted(seq.items()):
    rows.append(dict(service_id=f"SVC|{st}|{te}", operator="LIRR", kind="rail_route",
                     origin=st, terminal=te, n_links=len(r), link_ids="|".join(r),
                     Cmax=float(min(cap[a] for a in r))))
covered = {a for r in seq.values() for a in r}
own = L[L.operator.isin(["LIRR", "LOCAL"])]
for a, t, o in zip(own.link_id, own.link_type, own.operator):
    if t == "RAIL" and a in covered:
        continue
    o = "PARK" if t == "PARK" else o
    rows.append(dict(service_id=f"LINK|{a}", operator=o, kind="link" if t != "RAIL" else "rail_link",
                     origin="", terminal="", n_links=1, link_ids=a, Cmax=float(cap[a])))
S = pd.DataFrame(rows)
S.to_csv(f"{D}/services.csv", index=False)
print(S.groupby(["operator", "kind"]).size().to_string())
print("rail links covered by routes:", len(covered), "of", int((L.link_type == "RAIL").sum()))

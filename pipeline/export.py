# -*- coding: utf-8 -*-
"""Stage 2: link loading, observation model, and file export."""
import os, json, pickle, math
import numpy as np
import pandas as pd
import build_network as B

OUT = B.OUT
RNG = np.random.default_rng(7)
R = pickle.load(open(os.path.join(OUT, "_state.pkl"), "rb"))

tr, st, nyc = R["tr"], R["st"], R["nyc"]
acc, rail, egress = R["acc"], R["rail"], R["egress"]
out_df, pth, park = R["out_df"], R["pth"], R["park"]
skim, P = R["skim"], R["P"]
sid = dict(zip(st["name"], st.node_id))

# ============================================================ 1. node table
nodes = []
for _, r in tr.iterrows():
    nodes.append(dict(node_id=r.node_id, node_type="CENTROID_LI", name=r["name"],
                      geoid=str(r.geoid), county=r.county, lat=r.lat, lon=r.lon,
                      population=r.population, workers=round(r.workers, 1),
                      nyc_share=round(r.nyc_share, 4),
                      prod_am=round(r.O_am, 2), fare_zone="", branch=""))
for _, r in nyc.iterrows():
    nodes.append(dict(node_id=r.node_id, node_type="CENTROID_NYC", name=r["name"],
                      geoid="", county="NYC", lat=r.lat, lon=r.lon,
                      population="", workers="", nyc_share="",
                      prod_am="", fare_zone="", branch=""))
# One parking node per Long Island station.  All drive access funnels through
# a single virtual PARK link into the station, which is where the lot capacity
# lives: without it nothing stops 6,000 people parking in 649 spaces.
li_names = list(R["park"].station)
park_node = {nm: "K" + str(i + 1).zfill(3) for i, nm in enumerate(li_names)}

stn_ll = st.set_index("name")[["lat", "lon"]]
for nm in li_names:
    nodes.append(dict(node_id=park_node[nm], node_type="PARK", name=nm + " parking",
                      geoid="", county="", lat=float(stn_ll.loc[nm, "lat"]) - 0.0015,
                      lon=float(stn_ll.loc[nm, "lon"]) - 0.0015,
                      population="", workers="", nyc_share="", prod_am="",
                      fare_zone="", branch=""))

stn_branch = {}
for b, cfg in B.BRANCHES.items():
    for s in cfg["stops"]:
        stn_branch.setdefault(s, []).append(b)
for _, r in st.iterrows():
    nodes.append(dict(node_id=r.node_id, node_type="STATION", name=r["name"],
                      geoid="", county=r.county, lat=r.lat, lon=r.lon,
                      population="", workers="", nyc_share="", prod_am="",
                      fare_zone=r.fare_zone, branch="|".join(stn_branch.get(r["name"], []))))
nodes = pd.DataFrame(nodes)

# ============================================================ 2. link table
LINKS = []
def add(**kw):
    LINKS.append(kw)

pk = park.set_index("station")
for _, a in acc.iterrows():
    drive = a["mode"] == "DRIVE"
    if drive:
        add(link_id=f"ACC_{a.tract}_{park_node[a.station]}_D", from_node=a.tract,
            to_node=park_node[a.station], link_type="ACCESS_DRIVE", operator="NONE",
            mode="AUTO", branch="", length_mi=a.length_mi, ff_time_min=a.time_min,
            capacity_period=np.nan, fare_usd=0.0,
            user_cost_usd=round(a.length_mi * P["auto_op_cost_mi"], 2),
            prov_cost_per_pax=0.0, cap_cost_per_unit=0.0,
            bpr_b=P["park_bpr_b"], bpr_power=P["park_bpr_pow"],
            shared_capacity_key="")
    else:
        cap = float(pk.loc[a.station, "bus_cap_persons"])
        add(link_id=f"ACC_{a.tract}_{sid[a.station]}_T", from_node=a.tract,
            to_node=sid[a.station], link_type="ACCESS_TRANSIT", operator="LOCAL",
            mode="BUS", branch="", length_mi=a.length_mi, ff_time_min=a.time_min,
            capacity_period=round(cap, 1), fare_usd=P["fare_local"],
            user_cost_usd=a.cost,
            prov_cost_per_pax=round(a.length_mi * P["bus_cost_seat_mi"] * 0.35, 4),
            cap_cost_per_unit=round(a.length_mi * P["bus_cost_seat_mi"], 4),
            bpr_b=0.15, bpr_power=4.0,
            shared_capacity_key=f"BUS_{sid[a.station]}")

spark = R["station_park"].set_index("station")
for nm in li_names:
    cap = float(pk.loc[nm, "park_cap_persons"])
    fee = float(spark.loc[nm, "park_usd"])
    add(link_id=f"PARK_{park_node[nm]}", from_node=park_node[nm], to_node=sid[nm],
        link_type="PARK", operator="LIRR", mode="PARK", branch="",
        length_mi=0.25, ff_time_min=4.0, capacity_period=round(cap, 1),
        fare_usd=fee, user_cost_usd=fee,
        prov_cost_per_pax=0.02, cap_cost_per_unit=0.45,
        bpr_b=0.0, bpr_power=1.0, shared_capacity_key="")

for L in rail:
    cap_p = L["capacity_hr"] * P["peak_equiv_hours"]
    add(link_id=f"RAIL_{sid[L['u']]}_{sid[L['v']]}_{L['branch'][:4].upper()}",
        from_node=sid[L["u"]], to_node=sid[L["v"]], link_type="RAIL",
        operator="LIRR", mode="RAIL", branch=L["branch"],
        length_mi=L["length_mi"], ff_time_min=L["ff_time_min"],
        capacity_period=round(cap_p, 1), fare_usd=0.0, user_cost_usd=0.0,
        prov_cost_per_pax=round(L["length_mi"] * 0.012, 4),
        cap_cost_per_unit=round(L["length_mi"] * P["lirr_cost_seat_mi"], 4),
        bpr_b=0.10, bpr_power=4.0, shared_capacity_key="")

for e in egress:
    add(link_id=f"EGR_{e['u_id']}_{e['v']}", from_node=e["u_id"], to_node=e["v"],
        link_type="EGRESS", operator=e["operator"],
        mode="WALK" if e["operator"] == "NONE" else "SUBWAY", branch="",
        length_mi=e["length_mi"], ff_time_min=e["time_min"],
        capacity_period=np.nan, fare_usd=e["fare"], user_cost_usd=e["fare"],
        prov_cost_per_pax=round(e["length_mi"]*0.05, 4) if e["operator"] != "NONE" else 0.0,
        cap_cost_per_unit=0.0, bpr_b=0.0, bpr_power=1.0, shared_capacity_key="")

for _, o in out_df.iterrows():
    add(link_id=f"OUT_{o.tract}_{o.zone}", from_node=o.tract, to_node=o.zone,
        link_type="OUTSIDE", operator="NONE", mode="AUTO", branch="",
        length_mi=o.length_mi, ff_time_min=o.time_min, capacity_period=np.nan,
        fare_usd=0.0, user_cost_usd=o.cost, prov_cost_per_pax=0.0,
        cap_cost_per_unit=0.0, bpr_b=0.15, bpr_power=4.0, shared_capacity_key="")

links = pd.DataFrame(LINKS)

# ================================== 3. path -> link incidence & link loading
# rebuild rail sequences from the Dijkstra predecessors stored in the skim
seq_cache = {}
def rail_seq(station, zi):
    key = (station, zi)
    if key in seq_cache:
        return seq_cache[key]
    sk = skim.get(key)
    if sk is None:
        seq_cache[key] = None
        return None
    dp = skim[("__prev__", station)]
    seq = B.rail_sequence(dp, sk["end_state"])
    seq_cache[key] = (seq, sk["terminal"])
    return seq_cache[key]

rail_lid = {}
for L in rail:
    rail_lid[(L["u"], L["v"], L["branch"])] = \
        f"RAIL_{sid[L['u']]}_{sid[L['v']]}_{L['branch'][:4].upper()}"
    rail_lid[(L["v"], L["u"], L["branch"])] = rail_lid[(L["u"], L["v"], L["branch"])]

egr_lid = {(e["u"], e["zone_idx"]): f"EGR_{e['u_id']}_{e['v']}" for e in egress}

flows = {}
path_rows = []
for pid, p in enumerate(pth.itertuples(index=False)):
    lids = []
    if p.kind == "OUTSIDE":
        lids.append(f"OUT_{p.tract}_{p.zone}")
    else:
        if p.access_mode == "DRIVE":
            lids.append(f"ACC_{p.tract}_{park_node[p.station]}_D")
            lids.append(f"PARK_{park_node[p.station]}")
        else:
            lids.append(f"ACC_{p.tract}_{sid[p.station]}_T")
        rs = rail_seq(p.station, p.zone_idx)
        if rs is None:
            continue
        seq, term = rs
        for (u, v, br) in seq:
            lids.append(rail_lid[(u, v, br)])
        lids.append(egr_lid[(term, p.zone_idx)])
    for lid in lids:
        flows[lid] = flows.get(lid, 0.0) + p.flow
    path_rows.append(dict(path_id=f"P{pid:06d}", tract=p.tract, zone=p.zone,
                          kind=p.kind, station=p.station, access_mode=p.access_mode,
                          terminal=p.terminal, n_links=len(lids),
                          time_min=p.time_min, money_usd=p.money,
                          gen_cost_usd=p.gen_cost_eq, station_asc=p.station_asc,
                          flow=round(p.flow, 5), link_ids="|".join(lids)))
paths_out = pd.DataFrame(path_rows)

links["flow_true"] = links.link_id.map(flows).fillna(0.0).round(4)
links["voc"] = (links.flow_true / links.capacity_period.replace(0, np.nan)).round(4)

# shared parking capacity: report station-level v/c on drive access links
links["station_park_voc"] = np.where(
    links.link_type == "PARK",
    links.flow_true / links.capacity_period.replace(0, np.nan), np.nan).round(4)

# ======================================================= 4. observation model
obs = links[links.flow_true > 0].copy()
eps = RNG.normal(0, P["count_cv"], len(obs))
obs["count_obs"] = np.maximum(0, obs.flow_true * np.exp(eps - P["count_cv"]**2/2)).round(2)
obs["count_cv"] = P["count_cv"]
# link counts are only realistically observed on rail + a screenline sample
obs["observed"] = np.where(obs.link_type == "RAIL", 1,
                    np.where(RNG.random(len(obs)) < 0.18, 1, 0))

# station boardings
bo = pth[pth.kind == "RAIL"].groupby(["station", "access_mode"]).flow.sum().unstack(fill_value=0)
bo["boardings_true"] = bo.sum(axis=1)
bo = bo.join(park.set_index("station")[["spaces", "park_cap_persons", "utilisation",
                                        "park_delay_min", "bus_veh_hr",
                                        "bus_cap_persons", "asc_dollars",
                                        "target_boardings"]])
bo["boardings_obs"] = np.maximum(0, bo.boardings_true *
      np.exp(RNG.normal(0, P["boarding_cv"], len(bo)) - P["boarding_cv"]**2/2)).round(1)
bo = bo.reset_index().rename(columns={"DRIVE": "board_drive_access",
                                      "TRANSIT": "board_transit_access"})
bo = bo.merge(st[["name", "node_id", "fare_zone", "county"]],
              left_on="station", right_on="name", how="left").drop(columns="name")
bo = bo.merge(R["station_park"][["station", "d_manhattan", "park_usd"]],
              on="station", how="left")
bo["peak_hour_boardings_true"] = (bo.boardings_true * 0.34).round(1)

# ============================================================== 5. OD table
Tij, tracts = R["Tij"], R["tracts"]
od = []
for i, tname in enumerate(tracts):
    for j, z in nyc.iterrows():
        if Tij[i, j] > 1e-6:
            od.append(dict(o_node=tname, d_node=z.node_id, d_name=z["name"],
                           trips_am=round(Tij[i, j], 5)))
od = pd.DataFrame(od)
od["trips_peak_hour"] = (od.trips_am * 0.34).round(5)

# ========================================================= 6. operator table
def opsum(op):
    L = links[links.operator == op]
    pool = L[L.shared_capacity_key != ""]
    uniq = (pool.groupby("shared_capacity_key")
                .agg(cap=("capacity_period", "first"), mi=("length_mi", "mean")))
    solo = L[L.shared_capacity_key == ""].dropna(subset=["capacity_period"])
    seat_mi = (uniq.cap * uniq.mi).sum() + (solo.capacity_period * solo.length_mi).sum()
    cap_usd = ((uniq.cap * uniq.mi).sum() * (P["bus_cost_seat_mi"] if op == "LOCAL" else 0)
               + (solo.capacity_period * solo.cap_cost_per_unit).sum())
    return dict(operator=op, n_links=len(L),
                pax_trips=round(L.flow_true.sum(), 1),
                pax_miles=round((L.flow_true * L.length_mi).sum(), 1),
                capacity_seat_miles=round(seat_mi, 1),
                capacity_budget_usd=round(cap_usd, 2),
                provision_cost_usd=round((L.flow_true * L.prov_cost_per_pax).sum(), 2),
                fare_revenue_usd=round((L.flow_true * L.fare_usd).sum(), 2))
ops = pd.DataFrame([opsum("LIRR"), opsum("LOCAL"), opsum("NONE")])
# LIRR fare revenue is charged at the path level, not the link level
_zn = dict(zip(st["name"], st.fare_zone))
lirr_rev = sum(r.flow * B.lirr_fare(_zn[r.station], r.terminal)
               for r in pth[pth.kind == "RAIL"].itertuples())
ops.loc[ops.operator == "LIRR", "fare_revenue_usd"] = round(lirr_rev, 2)

# ========================================================== 7. ground truth
prm = [("mu_logit", P["mu"], "path-choice logit scale, 1/$"),
       ("beta_gravity", P["beta_gravity"], "deterrence on logsum, 1/$"),
       ("vot_per_min", P["vot_per_min"], "$/min"),
       ("xfer_penalty_min", P["xfer_penalty_min"], "min"),
       ("park_bpr_b", P["park_bpr_b"], "-"),
       ("park_bpr_pow", P["park_bpr_pow"], "-"),
       ("fare_local", P["fare_local"], "$"),
       ("park_floor", P["park_floor"], "$"),
       ("park_amp", P["park_amp"], "$"),
       ("park_decay", P["park_decay"], "mi"),
       ("crz_toll", P["crz_toll"], "$"),
       ("am_share", P["am_share"], "-")]
params = pd.DataFrame(prm, columns=["parameter", "true_value", "units"])
asc = park[["station", "asc_dollars"]].copy()
asc.columns = ["parameter", "true_value"]
asc["parameter"] = "asc_" + asc.parameter
asc["units"] = "$ (station constant)"
params = pd.concat([params, asc], ignore_index=True)

# ================================================================ 8. write
nodes.to_csv(f"{OUT}/nodes.csv", index=False)
links.drop(columns=["flow_true"]).to_csv(f"{OUT}/links.csv", index=False)
od.to_csv(f"{OUT}/od_am.csv", index=False)
paths_out.to_csv(f"{OUT}/paths.csv", index=False)
paths_out[["path_id", "flow"]].to_csv(f"{OUT}/path_flows_true.csv", index=False)
links[["link_id", "link_type", "operator", "flow_true", "capacity_period", "voc"]] \
    .to_csv(f"{OUT}/link_flows_true.csv", index=False)
obs[["link_id", "link_type", "operator", "observed", "count_obs", "count_cv"]] \
    .to_csv(f"{OUT}/link_counts_obs.csv", index=False)
bo.to_csv(f"{OUT}/station_boardings.csv", index=False)
ops.to_csv(f"{OUT}/operators.csv", index=False)
params.to_csv(f"{OUT}/params_true.csv", index=False)

# --------------------------------------------------------------- TNTP form
lid = {n: i+1 for i, n in enumerate(nodes.node_id)}
with open(f"{OUT}/LIRRAccess_net.tntp", "w") as f:
    f.write(f"<NUMBER OF ZONES> {len(tr)+len(nyc)}\n<NUMBER OF NODES> {len(nodes)}\n")
    f.write(f"<FIRST THRU NODE> {len(tr)+len(nyc)+1}\n<NUMBER OF LINKS> {len(links)}\n")
    f.write("<END OF METADATA>\n~\tInit\tTerm\tCapacity\tLength\tFreeFlowTime\tB\tPower\tSpeed\tToll\tType\t;\n")
    tmap = {"ACCESS_DRIVE": 1, "ACCESS_TRANSIT": 2, "RAIL": 3, "EGRESS": 4,
            "OUTSIDE": 5, "PARK": 6}
    for _, L in links.iterrows():
        f.write(f"\t{lid[L.from_node]}\t{lid[L.to_node]}\t{L.capacity_period:.1f}\t"
                f"{L.length_mi:.3f}\t{L.ff_time_min:.2f}\t{L.bpr_b}\t{L.bpr_power}\t0\t"
                f"{L.user_cost_usd + L.fare_usd:.2f}\t{tmap[L.link_type]}\t;\n")
with open(f"{OUT}/LIRRAccess_trips.tntp", "w") as f:
    f.write(f"<NUMBER OF ZONES> {len(tr)+len(nyc)}\n"
            f"<TOTAL OD FLOW> {od.trips_am.sum():.2f}\n<END OF METADATA>\n\n")
    for o, g in od.groupby("o_node", sort=False):
        f.write(f"Origin \t{lid[o]}\n")
        f.write("    " + "; ".join(f"{lid[r.d_node]} :{r.trips_am:12.4f}"
                                   for r in g.itertuples()) + "; \n\n")

# ---------------------------------------------------------------- GeoJSON
pos = dict(zip(nodes.node_id, zip(nodes.lon, nodes.lat)))
feat = []
for _, n in nodes.iterrows():
    feat.append(dict(type="Feature", geometry=dict(type="Point", coordinates=[n.lon, n.lat]),
                     properties={k: (None if pd.isna(v) else v) for k, v in n.items()}))
bmap = {b: c["color"] for b, c in B.BRANCHES.items()}
for _, L in links.iterrows():
    if L.link_type in ("ACCESS_DRIVE", "OUTSIDE"):
        continue                                    # 4.6k links, kept out of GeoJSON
    feat.append(dict(type="Feature",
        geometry=dict(type="LineString",
                      coordinates=[list(pos[L.from_node]), list(pos[L.to_node])]),
        properties=dict(link_id=L.link_id, link_type=L.link_type, operator=L.operator,
                        branch=L.branch, length_mi=L.length_mi,
                        flow=float(L.flow_true), voc=float(L.voc) if pd.notna(L.voc) else None,
                        color=bmap.get(L.branch, "#8892a0"))))
json.dump(dict(type="FeatureCollection", features=feat),
          open(f"{OUT}/network.geojson", "w"))

print(f"nodes {len(nodes)} | links {len(links)} | od pairs {len(od)} | paths {len(paths_out)}")
print(links.groupby(["link_type", "operator"]).agg(
    n=("link_id", "size"), flow=("flow_true", "sum")).round(0))
print(ops.to_string(index=False))
print("\ntop stations by AM boardings:")
print(bo.nlargest(12, "boardings_true")[
    ["station", "boardings_true", "board_drive_access", "board_transit_access",
     "spaces", "utilisation"]].round(0).to_string(index=False))
print("\nrail links over capacity:", int((links[links.link_type=='RAIL'].voc > 1).sum()),
      "/", int((links.link_type=='RAIL').sum()),
      "| max v/c:", round(links[links.link_type=='RAIL'].voc.max(), 2))

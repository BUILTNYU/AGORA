# -*- coding: utf-8 -*-
"""
Aggregate network assignment game test instance: LIRR access network.

Two operators (LIRR commuter rail, LOCAL feeder transit) plus operator-less
links (drive-and-park access, drive-all-the-way outside option).

Outputs to out/:
    nodes.csv              centroids (LI tracts + NYC zones) and stations
    links.csv              all links with operator, cost, capacity attributes
    od_am.csv              AM-period OD matrix, tract -> NYC zone
    paths.csv              enumerated path set (1 outside + 3 stations x 2 modes)
    path_flows_true.csv    ground-truth path flows
    link_flows_true.csv    ground-truth link flows
    link_counts_obs.csv    noisy "observed" link counts  (E4 data regime)
    station_boardings.csv  true + observed AM boardings by station
    operators.csv          operator capacity budgets and unit costs
    network.geojson        geometry for GIS
    params_true.csv        ground-truth parameters (recovery target)
"""
import json
import math
import os
import heapq
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.geo import (STATIONS, BRANCHES, NYC_ZONES, ANCHORS, COUNTY_POP,
                      COUNTY_TRACTS, COUNTY_FIPS, MAJOR_STATIONS, ANCHOR_SHARES,
                      AM_PEAK_BOARDINGS, BRANCH_2025, LI_ORIGIN_ANNUAL,
                      AM_PEAK_MANHATTAN_TERMINAL,
                      AM_PEAK_BOARDINGS_2012_2014)

RNG = np.random.default_rng(20260822)
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# ===================================================================== params
P = dict(
    # --- demand ---
    worker_rate        = 0.50,   # employed residents per capita
    am_share           = 0.62,   # share of work trips departing 6-10 AM
    nyc_share_nassau   = 0.30,   # Nassau workers with NYC workplace
    nyc_share_suffolk  = 0.14,
    # --- behaviour ---
    vot_per_min        = 0.367,  # $22/hr
    mu                 = 0.55,   # path-choice logit scale (1/$)
    beta_gravity       = 0.045,  # deterrence on logsum ($^-1)
    xfer_penalty_min   = 5.0,
    # --- rail supply ---
    rail_mph_electric  = 38.0,
    rail_mph_diesel    = 34.0,
    seats_per_car      = 110,
    load_factor        = 1.15,
    peak_equiv_hours   = 3.2,    # effective peak-frequency hours in the 4h AM
    # --- access supply ---
    drive_mph_local    = 26.0,
    hwy_mph            = 38.0,   # AM peak LIE / Southern State, westbound
    core_miles         = 8.0,    # congested approach charged at the core speed
    bus_mph            = 13.0,
    bus_headway_min    = 22.0,
    bus_seats          = 38,
    bus_veh_per_hr_lo  = 2.0,
    circuity           = 1.28,   # network distance / great-circle
    # --- money ---
    fare_local         = 2.75,
    # Station parking is priced by distance to Manhattan, a proxy for land
    # value and permit scarcity: close-in Nassau village lots are permit-only
    # and full by 7am, eastern Suffolk lots are large and cheap or free.
    # park = clip(park_floor + park_amp * exp(-(d - 8) / park_decay), floor, cap)
    park_floor         = 0.75,
    park_amp           = 7.50,
    park_decay         = 18.0,
    park_cap           = 8.00,
    auto_op_cost_mi    = 0.32,
    crz_toll           = 9.00,   # Manhattan congestion relief zone
    # --- control totals ---
    am_lirr_boardings  = AM_PEAK_BOARDINGS,  # MTA 2025: (256,000 - 111,740)/2
    # --- operator economics ---
    lirr_cost_seat_mi  = 27.90 / 110,   # NTD 2023, LIRR commuter rail (ID 20100): opex $27.90 per vehicle revenue mile, 110 seats/car
    bus_cost_seat_mi   = 0.30,
    # --- measurement ---
    count_cv           = 0.08,
    boarding_cv        = 0.05,
    # --- station parking (binding constraint on drive access) ---
    park_spaces_total  = 58000,  # Nassau + Suffolk LIRR commuter spaces
    park_turnover      = 1.0,    # AM-period vehicles per space
    park_occupancy     = 1.08,   # persons per arriving auto (incl. drop-off)
    park_bpr_b         = 0.60,
    park_bpr_pow       = 6.0,
)

# LIRR peak one-way fares as published: MTA, "LIRR Stations and Fare Zones /
# Fare Chart - Ticket Types", effective January 4, 2026 (mta.info/document/194866).
# Origin fare zone -> City Terminal Zone (zone 1), and origin zone -> zone 3
# (Jamaica). Note the CityTicket rate makes zones 1 and 3 a flat $7.25 to zone 1,
# so the fare is not a function of the zone gap and is looked up by zone pair.
ZONE_ORDER = {1: 0, 3: 1, 4: 2, 7: 3, 9: 4, 10: 5, 12: 6, 14: 7}
FARE_PEAK_2026 = {
    1: {1: 7.25, 3: 7.25, 4: 13.50, 7: 15.25, 9: 18.25, 10: 21.50, 12: 25.50, 14: 33.00},
    3: {1: 7.25, 3: 6.00, 4: 9.00, 7: 10.75, 9: 13.25, 10: 16.50, 12: 22.00, 14: 28.25},
}
FARE_SCHEDULE = "MTA LIRR peak one-way, effective 2026-01-04"
TERMINAL_ZONE = {"Jamaica": 3}          # everything else is City Terminal, zone 1
def lirr_fare(orig_zone, terminal):
    dz = TERMINAL_ZONE.get(terminal, 1)
    return FARE_PEAK_2026[dz][orig_zone]
FARE_TO_ZONE1 = dict(FARE_PEAK_2026[1])
DIESEL_BRANCHES = {"Montauk", "Greenport", "Oyster Bay", "Port Jefferson"}


def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def hav_vec(lat, lon, lats, lons):
    R = 3958.8
    p1 = np.radians(lat); p2 = np.radians(lats)
    dp = p2 - p1; dl = np.radians(lons - lon)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ============================================================ 1. tract layer
def build_tracts():
    """Real Nassau and Suffolk tract centroids from the 2025 Census Gazetteer,
    with NYC-bound worker counts from LEHD LODES8 (ny_od_main_JT00_2022).

    Replaces the population-kernel synthesis this build previously used.  The
    tracts, their centroids and their NYC-bound worker counts are all observed;
    nothing here is generated.
    """
    D_ = os.path.dirname(os.path.abspath(__file__))
    tr = pd.read_csv(f"{D_}/data/li_tracts.csv", dtype={"GEOID": str})
    od = pd.read_csv(f"{D_}/data/li_to_nyc_od.csv", dtype={"h_tract": str})
    w = od.groupby("h_tract").S000.sum()
    tr["nyc_workers"] = tr.GEOID.map(w).fillna(0.0)
    tr = tr.rename(columns={"GEOID": "geoid"})
    tr["name"] = tr.county + " " + tr.geoid.str[5:]
    tr["place"] = tr.county
    # LODES has no population; area-weighted density is not needed downstream,
    # so population is carried only as a label and left blank.
    tr["population"] = np.nan
    return tr[["geoid", "name", "county", "place", "lat", "lon",
               "population", "nyc_workers"]].reset_index(drop=True)


def nyc_zone_shares():
    """Destination shares from LODES borough totals.

    The six zones are Manhattan x3, Brooklyn, and Queens x2.  Bronx and Staten
    Island have no zone and no LIRR service, so their 8.7% is dropped and the
    remainder rescaled.  Manhattan is split 70/30 midtown/downtown, and the
    midtown part 60/40 Penn/Grand Central per the MTA 2025 report; Queens is
    split 35/65 between Long Island City and the rest.
    """
    D_ = os.path.dirname(os.path.abspath(__file__))
    od = pd.read_csv(f"{D_}/data/li_to_nyc_od.csv")
    b = od.groupby("borough").S000.sum()
    man, qns, bkn = b["Manhattan"], b["Queens"], b["Brooklyn"]
    t = AM_PEAK_MANHATTAN_TERMINAL
    raw = {"Midtown West / Penn":     man * 0.70 * t["Penn Station"],
           "Midtown East / GCM":      man * 0.70 * t["Grand Central Madison"],
           "Lower Manhattan":         man * 0.30,
           "Downtown Brooklyn":       bkn,
           "Long Island City":        qns * 0.35,
           "Central Queens/Jamaica":  qns * 0.65}
    tot = sum(raw.values())
    return {k: v / tot for k, v in raw.items()}


# ============================================================= 2. rail graph
def build_rail_graph(st):
    """Returns rail link list and a (station, branch) state graph."""
    idx = {r["name"]: r for _, r in st.iterrows()}
    links, seen = [], set()
    for br, cfg in BRANCHES.items():
        mph = P["rail_mph_diesel"] if br in DIESEL_BRANCHES else P["rail_mph_electric"]
        stops = cfg["stops"]
        for a, b in zip(stops[:-1], stops[1:]):
            key = (a, b, br)
            if key in seen:
                continue
            seen.add(key)
            A, B = idx[a], idx[b]
            d = haversine(A["lat"], A["lon"], B["lat"], B["lon"]) * 1.09
            t = d / mph * 60 + 0.7                       # + dwell
            cap_hr = cfg["peak_tph"] * cfg["cars"] * P["seats_per_car"] * P["load_factor"]
            links.append(dict(branch=br, u=a, v=b, length_mi=round(d, 3),
                              ff_time_min=round(t, 2),
                              capacity_hr=round(cap_hr, 0),
                              seats_hr=cfg["peak_tph"]*cfg["cars"]*P["seats_per_car"],
                              peak_tph=cfg["peak_tph"], cars=cfg["cars"],
                              color=cfg["color"]))
    return links


def shortest_rail(rail_links, egress, origins, zones):
    """Label-correcting Dijkstra over (station, branch) states with transfer
    penalties. Returns {(station, zone): (gen_cost_min, fare_zone_pair, seq)}."""
    adj = {}
    for L in rail_links:
        for u, v in ((L["u"], L["v"]), (L["v"], L["u"])):
            adj.setdefault((u, L["branch"]), []).append(
                (v, L["branch"], L["ff_time_min"], L))
    # wait time by branch
    wait = {b: min(15.0, 30.0 / max(cfg["peak_tph"], 0.5))
            for b, cfg in BRANCHES.items()}
    egr = {}
    for e in egress:
        egr.setdefault(e["u"], []).append(e)

    out = {}
    for o in origins:
        starts = [(b, wait[b]) for (s, b) in adj if s == o]
        if not starts:
            continue
        dist, prev = {}, {}
        pq = []
        for b, w in starts:
            dist[(o, b)] = w
            heapq.heappush(pq, (w, (o, b)))
        while pq:
            d, state = heapq.heappop(pq)
            if d > dist.get(state, 1e18) + 1e-9:
                continue
            s, b = state
            for (v, vb, t, L) in adj.get(state, []):
                nd = d + t
                if nd < dist.get((v, vb), 1e18) - 1e-9:
                    dist[(v, vb)] = nd
                    prev[(v, vb)] = (state, L)
                    heapq.heappush(pq, (nd, (v, vb)))
            # transfer at this station to another branch
            for (s2, b2) in adj:
                if s2 == s and b2 != b:
                    nd = d + P["xfer_penalty_min"] + wait[b2]
                    if nd < dist.get((s2, b2), 1e18) - 1e-9:
                        dist[(s2, b2)] = nd
                        prev[(s2, b2)] = (state, None)
                        heapq.heappush(pq, (nd, (s2, b2)))
        for zi, z in enumerate(zones):
            best = None
            for e in egress:
                term = e["u"]
                cand = [(dist[(term, b)], (term, b)) for (t2, b) in dist
                        if t2 == term] if False else None
                for (s2, b2), dv in dist.items():
                    if s2 != term:
                        continue
                    if e["zone_idx"] != zi:
                        continue
                    tot = dv + e["time_min"]
                    money = e["fare"]
                    if best is None or tot + money / P["vot_per_min"] < best[0]:
                        best = (tot + money / P["vot_per_min"], tot, money,
                                term, (s2, b2))
            if best is not None:
                out[(o, zi)] = dict(gen_min=best[0], time_min=best[1],
                                    egress_fare=best[2], terminal=best[3],
                                    end_state=best[4])
        # reconstruct sequences lazily below
        out.setdefault(("__prev__", o), (dist, prev))
    return out


def rail_sequence(dist_prev, end_state):
    dist, prev = dist_prev
    seq, cur = [], end_state
    while cur in prev:
        st, L = prev[cur]
        if L is not None:
            seq.append((st[0], cur[0], L["branch"]))
        cur = st
    return list(reversed(seq))


def main():
    # ------------------------------------------------------------ geography
    tr = build_tracts()
    st = pd.DataFrame(STATIONS, columns=["name", "lat", "lon", "fare_zone", "county"])
    st["node_id"] = ["S" + str(i+1).zfill(3) for i in range(len(st))]
    tr["node_id"] = ["T" + str(i+1).zfill(4) for i in range(len(tr))]
    nyc = pd.DataFrame(NYC_ZONES, columns=["name", "lat", "lon", "attr_share", "walk_terms"])
    nyc["node_id"] = ["N" + str(i+1) for i in range(len(nyc))]

    print(f"tracts={len(tr)}  stations={len(st)}  nyc_zones={len(nyc)}")

    # --------------------------------------------------------- egress links
    egress = []
    for zi, z in nyc.iterrows():
        elig = st[(st.fare_zone == 1) | (st['name'].isin(z.walk_terms))]
        for _, s in elig.iterrows():
            d = haversine(s.lat, s.lon, z.lat, z.lon) * 1.2
            walk = s["name"] in z.walk_terms
            if walk:
                t, fare, op = max(3.0, d / 3.0 * 60), 0.0, "NONE"
            else:
                if d > 9:
                    continue
                t, fare, op = 6.0 + d / 12.0 * 60, 2.90, "LOCAL"   # subway
            egress.append(dict(u=s["name"], u_id=s.node_id, v=z.node_id,
                               zone_idx=zi, length_mi=round(d, 3),
                               time_min=round(t, 2), fare=fare, operator=op))
    # Jamaica -> Central Queens zone is a walk-out
    print(f"egress links={len(egress)}")

    # ----------------------------------------------------------- rail links
    rail = build_rail_graph(st)
    print(f"rail links={len(rail)}")

    # ------------------------------- station -> zone skims (rail + egress)
    li_st = st[st.county.isin(["Nassau", "Suffolk"])]
    skim = shortest_rail(rail, egress, list(li_st["name"]), list(range(len(nyc))))

    # ---------------------------------------------- access links: 3 nearest
    MAN = (40.7550, -73.9800)
    st_dman = np.array([haversine(a, b, *MAN) for a, b in
                        zip(li_st.lat, li_st.lon)])
    st_park = np.clip(P["park_floor"] + P["park_amp"]
                      * np.exp(-(st_dman - 8.0) / P["park_decay"]),
                      P["park_floor"], P["park_cap"]).round(2)

    sl, so = li_st.lat.to_numpy(), li_st.lon.to_numpy()
    snames = li_st["name"].to_numpy()
    sids = li_st.node_id.to_numpy()
    access = []
    for _, t in tr.iterrows():
        d = hav_vec(t.lat, t.lon, sl, so) * P["circuity"]
        order = np.argsort(d)[:3]
        for rank, k in enumerate(order):
            dd = float(d[k])
            # drive-and-park: operator-less
            access.append(dict(tract=t.node_id, station=snames[k],
                               station_id=sids[k], rank=rank+1,
                               length_mi=round(dd, 3), mode="DRIVE",
                               operator="NONE",
                               time_min=round(dd / P["drive_mph_local"] * 60 + 3.0, 2),
                               park_usd=float(st_park[k]),
                               cost=round(dd * P["auto_op_cost_mi"]
                                          + st_park[k], 2)))
            # local feeder transit
            headway = P["bus_headway_min"] * (1.0 if dd < 5 else 1.5)
            access.append(dict(tract=t.node_id, station=snames[k],
                               station_id=sids[k], rank=rank+1,
                               length_mi=round(dd * 1.15, 3), mode="TRANSIT",
                               operator="LOCAL",
                               time_min=round(dd*1.15 / P["bus_mph"] * 60 + headway/2 + 4.0, 2),
                               cost=P["fare_local"]))
    acc = pd.DataFrame(access)
    print(f"access links={len(acc)}")
    print(f"station parking: ${st_park.min():.2f}-${st_park.max():.2f}, "
          f"median ${np.median(st_park):.2f} (was a flat $3.00)")
    inst_park = pd.DataFrame(dict(station=li_st["name"].to_numpy(),
                                  node_id=li_st.node_id.to_numpy(),
                                  d_manhattan=st_dman.round(2),
                                  park_usd=st_park))

    # ------------------------------------------------- outside option links
    outside = []
    for _, t in tr.iterrows():
        for zi, z in nyc.iterrows():
            d = haversine(t.lat, t.lon, z.lat, z.lon) * P["circuity"]
            crz = P["crz_toll"] if z["name"] in (
                "Midtown West / Penn", "Midtown East / GCM", "Lower Manhattan") else 0.0
            park = {"Midtown West / Penn": 13.0, "Midtown East / GCM": 13.0,
                    "Lower Manhattan": 13.0, "Downtown Brooklyn": 14.0,
                    "Long Island City": 8.0, "Central Queens/Jamaica": 4.5}[z["name"]]
            # Two regimes: the last few miles into the core crawl, the rest
            # runs at AM-peak highway speed. A single flat speed charged a
            # 70 mile Suffolk trip 177 min against a realistic 141, i.e. $13
            # of phantom cost on exactly the long ODs where rail already won.
            core = {"Midtown West / Penn": 13.0, "Midtown East / GCM": 13.0,
                    "Lower Manhattan": 13.0, "Downtown Brooklyn": 15.0,
                    "Long Island City": 17.0, "Central Queens/Jamaica": 20.0}[z["name"]]
            d_core = min(d, P["core_miles"])
            t_drive = ((d - d_core) / P["hwy_mph"] + d_core / core) * 60
            outside.append(dict(tract=t.node_id, zone=z.node_id, zone_idx=zi,
                                length_mi=round(d, 3), operator="NONE",
                                time_min=round(t_drive + 8.0, 2),
                                cost=round(d*P["auto_op_cost_mi"] + crz + park, 2)))
    out_df = pd.DataFrame(outside)

    # ================================================= 3. demand generation
    man_ll = (40.7550, -73.9800)
    tr["d_manhattan"] = [haversine(r.lat, r.lon, *man_ll) for _, r in tr.iterrows()]
    # Productions are observed: LODES NYC-bound workers per tract, times the
    # AM departure share.  worker_rate and the nyc_share decay function that
    # this build previously assumed are both gone.
    tr["workers"] = tr.nyc_workers
    tr["nyc_share"] = 1.0
    tr["O_am"] = tr.nyc_workers * P["am_share"]
    print(f"AM LI->NYC person-trips = {tr.O_am.sum():,.0f} "
          f"(LODES {tr.nyc_workers.sum():,.0f} workers x {P['am_share']} AM share)")

    # ============================== 4. path enumeration + generalized costs
    st_zone = {r["name"]: r.fare_zone for _, r in st.iterrows()}
    acc_by_tract = {k: v for k, v in acc.groupby("tract")}
    paths = []
    for _, t in tr.iterrows():
        tid = t.node_id
        for _, a in acc_by_tract[tid].iterrows():
            sname = a.station
            for zi, z in nyc.iterrows():
                sk = skim.get((sname, zi))
                if sk is None:
                    continue
                fare_rail = lirr_fare(st_zone[sname], sk["terminal"])
                time = a.time_min + sk["time_min"]
                money = a.cost + fare_rail + sk["egress_fare"]
                paths.append(dict(
                    tract=tid, zone=z.node_id, zone_idx=zi, kind="RAIL",
                    station=sname, access_mode=a["mode"],
                    access_operator=a.operator, rail_operator="LIRR",
                    terminal=sk["terminal"], access_mi=a.length_mi,
                    time_min=round(time, 2), money=round(money, 2),
                    gen_cost=round(money + time * P["vot_per_min"], 3)))
        for zi, z in nyc.iterrows():
            o = out_df[(out_df.tract == tid) & (out_df.zone_idx == zi)].iloc[0]
            paths.append(dict(
                tract=tid, zone=z.node_id, zone_idx=zi, kind="OUTSIDE",
                station="", access_mode="AUTO", access_operator="NONE",
                rail_operator="NONE", terminal="", access_mi=o.length_mi,
                time_min=o.time_min, money=o.cost,
                gen_cost=round(o.cost + o.time_min * P["vot_per_min"], 3)))
    pth = pd.DataFrame(paths)
    print(f"paths={len(pth)}")

    # ------------------------------------- logsum accessibility by OD pair
    mu = P["mu"]
    pth["expv"] = np.exp(-mu * pth.gen_cost)
    ls = pth.groupby(["tract", "zone_idx"]).expv.sum().rename("expsum").reset_index()
    ls["logsum_cost"] = -np.log(ls.expsum) / mu

    # ================================== 5. doubly-constrained gravity (IPF)
    tracts = tr.node_id.to_numpy()
    ti = {n: i for i, n in enumerate(tracts)}
    C = np.full((len(tracts), len(nyc)), np.nan)
    for _, r in ls.iterrows():
        C[ti[r.tract], int(r.zone_idx)] = r.logsum_cost
    C = np.where(np.isnan(C), np.nanmax(C), C)
    F = np.exp(-P["beta_gravity"] * (C - C.min()))

    O = tr.O_am.to_numpy()
    zshare = nyc_zone_shares()
    nyc["attr_share"] = nyc["name"].map(zshare)
    D = nyc.attr_share.to_numpy() * O.sum()
    print("  destination shares (LODES): "
          + ", ".join(f"{k.split(' /')[0].split('/')[0]} {v:.3f}"
                      for k, v in zshare.items()))
    a_i = np.ones(len(O)); b_j = np.ones(len(D))
    for _ in range(200):
        a_i = O / np.maximum((F * b_j).sum(1), 1e-12)
        b_j = D / np.maximum((F * a_i[:, None]).sum(0), 1e-12)
    Tij = a_i[:, None] * F * b_j
    print(f"OD total = {Tij.sum():,.0f}   max row err = "
          f"{np.abs(Tij.sum(1)-O).max():.4f}")

    # ======================= 6. station-choice ASCs calibrated to ridership
    # target AM boardings by LI station
    st_catch = acc.groupby("station").apply(
        lambda d: d.tract.nunique(), include_groups=False)
    rw = {1: 0.58, 2: 0.29, 3: 0.13}
    catch_pop = (acc[["tract", "station", "rank"]].drop_duplicates()
                 .merge(tr[["node_id", "O_am"]], left_on="tract", right_on="node_id")
                 .assign(w=lambda d: d["rank"].map(rw) * d.O_am)
                 .groupby("station").w.sum())
    svc = {}
    for b, cfg in BRANCHES.items():
        for s in cfg["stops"]:
            svc[s] = max(svc.get(s, 0), cfg["peak_tph"])
    tot = P["am_lirr_boardings"]
    # Observed AM peak westbound boardings by station, MTA 2012-2014 O&D
    # Report sheets 197-199.  Only the spatial distribution is used: the
    # level is rescaled to the 2025-derived control total, since the survey
    # predates Grand Central Madison and the post-COVID shift.
    obs = AM_PEAK_BOARDINGS_2012_2014
    names = list(li_st["name"])
    miss = [n for n in names if n not in obs]
    tv = np.array([float(obs.get(n, 0.0)) for n in names])
    if miss:                       # no survey row: fall back to the catchment
        fw = np.array([catch_pop.get(n, 0.0) * (0.55 + 0.06 * svc.get(n, 1))
                       for n in miss], float)
        fw = np.maximum(fw, 1.0)
        share = fw / fw.sum() * tv.sum() * (len(miss) / max(len(names), 1))
        for n, v in zip(miss, share):
            tv[names.index(n)] = v
    scale = tot / tv.sum()
    tv = tv * scale
    target = dict(zip(names, tv))
    print(f"  station profile: observed 2012-2014 O&D, {len(names)-len(miss)} of "
          f"{len(names)} stations; rescaled x{scale:.3f} to the 2025 control")
    br_of = {}
    for b, cfg in BRANCHES.items():
        for st_ in cfg["stops"]:
            br_of.setdefault(st_, b)
    for b in ("Port Jefferson", "Babylon"):
        m = np.array([br_of.get(nm) == b for nm in names])
        print(f"    {b:16s} {tv[m].sum()/tot:6.1%} of AM boardings "
              f"(MTA 2025 annual: {BRANCH_2025[b]/LI_ORIGIN_ANNUAL:.1%})")

    # calibrate additive station constants (in $) so predicted == target
    pth["od"] = pth.tract + "|" + pth.zone_idx.astype(str)
    od_index = {k: i for i, k in enumerate(sorted(pth.od.unique()))}
    pth["od_i"] = pth.od.map(od_index)
    Tflat = np.zeros(len(od_index))
    for i, tname in enumerate(tracts):
        for zi in range(len(nyc)):
            k = od_index.get(f"{tname}|{zi}")
            if k is not None:
                Tflat[k] = Tij[i, zi]
    stn_list = list(li_st["name"])
    stn_i = {s: i for i, s in enumerate(stn_list)}
    p_stn = pth.station.map(lambda s: stn_i.get(s, -1)).to_numpy()
    gc0 = pth.gen_cost.to_numpy().copy()
    od_i = pth.od_i.to_numpy()
    is_drive = (pth.access_mode == "DRIVE").to_numpy()
    alpha = np.zeros(len(stn_list))
    tvec = np.array([target[s] for s in stn_list])

    # --- parking supply: distribute spaces by target ridership, floor/cap ---
    cty_f = np.array([1.35 if c == "Suffolk" else 1.0 for c in li_st.county])
    jit = np.exp(np.random.default_rng(11).normal(0, 0.26, len(stn_list)))
    spaces = np.power(np.maximum(tvec, 1.0), 0.85) * cty_f * jit
    spaces = np.clip(spaces / spaces.sum() * P["park_spaces_total"], 45, 3800)
    spaces *= P["park_spaces_total"] / spaces.sum()
    park_cap = spaces * P["park_turnover"] * P["park_occupancy"]   # person-trips
    t0_park = 3.0                                                  # min, uncongested

    # --- MSA over parking congestion, with ASC recalibration inside ---
    drive_load = np.zeros(len(stn_list))
    for outer in range(150):
        util = drive_load / np.maximum(park_cap, 1.0)
        delay = t0_park * P["park_bpr_b"] * np.power(util, P["park_bpr_pow"])
        delay = np.minimum(delay, 45.0)
        gc = gc0 + np.where(is_drive & (p_stn >= 0),
                            delay[np.clip(p_stn, 0, None)] * P["vot_per_min"], 0.0)
        for it in range(150):
            adj = np.where(p_stn >= 0, alpha[np.clip(p_stn, 0, None)], 0.0)
            v = np.exp(-mu * (gc - adj))
            den = np.bincount(od_i, weights=v, minlength=len(od_index))
            flow = v / den[od_i] * Tflat[od_i]
            pred = np.bincount(np.clip(p_stn, 0, None),
                               weights=np.where(p_stn >= 0, flow, 0),
                               minlength=len(stn_list))
            err = np.abs(pred - tvec).sum() / tvec.sum()
            if err < 3e-4:
                break
            alpha += (np.log(np.maximum(tvec, 1e-6))
                      - np.log(np.maximum(pred, 1e-6))) / mu * 0.5
        new_drive = np.bincount(np.clip(p_stn, 0, None),
                                weights=np.where(is_drive & (p_stn >= 0), flow, 0),
                                minlength=len(stn_list))
        gap = np.abs(new_drive - drive_load).sum() / max(new_drive.sum(), 1)
        drive_load = drive_load + (new_drive - drive_load) / (outer + 2)
        if gap < 1e-3 and outer > 3:
            break
    print(f"parking equilibrium: {outer+1} outer iters, gap={gap:.2e}; "
          f"ASC rel err={err:.2e}")
    print(f"mean parking utilisation = {np.average(drive_load/park_cap, weights=park_cap):.2f}")
    pth["flow"] = flow
    pth["gen_cost_eq"] = gc
    pth["station_asc"] = np.where(p_stn >= 0, adj, 0.0)
    bus_veh_hr = np.clip(np.sqrt(tvec) / 9.0, 1.0, 14.0)
    bus_pool = bus_veh_hr * P["bus_seats"] * P["peak_equiv_hours"]
    park_df = pd.DataFrame(dict(station=stn_list,
                                bus_veh_hr=np.round(bus_veh_hr, 2),
                                bus_cap_persons=np.round(bus_pool, 1), spaces=np.round(spaces).astype(int),
                                park_cap_persons=np.round(park_cap, 1),
                                drive_load=np.round(drive_load, 1),
                                utilisation=np.round(drive_load/park_cap, 3),
                                park_delay_min=np.round(delay, 2),
                                asc_dollars=np.round(alpha, 4),
                                target_boardings=np.round(tvec, 1)))

    rail_tot = pth[pth.kind == "RAIL"].flow.sum()
    print(f"LIRR AM boardings = {rail_tot:,.0f}  "
          f"({rail_tot/Tij.sum()*100:.1f}% mode share); "
          f"outside = {pth[pth.kind=='OUTSIDE'].flow.sum():,.0f}")
    ac = pth[pth.kind == "RAIL"].groupby("access_mode").flow.sum()
    print("access split:", {k: f"{v:,.0f}" for k, v in ac.items()})

    return dict(tr=tr, st=st, nyc=nyc, acc=acc, rail=rail, egress=egress,
                out_df=out_df, pth=pth, Tij=Tij, skim=skim, target=target,
                park=park_df, li_st=li_st, tracts=tracts, P=P,
                station_park=inst_park,
                od_index=od_index, Tflat=Tflat, FARE=FARE_TO_ZONE1,
                st_zone=st_zone)


if __name__ == "__main__":
    R = main()
    import pickle
    with open(os.path.join(OUT, "_state.pkl"), "wb") as f:
        pickle.dump(R, f)
    print("stage 1 complete")

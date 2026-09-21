"""LIRR v2 as a P2Instance, destination segments, Eq.(243) normalisation.

phi is stored as absolute disutility (outside links run -158..-9.7), so each
segment's surpluses are shifted so its outside option carries phi = 0.  With
multi-origin (destination) segments the shift is per ORIGIN: every path from a
tract uses exactly one link leaving that tract, so subtracting that tract's
outside value from all links leaving it shifts every path of that origin by a
constant, which the origin potential absorbs.  Flows are unchanged.
"""
import numpy as np, pandas as pd
from algorithm2_p2 import P2Instance

def build(alpha_B=1.0, alpha_S=2.0, c_idle=None, with_z=False, n_feat=7):
    links = pd.read_csv('links_p2.csv'); nodes = pd.read_csv('nodes.csv')
    paths = pd.read_csv('paths.csv'); od = pd.read_csv('od_am.csv')
    segs = pd.read_csv('segments.csv')
    segs = segs[segs.segmentation == 'destination'].reset_index(drop=True)
    bn = pd.read_csv('b_node.csv'); bn = bn[bn.segmentation == 'destination']

    nidx = {n: i for i, n in enumerate(nodes.node_id)}
    aidx = {a: i for i, a in enumerate(links.link_id)}
    nL, nV, nQ = len(links), len(nodes), len(segs)
    tail = links.from_node.map(nidx).to_numpy()
    owner_map = {'LIRR': 0, 'LOCAL': 1}
    link_owner = np.array([owner_map.get(o, -1) for o in links.operator])
    owned = np.flatnonzero(link_owner >= 0)
    phi0 = links.phi_usd.to_numpy(float)

    avail = np.zeros((nQ, nL), bool)
    phi = np.zeros((nQ, nL))
    seg_i = {s: i for i, s in enumerate(segs.segment)}
    for i, s in enumerate(segs.segment):
        g = paths[paths.zone == s]
        R = np.array(sorted({aidx[x] for v in g.link_ids for x in v.split('|')}))
        avail[i, R] = True
        phi[i, R] = phi0[R]
    # per-origin shift so each tract's outside option carries phi = 0
    outside_link = np.zeros(nQ, int); seen = set()
    for r in od.itertuples():
        i = seg_i[r.d_node]; o = nidx[r.o_node]
        a0 = aidx[f'OUT_{r.o_node}_{r.d_node}']
        m = avail[i] & (tail == o)
        phi[i, m] -= phi0[a0]
        outside_link[i] = a0        # representative; all are 0 after the shift
        seen.add((i, a0))
    # outside cells with no OD row carry no demand; drop them rather than
    # leave an unshifted phi behind
    isout = np.array([str(l).startswith('OUT_') for l in links.link_id])
    for i in range(nQ):
        for a in np.flatnonzero(avail[i] & isout):
            if (i, int(a)) not in seen:
                avail[i, a] = False; phi[i, a] = 0.0

    cap = links.capacity_period.to_numpy(float)[owned]
    cap = np.where(np.isnan(cap) | (cap <= 0), 1e7, cap)

    b = np.zeros((nQ, nV))
    for r in bn.itertuples():
        i = seg_i[r.segment]
        b[i, nidx[r.node]] = r.b / segs.demand.to_numpy(float)[i]

    inst = P2Instance(
        tail=tail, head=links.to_node.map(nidx).to_numpy(),
        link_owner=link_owner, n_nodes=nV, D=segs.demand.to_numpy(float),
        origin=np.zeros(nQ, int),
        sink=np.array([nidx[s] for s in segs.sink_node]),
        F=np.array([1434827.52, 88948.18]), Cmax=cap,
        c=links.cap_cost_per_unit.to_numpy(float)[owned],
        ell=links.ell.to_numpy(float), phi=phi,
        alpha_B=alpha_B, alpha_S=alpha_S, avail=avail,
        b_supply=b,
        c_idle=(np.array([0.27, 0.72]) if c_idle is None
                else np.asarray(c_idle, float)),
        outside_link=outside_link,
        link_names=list(links.link_id), operator_names=['LIRR', 'LOCAL'])
    Zs = None
    if with_z:
        # features shifted per origin exactly as phi, so z_{i,a_i0} = 0 while
        # every other link keeps its attributes measured relative to driving
        lt = links.link_type.to_numpy()
        TIMED = ["RAIL", "ACCESS_DRIVE", "ACCESS_TRANSIT", "EGRESS", "OUTSIDE"]
        Z0 = np.zeros((nL, n_feat))
        for k, t in enumerate(TIMED):
            Z0[:, k] = links.ff_time_min.to_numpy() * (lt == t)
        Z0[:, 5] = (links.user_cost_usd - links.fare_usd).clip(lower=0).to_numpy()
        Z0[:, 6] = (lt == "OUTSIDE").astype(float)
        Zs = np.tile(Z0, (nQ, 1, 1))
        for r in od.itertuples():
            i = seg_i[r.d_node]; o = nidx[r.o_node]
            a0 = aidx[f'OUT_{r.o_node}_{r.d_node}']
            m = avail[i] & (tail == o)
            Zs[i, m, :] -= Z0[a0]
        Zs[~avail] = 0.0

    return inst, links, segs, aidx, nidx, Zs

"""
Cycle-projected solution of the inverse PURC link-flow aggregate network
assignment game IO1 ("Algorithm 4" for "IO2" in earlier drafts).

This is NOT the paper's Algorithm 3, which is qp/run_qp_sweep.py (see
MAPPING.md). Algorithm 3 imports this module's seller block and data loading,
and this module is the original implementation of the "Projection (R only)"
ablation arm. Equation and remark numbers below follow the earlier draft.

The buyer stationarity condition of Eq. (57) carries one node potential
difference per cell,

    q_ia = thetat' z_ia - alpha_B p_a - ( mu_{i,t(a)} - mu_{i,h(a)} ) ,
    q_ia = l_a f0_ia / D_i ,   p_a = tau0_ma + sigma_a ,

so a naive least squares would estimate |Q| |V| nuisance potentials alongside
Theta = (thetat, alpha_B). Instead the potentials are annihilated: on any cycle
of the active subgraph the differences telescope to zero, so left-multiplying
by a basis C of the cycle space removes mu exactly and leaves

    C q = C Z thetat - C p alpha_B ,

a bound-constrained linear least squares in Theta alone. The projection also
supplies the identification certificate of step 4: if [C Z, -C p] is rank
deficient, Theta is not identified on the sampled support, and no amount of
regularization repairs it.

    Step 1  Preprocess       partition A+ / A0, fix the rents of Eq. (41),
                             load factors, idle stock, weights, sigma^0 = 0.
    Step 2  Seller block     the seller QP (step 2 of the paper's Algorithm 3).
    Step 3  Cycle projection spanning forest per segment by union-find; one row
                             per non-forest edge.
    Step 4  Identification   rank test on [C Z, -C p].
    Step 5  Buyer block      bound-constrained least squares in Theta.
    Step 6  Fixed-price      Algorithm 2' at the estimated primitives returns
                             u, y, sigma~^r and every f_ia.
    Step 7  Support check    fraction of cells in V with psi_ia > 0.
    Step 8  Average and test sigma^r = (1 - gamma_r) sigma^{r-1} + gamma_r
                             sigma~^r, gamma_r = r^-0.6.
    Step 9  Clearing         Algorithm 2 for tau* and the wedge Delta.

Step 5 is consistent for Theta but not efficient: the residuals of Eq. (60) are
heteroskedastic with variance l_a / (D_i^2 f*_ia) under Assumption 4, and the
weights cannot be carried through the projection because C mixes cells. See
Remark 10.

Requires: numpy, scipy, and `algorithm2_p2.py` and `algorithm2prime_p2.py` in
the same directory.

Reference: Chow (2026), Sections 3.3 and 4.2.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from algorithm2_p2 import P2Instance, solve_p2
from algorithm2prime_p2 import solve_p2_fixed_price


# ----------------------------------------------------------------------------
# Observed data
# ----------------------------------------------------------------------------

@dataclass
class IO2Data:
    """Observations entering IO2, Assumption 4.

    Parameters
    ----------
    f_seg : (Q, A) float
        Segment-level link flows f0_ia. Zeros are informative: they place the
        cell in A0 and are used only by the support check of step 7. Use np.nan
        for a cell outside O_F.
    y : (A_own,) float
        Sampled allocated capacities y0_ma; np.nan for an unsampled link.
    f_link : (A_own,) float
        Aggregate link counts f0_a, used only for the load factors L_ma.
    tau0 : (A_own,) float
        Observed fares over owned links. Data, not a decision variable.
    z : (Q, A, n_feat) float
        Link features z_ia, zero on links in A_0 so that phi_{i,a_i0} = 0.
    c_prior : (A_own,) float
    c_idle_prior : (M,) float
    Y : (R,) float, optional
        Observed service capacities Y0_r (e.g. seats per service from the
        timetable); np.nan for an unobserved service. Needed when the instance
        has services (H != I); with H = I it defaults to y.
    """

    f_seg: np.ndarray
    y: np.ndarray
    f_link: np.ndarray
    tau0: np.ndarray
    z: np.ndarray
    c_prior: np.ndarray
    c_idle_prior: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        for nm in ("f_seg", "y", "f_link", "tau0", "z", "c_prior"):
            setattr(self, nm, np.asarray(getattr(self, nm), dtype=float))
        if self.c_idle_prior is None:
            self.c_idle_prior = np.zeros(0)
        self.c_idle_prior = np.asarray(self.c_idle_prior, dtype=float)
        if np.any(self.tau0 < 0):
            raise ValueError("observed fares tau0 must be non-negative")


# ----------------------------------------------------------------------------
# Solution container
# ----------------------------------------------------------------------------

@dataclass
class IO2Solution:
    theta: np.ndarray
    c: np.ndarray
    c_idle: np.ndarray
    pi: np.ndarray
    alpha_B: float
    alpha_S: float
    u: np.ndarray              # (Q, V) node potentials from Algorithm 2'
    v: np.ndarray              # (M,)
    f_seg: np.ndarray          # (Q, A) fitted flows, incl. unsampled cells
    sigma: np.ndarray          # (A_own,)
    phi: np.ndarray            # (Q, A) fitted link surpluses
    tau_star: np.ndarray       # (A_own,) clearing prices from Algorithm 2
    wedge: np.ndarray          # (A_own,) Delta_ma
    cycle_rank: np.ndarray     # (Q,) independent cycles per segment
    design_rank: int           # rank of [C Z, -C p]
    design_cond: float         # condition number of the same design
    identified: bool
    support_violations: float  # fraction of A0 cells with psi_ia > 0
    support_max_psi: float
    obj_buyer: float
    obj_seller: float
    n_outer: int
    wall_time: float
    converged: bool
    message: str
    history: Dict[str, List[float]] = field(default_factory=dict)

    def summary(self, inst: P2Instance, data: IO2Data) -> str:
        ln = [
            f"Algorithm 4  ({'converged' if self.converged else 'STOPPED'}: {self.message})",
            f"  outer iterations  : {self.n_outer}",
            f"  wall time         : {1e3 * self.wall_time:.1f} ms",
            f"  cycle rank / seg  : {self.cycle_rank.tolist()}"
            f"   (design rank {self.design_rank}, cond "
            f"{self.design_cond:.2e}, "
            f"{'identified' if self.identified else 'NOT IDENTIFIED'})",
            f"  buyer objective   : {self.obj_buyer:.6e}",
            f"  seller objective  : {self.obj_seller:.6e}",
            f"  alpha_B, alpha_S  : {self.alpha_B:.6f}, {self.alpha_S:.6f}",
            f"  support check     : {100 * self.support_violations:.2f}% of A0 "
            f"cells with psi > 0, max psi = {self.support_max_psi:+.4e}",
            "",
            "  theta             : " + np.array2string(self.theta, precision=6),
            "",
            "  link      c_ma     c0_ma      pi_ma    sigma_a     tau0     tau*      Delta",
        ]
        names = inst.link_names or [f"a{a}" for a in range(inst.n_links)]
        for j, a in enumerate(inst.owned_idx):
            ln.append(
                f"  {names[a]:<8}{self.c[j]:8.4f}  {data.c_prior[j]:8.4f}  "
                f"{self.pi[j]:8.4f}  {self.sigma[j]:8.4f}  {data.tau0[j]:7.3f}"
                f"{self.tau_star[j]:9.3f}{self.wedge[j]:+10.4f}"
            )
        ln += ["", "  operator       v_m      c_idle_m"]
        onames = inst.operator_names or [f"m{m}" for m in range(inst.n_op)]
        for m in range(inst.n_op):
            ln.append(f"  {onames[m]:<12}{self.v[m]:9.4f}{self.c_idle[m]:12.4f}")
        return "\n".join(ln)


# ----------------------------------------------------------------------------
# Step 2: seller block (step 2 of the paper's Algorithm 3)
# ----------------------------------------------------------------------------

def seller_block(inst, data, rho_c, rho_pi, eps_scale):
    """Step 2, on services when the instance has them (H != I).

    Service form of the seller QP:
        r_r  = ln(Y0_r / F_m) - aS sum_a H_ar tau0_a L_a + sum_a H_ar ct_a + pit_r + beta_m
        r_m0 = ln(Y0_m0 / F_m) + ct_idle,m + beta_m
    with ct = aS c the link costs (deviations from the prior penalized by rho_c)
    and pit = aS pi free only where the service is observed at its cap. With
    H = I and Y0 = y0 it is the link-level QP, which is kept below as
    `seller_block_links`.
    """
    if inst.n_svc == inst.n_own and data.Y is None and inst.n_aS == 1:
        Hc = inst.Hs.tocoo()
        if np.array_equal(Hc.row, Hc.col) and Hc.nnz == inst.n_own:
            return seller_block_links(inst, data, rho_c, rho_pi, eps_scale)
    return seller_block_svc(inst, data, rho_c, rho_pi, eps_scale)


def seller_block_svc(inst, data, rho_c, rho_pi, eps_scale):
    """Step 2 on services, Eq. (S9). One supply scale per group of operators
    (inst.aS_group); with a single group this is the common-alpha_S block.
    Returns (c, c_idle, pi, alpha_S, beta, obj), alpha_S a float for one group
    and a per-operator array otherwise."""
    nM = inst.n_op
    grp, nG = inst.aS_group, inst.n_aS
    Ydat = np.asarray(data.Y if data.Y is not None else data.y, dtype=float)
    so = inst.svc_owner
    obs = np.isfinite(Ydat) & (Ydat > 0)
    rj = np.flatnonzero(obs)
    if rj.size == 0:
        raise ValueError("the seller block needs at least one observed service")
    M = inst.HT[rj]                                   # (n_obs, A_own)
    J = np.unique(M.tocoo().col)                      # links touched
    MJ = M[:, J].tocsr()
    yl = np.asarray(data.y, dtype=float)
    L_all = np.minimum(1.0, np.where(np.isfinite(yl) & (yl > 0),
                                     data.f_link / np.maximum(yl, 1e-300), 1.0))
    t0L = MJ @ (data.tau0[J] * L_all[J])
    c0J = data.c_prior[J]
    lS = np.log(Ydat[rj] / inst.F[so[rj]])
    wS = Ydat[rj]
    opj = so[rj]
    gj = grp[opj]                                     # scale group of each service
    pi_free = Ydat[rj] >= inst.svc_Cmax[rj] - 1e-9
    y_idle = np.array([inst.F[m] - np.nansum(np.where(so == m, Ydat, np.nan))
                       for m in range(nM)])
    y_idle = np.maximum(y_idle, 1e-12)
    l0 = np.log(y_idle / inst.F)
    w0 = y_idle
    c0_idle = (data.c_idle_prior if data.c_idle_prior.size == nM else np.zeros(nM))
    # Track-cap rents kappa_a on shared links observed at their cap.
    K = np.array([a for a in inst.lcap_idx
                  if np.isfinite(yl[a]) and yl[a] >= inst.Cmax[a] * (1 - 1e-6)], dtype=int)
    MK = inst.HT[rj][:, K].tocsr() if K.size else None
    n, nJ, nK = rj.size, J.size, K.size
    o = 0
    sl = {}
    for key, size in (("cp", nJ), ("cm", nJ), ("pit", n), ("dp", nM), ("dm", nM),
                      ("aS", nG), ("beta", nM), ("kt", nK)):
        sl[key] = slice(o, o + size); o += size
    nvar = o
    MJT = MJ.T.tocsr()
    MKT = MK.T.tocsr() if nK else None
    Mc0 = MJ @ c0J

    def fun_grad(p):
        aG = p[sl["aS"]]
        aO = aG[grp]                                  # per operator
        aR = aG[gj]                                   # per observed service
        dev = p[sl["cp"]] - p[sl["cm"]]
        ct_idle = aO * c0_idle + p[sl["dp"]] - p[sl["dm"]]
        beta = p[sl["beta"]]
        r = lS - aR * t0L + aR * Mc0 + MJ @ dev + p[sl["pit"]] + beta[opj]
        if nK:
            r = r + MK @ p[sl["kt"]]
        r0 = l0 + ct_idle + beta
        obj = float(wS @ (r ** 2)) + float(w0 @ (r0 ** 2))
        obj += rho_c * (float(np.sum(p[sl["cp"]] + p[sl["cm"]]))
                        + float(np.sum(p[sl["dp"]] + p[sl["dm"]])))
        obj += rho_pi * (float(np.sum(p[sl["pit"]])) + float(np.sum(p[sl["kt"]])))
        g = np.zeros(nvar)
        gr, gr0 = 2.0 * wS * r, 2.0 * w0 * r0
        gl = MJT @ gr
        g[sl["cp"]] = gl + rho_c
        g[sl["cm"]] = -gl + rho_c
        g[sl["pit"]] = np.where(pi_free, gr + rho_pi, 0.0)
        g[sl["dp"]] = gr0 + rho_c
        g[sl["dm"]] = -gr0 + rho_c
        gA = np.zeros(nG)
        np.add.at(gA, gj, gr * (Mc0 - t0L))
        np.add.at(gA, grp, gr0 * c0_idle)
        g[sl["aS"]] = gA
        np.add.at(g, sl["beta"].start + opj, gr)
        g[sl["beta"]] += gr0
        if nK:
            g[sl["kt"]] = MKT @ gr + rho_pi
        return obj, g

    p0 = np.zeros(nvar)
    p0[sl["aS"]] = 1.0
    p0[sl["beta"]] = 1.0
    bounds = ([(0.0, None)] * (2 * nJ)
              + [(0.0, None) if fr else (0.0, 0.0) for fr in pi_free]
              + [(0.0, None)] * (2 * nM) + [(eps_scale, None)] * nG + [(None, None)] * nM
              + [(0.0, None)] * nK)
    res = minimize(fun_grad, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14, "maxcor": 20})
    aG = res.x[sl["aS"]]
    aO = aG[grp]
    aLink = aO[inst.owner_of_dual]                    # scale of each owned link
    c_hat = data.c_prior.copy()
    c_hat[J] = c0J + (res.x[sl["cp"]] - res.x[sl["cm"]]) / aLink[J]
    pi_hat = np.zeros(inst.n_svc)
    pi_hat[rj] = res.x[sl["pit"]] / aG[gj]
    c_idle_hat = np.maximum((aO * c0_idle + res.x[sl["dp"]] - res.x[sl["dm"]]) / aO, 0.0)
    seller_block_svc.last_kappa = dict(zip(K.tolist(), (res.x[sl["kt"]] / aLink[K]).tolist()))
    aS_out = float(aG[0]) if nG == 1 else aO.copy()
    return c_hat, c_idle_hat, pi_hat, aS_out, res.x[sl["beta"]], float(res.fun)


def seller_block_links(
    inst: P2Instance,
    data: IO2Data,
    rho_c: float,
    rho_pi: float,
    eps_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float]:
    """Step 2. Bound-constrained QP; returns (c, c_idle, pi, alpha_S, beta, obj).

        r_ma = l^S_ma - alpha_S tau0_ma L_ma + ct_ma + pit_ma + beta_m
        r_m0 = ln(y0_m0 / F_m) + ct_idle,m + beta_m

    in the reduced form ct = alpha_S c, pit = alpha_S pi, with the deviation
    splits of Eqs. (42) and (43). Rents are free only where the link is
    observed at its cap, Eq. (41).
    """
    nO, nM = inst.n_own, inst.n_op
    owner = inst.owner_of_dual

    obs = np.isfinite(data.y) & (data.y > 0)
    j = np.flatnonzero(obs)
    if j.size == 0:
        raise ValueError("the seller block needs at least one sampled link")

    lS = np.log(data.y[j] / inst.F[owner[j]])
    wS = data.y[j]
    L = np.minimum(1.0, np.where(data.y[j] > 0, data.f_link[j] / data.y[j], 0.0))
    t0L = data.tau0[j] * L
    c0 = data.c_prior[j]
    opj = owner[j]
    pi_free = data.y[j] >= inst.Cmax[j] - 1e-9

    y_idle = np.array([
        inst.F[m] - np.nansum(np.where(owner == m, data.y, np.nan))
        for m in range(nM)
    ])
    y_idle = np.maximum(y_idle, 1e-12)
    l0 = np.log(y_idle / inst.F)
    w0 = y_idle
    c0_idle = (data.c_idle_prior if data.c_idle_prior.size == nM
               else np.zeros(nM))

    n = j.size
    sl = {"cp": slice(0, n), "cm": slice(n, 2 * n), "pit": slice(2 * n, 3 * n),
          "dp": slice(3 * n, 3 * n + nM), "dm": slice(3 * n + nM, 3 * n + 2 * nM),
          "aS": slice(3 * n + 2 * nM, 3 * n + 2 * nM + 1),
          "beta": slice(3 * n + 2 * nM + 1, 3 * n + 3 * nM + 1)}
    nvar = 3 * n + 3 * nM + 1

    def fun_grad(p: np.ndarray) -> Tuple[float, np.ndarray]:
        aS = p[sl["aS"]][0]
        ct = aS * c0 + p[sl["cp"]] - p[sl["cm"]]
        ct_idle = aS * c0_idle + p[sl["dp"]] - p[sl["dm"]]
        beta = p[sl["beta"]]
        r = lS - aS * t0L + ct + p[sl["pit"]] + beta[opj]
        r0 = l0 + ct_idle + beta

        obj = float(wS @ (r ** 2)) + float(w0 @ (r0 ** 2))
        obj += rho_c * (float(np.sum(p[sl["cp"]] + p[sl["cm"]]))
                        + float(np.sum(p[sl["dp"]] + p[sl["dm"]])))
        obj += rho_pi * float(np.sum(p[sl["pit"]]))

        g = np.zeros(nvar)
        gr, gr0 = 2.0 * wS * r, 2.0 * w0 * r0
        g[sl["cp"]] = gr + rho_c
        g[sl["cm"]] = -gr + rho_c
        g[sl["pit"]] = np.where(pi_free, gr + rho_pi, 0.0)
        g[sl["dp"]] = gr0 + rho_c
        g[sl["dm"]] = -gr0 + rho_c
        g[sl["aS"]] = float(gr @ (c0 - t0L)) + float(gr0 @ c0_idle)
        np.add.at(g, sl["beta"].start + opj, gr)
        g[sl["beta"]] += gr0
        return obj, g

    p0 = np.zeros(nvar)
    p0[sl["aS"]] = 1.0
    p0[sl["beta"]] = 1.0
    bounds = ([(0.0, None)] * (2 * n)
              + [(0.0, None) if fr else (0.0, 0.0) for fr in pi_free]
              + [(0.0, None)] * (2 * nM)
              + [(eps_scale, None)]
              + [(None, None)] * nM)

    res = minimize(fun_grad, p0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14,
                            "maxcor": 20})

    aS = float(res.x[sl["aS"]][0])
    c_hat = data.c_prior.copy()
    pi_hat = np.zeros(nO)
    c_hat[j] = (aS * c0 + res.x[sl["cp"]] - res.x[sl["cm"]]) / aS
    pi_hat[j] = res.x[sl["pit"]] / aS
    # Idle cost is a cost: clip at zero.  c_idle is pinned only by the idle
    # row of Eq. (61c), conditional on beta_m, so it is weakly identified and
    # can come back negative (-0.159 on the LIRR 10% draw), which has no
    # economic reading.
    c_idle_hat = np.maximum(
        (aS * c0_idle + res.x[sl["dp"]] - res.x[sl["dm"]]) / aS, 0.0)
    return c_hat, c_idle_hat, pi_hat, aS, res.x[sl["beta"]], float(res.fun)


# ----------------------------------------------------------------------------
# Step 3: cycle projection
# ----------------------------------------------------------------------------

def cycle_basis(
    tail: np.ndarray,
    head: np.ndarray,
    cells: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """One row per independent cycle of the active subgraph of a segment.

    `cells` indexes the active links of this segment. Parallel links are kept
    as distinct edges, so two links sharing endpoints already close a cycle.
    Row e has +1 on the non-forest edge and -1 times the signed forest path
    from its tail to its head, so that row . d = 0 for any vector of potential
    differences d_a = u_{t(a)} - u_{h(a)}.
    """
    m = cells.size
    parent = np.arange(n_nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    adj: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_nodes)]
    forest: List[int] = []
    extra: List[int] = []
    for q, a in enumerate(cells):
        t, h = int(tail[a]), int(head[a])
        rt, rh = find(t), find(h)
        if rt != rh or t == h:
            if rt != rh:
                parent[rt] = rh
                adj[t].append((h, q, +1))
                adj[h].append((t, q, -1))
                forest.append(q)
            else:
                extra.append(q)          # self-loop closes a cycle on its own
        else:
            extra.append(q)

    rows = np.zeros((len(extra), m))
    for r, q in enumerate(extra):
        a = cells[q]
        t, h = int(tail[a]), int(head[a])
        rows[r, q] = 1.0
        if t == h:
            continue
        # BFS on the forest from t to h, recording the signed edges used
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
            continue                      # different components: no cycle
        node = h
        while node != t:
            n0, q1, s1 = prev[node]
            rows[r, q1] -= s1             # subtract the forest path
            node = n0
    return rows


def build_projection(
    inst: P2Instance,
    data: IO2Data,
    active: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step 3. Stack the per-segment cycle bases.

    Returns (Cq, CZ, Cp_idx, cycle_rank, cells) where Cq is the projected
    left-hand side, CZ the projected feature matrix, and Cp_idx the projected
    incidence onto the owned-link price vector, so that the price term is
    Cp_idx @ (tau0 + sigma) for any sigma.
    """
    Cq_blocks, CZ_blocks, CP_blocks = [], [], []
    cyc = np.zeros(inst.n_seg, dtype=int)
    for i in range(inst.n_seg):
        cells = np.flatnonzero(active[i])
        if cells.size == 0:
            continue
        C = cycle_basis(inst.tail, inst.head, cells, inst.n_nodes)
        cyc[i] = C.shape[0]
        if C.shape[0] == 0:
            continue
        q = inst.ell[cells] * data.f_seg[i, cells] / inst.D[i]
        Z = data.z[i, cells, :]
        # map each cell to its owned-link dual index, -1 for links in A_0
        P = np.zeros((cells.size, inst.n_own))
        for r, a in enumerate(cells):
            d = inst.dual_of_link[a]
            if d >= 0:
                P[r, d] = 1.0
        Cq_blocks.append(C @ q)
        CZ_blocks.append(C @ Z)
        CP_blocks.append(C @ P)

    if not Cq_blocks:
        raise ValueError("no cycles on the sampled support: Theta is not "
                         "identified, see step 4")
    return (np.concatenate(Cq_blocks), np.vstack(CZ_blocks),
            np.vstack(CP_blocks), cyc, active)


# ----------------------------------------------------------------------------
# Steps 4 and 5: identification test and buyer block
# ----------------------------------------------------------------------------

def buyer_block(
    Cq: np.ndarray,
    CZ: np.ndarray,
    CP: np.ndarray,
    price: np.ndarray,
    eps_scale: float,
    fix_alpha_B: bool = False,
    ridge_lambda: float = 0.0,
    ridge_anchor: Optional[np.ndarray] = None,
    ridge_scale: str = "none",
) -> Tuple[np.ndarray, float, float, int, int, bool, float]:
    """Steps 4-5. Rank test, then bound-constrained least squares in Theta.

        min_{thetat, alpha_B >= eps}  || Cq - CZ thetat + (CP p) alpha_B ||^2

    Returns (thetat, alpha_B, objective, design rank, n_params, identified,
    condition number).

    The rank threshold is relative, not absolute.  An absolute cutoff does not
    scale with the design: the last column is Cp = CP @ (tau0 + sigma), and
    sigma is estimated, so A rescales between runs and a borderline singular
    value can cross a fixed threshold.  That made the reported rank depend on
    the solver tolerance -- the same case gave rank 7 at inner_tol = 1e-7 and
    rank 8 at 1e-4.  numpy's default S.max() * max(shape) * eps is
    scale-invariant and is what is used here.

    The condition number is returned alongside: a design may satisfy the rank
    test yet be badly determined, and the two together are far more
    informative than the binary flag.

    ridge_lambda, ridge_anchor add a Tikhonov term lambda ||thetat - anchor||^2.
    In the SVD of CZ this replaces the pseudo-inverse weight 1/s_k by
    s_k/(s_k^2 + lambda), so each direction is a convex combination

        thetat_k = [s_k^2/(s_k^2+L)] thetat_LS,k + [L/(s_k^2+L)] anchor_k

    i.e. the shrinkage toward the anchor is automatically heavier the worse
    identified the direction is. With anchor = thetat from the previous outer
    iteration this is a proximal-point step (MSA in theta-space) and leaves the
    fixed point unchanged, since the penalty vanishes when thetat stops moving.
    With a constant anchor it instead scales the sigma -> theta gain by
    s_k^2/(s_k^2+L), which is what can move the loop gain below 1.
    """
    Cp = CP @ price
    # With the scale fixed, alpha_B leaves the design and the fare term moves
    # to the right-hand side.  Under a misspecified buyer specification the
    # fitted fare coefficient is pushed negative, alpha_B pins at its floor,
    # and theta = theta~/alpha_B diverges -- which then feeds a degenerate
    # sigma back into the design and collapses its conditioning (rank 7 of 8
    # at cond 7.4e17 on the LIRR 10% draw).  Fixing the scale puts the
    # misspecification into theta, where it belongs.
    A = CZ if fix_alpha_B else np.column_stack([CZ, -Cp])
    sv = np.linalg.svd(A, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A))
    n_params = A.shape[1]
    identified = rank >= n_params
    cond = float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf")

    nF = CZ.shape[1]
    lam = float(max(ridge_lambda, 0.0))
    anchor = (np.zeros(nF) if ridge_anchor is None
              else np.asarray(ridge_anchor, float).ravel()[:nF])

    # Per-feature ridge scaling. A SCALAR lambda penalises every coefficient
    # equally in RAW units, which on this design is arbitrary and actively
    # misleading: the columns of CZ span 3.87 (drive, a 0/1 mode dummy) to
    # 316.6 (t_OUT, minutes), a factor of 82. In raw units cond(CZ) = 856 and
    # the smallest right singular vector loads -0.9996 on `drive` -- so the
    # ridge clamps `drive` and leaves everything else alone. Standardise the
    # columns and cond falls to 58.9 and the near-null direction becomes
    # t_OUT +0.859, t_RAIL +0.412, money -0.254. The apparent
    # non-identification of `drive` was a units artefact; the direction the
    # data really fails to pin down is the one carrying almost all of the GS
    # sensitivity (S = -5.34e6, +3.30e6, -2.11e6 against drive's -7.1e4).
    #
    # scale = ||CZ_k|| gives the penalty  lam * sum_k ||CZ_k||^2 (th_k - a_k)^2,
    # identical to ridging the design with unit-norm columns:
    #     min ||(Cq - CZ a) - CZ' u||^2 + lam ||u||^2,  CZ' = CZ D^-1, u = D(th-a)
    # so the filter becomes s'^2/(s'^2 + lam) on the STANDARDISED spectrum.
    # Least squares is scale-equivariant, so lam = 0 is unaffected either way.
    if ridge_scale == "column":
        scale = np.linalg.norm(CZ, axis=0)
        scale = np.where(scale > 0, scale, 1.0)
    elif ridge_scale in ("none", None):
        scale = np.ones(nF)
    else:
        raise ValueError(f"unknown ridge_scale {ridge_scale!r}")
    scale2 = scale ** 2

    if fix_alpha_B:
        # alpha_B = 1: the fare term is known and moves to the right-hand side
        rhs = Cq + Cp

        def fun_grad(p: np.ndarray) -> Tuple[float, np.ndarray]:
            r = rhs - CZ @ p
            f = float(r @ r)
            g = -(CZ.T @ (2.0 * r))
            if lam > 0.0:
                diff = p - anchor
                d = scale * diff
                f += lam * float(d @ d)
                g = g + 2.0 * lam * scale2 * diff
            return f, g

        x0 = anchor.copy() if lam > 0.0 else np.zeros(nF)
        res = minimize(fun_grad, x0, jac=True, method="L-BFGS-B",
                       bounds=[(None, None)] * nF,
                       options={"maxiter": 20000, "ftol": 1e-18,
                                "gtol": 1e-14, "maxcor": 20})
        # Report the DATA-fit objective only, so obj_buyer stays comparable
        # across lambda; the penalty is a regulariser, not part of Z_B.
        rd = rhs - CZ @ res.x
        return (res.x, 1.0, float(rd @ rd), rank, n_params, identified, cond)

    def fun_grad(p: np.ndarray) -> Tuple[float, np.ndarray]:
        th, aB = p[:nF], p[nF]
        r = Cq - CZ @ th + Cp * aB
        g = 2.0 * r
        return float(r @ r), np.concatenate([-(CZ.T @ g), [float(g @ Cp)]])

    p0 = np.zeros(nF + 1)
    p0[nF] = 1.0
    res = minimize(fun_grad, p0, jac=True, method="L-BFGS-B",
                   bounds=[(None, None)] * nF + [(eps_scale, None)],
                   options={"maxiter": 20000, "ftol": 1e-18, "gtol": 1e-14,
                            "maxcor": 20})
    return (res.x[:nF], float(res.x[nF]), float(res.fun), rank, n_params,
            identified, cond)


# ----------------------------------------------------------------------------
# Cycle-projection estimator
# ----------------------------------------------------------------------------

def solve_io2(
    inst: P2Instance,
    data: IO2Data,
    rho_c: float = 1.0,
    rho_pi: float = 0.0,
    eps_scale: float = 1e-6,
    tol_sigma: float = 1e-6,
    tol_gap: Optional[float] = None,
    max_outer: int = 100,
    gamma_exponent: float = 0.6,
    inner_tol: float = 1e-7,
    clearing_max_iter: int = 20000,
    skip_clearing: bool = False,
    step6_max_iter: int = 40,
    fix_alpha_B: bool = False,
    fix_sigma: Optional[np.ndarray] = None,
    fix_seller: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
    ridge_lambda: float = 0.0,
    ridge_prior: str = "moving",
    ridge_anchor_vec: Optional[np.ndarray] = None,
    ridge_lambda_prox: float = 0.0,
    ridge_scale: str = "none",
    verbose: bool = False,
    trace_hook: Optional[Callable[[dict], None]] = None,
) -> IO2Solution:
    """Cycle-projection estimator of IO1 (not the paper's Algorithm 3).

    Parameters
    ----------
    rho_c, rho_pi : L1 weights. rho_c must be strictly positive; the seller
        residual bundles c_ma, pi_ma and alpha_S on each link, so the prior is
        what separates them (Remark 7).
    tol_sigma : outer tolerance on ||sigma^r - sigma^{r-1}||_inf / max tau0.
    fix_sigma : hold the rationing prices at this value and skip steps 6-8.
        Pass zeros when the observed fares are known to clear.
    trace_hook : called once per outer iteration r, after step 8 forms
        sigma^{r+1}, with a dict of that iteration's state. Every quantity the
        outer loop recomputes -- theta_hat, alpha_B, the rank test, the step 6
        solve and its flows -- is passed through, so a caller with the ground
        truth can score each iterate without this module knowing about it.

        Note ``sigma_change`` is the MSA-damped step, gamma * gap, and gamma =
        r ** -gamma_exponent decays with r. It is therefore NOT comparable
        across iterations: at r = 20 with the default exponent it is deflated
        ~7.6x relative to r = 1, so tol_sigma can be met by damping alone.
        ``sigma_gap_raw`` is the undamped ||sigma_target - sigma^r||_inf /
        max tau0 and is the one to read as a convergence diagnostic.
    """
    t_start = time.perf_counter()
    nO = inst.n_own

    # --- Step 1. Preprocess -------------------------------------------------
    sampled = np.isfinite(data.f_seg)
    A_pos = sampled & (data.f_seg > 0)
    A_zero = sampled & (data.f_seg == 0)
    if inst.avail is not None:
        A_pos &= inst.avail
        A_zero &= inst.avail
    sigma = np.zeros(nO) if fix_sigma is None else np.asarray(fix_sigma, float)
    history: Dict[str, List[float]] = {"sigma_change": [], "obj_buyer": []}

    # --- Step 2. Seller block, solved once ----------------------------------
    c_hat, c_idle_hat, pi_hat, aS, beta, obj_S = seller_block(
        inst, data, rho_c, rho_pi, eps_scale
    )
    if fix_seller is not None:
        # Diagnostic: replace the ESTIMATED seller primitives with known ones.
        # Step 6 consumes only c, c_idle and alpha_S from this block (pi_hat and
        # beta are report-only), so this isolates how much of the sigma error is
        # inherited from the seller block rather than produced by the outer loop.
        c_o, c_idle_o, aS_o = fix_seller
        c_hat = np.array(c_o, float)
        c_idle_hat = np.array(c_idle_o, float)
        aS = aS_o if np.ndim(aS_o) else float(aS_o)
        if verbose:
            print(f"  seller block overridden: alpha_S={aS}, "
                  f"c_idle={c_idle_hat}, mean c={c_hat.mean():.4f}")

    # --- Step 3. Cycle projection, independent of sigma ---------------------
    Cq, CZ, CP, cyc, _ = build_projection(inst, data, A_pos)

    theta = np.zeros(data.z.shape[2])
    aB, obj_B, rank, identified = 1.0, np.nan, 0, False
    u = np.zeros((inst.n_seg, inst.n_nodes))
    v = beta / aS - np.log(inst.F) / aS
    f_fit = np.where(sampled, np.nan_to_num(data.f_seg), 0.0)
    viol, max_psi = np.nan, np.nan
    converged, message, r = False, "outer iteration cap reached", 0


    # Ridge anchor. "moving": the previous outer iteration's thetat, so the
    # first iteration is unregularised and every later one is a proximal step.
    # The fixed point is unchanged, and so is the stability condition g < 1 --
    # which is why the moving prior damps the steps without making the loop
    # converge.
    #
    # A CONSTANT anchor is the variant that scales the sigma -> theta gain:
    # the condition becomes (1 - w) g < 1 with w = L/(s^2 + L) on the relevant
    # direction, so it can be satisfied by choice of lambda. That buys a fixed
    # point at the price of shrinkage toward the anchor, so WHICH constant is
    # a substantive modelling choice, not a detail:
    #
    #   "zero"  -- anchor at thetat = 0. Stabilises, but shrinks every
    #              coefficient toward "no preference at all"; on the worst
    #              identified direction (drive) the shrinkage is heaviest,
    #              which is precisely where the bias is then largest.
    #   "first" -- anchor at thetat from the FIRST outer iteration, i.e. the
    #              estimate under the maintained assumption sigma = 0 (posted
    #              fares clear). Truth-free, data-driven, and economically
    #              readable: shrink toward the no-rationing estimate, in
    #              proportion to how poorly identified each direction is.
    #   "prior" -- anchor at an externally supplied thetat, passed in as
    #              ridge_anchor_vec. This is the buyer-side counterpart of the
    #              Assumption 5 cost anchor: the analyst holds an independent
    #              estimate of the taste parameters (a published value-of-time
    #              study, say) and the ridge shrinks toward it, hardest on the
    #              directions the data identifies worst. Note this anchor is
    #              informative about thetat in a way "zero" and "first" are
    #              not, so a benchmark result under it is conditional on the
    #              anchor's accuracy and must be reported as such.
    #
    # MIXED: a constant anchor AND the moving proximal term together, via
    # ridge_lambda_prox = L_p alongside ridge_lambda = L_a. The penalty is
    #
    #     L_p ||thetat - thetat^(r-1)||^2  +  L_a ||thetat - a||^2
    #
    # which, completing the square, is one ridge at L_eff = L_p + L_a with a
    # blended anchor
    #
    #     (L_p thetat^(r-1) + L_a a) / L_eff .
    #
    # So nothing in buyer_block changes; only the anchor handed to it does.
    # At a fixed point thetat^(r-1) = thetat*, and the penalty gradient
    # collapses to 2 L_a (thetat* - a): the moving part drops out exactly, so
    # the fixed point is the constant-anchor one and the proximal term only
    # damps the path taken to reach it. L_a -> 0 recovers the pure moving
    # prior (fixed point unchanged, no stabilisation); L_p -> 0 recovers the
    # pure constant anchor.
    #
    # Caution for the write-up: the mixed recursion carries an explicit
    # thetat^(r-1) feedback of weight w_eff L_p / L_eff on top of the
    # (1 - w_eff) g gain, so its stability condition is NOT (1-w) g < 1 -- it
    # is a 2x2 system and needs deriving. What IS immediate is that
    # w_eff = L_eff/(s^2 + L_eff) exceeds the pure-constant w_a at the same
    # L_a, so the sigma -> theta gain is no larger.
    anchor_const: Optional[np.ndarray] = None
    if ridge_lambda > 0.0 and ridge_prior in ("zero", "fixed"):
        anchor_const = np.zeros(data.z.shape[2])
    elif ridge_lambda > 0.0 and ridge_prior == "prior":
        if ridge_anchor_vec is None:
            raise ValueError("ridge_prior='prior' requires ridge_anchor_vec")
        anchor_const = np.asarray(ridge_anchor_vec, float).ravel()[:data.z.shape[2]]
        if anchor_const.size != data.z.shape[2]:
            raise ValueError(
                f"ridge_anchor_vec has {anchor_const.size} entries, "
                f"expected {data.z.shape[2]}")
    lam_prox = float(max(ridge_lambda_prox, 0.0))
    if lam_prox > 0.0 and ridge_prior == "moving":
        raise ValueError(
            "ridge_prior='moving' IS the proximal term; combining it with "
            "ridge_lambda_prox would double it. Pair the proximal weight with "
            "a constant anchor: ridge_prior in {first, zero, prior}")
    theta_prev: Optional[np.ndarray] = None

    for r in range(1, max_outer + 1):
        # --- Steps 4-5. Identification test and buyer block -----------------
        # One ridge term, whose anchor is the L-weighted blend of the constant
        # anchor and the previous iterate (see the note above). Either weight
        # may be zero, giving the pure moving prior or the pure constant.
        if ridge_prior == "moving":
            anchor = theta_prev
            lam_r = (ridge_lambda if (ridge_lambda > 0.0 and anchor is not None)
                     else 0.0)
        else:
            l_p = lam_prox if theta_prev is not None else 0.0
            l_a = ridge_lambda if anchor_const is not None else 0.0
            lam_r = l_p + l_a
            if lam_r > 0.0:
                blend = np.zeros(data.z.shape[2])
                if l_p > 0.0:
                    blend = blend + l_p * theta_prev
                if l_a > 0.0:
                    blend = blend + l_a * anchor_const
                anchor = blend / lam_r
            else:
                anchor = None

        th_t, aB, obj_B, rank, n_params, identified, design_cond = buyer_block(
            Cq, CZ, CP, data.tau0 + sigma, eps_scale, fix_alpha_B,
            ridge_lambda=lam_r, ridge_anchor=anchor,
            ridge_scale=ridge_scale,
        )
        anchor_dist = (float(np.abs(th_t - anchor).max())
                       if anchor is not None else float("nan"))
        # Distance to the CONSTANT anchor is the one that reports shrinkage
        # bias; distance to the blend is contaminated by the proximal part.
        anchor_const_dist = (float(np.abs(th_t - anchor_const).max())
                             if anchor_const is not None else float("nan"))
        if ridge_lambda > 0.0 and ridge_prior == "first" and anchor_const is None:
            anchor_const = np.array(th_t, float)  # sigma = 0 estimate; frozen
        theta_prev = np.array(th_t, float)
        if not identified:
            message = (f"Theta not identified on O_F: rank {rank} of "
                       f"{n_params}")
            if verbose:
                print("  " + message)
            if trace_hook is not None:
                trace_hook(dict(
                    r=r, identified=False, design_rank=int(rank),
                    n_params=int(n_params), design_cond=float(design_cond),
                    obj_buyer=float(obj_B), alpha_B=float(aB),
                    theta=th_t / aB, sigma=sigma, message=message,
                ))
            break
        theta = th_t / aB
        phi = data.z @ theta
        if inst.avail is not None:
            phi = np.where(inst.avail, phi, 0.0)

        # --- Step 6. Fixed-price allocation via Algorithm 2' ----------------
        est = P2Instance(
            tail=inst.tail, head=inst.head, link_owner=inst.link_owner,
            n_nodes=inst.n_nodes, D=inst.D, origin=inst.origin, sink=inst.sink,
            F=inst.F, Cmax=inst.Cmax, c=c_hat, ell=inst.ell, phi=phi,
            alpha_B=aB, alpha_S=aS, avail=inst.avail, c_idle=c_idle_hat,
            link_names=inst.link_names, segment_names=inst.segment_names,
            operator_names=inst.operator_names, node_names=inst.node_names,
            outside_link=inst.outside_link, b_supply=inst.b_supply,
        )
        # Step 6's inner MSA loop dominates the cycle-projection estimator's cost: at the
        # estimated phi it does not converge as readily as at the true
        # parameters, so it runs to its cap every outer iteration.  The outer
        # sigma loop averages across iterations anyway, so an inexact inner
        # solve is tolerable -- MSA is built for that.  Capping it takes a
        # case from ~24 min to ~6 min on the LIRR instance.
        fp = solve_p2_fixed_price(est, data.tau0, tol=inner_tol,
                                  max_iter=step6_max_iter)
        u, f_fit = fp.u, fp.f_seg

        # --- Step 7. Support check on the zero cells ------------------------
        price_all = np.zeros(inst.n_links)
        price_all[inst.owned_idx] = data.tau0 + sigma
        psi = phi - price_all[None, :] - (u[:, inst.tail] - u[:, inst.head])
        if A_zero.any():
            viol = float(np.mean(psi[A_zero] > 0))
            max_psi = float(np.max(psi[A_zero]))
        else:
            viol, max_psi = 0.0, -np.inf

        # --- Step 8. Average and test ---------------------------------------
        if fix_sigma is not None:
            converged, message = True, "sigma held fixed"
            break
        gamma = float(r) ** (-gamma_exponent)
        sigma_new = (1.0 - gamma) * sigma + gamma * fp.sigma
        denom = max(float(np.max(data.tau0)), 1e-12)
        change = float(np.abs(sigma_new - sigma).max() / denom)
        # Undamped distance to step 6's target, i.e. change without the MSA
        # gamma. Unlike `change` this is comparable across iterations.
        gap_raw = float(np.abs(fp.sigma - sigma).max() / denom)
        sigma_prev = sigma
        sigma = sigma_new
        history["sigma_change"].append(change)
        history["obj_buyer"].append(obj_B)
        history.setdefault("sigma_gap_raw", []).append(gap_raw)
        history.setdefault("theta", []).append(np.array(theta, float))
        if trace_hook is not None:
            trace_hook(dict(
                r=r, identified=True, gamma=gamma,
                sigma_change=change, sigma_gap_raw=gap_raw,
                design_rank=int(rank), n_params=int(n_params),
                design_cond=float(design_cond), obj_buyer=float(obj_B),
                obj_seller=float(obj_S), alpha_B=float(aB), alpha_S=float(np.atleast_1d(aS)[0]),
                theta=np.array(theta, float),
                sigma=sigma_new, sigma_target=fp.sigma, sigma_prev=sigma_prev,
                support_violations=viol, support_max_psi=max_psi,
                inner_n_iter=int(fp.n_iter), inner_converged=bool(fp.converged),
                inner_residual=float(fp.residual),
                inner_residual_natural=float(fp.residual_natural),
                inner_history=fp.history,
                ridge_lambda=lam_r, ridge_prior=ridge_prior,
                ridge_lambda_prox=lam_prox, ridge_scale=ridge_scale,
                ridge_anchor_dist=anchor_dist,
                ridge_anchor_const_dist=anchor_const_dist,
                f_seg=f_fit, phi=phi, c=c_hat, pi=pi_hat, u=u,
            ))
        if verbose:
            print(f"  r={r:3d}  ||dsigma||/max tau0 = {change:.3e}  "
                  f"ZB = {obj_B:.6e}  rank {rank}")
        # Stop only when BOTH the damped step and the UNDAMPED residual are
        # small. Testing `change` alone is unsound: change = gamma_r * gap_raw
        # and gamma_r -> 0 by construction, so the criterion is eventually met
        # by damping even with the residual frozen -- at gamma exponent a and a
        # stalled gap g, `change` crosses tol at r ~ (g/tol)^(1/a) regardless of
        # whether sigma has gone anywhere. gap_raw is the quantity that is
        # actually comparable across iterations.
        tg = tol_sigma if tol_gap is None else float(tol_gap)
        if change <= tol_sigma and gap_raw <= tg:
            converged, message = True, "rationing prices converged"
            break
        if change <= tol_sigma and gap_raw > tg:
            message = (f"damped step below tol_sigma at r={r} but the undamped "
                       f"residual is {gap_raw:.3e} > {tg:.3e} -- gamma is "
                       f"shrinking the step, sigma has not converged")
            if verbose:
                print("  " + message)

    # --- Step 9. Clearing prices and wedge ----------------------------------
    phi = data.z @ theta
    if inst.avail is not None:
        phi = np.where(inst.avail, phi, 0.0)
    est = P2Instance(
        tail=inst.tail, head=inst.head, link_owner=inst.link_owner,
        n_nodes=inst.n_nodes, D=inst.D, origin=inst.origin, sink=inst.sink,
        F=inst.F, Cmax=inst.Cmax, c=c_hat, ell=inst.ell, phi=phi,
        alpha_B=aB, alpha_S=aS, avail=inst.avail, c_idle=c_idle_hat,
        link_names=inst.link_names, segment_names=inst.segment_names,
        operator_names=inst.operator_names, node_names=inst.node_names,
        outside_link=inst.outside_link, b_supply=inst.b_supply,
    )
    # Step 9 supplies tau* and the wedge only; the flow-fit and GS-at-tau^0
    # metrics do not depend on it, and the clearing solve is the single most
    # expensive call in the cycle-projection estimator.  skip_clearing lets a sweep defer it and
    # compute tau* once per sampling rate instead of once per (seed, rate).
    if skip_clearing:
        tau_star = np.full_like(data.tau0, np.nan)
    else:
        clearing = solve_p2(est, tol=inner_tol, max_iter=clearing_max_iter)
        tau_star = clearing.tau
        # A capped clearing solve returns a tau* that has not cleared, and the
        # wedge tau^0 - tau* is then not the wedge. Record the stopping state
        # so a truncated step 9 cannot be mistaken for a converged one.
        history["clearing_n_iter"] = [int(clearing.n_iter)]
        history["clearing_converged"] = [bool(clearing.converged)]
        history["clearing_residual"] = [float(clearing.residual)]

    return IO2Solution(
        theta=theta, c=c_hat, c_idle=c_idle_hat, pi=pi_hat,
        alpha_B=aB, alpha_S=aS, u=u, v=beta / aS - np.log(inst.F) / aS,
        f_seg=f_fit, sigma=sigma, phi=phi,
        tau_star=tau_star, wedge=data.tau0 - tau_star,
        cycle_rank=cyc, design_rank=rank, design_cond=design_cond,
        identified=identified,
        support_violations=viol, support_max_psi=max_psi,
        obj_buyer=obj_B, obj_seller=obj_S, n_outer=r,
        wall_time=time.perf_counter() - t_start,
        converged=converged, message=message, history=history,
    )


# ----------------------------------------------------------------------------
# Toy network of Fig. 1
# ----------------------------------------------------------------------------

def toy_link_features(inst: P2Instance) -> np.ndarray:
    """z_ia over (Q, A, 3); zero on links in A_0 so that phi_{i,a_i0} = 0."""
    z_own = np.array([
        [8.0, 20.0, 5.0],
        [10.0, 25.0, 5.0],
        [8.0, 12.0, 2.0],
        [18.0, 30.0, 2.0],
    ])
    z = np.zeros((inst.n_seg, inst.n_links, 3))
    for j, a in enumerate(inst.owned_idx):
        z[:, a, :] = z_own[j]
    return z


def toy_data(inst: P2Instance, tau0: np.ndarray, seed: int = 20260826) -> IO2Data:
    """Ground truth from Algorithm 2' at the administered fares tau0."""
    rng = np.random.default_rng(seed)
    truth = solve_p2_fixed_price(inst, tau0, tol=1e-9)
    return IO2Data(
        f_seg=truth.f_seg, y=truth.y, f_link=truth.f, tau0=tau0,
        z=toy_link_features(inst),
        c_prior=inst.c * (1.0 + rng.uniform(-0.15, 0.15, inst.n_own)),
        c_idle_prior=np.zeros(inst.n_op),
    )


if __name__ == "__main__":
    from algorithm2_p2 import toy_instance_p2
    from algorithm2prime_p2 import toy_tau0 as toy_tau0_p2

    inst = toy_instance_p2()
    tau0 = toy_tau0_p2() if callable(toy_tau0_p2) else np.array([2.75, 2.75, 3.50, 6.00])
    data = toy_data(inst, tau0)

    truth = solve_p2_fixed_price(inst, tau0, tol=1e-9)
    sol = solve_io2(inst, data, rho_c=1.0, rho_pi=0.0,
                    fix_sigma=truth.sigma, verbose=True)
    print()
    print(sol.summary(inst, data))

    theta_true = np.array([1.20, -0.25, -0.50])
    err = np.linalg.norm(sol.theta - theta_true) / np.linalg.norm(theta_true)
    print()
    print(f"  ||theta_hat - theta|| / ||theta||  = {err:.3e}")
    print(f"  alpha_B  {sol.alpha_B:.6f}  (true {inst.alpha_B})")
    print(f"  alpha_S  {sol.alpha_S:.6f}  (true {inst.alpha_S})")
    print(f"  ||c_hat - c||_1                    = "
          f"{np.abs(sol.c - inst.c).sum():.4f}")

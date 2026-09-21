"""
Algorithm 3 - L-BFGS-B dual solution of the PURC-based link formulation of the
aggregate network assignment game P2.

Solves the dual of Eq. (59) over free node potentials u and the box tau >= 0,
pi >= 0:

    min_{u, tau >= 0, pi >= 0}  D'(u, tau, pi)

      = sum_i sum_a ( aB D_i / (2 l_a) ) [ psi_ia ]_+^2
      + sum_i D_i u_{i,o_i}
      + (1/aS) sum_m F_m [ ln( exp(-aS c_idle_m) + sum_{a in A_m}
                                 exp( aS ( tau_ma - c_ma - pi_ma ) ) ) - ln F_m + 1 ]
      + sum_m sum_{a in A_m} pi_ma Cmax_ma

with the reduced link margin

    psi_ia = phi_ia - tau_ma - ( u_{i,t(a)} - u_{i,h(a)} )

and gradients

    dD'/du_in    = -eta_in ,
    dD'/dtau_ma  = y_ma - f_a ,
    dD'/dpi_ma   = Cmax_ma - y_ma ,

where eta_in = sum_{a:t(a)=n} f_ia - sum_{a:h(a)=n} f_ia - D_i b_in is the node
flow residual. Link flows come from Eq. (57), capacities from Eq. (33), v* from
Eq. (32). Convergence is monitored by Eq. (38a).

ON THE SIGN OF psi. The grouping in psi_ia is as the draft has it throughout:
relaxing the conservation constraint of Eq. (50) contributes -u_{i,t(a)} +
u_{i,h(a)} to dL'/df_ia, matching the notation block, Eq. (56), Eq. (57),
Eq. (58) and Eq. (59). Note that -(u_{i,t(a)} - u_{i,h(a)}) is authored as an
OMML delimiter object, so plain-text extraction of the .docx drops the
parentheses and makes it read as "- u_{i,t(a)} - u_{i,h(a)}". That is an
artifact of extraction, not of the draft.

SEGMENT LINK AVAILABILITY. P2 is link-based, but each segment still has its own
subnetwork: the outside link a_i0 belongs to segment i alone, and the toy
network of Fig. 1 does not let segment 3 substitute l5 or l6 for l1/l2. The
`avail` mask carries the P1 path-set structure over to the link formulation.
Leaving it at the default (every link available to every segment) changes the
solution.

Requires: numpy, scipy, and `algorithm1_p1.py` in the same directory.

Reference: Chow (2026), Section 3.3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from algorithm1_p1 import _segment_logsumexp

_INF = np.inf


# ----------------------------------------------------------------------------
# Problem instance
# ----------------------------------------------------------------------------

@dataclass
class P2Instance:
    """Inputs to P2, Eqs. (49)-(55).

    Links are indexed over the whole of A. Owned links carry an operator index
    in `link_owner`; outside links in A_0 carry -1 and have no cost transfer, no
    capacity constraint and no market clearing. The dual vectors tau, pi live
    only on owned links, in the order they appear in `link_owner`.

    Parameters
    ----------
    tail, head : (A,) int
        Node-link maps t(a), h(a).
    link_owner : (A,) int
        Operator m(a) for owned links, -1 for links in A_0.
    n_nodes : int
        Size of V.
    D : (Q,) float
        Segment demands D_i > 0.
    origin, sink : (Q,) int
        o_i and s_i. b_in is derived from these.
    F : (M,) float
        Operator endowments F_m > 0.
    Cmax, c : (A_own,) float
        Caps C^max_ma and provision costs c_ma, over owned links only.
    c_idle : (M,) float
        Reserve cost c_idle,m. Defaults to zeros (the notation block's c_m0 = 0).
    ell : (A,) float
        Length weights l_a > 0 of the quadratic perturbation H_a(z) = l_a z^2/2.
    phi : (Q, A) float
        Segment-link surplus rates phi_ia, with phi_{i,a_i0} = 0 on the outside
        link. Entries where `avail` is False are ignored.
    avail : (Q, A) bool, optional
        Whether segment i may use link a. Defaults to all True.
    alpha_B, alpha_S : float
        Scale parameters.
    link_names, segment_names, operator_names, node_names : optional labels.
    """

    tail: np.ndarray
    head: np.ndarray
    link_owner: np.ndarray
    n_nodes: int
    D: np.ndarray
    origin: np.ndarray
    sink: np.ndarray
    F: np.ndarray
    Cmax: np.ndarray
    c: np.ndarray
    ell: np.ndarray
    phi: np.ndarray
    alpha_B: float
    alpha_S: float
    avail: Optional[np.ndarray] = None
    c_idle: Optional[np.ndarray] = None
    link_names: Optional[List[str]] = None
    segment_names: Optional[List[str]] = None
    operator_names: Optional[List[str]] = None
    node_names: Optional[List[str]] = None
    outside_link: Optional[np.ndarray] = None
    b_supply: Optional[np.ndarray] = None

    # derived
    n_links: int = field(init=False)
    n_own: int = field(init=False)
    n_seg: int = field(init=False)
    n_op: int = field(init=False)
    owned_idx: np.ndarray = field(init=False)   # (A_own,) index into A
    dual_of_link: np.ndarray = field(init=False)  # (A,) index into tau, -1 if A_0
    owner_of_dual: np.ndarray = field(init=False)  # (A_own,) operator index
    b: np.ndarray = field(init=False)           # (Q, V) net supply indicator
    p_seg: np.ndarray = field(init=False)       # (P,) segment of each (i,a) pair
    p_link: np.ndarray = field(init=False)      # (P,) link of each pair
    p_phi: np.ndarray = field(init=False)       # (P,) phi_ia
    p_coef: np.ndarray = field(init=False)      # (P,) aB D_i / l_a
    u_free: np.ndarray = field(init=False)      # (Q, V) bool, free potentials

    def __post_init__(self) -> None:
        self.tail = np.asarray(self.tail, dtype=int)
        self.head = np.asarray(self.head, dtype=int)
        self.link_owner = np.asarray(self.link_owner, dtype=int)
        self.D = np.asarray(self.D, dtype=float)
        self.origin = np.asarray(self.origin, dtype=int)
        self.sink = np.asarray(self.sink, dtype=int)
        self.F = np.asarray(self.F, dtype=float)
        self.Cmax = np.asarray(self.Cmax, dtype=float)
        self.c = np.asarray(self.c, dtype=float)
        self.ell = np.asarray(self.ell, dtype=float)
        self.phi = np.asarray(self.phi, dtype=float)

        self.n_links = self.tail.size
        self.n_seg = self.D.size
        self.n_op = self.F.size

        if self.avail is None:
            self.avail = np.ones((self.n_seg, self.n_links), dtype=bool)
        self.avail = np.asarray(self.avail, dtype=bool)
        if self.c_idle is None:
            self.c_idle = np.zeros(self.n_op)
        self.c_idle = np.asarray(self.c_idle, dtype=float)

        self.owned_idx = np.flatnonzero(self.link_owner >= 0)
        self.n_own = self.owned_idx.size
        self.owner_of_dual = self.link_owner[self.owned_idx]
        self.dual_of_link = np.full(self.n_links, -1, dtype=int)
        self.dual_of_link[self.owned_idx] = np.arange(self.n_own)

        # Net supply indicator b_in of Eq. (50).
        self.b = np.zeros((self.n_seg, self.n_nodes))
        if self.b_supply is None:
            self.b[np.arange(self.n_seg), self.origin] += 1.0
            self.b[np.arange(self.n_seg), self.sink] -= 1.0
        else:
            # Multi-origin segments: Eq. (50) is stated in terms of b_in, so a
            # segment may draw from many origins (e.g. destination-based
            # segmentation).  Rows must sum to zero and are net of the sink.
            self.b_supply = np.asarray(self.b_supply, dtype=float)
            if self.b_supply.shape != (self.n_seg, self.n_nodes):
                raise ValueError("b_supply must be shaped (Q, V).")
            rs = self.b_supply.sum(axis=1)
            scale = np.maximum(np.abs(self.b_supply).sum(axis=1), 1e-30)
            if np.max(np.abs(rs) / scale) > 1e-8:
                raise ValueError("each row of b_supply must sum to zero.")
            self.b = self.b_supply.copy()
            # absorb round-off at the sink so conservation is exact
            self.b[np.arange(self.n_seg), self.sink] -= rs

        # Flat (i, a) pair list over available cells.
        self.p_seg, self.p_link = np.nonzero(self.avail)
        self.p_phi = self.phi[self.p_seg, self.p_link]
        self.p_coef = self.alpha_B * self.D[self.p_seg] / self.ell[self.p_link]

        # Free node potentials: nodes touched by an available link, plus the OD
        # pair, minus the sink (normalized to u_{i,s_i} = 0 by elimination).
        touched = np.zeros((self.n_seg, self.n_nodes), dtype=bool)
        touched[self.p_seg, self.tail[self.p_link]] = True
        touched[self.p_seg, self.head[self.p_link]] = True
        if self.b_supply is None:
            touched[np.arange(self.n_seg), self.origin] = True
        else:
            touched |= self.b != 0.0        # every node with net supply
        touched[np.arange(self.n_seg), self.sink] = True
        touched[np.arange(self.n_seg), self.sink] = False
        self.u_free = touched

        self._validate()

    def _validate(self) -> None:
        if np.any(self.D <= 0):
            raise ValueError("Assumption 1: every D_i must be strictly positive.")
        if np.any(self.F <= 0):
            raise ValueError("Assumption 1: every F_m must be strictly positive.")
        if np.any(self.ell <= 0):
            raise ValueError("Length weights l_a must be strictly positive.")
        if self.alpha_B <= 0 or self.alpha_S <= 0:
            raise ValueError("alpha_B and alpha_S must be strictly positive.")
        if self.Cmax.size != self.n_own or self.c.size != self.n_own:
            raise ValueError("Cmax and c must have one entry per owned link.")
        if self.phi.shape != (self.n_seg, self.n_links):
            raise ValueError("phi must be shaped (Q, A).")
        if self.b_supply is None and np.any(self.origin == self.sink):
            raise ValueError("Each segment needs distinct origin and sink.")

        # Outside links carry phi_{i,a_i0} = 0.  A_0 is the outside option of
        # each segment, not every unowned link: an instance may contain
        # unpriced links (drive access, walk egress) that are ordinary links
        # carrying surplus.  If `outside_link` is supplied, only those cells
        # are checked; otherwise every unowned link is, as before.
        if getattr(self, "outside_link", None) is not None:
            ol = np.asarray(self.outside_link, dtype=int)
            bad = np.abs(self.phi[np.arange(self.n_seg), ol]) > 1e-9
            if np.any(bad):
                raise ValueError(
                    "Outside links must carry phi_{i,a_i0} = 0 "
                    "(the normalization of Assumption 3)."
                )
        else:
            out = np.flatnonzero(self.link_owner < 0)
            if out.size and np.any(np.abs(self.phi[:, out][self.avail[:, out]]) > 0):
                raise ValueError(
                    "Links in A_0 must carry phi_ia = 0 (the outside-link "
                    "normalization of Assumption 3)."
                )

        cap_by_op = np.zeros(self.n_op)
        np.add.at(cap_by_op, self.owner_of_dual, self.Cmax)
        bad = np.flatnonzero(self.F >= cap_by_op)
        if bad.size:
            raise ValueError(
                "Assumption 1 violated: need F_m < sum_a Cmax_ma strictly for "
                f"operator(s) {bad.tolist()}."
            )

        # Assumption 3: no directed cycle with positive surplus, checked per
        # segment by Bellman-Ford on -phi. Also the condition that makes the
        # shortest-path initialization of step 1 well posed.
        for i in range(self.n_seg):
            if _has_negative_cycle(self, i, np.zeros(self.n_own)):
                raise ValueError(
                    f"Assumption 3 violated: segment {i} has a directed cycle "
                    "with strictly positive surplus."
                )


# ----------------------------------------------------------------------------
# Step 1: shortest-path initialization of the node potentials
# ----------------------------------------------------------------------------

def _arc_costs(inst: P2Instance, tau: np.ndarray) -> np.ndarray:
    """Per-pair arc cost -phi_ia + tau_ma used by the shortest-path pass."""
    w = -inst.p_phi.copy()
    d = inst.dual_of_link[inst.p_link]
    owned = d >= 0
    w[owned] += tau[d[owned]]
    return w


def _bellman_ford(
    inst: P2Instance, i: int, tau: np.ndarray
) -> Tuple[np.ndarray, bool]:
    """Min-cost distance from every node to s_i on arc costs -phi_ia + tau.

    Relaxation runs against the arc direction (a node's distance is improved
    through its outgoing arcs), so this is a single-sink shortest path.
    Returns (dist, negative_cycle_found).
    """
    sel = inst.p_seg == i
    links = inst.p_link[sel]
    w = _arc_costs(inst, tau)[sel]
    t, h = inst.tail[links], inst.head[links]

    dist = np.full(inst.n_nodes, _INF)
    dist[inst.sink[i]] = 0.0
    for _ in range(inst.n_nodes):
        cand = dist[h] + w
        new = np.full(inst.n_nodes, _INF)
        np.minimum.at(new, t, cand)
        upd = new < dist - 1e-12
        if not upd.any():
            return dist, False
        dist = np.where(upd, new, dist)
    # One more pass that still improves implies a negative cycle.
    cand = dist[h] + w
    new = np.full(inst.n_nodes, _INF)
    np.minimum.at(new, t, cand)
    return dist, bool(np.any(new < dist - 1e-9))


def _has_negative_cycle(inst: P2Instance, i: int, tau: np.ndarray) -> bool:
    return _bellman_ford(inst, i, tau)[1]


def initial_potentials(inst: P2Instance, tau: np.ndarray) -> np.ndarray:
    """Step 1. u_in from a shortest-path pass on -phi_ia, with u_{i,s_i} = 0.

    u_in is minus the min-cost-to-sink, i.e. the best attainable surplus from n
    to s_i. Nodes that cannot reach s_i are set to 0; no available link carries
    flow through them at the optimum.
    """
    u = np.zeros((inst.n_seg, inst.n_nodes))
    for i in range(inst.n_seg):
        dist, _ = _bellman_ford(inst, i, tau)
        u[i] = np.where(np.isfinite(dist), -dist, 0.0)
    u[np.arange(inst.n_seg), inst.sink] = 0.0
    return u


# ----------------------------------------------------------------------------
# Steps 2(a) and 2(b): buyer and seller loading
# ----------------------------------------------------------------------------

def buyer_loading(
    inst: P2Instance, u: np.ndarray, tau: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Step 2(a). Eq. (57).

    Returns (f_pair, f_link, eta, psi_pair, quad) where f_pair is f_ia over the
    available cells, f_link the owned-link loads f_a, eta the node residuals,
    and quad the buyer term sum_i sum_a (aB D_i / 2 l_a) [psi_ia]_+^2.
    """
    d = inst.dual_of_link[inst.p_link]
    t, h = inst.tail[inst.p_link], inst.head[inst.p_link]

    psi = inst.p_phi - u[inst.p_seg, t] + u[inst.p_seg, h]
    owned = d >= 0
    psi[owned] -= tau[d[owned]]

    psi_plus = np.maximum(psi, 0.0)
    f_pair = inst.p_coef * psi_plus                # Eq. (57)
    quad = 0.5 * float(f_pair @ psi_plus)

    f_link = np.zeros(inst.n_own)
    np.add.at(f_link, d[owned], f_pair[owned])

    eta = np.zeros((inst.n_seg, inst.n_nodes))
    np.add.at(eta, (inst.p_seg, t), f_pair)
    np.subtract.at(eta, (inst.p_seg, h), f_pair)
    eta -= inst.D[:, None] * inst.b

    return f_pair, f_link, eta, psi, quad


def seller_loading(
    inst: P2Instance, tau: np.ndarray, pi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 2(b). Eq. (33), unchanged from P1.

    Returns (y, y_idle, S) with S_m the operator inclusive value.
    """
    omega = tau - inst.c - pi
    S, w, w_null = _segment_logsumexp(
        inst.alpha_S * omega,
        inst.owner_of_dual,
        inst.n_op,
        -inst.alpha_S * inst.c_idle,
    )
    y = inst.F[inst.owner_of_dual] * w
    y_idle = inst.F * w_null
    return y, y_idle, S


def natural_residual(
    inst: P2Instance,
    tau: np.ndarray,
    pi: np.ndarray,
    f_link: np.ndarray,
    y: np.ndarray,
    eta: np.ndarray,
    clearing: bool = True,
) -> float:
    """Eq. (38a). With clearing=False it collapses to max|eta| / max_a f_a."""
    scale = f_link.max() if f_link.size else 0.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    parts = [np.abs(eta[inst.u_free]).max() if inst.u_free.any() else 0.0]
    if clearing:
        parts.append(np.abs(np.minimum(tau, y - f_link)).max())
        parts.append(np.abs(np.minimum(pi, inst.Cmax - y)).max())
    return float(max(parts) / scale)


# ----------------------------------------------------------------------------
# Solution container
# ----------------------------------------------------------------------------

@dataclass
class P2Solution:
    u: np.ndarray              # (Q, V) node potentials u*_in, u*_{i,s_i} = 0
    tau: np.ndarray            # (A_own,) cost transfers
    pi: np.ndarray             # (A_own,) capacity rents
    v: np.ndarray              # (M,) v*_m, Eq. (32)
    v_tilde: np.ndarray        # (M,) recentered v_m = v*_m + ln(F_m)/aS
    f_seg: np.ndarray          # (Q, A) segment-link flows f*_ia
    f: np.ndarray              # (A_own,) owned-link loads f_a
    f_all: np.ndarray          # (A,) loads on every link, incl. A_0
    y: np.ndarray              # (A_own,) allocated capacity
    y_idle: np.ndarray         # (M,) undeployed capacity
    u_origin: np.ndarray       # (Q,) supply-weighted mean potential (accessibility)
    dual_obj: float            # D'* of Eq. (59)
    gross_surplus: float       # sum phi f - sum c y - sum c_idle y_m0
    social_surplus: float      # Z~' of Eq. (49a)
    residual: float            # varrho of Eq. (38a)
    n_iter: int
    wall_time: float
    converged: bool
    message: str
    history: Dict[str, List[float]]

    def summary(self, inst: P2Instance) -> str:
        lnames = inst.link_names or [f"a{a}" for a in range(inst.n_links)]
        out = [
            f"Algorithm 3  ({'converged' if self.converged else 'STOPPED'}: {self.message})",
            f"  iterations      : {self.n_iter}",
            f"  wall time       : {1e3 * self.wall_time:.1f} ms",
            f"  dual objective  : {self.dual_obj:.3f}",
            f"  natural residual: {self.residual:.3e}",
            f"  gross surplus   : {self.gross_surplus:.2f}",
            f"  social surplus  : {self.social_surplus:.2f}",
            "",
            "  link      owner     f_a       y_ma      tau*      pi*     Cmax",
        ]
        onames = inst.operator_names or [f"m{m}" for m in range(inst.n_op)]
        for a in range(inst.n_links):
            d = inst.dual_of_link[a]
            if d >= 0:
                out.append(
                    f"  {lnames[a]:<8}{onames[inst.link_owner[a]]:<8}"
                    f"{self.f[d]:9.3f} {self.y[d]:9.3f} {self.tau[d]:9.4f} "
                    f"{self.pi[d]:8.4f} {inst.Cmax[d]:8.1f}"
                )
            else:
                out.append(
                    f"  {lnames[a]:<8}{'A_0':<8}{self.f_all[a]:9.3f}"
                    f"{'':>9} {'--':>9} {'--':>8} {'--':>8}"
                )
        nnames = inst.node_names or [f"n{n}" for n in range(inst.n_nodes)]
        out += ["", "  node potentials u*_in  (-- = outside the subnetwork)",
                "  segment    " + "".join(f"{nm:>10}" for nm in nnames)]
        snames = inst.segment_names or [f"i{i}" for i in range(inst.n_seg)]
        for i in range(inst.n_seg):
            row = ""
            for n in range(inst.n_nodes):
                if n == inst.sink[i]:
                    row += f"{0.0:>10.2f}"
                elif inst.u_free[i, n]:
                    row += f"{self.u[i, n]:>10.2f}"
                else:
                    row += f"{'--':>10}"
            out.append(f"  {snames[i]:<11}{row}")
        out += ["", "  segment    u*_{i,o_i} (accessibility)"]
        for i in range(inst.n_seg):
            out.append(f"  {snames[i]:<11}{self.u_origin[i]:12.4f}")
        out += ["", "  operator     v*        v~        y_m0    deployed"]
        for m in range(inst.n_op):
            out.append(
                f"  {onames[m]:<10}{self.v[m]:9.4f} {self.v_tilde[m]:9.4f} "
                f"{self.y_idle[m]:9.3f} {100 * (1 - self.y_idle[m] / inst.F[m]):8.2f}%"
            )
        return "\n".join(out)


# ----------------------------------------------------------------------------
# Algorithm 3
# ----------------------------------------------------------------------------

def solve_p2(
    inst: P2Instance,
    tol: float = 1e-7,
    max_iter: int = 2000,
    mode: str = "clearing",
    tau_fixed: Optional[np.ndarray] = None,
    u0: Optional[np.ndarray] = None,
    tau0: Optional[np.ndarray] = None,
    pi0: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> P2Solution:
    """Algorithm 3. L-BFGS-B on the dual of Eq. (59).

    Parameters
    ----------
    tol : tolerance epsilon on Eq. (38a).
    mode : "clearing" minimizes over (u, tau, pi) jointly. "buyer" holds the
        fares at `tau_fixed`, minimizes over u alone, omits step 2(b), and
        tests the collapsed residual max|eta| / max_a f_a. Market clearing is
        not imposed in buyer mode, so f_a != y_ma in general.
    tau_fixed : (A_own,) administered fares, required when mode == "buyer".
    u0, tau0, pi0 : optional warm starts. Step 1 defaults are tau = pi = 0 and
        u from `initial_potentials`.
    """
    if mode not in ("clearing", "buyer"):
        raise ValueError("mode must be 'clearing' or 'buyer'.")
    if mode == "buyer" and tau_fixed is None:
        raise ValueError("buyer mode needs tau_fixed.")

    n_own = inst.n_own
    t_start = time.perf_counter()

    # --- Step 1. Initialize -------------------------------------------------
    tau_start = (
        np.asarray(tau_fixed, float) if mode == "buyer"
        else (np.zeros(n_own) if tau0 is None else np.asarray(tau0, float))
    )
    pi_start = np.zeros(n_own) if pi0 is None else np.asarray(pi0, float)
    u_start = initial_potentials(inst, tau_start) if u0 is None else np.array(u0, float)

    mask = inst.u_free
    n_u = int(mask.sum())

    def pack(u, tau, pi):
        if mode == "buyer":
            return u[mask].copy()
        return np.concatenate([u[mask], tau, pi])

    def unpack(z):
        u = np.zeros((inst.n_seg, inst.n_nodes))
        u[mask] = z[:n_u]
        if mode == "buyer":
            return u, tau_start, np.zeros(n_own)
        return u, z[n_u:n_u + n_own], z[n_u + n_own:]

    z0 = pack(u_start, tau_start, pi_start)
    history: Dict[str, List[float]] = {"dual_obj": [], "residual": []}
    cache: Dict[str, object] = {}

    # --- Step 2(a)-(c). Objective and gradient ------------------------------
    def fun_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        u, tau, pi = unpack(z)
        f_pair, f_link, eta, psi, quad = buyer_loading(inst, u, tau)

        # see note in algorithm2prime_p2.buyer_step: use b, not origin
        obj = quad + float((inst.D[:, None] * inst.b * u).sum())
        g_u = -eta[mask]                                    # dD'/du = -eta

        if mode == "buyer":
            cache.update(f_pair=f_pair, f_link=f_link, eta=eta, obj=obj)
            return float(obj), g_u

        y, y_idle, S = seller_loading(inst, tau, pi)
        obj += (
            float(np.sum(inst.F * (S - np.log(inst.F) + 1.0))) / inst.alpha_S
            + float(pi @ inst.Cmax)
        )
        grad = np.concatenate([g_u, y - f_link, inst.Cmax - y])  # Eq. (37)
        cache.update(f_pair=f_pair, f_link=f_link, eta=eta, y=y,
                     y_idle=y_idle, S=S, obj=obj)
        return float(obj), grad

    # --- Step 2(d). Stop on Eq. (38a) ---------------------------------------
    state = {"n_iter": 0, "stopped": False}

    def callback(z_or_res):
        z = np.asarray(getattr(z_or_res, "x", z_or_res), dtype=float)
        u, tau, pi = unpack(z)
        _, f_link, eta, _, _ = buyer_loading(inst, u, tau)
        if mode == "clearing":
            y, _, _ = seller_loading(inst, tau, pi)
        else:
            y = f_link
        rho = natural_residual(inst, tau, pi, f_link, y, eta,
                               clearing=(mode == "clearing"))
        state["n_iter"] += 1
        history["residual"].append(rho)
        history["dual_obj"].append(float(cache.get("obj", np.nan)))
        if verbose:
            print(f"  t={state['n_iter']:5d}  D'={cache.get('obj', float('nan')):15.6f}"
                  f"  rho={rho:.3e}")
        if rho <= tol:
            state["stopped"] = True
            raise StopIteration

    if mode == "buyer":
        bounds = [(None, None)] * n_u
    else:
        bounds = [(None, None)] * n_u + [(0.0, None)] * (2 * n_own)

    try:
        res = minimize(
            fun_grad, z0, jac=True, method="L-BFGS-B", bounds=bounds,
            callback=callback,
            options={"maxiter": max_iter, "maxfun": 20 * max_iter,
                     "ftol": 1e-16, "gtol": 1e-12, "maxcor": 30},
        )
        z_star, message = res.x, str(res.message)
    except StopIteration:
        z_star = minimize(
            fun_grad, z0, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": max(state["n_iter"], 1), "ftol": 1e-16,
                     "gtol": 1e-12, "maxcor": 30},
        ).x
        message = "natural residual below tolerance"
    if state["stopped"]:
        message = "natural residual below tolerance"

    # --- Step 3. Recover primal and duals -----------------------------------
    u, tau, pi = unpack(z_star)
    tau = np.maximum(tau, 0.0)
    pi = np.maximum(pi, 0.0)
    obj, _ = fun_grad(pack(u, tau, pi))

    f_pair, f_link, eta, psi, quad = buyer_loading(inst, u, tau)
    f_seg = np.zeros((inst.n_seg, inst.n_links))
    f_seg[inst.p_seg, inst.p_link] = f_pair
    f_all = f_seg.sum(axis=0)

    if mode == "clearing":
        y, y_idle, S = seller_loading(inst, tau, pi)
    else:
        y, y_idle, S = f_link.copy(), np.zeros(inst.n_op), np.zeros(inst.n_op)

    v = (S - np.log(inst.F)) / inst.alpha_S           # Eq. (32)
    v_tilde = S / inst.alpha_S
    # Accessibility: the supply-weighted mean potential over a segment's
    # origins.  Reduces to u_{i,o_i} for single-origin segments; with
    # multi-origin b the single `origin` entry is not meaningful.
    _w = np.maximum(inst.b, 0.0)
    _ws = _w.sum(axis=1, keepdims=True)
    u_origin = np.einsum('qv,qv->q', np.where(_ws > 0, _w / np.maximum(_ws, 1e-300), 0.0), u)

    gross = (
        float((inst.phi * f_seg).sum())
        - float(inst.c @ y)
        - float(inst.c_idle @ y_idle)
    )
    # Z' of Eq. (49) at the primal optimum, then Eq. (49a) for social surplus.
    # The buyer perturbation is already demand-scaled, so only the seller side
    # carries dropped constants.
    pert = float(np.sum(inst.ell[inst.p_link] * f_pair ** 2
                        / (2.0 * inst.D[inst.p_seg]))) / inst.alpha_B
    ent_s = _xlogx_minus(np.concatenate([y, y_idle])) / inst.alpha_S
    Zp = gross - pert - ent_s
    social = Zp + (
        float(inst.F @ np.log(inst.F)) - float(inst.F.sum())
    ) / inst.alpha_S

    rho = natural_residual(inst, tau, pi, f_link, y, eta,
                           clearing=(mode == "clearing"))

    return P2Solution(
        u=u, tau=tau, pi=pi, v=v, v_tilde=v_tilde,
        f_seg=f_seg, f=f_link, f_all=f_all, y=y, y_idle=y_idle,
        u_origin=u_origin, dual_obj=float(obj), gross_surplus=gross,
        social_surplus=float(social), residual=rho, n_iter=state["n_iter"],
        wall_time=time.perf_counter() - t_start,
        converged=rho <= tol, message=message, history=history,
    )


def _xlogx_minus(t: np.ndarray) -> float:
    """sum t (ln t - 1), with the 0 ln 0 = 0 convention."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    out[pos] = t[pos] * (np.log(t[pos]) - 1.0)
    return float(out.sum())


# ----------------------------------------------------------------------------
# Toy network of Fig. 1, in link form
# ----------------------------------------------------------------------------

def toy_instance_p2() -> P2Instance:
    """The three-node toy network as a link-based instance.

    Nodes 0, 1, 2 are the suburb (1), hub (2) and downtown (3) of Fig. 1.
    BLUE owns l1, l2; RED owns l3, l4; l5, l6, l7 are the outside links in A_0.
    Length weights are the link distances d_a, which is what reproduces the
    node potentials of Fig. 4 through Remark 8.

    `avail` mirrors the P1 path sets: segment 3 may use l1, l2, l3, l4 and its
    own outside link l7, but not l5 or l6, which belong to segments 1 and 2.
    """
    theta = np.array([1.20, -0.25, -0.50])          # (d, IVT, OVT), $ units
    z = np.array([                                   # l1..l4
        [8.0, 20.0, 5.0],
        [10.0, 25.0, 5.0],
        [8.0, 12.0, 2.0],
        [18.0, 30.0, 2.0],
    ])
    phi_own = z @ theta                              # (2.10, 3.25, 5.60, 13.10)
    phi_link = np.concatenate([phi_own, [0.0, 0.0, 0.0]])   # l5, l6, l7 carry 0

    #            l1     l2     l3     l4     l5     l6     l7
    tail = np.array([0, 1, 0, 0, 0, 1, 0])
    head = np.array([1, 2, 1, 2, 1, 2, 2])
    owner = np.array([0, 0, 1, 1, -1, -1, -1])
    ell = np.array([8.0, 10.0, 8.0, 18.0, 8.0, 10.0, 18.0])

    avail = np.array([
        [1, 0, 1, 0, 1, 0, 0],      # OD(1,2): l1, l3, outside l5
        [0, 1, 0, 0, 0, 1, 0],      # OD(2,3): l2, outside l6
        [1, 1, 1, 1, 0, 0, 1],      # OD(1,3): l1, l2, l3, l4, outside l7
    ], dtype=bool)

    return P2Instance(
        tail=tail, head=head, link_owner=owner, n_nodes=3,
        D=np.array([400.0, 600.0, 800.0]),
        origin=np.array([0, 1, 0]),
        sink=np.array([1, 2, 2]),
        F=np.array([1200.0, 250.0]),
        Cmax=np.array([900.0, 900.0, 200.0, 150.0]),
        c=np.array([1.40, 1.70, 0.80, 1.50]),
        c_idle=np.array([0.0, 0.0]),
        ell=ell,
        phi=np.tile(phi_link, (3, 1)),
        avail=avail,
        alpha_B=1.0,
        alpha_S=0.8,
        link_names=["l1", "l2", "l3", "l4", "l5", "l6", "l7"],
        segment_names=["OD(1,2)", "OD(2,3)", "OD(1,3)"],
        operator_names=["BLUE", "RED"],
        node_names=["n1", "n2", "n3"],
    )


def _check_figure4(sol: P2Solution, inst: P2Instance) -> str:
    """Compare against the reported values of Fig. 4 and Section 4."""
    rows = [
        ("f/y l1", sol.f[0], 430.44), ("f/y l2", sol.f[1], 614.36),
        ("f/y l3", sol.f[2], 99.77), ("f/y l4", sol.f[3], 150.00),
        ("tau l1", sol.tau[0], 2.68), ("tau l2", sol.tau[1], 3.42),
        ("tau l3", sol.tau[2], 8.38), ("tau l4", sol.tau[3], 17.16),
        ("pi  l4", sol.pi[3], 7.57),
        ("f   l5", sol.f_all[4], 189.24), ("f   l6", sol.f_all[5], 305.09),
        ("f   l7", sol.f_all[6], 330.55),
        ("u 1,n1", sol.u[0, 0], -3.78), ("u 2,n2", sol.u[1, 1], -5.08),
        ("u 3,n1", sol.u[2, 0], -7.44), ("u 3,n2", sol.u[2, 1], -4.16),
        ("v~ BLUE", sol.v_tilde[0], 2.56), ("v~ RED", sol.v_tilde[1], 8.73),
        ("D'*", sol.dual_obj, -10038.82),
        ("gross", sol.gross_surplus, 3472.44),
    ]
    out = ["", "  Fig. 4 / Section 4 check", "  quantity     computed    reported     diff"]
    for name, got, want in rows:
        out.append(f"  {name:<11}{got:11.3f} {want:11.2f} {got - want:9.3f}")
    return "\n".join(out)


if __name__ == "__main__":
    inst = toy_instance_p2()
    sol = solve_p2(inst, tol=1e-7)
    print(sol.summary(inst))
    print(_check_figure4(sol, inst))

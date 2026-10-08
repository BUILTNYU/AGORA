"""
Algorithm 1 - L-BFGS-B dual solution of the path-flow formulation of the
aggregate network assignment game P1.

Solves the box-constrained concentrated dual, Eq. (34):

    min_{tau >= 0, pi >= 0}  D(tau, pi)

      = (1/aB) sum_i D_i [ ln( 1 + sum_{k in K_i \\ k_i0}
                                 exp( aB ( Phi_ik - T_k ) ) ) - ln D_i + 1 ]
      + (1/aS) sum_m F_m [ ln( exp(-aS c_idle_m) + sum_{a in A_m}
                                 exp( aS ( tau_ma - c_ma - pi_ma ) ) ) - ln F_m + 1 ]
      + sum_m sum_{a in A_m} pi_ma Cmax_ma

with T_k = sum_{a in A_k \\ A_0} tau_ma, and gradients (Eq. 37)

    dD/dtau_ma = y_ma - f_a ,      dD/dpi_ma = Cmax_ma - y_ma .

Primal quantities are recovered by the multinomial logit forms of Eqs. (28)
and (33); duals u*, v* by Eqs. (29) and (32). Convergence is monitored by the
natural-residual gap of Eq. (38).

Requires: numpy, scipy.

Reference: Chow (2026), Section 3.1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------------
# Numerics
# ----------------------------------------------------------------------------

def _segment_logsumexp(
    values: np.ndarray,
    seg_idx: np.ndarray,
    n_seg: int,
    null_exponent: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable log-sum-exp within groups, with one extra "null alternative".

    For each group g, computes

        L_g = ln( exp(null_exponent[g]) + sum_{j : seg_idx[j] == g} exp(values[j]) )

    Returns
    -------
    L         : (n_seg,)          the inclusive values
    w         : (len(values),)    exp(values - L[seg_idx]), the logit shares
    w_null    : (n_seg,)          exp(null_exponent - L), the null-alternative shares

    The null alternative is the outside path k_i0 on the buyer side
    (null_exponent = aB * Phi_i0 = 0) and idle capacity y_m0 on the seller side
    (null_exponent = -aS * c_idle_m). Shifting by the group max keeps every
    exponential in [0, 1], so the routine is safe for large aB, aS.
    """
    mx = np.array(null_exponent, dtype=float, copy=True)
    if values.size:
        np.maximum.at(mx, seg_idx, values)

    acc = np.exp(null_exponent - mx)  # null-alternative contribution
    if values.size:
        shifted = np.exp(values - mx[seg_idx])
        np.add.at(acc, seg_idx, shifted)

    L = mx + np.log(acc)
    w = np.exp(values - L[seg_idx]) if values.size else np.zeros(0)
    w_null = np.exp(null_exponent - L)
    return L, w, w_null


# ----------------------------------------------------------------------------
# Problem instance
# ----------------------------------------------------------------------------

@dataclass
class P1Instance:
    """Inputs to P1, Eqs. (20)-(25).

    Links are split into owned links (each belonging to exactly one operator m,
    so the owner map m(a) is well defined) and outside links in A_0, which carry
    no market-clearing constraint, no flow-capacity constraint and no cost
    transfer. Only owned links appear in the dual vector.

    Parameters
    ----------
    link_owner : (A_own,) int
        Operator index m(a) for each owned link.
    D : (Q,) float
        Segment demands D_i > 0.
    F : (M,) float
        Operator endowments F_m > 0.
    Cmax : (A_own,) float
        Flow capacity caps C^max_ma.
    c : (A_own,) float
        Per-unit provision costs c_ma.
    c_idle : (M,) float
        Per-unit cost of undeployed capacity c_idle,m. The notation block of
        the paper sets c_m0 = 0 while Eq. (34) carries c_idle,m; this argument
        makes the choice explicit and defaults to zeros.
    paths : list of (segment, link list)
        Interior paths only. The outside path k_i0 is NOT listed: it is handled
        analytically as the null alternative, contributing the term 1 to the
        denominator of Eq. (28) under the normalization Phi_i0 = 0.
    Phi : (K,) float
        Gross surplus Phi_ik of each interior path, in the order of `paths`.
    alpha_B, alpha_S : float
        Logit scale parameters.
    link_names, segment_names, operator_names : optional labels for reporting.
    """

    link_owner: np.ndarray
    D: np.ndarray
    F: np.ndarray
    Cmax: np.ndarray
    c: np.ndarray
    paths: List[Tuple[int, Sequence[int]]]
    Phi: np.ndarray
    alpha_B: float
    alpha_S: float
    c_idle: Optional[np.ndarray] = None
    link_names: Optional[List[str]] = None
    segment_names: Optional[List[str]] = None
    operator_names: Optional[List[str]] = None

    # derived
    n_links: int = field(init=False)
    n_seg: int = field(init=False)
    n_op: int = field(init=False)
    path_seg: np.ndarray = field(init=False)
    incidence: np.ndarray = field(init=False)  # (K, A_own) 0/1, delta_ak

    def __post_init__(self) -> None:
        self.link_owner = np.asarray(self.link_owner, dtype=int)
        self.D = np.asarray(self.D, dtype=float)
        self.F = np.asarray(self.F, dtype=float)
        self.Cmax = np.asarray(self.Cmax, dtype=float)
        self.c = np.asarray(self.c, dtype=float)
        self.Phi = np.asarray(self.Phi, dtype=float)

        self.n_links = self.link_owner.size
        self.n_seg = self.D.size
        self.n_op = self.F.size

        if self.c_idle is None:
            self.c_idle = np.zeros(self.n_op)
        self.c_idle = np.asarray(self.c_idle, dtype=float)

        self.path_seg = np.array([s for s, _ in self.paths], dtype=int)
        self.incidence = np.zeros((len(self.paths), self.n_links))
        for j, (_, links) in enumerate(self.paths):
            for a in links:
                self.incidence[j, a] = 1.0

        self._validate()

    def _validate(self) -> None:
        if np.any(self.D <= 0):
            raise ValueError("Assumption 1: every D_i must be strictly positive.")
        if np.any(self.F <= 0):
            raise ValueError("Assumption 1: every F_m must be strictly positive.")
        if self.alpha_B <= 0 or self.alpha_S <= 0:
            raise ValueError("alpha_B and alpha_S must be strictly positive.")
        for arr, name in ((self.Cmax, "Cmax"), (self.c, "c")):
            if arr.size != self.n_links:
                raise ValueError(f"{name} must have one entry per owned link.")
        if self.Phi.size != len(self.paths):
            raise ValueError("Phi must have one entry per interior path.")
        if np.any(self.path_seg < 0) or np.any(self.path_seg >= self.n_seg):
            raise ValueError("A path references an unknown segment.")

        # Assumption 1: F_m < sum_{a in A_m} Cmax_ma, strictly. Strictness is
        # what makes the Slater point of proof (d) interior on the cap
        # constraints, so it is enforced rather than warned about.
        cap_by_op = np.zeros(self.n_op)
        np.add.at(cap_by_op, self.link_owner, self.Cmax)
        bad = np.where(self.F >= cap_by_op)[0]
        if bad.size:
            raise ValueError(
                "Assumption 1 violated: need F_m < sum_a Cmax_ma strictly for "
                f"operator(s) {bad.tolist()} (F={self.F[bad]}, "
                f"sum Cmax={cap_by_op[bad]})."
            )

        # Every segment needs at least an outside path; interior paths optional.
        # The outside path is implicit, so nothing to check beyond indices.


# ----------------------------------------------------------------------------
# Solution container
# ----------------------------------------------------------------------------

@dataclass
class P1Solution:
    tau: np.ndarray            # (A_own,) cost transfers tau*_ma
    pi: np.ndarray             # (A_own,) capacity rents pi*_ma
    x: np.ndarray              # (K,) interior path flows x*_ik
    x_out: np.ndarray          # (Q,) outside-path flows x*_i0
    f: np.ndarray              # (A_own,) link loads f_a
    y: np.ndarray              # (A_own,) allocated capacities y*_ma
    y_idle: np.ndarray         # (M,) undeployed capacity y*_m0
    u: np.ndarray              # (Q,) u*_i, Eq. (29)
    v: np.ndarray              # (M,) v*_m, Eq. (32)
    u_tilde: np.ndarray        # (Q,) recentered u_i = u*_i + ln(D_i)/aB
    v_tilde: np.ndarray        # (M,) recentered v_m = v*_m + ln(F_m)/aS
    dual_obj: float            # D* of Eq. (34)
    gross_surplus: float       # sum Phi x - sum c y - sum c_idle y_m0
    social_surplus: float      # Z~* of Eq. (20a)
    residual: float            # varrho of Eq. (38) at termination
    n_iter: int
    wall_time: float
    converged: bool
    message: str
    history: Dict[str, List[float]]

    def summary(self, inst: P1Instance) -> str:
        ln = [
            f"Algorithm 1  ({'converged' if self.converged else 'STOPPED'}: {self.message})",
            f"  iterations      : {self.n_iter}",
            f"  wall time       : {1e3 * self.wall_time:.1f} ms",
            f"  dual objective  : {self.dual_obj:.3f}",
            f"  natural residual: {self.residual:.3e}",
            f"  gross surplus   : {self.gross_surplus:.2f}",
            f"  social surplus  : {self.social_surplus:.2f}",
            "",
            "  link      tau*      pi*        f_a        y_ma     Cmax",
        ]
        names = inst.link_names or [f"a{a}" for a in range(inst.n_links)]
        for a in range(inst.n_links):
            ln.append(
                f"  {names[a]:<8}{self.tau[a]:8.4f}  {self.pi[a]:8.4f}  "
                f"{self.f[a]:9.3f}  {self.y[a]:9.3f}  {inst.Cmax[a]:7.1f}"
            )
        ln += ["", "  segment      u*        u~ (exp. utility)      x_i0"]
        snames = inst.segment_names or [f"i{i}" for i in range(inst.n_seg)]
        for i in range(inst.n_seg):
            ln.append(
                f"  {snames[i]:<10}{self.u[i]:9.4f}  {self.u_tilde[i]:15.4f}  "
                f"{self.x_out[i]:12.3f}"
            )
        ln += ["", "  operator     v*        v~ (marg. surplus)   y_m0   deployed"]
        onames = inst.operator_names or [f"m{m}" for m in range(inst.n_op)]
        for m in range(inst.n_op):
            deployed = 1.0 - self.y_idle[m] / inst.F[m]
            ln.append(
                f"  {onames[m]:<10}{self.v[m]:9.4f}  {self.v_tilde[m]:15.4f}  "
                f"{self.y_idle[m]:9.3f}  {100 * deployed:6.2f}%"
            )
        return "\n".join(ln)


# ----------------------------------------------------------------------------
# Steps 2(a) and 2(b): buyer and seller loading
# ----------------------------------------------------------------------------

def buyer_loading(
    inst: P1Instance, tau: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step 2(a). Eq. (28).

    Returns (x, x_out, f, B) where B_i is the buyer inclusive value
    ln( 1 + sum_k exp( aB ( Phi_ik - T_k ) ) ).
    """
    T = inst.incidence @ tau                       # path prices T_k
    z = inst.alpha_B * (inst.Phi - T)              # systematic index
    B, w, w_null = _segment_logsumexp(
        z, inst.path_seg, inst.n_seg, np.zeros(inst.n_seg)
    )
    x = inst.D[inst.path_seg] * w                  # interior path flows
    x_out = inst.D * w_null                        # outside-path flows x_i0
    f = inst.incidence.T @ x                       # link loads f_a
    return x, x_out, f, B


def seller_loading(
    inst: P1Instance, tau: np.ndarray, pi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 2(b). Eq. (33).

    Returns (y, y_idle, S) where S_m is the seller inclusive value
    ln( exp(-aS c_idle_m) + sum_a exp( aS ( tau_ma - c_ma - pi_ma ) ) ).
    """
    omega = tau - inst.c - pi                      # affine link margin
    S, w, w_null = _segment_logsumexp(
        inst.alpha_S * omega,
        inst.link_owner,
        inst.n_op,
        -inst.alpha_S * inst.c_idle,
    )
    y = inst.F[inst.link_owner] * w
    y_idle = inst.F * w_null
    return y, y_idle, S


def natural_residual(
    inst: P1Instance, tau: np.ndarray, pi: np.ndarray, f: np.ndarray, y: np.ndarray
) -> float:
    """Eq. (38): sup-norm of min(z_j, dD/dz_j), scaled by the busiest link load."""
    r_clear = np.minimum(tau, y - f)
    r_cap = np.minimum(pi, inst.Cmax - y)
    scale = f.max()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0                                 # guard the early iterate f = 0
    return float(max(np.abs(r_clear).max(), np.abs(r_cap).max()) / scale)


# ----------------------------------------------------------------------------
# Algorithm 1
# ----------------------------------------------------------------------------

def solve_p1(
    inst: P1Instance,
    tol: float = 1e-7,
    max_iter: int = 500,
    tau0: Optional[np.ndarray] = None,
    pi0: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> P1Solution:
    """Algorithm 1. L-BFGS-B on the concentrated dual of Eq. (34).

    Parameters
    ----------
    tol : float
        Tolerance epsilon on the natural residual of Eq. (38).
    max_iter : int
        Iteration cap for L-BFGS-B.
    tau0, pi0 : optional warm starts. Step 1 default is tau = 0, pi = 0.
    """
    n = inst.n_links
    t_start = time.perf_counter()

    # --- Step 1. Initialize -------------------------------------------------
    z0 = np.concatenate([
        np.zeros(n) if tau0 is None else np.asarray(tau0, float),
        np.zeros(n) if pi0 is None else np.asarray(pi0, float),
    ])

    history: Dict[str, List[float]] = {"dual_obj": [], "residual": []}
    cache: Dict[str, object] = {}

    def unpack(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return z[:n], z[n:]

    # --- Step 2(a)-(c). Objective and gradient ------------------------------
    def fun_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        tau, pi = unpack(z)
        x, x_out, f, B = buyer_loading(inst, tau)
        y, y_idle, S = seller_loading(inst, tau, pi)

        obj = (
            np.sum(inst.D * (B - np.log(inst.D) + 1.0)) / inst.alpha_B
            + np.sum(inst.F * (S - np.log(inst.F) + 1.0)) / inst.alpha_S
            + float(pi @ inst.Cmax)
        )
        grad = np.concatenate([y - f, inst.Cmax - y])   # Eq. (37)

        cache.update(x=x, x_out=x_out, f=f, B=B, y=y, y_idle=y_idle, S=S, obj=obj)
        return float(obj), grad

    # --- Step 2(d). Stop on the natural residual ----------------------------
    state = {"n_iter": 0, "stopped": False}

    def callback(z_or_res):
        z = getattr(z_or_res, "x", z_or_res)
        tau, pi = unpack(np.asarray(z, dtype=float))
        _, _, f, _ = buyer_loading(inst, tau)
        y, _, _ = seller_loading(inst, tau, pi)
        rho = natural_residual(inst, tau, pi, f, y)
        state["n_iter"] += 1
        history["residual"].append(rho)
        history["dual_obj"].append(float(cache.get("obj", np.nan)))
        if verbose:
            print(f"  t={state['n_iter']:4d}  D={cache.get('obj', float('nan')):14.6f}  rho={rho:.3e}")
        if rho <= tol:
            state["stopped"] = True
            raise StopIteration

    bounds = [(0.0, None)] * (2 * n)                  # the box tau >= 0, pi >= 0
    try:
        res = minimize(
            fun_grad, z0, jac=True, method="L-BFGS-B", bounds=bounds,
            callback=callback,
            options={"maxiter": max_iter, "ftol": 1e-16, "gtol": 1e-12,
                     "maxcor": 20},
        )
        z_star, message = res.x, str(res.message)
    except StopIteration:
        # SciPy < 1.11 propagates the exception instead of halting gracefully;
        # the last accepted iterate is still the one in `cache`, so re-derive it.
        z_star = _last_iterate(fun_grad, z0, bounds, state["n_iter"], max_iter)
        message = "natural residual below tolerance"

    if state["stopped"]:
        message = "natural residual below tolerance"

    # --- Step 3. Recover primal and duals -----------------------------------
    tau, pi = unpack(z_star)
    tau = np.maximum(tau, 0.0)
    pi = np.maximum(pi, 0.0)
    obj, _ = fun_grad(np.concatenate([tau, pi]))
    x, x_out = cache["x"], cache["x_out"]
    f, y, y_idle = cache["f"], cache["y"], cache["y_idle"]
    B, S = cache["B"], cache["S"]

    u = (B - np.log(inst.D)) / inst.alpha_B           # Eq. (29)
    v = (S - np.log(inst.F)) / inst.alpha_S           # Eq. (32)
    u_tilde = B / inst.alpha_B                        # recentered, Section 3.2
    v_tilde = S / inst.alpha_S

    gross = (
        float(inst.Phi @ x)
        - float(inst.c @ y)
        - float(inst.c_idle @ y_idle)
    )
    # Z of Eq. (20) at the primal optimum, then Eq. (20a) for the social surplus.
    ent_b = _xlogx_minus(np.concatenate([x, x_out])) / inst.alpha_B
    ent_s = _xlogx_minus(np.concatenate([y, y_idle])) / inst.alpha_S
    Z = gross - ent_b - ent_s
    social = Z + (
        float(inst.D @ np.log(inst.D)) - float(inst.D.sum())
    ) / inst.alpha_B + (
        float(inst.F @ np.log(inst.F)) - float(inst.F.sum())
    ) / inst.alpha_S

    rho = natural_residual(inst, tau, pi, f, y)

    return P1Solution(
        tau=tau, pi=pi, x=x, x_out=x_out, f=f, y=y, y_idle=y_idle,
        u=u, v=v, u_tilde=u_tilde, v_tilde=v_tilde,
        dual_obj=float(obj), gross_surplus=gross, social_surplus=float(social),
        residual=rho, n_iter=state["n_iter"],
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


def _last_iterate(fun_grad, z0, bounds, n_iter, max_iter):
    """Re-run without a callback for `n_iter` iterations (SciPy < 1.11 path)."""
    res = minimize(
        fun_grad, z0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": max(n_iter, 1), "ftol": 1e-16, "gtol": 1e-12,
                 "maxcor": 20},
    )
    return res.x


# ----------------------------------------------------------------------------
# Toy network of Fig. 1
# ----------------------------------------------------------------------------

def toy_instance() -> P1Instance:
    """The three-node toy network: suburb (1) - hub (2) - downtown (3).

    BLUE (commuter bus) owns l1, l2; RED (on-demand) owns l3, l4.
    Outside links l5, l6, l7 are in A_0 and carry Phi = 0.
    """
    theta = np.array([1.20, -0.25, -0.50])          # (d, IVT, OVT), $ units
    z = np.array([                                   # owned links l1..l4
        [8.0, 20.0, 5.0],
        [10.0, 25.0, 5.0],
        [8.0, 12.0, 2.0],
        [18.0, 30.0, 2.0],
    ])
    phi = z @ theta                                  # (2.10, 3.25, 5.60, 13.10)

    L1, L2, L3, L4 = 0, 1, 2, 3
    paths = [
        (0, [L1]), (0, [L3]),                        # segment 1: OD (1,2)
        (1, [L2]),                                   # segment 2: OD (2,3)
        (2, [L4]), (2, [L1, L2]), (2, [L3, L2]),     # segment 3: OD (1,3)
    ]
    Phi = np.array([sum(phi[a] for a in links) for _, links in paths])

    return P1Instance(
        link_owner=np.array([0, 0, 1, 1]),           # 0 = BLUE, 1 = RED
        D=np.array([400.0, 600.0, 800.0]),
        F=np.array([1200.0, 250.0]),
        Cmax=np.array([900.0, 900.0, 200.0, 150.0]),
        c=np.array([1.40, 1.70, 0.80, 1.50]),
        c_idle=np.array([0.0, 0.0]),
        paths=paths,
        Phi=Phi,
        alpha_B=1.0,
        alpha_S=0.8,
        link_names=["l1", "l2", "l3", "l4"],
        segment_names=["OD(1,2)", "OD(2,3)", "OD(1,3)"],
        operator_names=["BLUE", "RED"],
    )


if __name__ == "__main__":
    inst = toy_instance()
    sol = solve_p1(inst, tol=1e-7, verbose=False)
    print(sol.summary(inst))

"""
Algorithm 3' - Solution to the aggregate network assignment game P2 (PURC-based
link formulation) with fixed (administered) prices tau0.

As in Algorithm 1', at an administered fare tau0 the market need not clear, so
Eq. (51) is carried explicitly rather than dualized and travelers face the
effective price tau0_ma + sigma_a with the rationing price sigma_a >= 0. Unlike
Algorithm 1', the node potentials u have no closed form and enter the buyer step
as free variables alongside sigma.

    Step 1  Initialize    y^1 from the seller step at L == 1; u from a
                          shortest-path pass on -phi_ia + tau0_ma from each s_i.

    Step 2  Buyer step    given y^t, minimize over u free and sigma >= 0

              sum_i sum_a ( aB D_i / (2 l_a) ) [ psi_ia ]_+^2
              + sum_i sum_n D_i b_in u_in
              + sigma^T y^t

            with psi_ia = phi_ia - tau0_ma - sigma_a - ( u_{i,t(a)} - u_{i,h(a)} )
            and f_ia = ( aB D_i / l_a ) [ psi_ia ]_+, gradients

              dD'/du_in    = -eta_in ,
              dD'/dsigma_a = y^t_a - f_a .

    Step 3  Seller step   identical to Algorithm 1' step 3, at the load factors
                          L^t_ma = min(1, f^t_a / y^t_ma) with f^t_a = sum_i f_ia.

    Step 4  Average       y^{t+1} = (1 - gamma_t) y^t + gamma_t y~^t; stop on rho.

Existence of the fixed point is Theorem 3; convergence is not established there,
so this module reports the residual trajectory rather than asserting it.

ON THE SIGN OF psi. The grouping -( u_{i,t(a)} - u_{i,h(a)} ) is as the draft
has it in step 2 here and throughout Section 3.3, and is what the Lagrangian of
Eq. (50) produces.

NOTE ON STEP 1. The draft writes the shortest-path pass as being on
"-phi_ia - tau0_ma". The arc cost of the negated net surplus is
-(phi_ia - tau0_ma) = -phi_ia + tau0_ma, which is what is used here.

Requires: numpy, scipy, and `algorithm1_p1.py`, `p1_fixed_price.py`,
`algorithm2_p2.py` in the same directory.

Reference: Chow (2026), Section 3.3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from p1_fixed_price import (
    _operator_allocation,
    fixed_point_residual,
    load_factors,
)
from algorithm2_p2 import (
    P2Instance,
    _xlogx_minus,
    buyer_loading,
    initial_potentials,
    toy_instance_p2,
)


# ----------------------------------------------------------------------------
# Step 3: seller step (Algorithm 1' step 3, on P2 indexing)
# ----------------------------------------------------------------------------

def seller_step(
    inst: P2Instance, tau0: np.ndarray, L: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step 3. Per-operator capped entropic allocation at margins tau0*L - c.

    Identical to step 3 of Algorithm 1'; only the owner map differs, since P2
    indexes tau, pi over `inst.owned_idx` rather than over all links.
    Returns (y_tilde, y_idle_tilde, pi, v).
    """
    g = tau0 * L - inst.c
    y = np.zeros(inst.n_own)
    pi = np.zeros(inst.n_own)
    y_idle = np.zeros(inst.n_op)
    v = np.zeros(inst.n_op)
    for m in range(inst.n_op):
        idx = np.flatnonzero(inst.owner_of_dual == m)
        y[idx], y0, pi[idx], v[m] = _operator_allocation(
            g[idx], inst.Cmax[idx], float(inst.F[m]),
            float(inst.c_idle[m]), inst.alpha_S,
        )
        y_idle[m] = y0
    return y, y_idle, pi, v


# ----------------------------------------------------------------------------
# Step 2: buyer step over (u, sigma)
# ----------------------------------------------------------------------------

def buyer_step(
    inst: P2Instance,
    tau0: np.ndarray,
    y_link: np.ndarray,
    u_init: Optional[np.ndarray] = None,
    sigma_init: Optional[np.ndarray] = None,
    max_iter: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Step 2. Minimize over u free and sigma >= 0 at capacities y^t.

    This is the Lagrangian dual, at capacity vector y^t, of the strictly concave
    buyer program of Eq. (49) subject to Eqs. (50)-(51), so sigma is the
    multiplier on link market clearing at the administered fare and u the
    multiplier on flow conservation.

    Returns (u, sigma, f_pair, f_link, eta, objective).
    """
    mask = inst.u_free
    n_u = int(mask.sum())
    n_own = inst.n_own
    seg = np.arange(inst.n_seg)

    u0 = initial_potentials(inst, tau0) if u_init is None else np.array(u_init, float)
    s0 = np.zeros(n_own) if sigma_init is None else np.maximum(sigma_init, 0.0)

    def unpack(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.zeros((inst.n_seg, inst.n_nodes))
        u[mask] = z[:n_u]
        return u, z[n_u:]

    def fun_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        u, sigma = unpack(z)
        # The effective price tau0 + sigma enters exactly where tau does in
        # Algorithm 3, so the Eq. (57) loading is reused unchanged.
        _, f_link, eta, _, quad = buyer_loading(inst, u, tau0 + sigma)
        # Eq. (60) linear term is sum_i sum_n D_i b_in u_in; D_i u_{i,o_i} is
        # equivalent only for single-origin segments, and otherwise disagrees
        # with the gradient, which buyer_loading forms from inst.b.
        obj = (quad + float((inst.D[:, None] * inst.b * u).sum())
               + float(sigma @ y_link))
        grad = np.concatenate([-eta[mask], y_link - f_link])
        return float(obj), grad

    z0 = np.concatenate([u0[mask], s0])
    res = minimize(
        fun_grad, z0, jac=True, method="L-BFGS-B",
        bounds=[(None, None)] * n_u + [(0.0, None)] * n_own,
        options={"maxiter": max_iter, "maxfun": 20 * max_iter,
                 "ftol": 1e-16, "gtol": 1e-12, "maxcor": 30},
    )
    u, sigma = unpack(res.x)
    sigma = np.maximum(sigma, 0.0)
    f_pair, f_link, eta, _, _ = buyer_loading(inst, u, tau0 + sigma)
    return u, sigma, f_pair, f_link, eta, float(res.fun)


# ----------------------------------------------------------------------------
# Residuals
# ----------------------------------------------------------------------------

def natural_residual_fixed_price(
    inst: P2Instance,
    sigma: np.ndarray,
    pi: np.ndarray,
    f_link: np.ndarray,
    y: np.ndarray,
    eta: np.ndarray,
) -> float:
    """Eq. (38a) with sigma in place of tau.

    Diagnostic only. Every term is driven to zero *within* a step by
    construction: the buyer step enforces eta = 0 and 0 <= sigma _|_ (y - f) >= 0
    at its own y^t, and the seller step enforces 0 <= pi _|_ (Cmax - y) >= 0. So
    this measures inconsistency between the two blocks, not distance to the
    fixed point. Use `fixed_point_residual` for the stopping test.
    """
    scale = f_link.max() if f_link.size else 0.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    parts = [
        np.abs(eta[inst.u_free]).max() if inst.u_free.any() else 0.0,
        np.abs(np.minimum(sigma, y - f_link)).max(),
        np.abs(np.minimum(pi, inst.Cmax - y)).max(),
    ]
    return float(max(parts) / scale)


# ----------------------------------------------------------------------------
# Solution container
# ----------------------------------------------------------------------------

@dataclass
class P2FixedPriceSolution:
    u: np.ndarray              # (Q, V) node potentials u†, u†_{i,s_i} = 0
    u_origin: np.ndarray       # (Q,) supply-weighted mean potential (accessibility)
    v: np.ndarray              # (M,) v†, budget multipliers from the seller step
    v_tilde: np.ndarray        # (M,) recentered v_m = v†_m + ln(F_m)/aS
    sigma: np.ndarray          # (A_own,) rationing prices sigma†
    pi: np.ndarray             # (A_own,) capacity rents pi†
    f_seg: np.ndarray          # (Q, A) segment-link flows f*_ia
    f: np.ndarray              # (A_own,) owned-link loads f_a
    f_all: np.ndarray          # (A,) loads on every link, incl. A_0
    y: np.ndarray              # (A_own,) allocated capacity
    y_idle: np.ndarray         # (M,) undeployed capacity
    L: np.ndarray              # (A_own,) load factors L†
    tau_eff: np.ndarray        # (A_own,) effective price tau0 + sigma
    excess_demand: np.ndarray  # (A_own,) f - y, positive where rationed
    gross_surplus: float
    residual: float
    residual_natural: float
    n_iter: int
    wall_time: float
    converged: bool
    history: Dict[str, List[float]] = field(default_factory=dict)

    def summary(self, inst: P2Instance, tau0: np.ndarray) -> str:
        lnames = inst.link_names or [f"a{a}" for a in range(inst.n_links)]
        onames = inst.operator_names or [f"m{m}" for m in range(inst.n_op)]
        out = [
            f"Algorithm 3'  ({'converged' if self.converged else 'NOT converged'})",
            f"  MSA iterations   : {self.n_iter}",
            f"  wall time        : {1e3 * self.wall_time:.1f} ms",
            f"  fixed-pt residual: {self.residual:.3e}",
            f"  natural residual : {self.residual_natural:.3e}",
            f"  gross surplus    : {self.gross_surplus:.2f}",
            "",
            "  link    owner    tau0   sigma†  tau_eff    pi†      f_a       y_ma"
            "      L†      f-y",
        ]
        for a in range(inst.n_links):
            d = inst.dual_of_link[a]
            if d >= 0:
                out.append(
                    f"  {lnames[a]:<7}{onames[inst.link_owner[a]]:<7}"
                    f"{tau0[d]:7.3f} {self.sigma[d]:7.3f} {self.tau_eff[d]:8.3f} "
                    f"{self.pi[d]:7.3f} {self.f[d]:9.3f} {self.y[d]:9.3f} "
                    f"{self.L[d]:7.4f} {self.excess_demand[d]:8.3f}"
                )
            else:
                out.append(
                    f"  {lnames[a]:<7}{'A_0':<7}{'--':>7} {'--':>7} {'--':>8} "
                    f"{'--':>7} {self.f_all[a]:9.3f} {'--':>9} {'--':>7} {'--':>8}"
                )
        nnames = inst.node_names or [f"n{n}" for n in range(inst.n_nodes)]
        snames = inst.segment_names or [f"i{i}" for i in range(inst.n_seg)]
        out += ["", "  node potentials u†_in  (-- = outside the subnetwork)",
                "  segment    " + "".join(f"{nm:>10}" for nm in nnames)]
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
        out += ["", "  operator     v†        v~        y_m0    deployed"]
        for m in range(inst.n_op):
            out.append(
                f"  {onames[m]:<10}{self.v[m]:9.4f} {self.v_tilde[m]:9.4f} "
                f"{self.y_idle[m]:9.3f} {100 * (1 - self.y_idle[m] / inst.F[m]):8.2f}%"
            )
        return "\n".join(out)


# ----------------------------------------------------------------------------
# Algorithm 3'
# ----------------------------------------------------------------------------

def _supply_weighted_u(inst, u):
    """Supply-weighted mean potential over each segment's origins.

    Reduces to u_{i,o_i} for single-origin segments; with multi-origin b the
    single `origin` entry is a placeholder and carries no meaning.
    """
    w = np.maximum(inst.b, 0.0)
    ws = w.sum(axis=1, keepdims=True)
    return np.einsum('qv,qv->q', np.where(ws > 0, w / np.maximum(ws, 1e-300), 0.0), u)


def solve_p2_fixed_price(
    inst: P2Instance,
    tau0: np.ndarray,
    tol: float = 1e-7,
    max_iter: int = 500,
    gamma_exponent: float = 0.6,
    residual: str = "fixed_point",
    y_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> P2FixedPriceSolution:
    """Algorithm 3'.

    Parameters
    ----------
    tau0 : (A_own,) observed / administered fares, tau0 >= 0, over owned links.
    tol : stopping tolerance epsilon on the residual.
    max_iter : cap on MSA iterations.
    gamma_exponent : the 0.6 in gamma_t = t^-gamma_exponent. Any value in
        (0.5, 1] gives sum gamma_t = inf, sum gamma_t^2 < inf.
    residual : "fixed_point" (default) tests ||y~ - y||_inf / ||y||_inf;
        "natural" tests Eq. (38a) with sigma in place of tau. Step 4 of the
        draft says "if rho^t <= eps" without saying which, and Eq. (38a) is not
        the right test here because the market does not clear at tau0 -- see
        `natural_residual_fixed_price`.
    y_init : optional warm start, ordered as the owned links followed by the M
        idle entries.

    Notes
    -----
    Step 4 averages the idle entries y_m0 alongside the deployed y_ma. Both y^t
    and y~^t satisfy the budget equality and the caps, so the convex combination
    does too -- the feasibility claim at the end of Theorem 3, which relies on
    convexity of K_{eta0} and fails if the idle entries are left un-averaged.
    """
    tau0 = np.asarray(tau0, dtype=float)
    if tau0.size != inst.n_own:
        raise ValueError("tau0 must have one entry per owned link.")
    if np.any(tau0 < 0):
        raise ValueError("Administered fares must satisfy tau0 >= 0.")
    if residual not in ("fixed_point", "natural"):
        raise ValueError("residual must be 'fixed_point' or 'natural'.")

    t_start = time.perf_counter()
    history: Dict[str, List[float]] = {
        "residual": [], "residual_natural": [], "gamma": [], "buyer_obj": [],
    }

    # --- Step 1. Initialize -------------------------------------------------
    if y_init is None:
        y, y_idle, pi, v = seller_step(inst, tau0, np.ones(inst.n_own))
    else:
        y_init = np.asarray(y_init, dtype=float)
        y, y_idle = y_init[: inst.n_own].copy(), y_init[inst.n_own:].copy()
        pi, v = np.zeros(inst.n_own), np.zeros(inst.n_op)

    u = initial_potentials(inst, tau0)
    sigma = np.zeros(inst.n_own)
    converged = False
    t = 0

    for t in range(1, max_iter + 1):
        # --- Step 2. Buyer step at the current capacities -------------------
        u, sigma, f_pair, f_link, eta, buyer_obj = buyer_step(
            inst, tau0, y, u_init=u, sigma_init=sigma
        )

        # --- Step 3. Seller step at the induced load factors ----------------
        L = load_factors(f_link, y)
        y_new, y_idle_new, pi, v = seller_step(inst, tau0, L)

        rho_fp = fixed_point_residual(y_new, y)
        rho_nat = natural_residual_fixed_price(inst, sigma, pi, f_link, y, eta)
        rho = rho_fp if residual == "fixed_point" else rho_nat

        history["residual"].append(rho_fp)
        history["residual_natural"].append(rho_nat)
        history["buyer_obj"].append(buyer_obj)

        if verbose:
            print(f"  t={t:4d}  rho_fp={rho_fp:.3e}  rho_nat={rho_nat:.3e}  "
                  f"Psi={buyer_obj:14.4f}")

        if rho <= tol:
            y, y_idle = y_new, y_idle_new
            converged = True
            history["gamma"].append(0.0)
            break

        # --- Step 4. Average and continue -----------------------------------
        gamma = float(t) ** (-gamma_exponent)
        history["gamma"].append(gamma)
        y = (1.0 - gamma) * y + gamma * y_new
        y_idle = (1.0 - gamma) * y_idle + gamma * y_idle_new

    # Final buyer response at the terminal capacities, so that f, sigma and u
    # are mutually consistent with the reported y.
    u, sigma, f_pair, f_link, eta, _ = buyer_step(
        inst, tau0, y, u_init=u, sigma_init=sigma
    )
    L = load_factors(f_link, y)

    f_seg = np.zeros((inst.n_seg, inst.n_links))
    f_seg[inst.p_seg, inst.p_link] = f_pair
    f_all = f_seg.sum(axis=0)

    gross = (
        float((inst.phi * f_seg).sum())
        - float(inst.c @ y)
        - float(inst.c_idle @ y_idle)
    )

    return P2FixedPriceSolution(
        u=u, u_origin=_supply_weighted_u(inst, u),
        v=v, v_tilde=v + np.log(inst.F) / inst.alpha_S,
        sigma=sigma, pi=pi,
        f_seg=f_seg, f=f_link, f_all=f_all, y=y, y_idle=y_idle, L=L,
        tau_eff=tau0 + sigma, excess_demand=f_link - y,
        gross_surplus=gross,
        residual=history["residual"][-1] if history["residual"] else np.nan,
        residual_natural=(history["residual_natural"][-1]
                          if history["residual_natural"] else np.nan),
        n_iter=t, wall_time=time.perf_counter() - t_start,
        converged=converged, history=history,
    )


# ----------------------------------------------------------------------------
# The t = 1 pre-rationing iterate
# ----------------------------------------------------------------------------

def initial_iterate(
    inst: P2Instance, tau0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The buyer response at sigma == 0 against y^1 from the seller step at L == 1.

    This is NOT a fixed-price equilibrium: sigma is held at zero, so Eq. (51) is
    violated wherever f_a > y_ma. It is provided because it is the quantity
    Section 4 reports as the "generated flows" at the suboptimal tolls (the
    f_a - y_ma figures quoted there), and is therefore what the IO2 ground-truth
    data appear to have been drawn from. Returns (u, f_link, y, y_idle).
    """
    y, y_idle, _, _ = seller_step(inst, tau0, np.ones(inst.n_own))
    mask = inst.u_free
    n_u = int(mask.sum())
    seg = np.arange(inst.n_seg)

    def fun_grad(zu: np.ndarray) -> Tuple[float, np.ndarray]:
        u = np.zeros((inst.n_seg, inst.n_nodes))
        u[mask] = zu
        _, _, eta, _, quad = buyer_loading(inst, u, tau0)
        return (float(quad + (inst.D[:, None] * inst.b * u).sum()), -eta[mask])

    res = minimize(
        fun_grad, initial_potentials(inst, tau0)[mask], jac=True,
        method="L-BFGS-B", bounds=[(None, None)] * n_u,
        options={"maxiter": 5000, "ftol": 1e-16, "gtol": 1e-12, "maxcor": 30},
    )
    u = np.zeros((inst.n_seg, inst.n_nodes))
    u[mask] = res.x
    _, f_link, _, _, _ = buyer_loading(inst, u, tau0)
    return u, f_link, y, y_idle


# ----------------------------------------------------------------------------
# Toy network of Fig. 1
# ----------------------------------------------------------------------------

def toy_tau0() -> np.ndarray:
    """Administered fares tau0 for l1..l4 from Fig. 1."""
    return np.array([2.75, 2.75, 3.50, 6.00])


if __name__ == "__main__":
    inst = toy_instance_p2()
    tau0 = toy_tau0()

    sol = solve_p2_fixed_price(inst, tau0, tol=1e-7)
    print(sol.summary(inst, tau0))

    # Section 4's reported "generated flows" at the suboptimal tolls.
    _, f1, y1, _ = initial_iterate(inst, tau0)
    print("\n  t = 1 pre-rationing iterate (sigma == 0, y^1 at L == 1)")
    print("    f_a - y_ma =", np.round(f1 - y1, 1),
          " [Section 4 reports -487.7, +109.6, +363.0, +288.5]")

    # Pricing wedge against the clearing solution of Algorithm 3.
    from algorithm2_p2 import solve_p2

    star = solve_p2(inst)
    print("\n  Pricing wedge  Delta_ma = tau0_ma - tau*_ma")
    lnames = inst.link_names or [f"a{a}" for a in range(inst.n_links)]
    for d, a in enumerate(inst.owned_idx):
        wedge = tau0[d] - star.tau[d]
        tag = "underpriced" if wedge < 0 else "overpriced "
        print(f"  {lnames[a]:<5}{wedge:9.4f}   {tag}   sigma†={sol.sigma[d]:.4f}")
    print(f"\n  Welfare cost of the observed fares: "
          f"{star.gross_surplus - sol.gross_surplus:.2f}")

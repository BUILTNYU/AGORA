"""
Fixed-price solver for P1 ("Algorithm 1'" in earlier drafts; the current draft
has no such algorithm). Solution to the aggregate network assignment game P1
with fixed (administered) prices tau0. algorithm2prime_p2.py imports helpers
from it.

At an administered fare tau0 the market need not clear, so the link capacity
constraint Eq. (22) is carried explicitly rather than dualized. Travelers face
the effective price tau0_ma + sigma_a, where sigma_a >= 0 is the rationing
price: the shadow price of scarce capacity, not revenue the operator receives.

The result is a fixed point in the capacity vector y, solved by Method of
Successive Averages (Powell and Sheffi, 1982) with weight gamma_t = t^-0.6.

    Step 1  Initialize    y^1 from the seller step at L == 1.
    Step 2  Buyer step    given y^t, minimize over sigma >= 0

              (1/aB) sum_i D_i [ ln( 1 + sum_k exp( aB ( Phi_ik - P_k(sigma) ) ) )
                                 - ln D_i + 1 ]  +  sigma^T y^t

            with P_k(sigma) = sum_{a in A_k ∩ A_own} ( tau0_ma + sigma_a ).
            Gradient: dPsi/dsigma_a = y^t_a - f_a.

    Step 3  Seller step   form L^t_ma = min(1, f^t_a / y^t_ma) and solve per m

              max_{y >= 0}  sum_a ( tau0_ma L^t_ma - c_ma ) y_ma - c_idle_m y_m0
                            - (1/aS) [ sum_a y_ma (ln y_ma - 1)
                                       + y_m0 (ln y_m0 - 1) ]
              s.t.  sum_a y_ma + y_m0 = F_m ,   y_ma <= Cmax_ma

            returning the auxiliary vector y~^t and the capacity rent pi^t.

    Step 4  Average       y^{t+1} = (1 - gamma_t) y^t + gamma_t y~^t; stop on rho.

Existence of the fixed point parallels Theorem 3, which the draft states for
P2. Convergence is not established there, so this module reports the
residual trajectory rather than asserting it.

Requires: numpy, scipy, and `algorithm1_p1.py` in the same directory.

Reference: Chow (2026), Section 3.1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq, minimize

from algorithm1_p1 import (
    P1Instance,
    _xlogx_minus,
    buyer_loading,
    toy_instance,
)

_EXP_CAP = 700.0     # exp() overflow guard
_TINY = 1e-300


# ----------------------------------------------------------------------------
# Step 3: seller step, one operator at a time
# ----------------------------------------------------------------------------

def _operator_allocation(
    g: np.ndarray,
    Cmax: np.ndarray,
    F: float,
    c_idle: float,
    alpha_S: float,
) -> Tuple[np.ndarray, float, np.ndarray, float]:
    """Solve one operator's capped entropic allocation.

    With margins g_a = tau0_ma L_ma - c_ma, budget multiplier v and cap
    multipliers pi_a >= 0, stationarity gives

        y_a  = exp( aS ( g_a - pi_a - v ) ) ,      y_0 = exp( -aS ( c_idle + v ) )

    and complementary slackness pi_a > 0 only where y_a = Cmax_a. Eliminating
    pi leaves the capped form

        y_a(v) = min( Cmax_a , exp( aS ( g_a - v ) ) ) ,
        pi_a   = max( 0 , g_a - v - ln(Cmax_a) / aS ) .

    The budget h(v) = sum_a y_a(v) + y_0(v) - F is continuous and strictly
    decreasing in v (the idle term is strictly decreasing and uncapped), so the
    root is unique and found by bisection. Returns (y, y_0, pi, v).
    """

    def total(v: float) -> float:
        expo = np.minimum(alpha_S * (g - v), np.log(np.maximum(Cmax, _TINY)))
        y = np.exp(np.clip(expo, -_EXP_CAP, _EXP_CAP))
        y0 = np.exp(np.clip(-alpha_S * (c_idle + v), -_EXP_CAP, _EXP_CAP))
        return float(y.sum() + y0)

    def h(v: float) -> float:
        return total(v) - F

    # Uncapped closed form (Eq. 32) is an upper bracket: capping only removes
    # capacity, so the budget-clearing v with caps is no larger.
    ex = np.concatenate([alpha_S * g, [-alpha_S * c_idle]])
    v_hi = (np.max(ex) + np.log(np.sum(np.exp(ex - np.max(ex)))) - np.log(F)) / alpha_S
    step = 1.0
    while h(v_hi) > 0.0:                     # numerical safety; normally false
        v_hi += step
        step *= 2.0

    v_lo, step = v_hi - 1.0, 1.0
    while h(v_lo) < 0.0:
        step *= 2.0
        v_lo = v_hi - step
        if step > 1e6:
            raise RuntimeError("Seller step: could not bracket the budget multiplier.")

    v = brentq(h, v_lo, v_hi, xtol=1e-14, rtol=1e-15, maxiter=200)

    expo = np.minimum(alpha_S * (g - v), np.log(np.maximum(Cmax, _TINY)))
    y = np.exp(np.clip(expo, -_EXP_CAP, _EXP_CAP))
    y0 = np.exp(np.clip(-alpha_S * (c_idle + v), -_EXP_CAP, _EXP_CAP))

    # Restore exact budget feasibility against bisection round-off, absorbing
    # the (machine-epsilon) drift into the uncapped idle margin.
    y0 = max(F - y.sum(), _TINY)

    pi = np.maximum(0.0, g - v - np.log(np.maximum(Cmax, _TINY)) / alpha_S)
    pi[y < Cmax - 1e-9 * np.maximum(1.0, Cmax)] = 0.0
    return y, float(y0), pi, float(v)


def seller_step(
    inst: P1Instance, tau0: np.ndarray, L: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step 3. Returns (y_tilde, y_idle_tilde, pi, v), one block per operator."""
    g = tau0 * L - inst.c                         # revenue net of provision cost
    y = np.zeros(inst.n_links)
    pi = np.zeros(inst.n_links)
    y_idle = np.zeros(inst.n_op)
    v = np.zeros(inst.n_op)
    for m in range(inst.n_op):
        idx = np.flatnonzero(inst.link_owner == m)
        y[idx], y0, pi[idx], v[m] = _operator_allocation(
            g[idx], inst.Cmax[idx], float(inst.F[m]),
            float(inst.c_idle[m]), inst.alpha_S,
        )
        y_idle[m] = y0
    return y, y_idle, pi, v


# ----------------------------------------------------------------------------
# Step 2: buyer step
# ----------------------------------------------------------------------------

def buyer_step(
    inst: P1Instance,
    tau0: np.ndarray,
    y_link: np.ndarray,
    sigma_init: Optional[np.ndarray] = None,
    max_iter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Step 2. Minimize the buyer objective over sigma >= 0 at capacities y^t.

    This is the Lagrangian dual, at capacity vector y^t, of the strictly
    concave buyer program of Eq. (20) under Eqs. (21)-(22), so sigma is the
    multiplier on link market clearing at the administered fare.

    Returns (sigma, x, x_out, f, B, objective).
    """
    n = inst.n_links
    s0 = np.zeros(n) if sigma_init is None else np.maximum(sigma_init, 0.0)

    def fun_grad(s: np.ndarray) -> Tuple[float, np.ndarray]:
        _, _, f, B = buyer_loading(inst, tau0 + s)
        obj = np.sum(inst.D * (B - np.log(inst.D) + 1.0)) / inst.alpha_B + float(s @ y_link)
        return float(obj), y_link - f          # dPsi/dsigma_a = y^t_a - f_a

    res = minimize(
        fun_grad, s0, jac=True, method="L-BFGS-B",
        bounds=[(0.0, None)] * n,
        options={"maxiter": max_iter, "ftol": 1e-16, "gtol": 1e-12, "maxcor": 20},
    )
    sigma = np.maximum(res.x, 0.0)
    x, x_out, f, B = buyer_loading(inst, tau0 + sigma)
    return sigma, x, x_out, f, B, float(res.fun)


# ----------------------------------------------------------------------------
# Residuals
# ----------------------------------------------------------------------------

def load_factors(f: np.ndarray, y: np.ndarray) -> np.ndarray:
    """L_ma = min(1, f_a / y_ma), with L = 1 where capacity has collapsed."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(y > _TINY, f / np.maximum(y, _TINY), 1.0)
    return np.minimum(1.0, ratio)


def fixed_point_residual(y_new: np.ndarray, y_old: np.ndarray) -> float:
    """Relative capacity change ||y~ - y||_inf / ||y||_inf."""
    denom = np.abs(y_old).max()
    if not np.isfinite(denom) or denom <= 0:
        denom = 1.0
    return float(np.abs(y_new - y_old).max() / denom)


def natural_residual_fixed_price(
    inst: P1Instance,
    sigma: np.ndarray,
    pi: np.ndarray,
    f: np.ndarray,
    y: np.ndarray,
) -> float:
    """Eq. (38) with sigma in place of tau.

    Diagnostic only. Both terms are driven to zero *within* each step by
    construction -- the buyer step enforces 0 <= sigma _|_ (y - f) >= 0 exactly
    at its own y^t, and the seller step enforces 0 <= pi _|_ (Cmax - y) >= 0 --
    so this measures inconsistency between the two blocks, not distance to the
    fixed point. Use `fixed_point_residual` for the stopping test.
    """
    r_clear = np.minimum(sigma, y - f)
    r_cap = np.minimum(pi, inst.Cmax - y)
    scale = f.max()
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return float(max(np.abs(r_clear).max(), np.abs(r_cap).max()) / scale)


# ----------------------------------------------------------------------------
# Solution container
# ----------------------------------------------------------------------------

@dataclass
class P1FixedPriceSolution:
    sigma: np.ndarray          # (A_own,) rationing prices sigma†
    pi: np.ndarray             # (A_own,) capacity rents pi†
    u: np.ndarray              # (Q,) u†, Eq. (29) at the effective price
    v: np.ndarray              # (M,) budget multipliers from the seller step
    u_tilde: np.ndarray        # (Q,) recentered u_i
    v_tilde: np.ndarray        # (M,) recentered v_m
    x: np.ndarray              # (K,) path flows
    x_out: np.ndarray          # (Q,) outside-path flows
    f: np.ndarray              # (A_own,) link loads
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

    def summary(self, inst: P1Instance, tau0: np.ndarray) -> str:
        names = inst.link_names or [f"a{a}" for a in range(inst.n_links)]
        ln = [
            f"Algorithm 1'  ({'converged' if self.converged else 'NOT converged'})",
            f"  MSA iterations   : {self.n_iter}",
            f"  wall time        : {1e3 * self.wall_time:.1f} ms",
            f"  fixed-pt residual: {self.residual:.3e}",
            f"  natural residual : {self.residual_natural:.3e}",
            f"  gross surplus    : {self.gross_surplus:.2f}",
            "",
            "  link     tau0    sigma†   tau_eff     pi†      f_a       y_ma      L†     f-y",
        ]
        for a in range(inst.n_links):
            ln.append(
                f"  {names[a]:<7}{tau0[a]:7.3f} {self.sigma[a]:8.4f} "
                f"{self.tau_eff[a]:9.4f} {self.pi[a]:7.4f} {self.f[a]:9.3f} "
                f"{self.y[a]:9.3f} {self.L[a]:7.4f} {self.excess_demand[a]:8.3f}"
            )
        ln += ["", "  segment      u†        u~           x_i0"]
        snames = inst.segment_names or [f"i{i}" for i in range(inst.n_seg)]
        for i in range(inst.n_seg):
            ln.append(
                f"  {snames[i]:<10}{self.u[i]:9.4f} {self.u_tilde[i]:10.4f} "
                f"{self.x_out[i]:14.3f}"
            )
        ln += ["", "  operator     v†        y_m0     deployed"]
        onames = inst.operator_names or [f"m{m}" for m in range(inst.n_op)]
        for m in range(inst.n_op):
            ln.append(
                f"  {onames[m]:<10}{self.v[m]:9.4f} {self.y_idle[m]:9.3f} "
                f"{100 * (1 - self.y_idle[m] / inst.F[m]):9.2f}%"
            )
        return "\n".join(ln)


# ----------------------------------------------------------------------------
# P1 fixed-price solver
# ----------------------------------------------------------------------------

def solve_p1_fixed_price(
    inst: P1Instance,
    tau0: np.ndarray,
    tol: float = 1e-7,
    max_iter: int = 500,
    gamma_exponent: float = 0.6,
    residual: str = "fixed_point",
    y_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> P1FixedPriceSolution:
    """P1 fixed-price solver.

    Parameters
    ----------
    tau0 : (A_own,) observed / administered fares, tau0 >= 0.
    tol : stopping tolerance epsilon on the residual.
    max_iter : cap on MSA iterations.
    gamma_exponent : the 0.6 in gamma_t = t^-gamma_exponent. Any value in
        (0.5, 1] gives sum gamma_t = inf, sum gamma_t^2 < inf.
    residual : "fixed_point" (default) tests ||y~ - y||_inf / ||y||_inf;
        "natural" tests Eq. (38) with sigma in place of tau. See the note in
        `natural_residual_fixed_price` -- step 4 of the draft says "if rho^t <=
        eps" without saying which, and Eq. (38) is not the right test here
        because the market does not clear at tau0.
    y_init : optional warm start on the capacity vector, ordered as the owned
        links followed by the M idle entries.

    Notes
    -----
    Step 4 averages the idle entries y_m0 alongside the deployed y_ma. Both y^t
    and y~^t satisfy the budget equality and the caps, so the convex
    combination does too -- this is the feasibility claim at the end of
    Theorem 3, and it fails if the idle entries are left un-averaged.
    """
    tau0 = np.asarray(tau0, dtype=float)
    if tau0.size != inst.n_links:
        raise ValueError("tau0 must have one entry per owned link.")
    if np.any(tau0 < 0):
        raise ValueError("Administered fares must satisfy tau0 >= 0.")
    if residual not in ("fixed_point", "natural"):
        raise ValueError("residual must be 'fixed_point' or 'natural'.")

    t_start = time.perf_counter()
    history: Dict[str, List[float]] = {
        "residual": [], "residual_natural": [], "gamma": [], "buyer_obj": [],
    }

    # --- Step 1. Initialize: seller step at L == 1 --------------------------
    if y_init is None:
        y, y_idle, pi, v = seller_step(inst, tau0, np.ones(inst.n_links))
    else:
        y_init = np.asarray(y_init, dtype=float)
        y, y_idle = y_init[: inst.n_links].copy(), y_init[inst.n_links:].copy()
        pi, v = np.zeros(inst.n_links), np.zeros(inst.n_op)

    sigma = np.zeros(inst.n_links)
    converged = False
    t = 0
    x = x_out = f = B = None

    for t in range(1, max_iter + 1):
        # --- Step 2. Buyer step at the current capacities -------------------
        sigma, x, x_out, f, B, buyer_obj = buyer_step(inst, tau0, y, sigma_init=sigma)

        # --- Step 3. Seller step at the induced load factors ----------------
        L = load_factors(f, y)
        y_new, y_idle_new, pi, v = seller_step(inst, tau0, L)

        rho_fp = fixed_point_residual(y_new, y)
        rho_nat = natural_residual_fixed_price(inst, sigma, pi, f, y)
        rho = rho_fp if residual == "fixed_point" else rho_nat

        history["residual"].append(rho_fp)
        history["residual_natural"].append(rho_nat)
        history["buyer_obj"].append(buyer_obj)

        if verbose:
            print(f"  t={t:4d}  rho_fp={rho_fp:.3e}  rho_nat={rho_nat:.3e}  "
                  f"Psi={buyer_obj:12.4f}")

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

    # Final buyer response at the terminal capacities, so that x, f, sigma and
    # u are mutually consistent with the reported y.
    sigma, x, x_out, f, B, _ = buyer_step(inst, tau0, y, sigma_init=sigma)
    L = load_factors(f, y)

    u = (B - np.log(inst.D)) / inst.alpha_B
    u_tilde = B / inst.alpha_B
    v_tilde = v + np.log(inst.F) / inst.alpha_S

    gross = (
        float(inst.Phi @ x) - float(inst.c @ y) - float(inst.c_idle @ y_idle)
    )

    return P1FixedPriceSolution(
        sigma=sigma, pi=pi, u=u, v=v, u_tilde=u_tilde, v_tilde=v_tilde,
        x=x, x_out=x_out, f=f, y=y, y_idle=y_idle, L=L,
        tau_eff=tau0 + sigma, excess_demand=f - y,
        gross_surplus=gross,
        residual=history["residual"][-1] if history["residual"] else np.nan,
        residual_natural=history["residual_natural"][-1] if history["residual_natural"] else np.nan,
        n_iter=t, wall_time=time.perf_counter() - t_start,
        converged=converged, history=history,
    )


# ----------------------------------------------------------------------------
# Toy network of Fig. 1
# ----------------------------------------------------------------------------

def toy_tau0() -> np.ndarray:
    """Observed (administered) fares tau0 for l1..l4 from Fig. 1."""
    return np.array([2.75, 2.75, 3.50, 6.00])


if __name__ == "__main__":
    inst = toy_instance()
    tau0 = toy_tau0()
    sol = solve_p1_fixed_price(inst, tau0, tol=1e-7, verbose=False)
    print(sol.summary(inst, tau0))

    # Pricing wedge against the clearing solution of Algorithm 1, Remark 5.
    from algorithm1_p1 import solve_p1

    star = solve_p1(inst)
    print("\n  Pricing wedge  Delta_ma = tau0_ma - tau*_ma")
    names = inst.link_names or [f"a{a}" for a in range(inst.n_links)]
    for a in range(inst.n_links):
        d = tau0[a] - star.tau[a]
        tag = "underpriced" if d < 0 else "overpriced "
        print(f"  {names[a]:<5}{d:9.4f}   {tag}   sigma†={sol.sigma[a]:.4f}")
    print(f"\n  Welfare cost of the observed fares: "
          f"{star.gross_surplus - sol.gross_surplus:.2f}")

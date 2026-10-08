# Paper ↔ code map

Draft: *Inverse Aggregate Network Assignment Game* (Chow, 2026), v3.

The working tree used an older algorithm numbering. The paper was renumbered
during revision, so the modules here are renamed to match the draft. This file
records the correspondence and the three places where code and draft do not
line up exactly.

## The four algorithms in the draft

| Draft | Title in the draft | Module here | Entry point | Was called |
|---|---|---|---|---|
| **Algorithm 1** | L-BFGS-B for dual solution of the path-flow formulation P1 | `algorithm1_p1.py` | `solve_p1` | same |
| **Algorithm 2** | L-BFGS-B for dual solution of the PURC-based link formulation P2 | `algorithm2_p2.py` | `solve_p2` | `algorithm3_p2.py` |
| **Algorithm 2′** | Solution to P2 with fixed prices τ⁰ | `algorithm2prime_p2.py` | `solve_p2_fixed_price` | `algorithm3prime_p2.py` |
| **Algorithm 3** | Solution to IO1 using cycle projection with inactive cell rows | four modules, below | `algorithm3.estimate_case` → `algorithm3_refit.refit` | `oneshot_io2.py` + others |

## Algorithm 3 is not a single file

Its steps are split across four modules, in draft step order:

| Draft step | What it does | Module / function |
|---|---|---|
| 1 Preprocess | partition A⁺/A⁰, fix π, load factors | `algorithm3.estimate_case` |
| 2 Seller block | the QP in (c, c_idle, π, α_S, β) | `algorithm3_blocks.seller_block` |
| 3 Support of σ | binding set S from f⁰ ≥ y⁰(1−ε_b) | `algorithm3.estimate_case` |
| 4 Cycle projection | spanning forest, cycle rows **C** | `algorithm3_blocks.cycle_basis`, `build_projection` |
| 4 Inactive cell rows | the **C⁰** block, w_ia = 1_a − λ_ia | `algorithm3_inactive.mixed_rows`, `build_blocks` |
| 5 Identification test | rank_θ vs rank((I − P_σ)**CZ**) | `algorithm3.estimate_case` |
| 6 Buyer block | the convex program with the ϑ_σ ridge and the ϑ_V squared hinge | `algorithm3_refit.refit` |
| 7 Fixed-price allocation | Algorithm 2′ at the estimated primitives | `algorithm2prime_p2.solve_p2_fixed_price` |
| 8 Support check | ψ_ia > 0 reported on R and on V∖R | `run_steps_7_8.py` |
| — τ\*, wedge, DWL | the clearing solve and the welfare metrics | `run_step_9.py` |

`algorithm3_refit.SCHEDULE` holds the per-rate ϑ_V used in the paper:
`{0.1: 1.0, 0.2: 0.1, 0.3: 0.1, 0.4: 0.1, 0.5: 0.03}`.

## Two modules that are NOT paper algorithms

Both are kept because the paper's algorithms import from them. Neither
corresponds to anything in the draft, and neither should be read as one.

- **`algorithm3_blocks.solve_io2`** — the *superseded* outer-loop estimator
  (σ damped by MSA across passes). The draft's Algorithm 3 is one-shot: σ's
  support is read off the data, so no outer loop is needed. Only this module's
  components (`IO2Data`, `seller_block`, `cycle_basis`, `build_projection`)
  are used by Algorithm 3. Do not run `solve_io2` expecting Algorithm 3.
- **`p1_fixed_price.py`** — the fixed-price solver for P1, an earlier
  "Algorithm 1′" that the current draft does not contain.
  `algorithm2prime_p2.py` imports two helpers from it.

The old `algorithm2_io1.py` (an inverse solver for P1 with explicit node
potentials) is **not** included: the draft has no such algorithm.

## Where code and draft do not match

Everything substantive reproduces — `verify_toy.py` checks 13 reported
quantities and all 13 agree, several to the last printed digit. Two reported
*iteration counts* do not:

| Quantity | Draft | Code |
|---|---|---|
| Algorithm 1, iterations at ε = 1e−7 | 19 | **18** |
| Algorithm 2′, iterations | 20 | **12** at tol 1e−7 |

Algorithm 1's looks like an off-by-one in the counting convention; every value
at termination matches. Algorithm 2′'s converged point is identical (gross
surplus agrees to four decimals), but no round tolerance reproduces exactly 20
— 1e−8 gives 16 and 1e−9 gives 22 — so the draft's count came from a different
tolerance or residual convention than the one recorded. Worth reconciling
before the numbers are quoted.

Wall times in the draft (38.5 ms, 6.2 ms, 37.2 ms) are machine-dependent and
are not checked here.

## 2026 revision

The draft's Algorithm 3 is now the single buyer QP: node potentials are kept, observed zero cells are imposed as g_ia ≤ 0, and σ is supported on 𝒮 with 0 ≤ σ ≤ κ.
- **Where it is implemented:** `run_qp_sweep.py` (function `buyer_qp`), shipped in the `qp_sweep_2026` folder.
- **What it replaces:** the cycle-projection modules (`algorithm3*.py`). Those remain here only because `run_qp_sweep.py` uses `algorithm3.estimate_case` for the sampling design and the seller block.

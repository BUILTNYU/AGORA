# Paper ↔ code map

Draft: *Inverse Aggregate Network Assignment Game* (Chow, 2026).

The working tree used an older algorithm numbering, so modules were renamed to match the draft. This file records the correspondence, the modules that are not paper algorithms, and the places where code and draft do not line up exactly.

The release is called **AGORA** and has two codes: **AGORA-F** (forward: Algorithms 1, 2 and 2′) and **AGORA-I** (inverse: Algorithm 3).

## The algorithms in the draft

| Draft | Title in the draft | Code | Module | Entry point |
|---|---|---|---|---|
| **Algorithm 1** | L-BFGS-B for dual solution of the path-flow formulation of the aggregate network assignment game P1 | AGORA-F | `code/algorithm1_p1.py` | `solve_p1` |
| **Algorithm 2** | L-BFGS-B for dual solution of P2 (implemented in service form; H = I gives the link form) | AGORA-F | `code/algorithm2_p2.py` | `solve_p2` |
| **Algorithm 2′** | Solution to P2 with fixed prices τ⁰ | AGORA-F | `code/algorithm2prime_p2.py` | `solve_p2_fixed_price` |
| **Algorithm 3** | Solution to IO1 with buyer and seller QPs (the buyer QP keeps the node potentials and imposes the observed zero cells) | AGORA-I | `qp/run_qp_sweep.py` plus helpers in `code/` | `run_qp_sweep.main` |

## Algorithm 3, step by step

| Step (draft title) | What it does | Module / function |
|---|---|---|
| 1 Preprocess | sampling design, observed cells A⁺ / A⁰, rents fixed at zero below the caps | `algorithm3.estimate_case` (design from `lirr_bench.nested_sets`) |
| 2 Seller block | the convex QP in (c, c_idle, π, κ, α_S per group, β), subject to Eqs. (54)–(58); κ here is the track-cap multiplier of the service form | `algorithm3_blocks.seller_block_svc` (called through `seller_block`) |
| 3 Support of the rationing prices | binding set 𝒮 = {f⁰ ≥ y⁰(1 − ε_b)} | `algorithm3.estimate_case` |
| 4 Buyer block | min Σ(q − g)² + ϑ_σ‖σ‖² s.t. g_ia ≤ 0 on observed zero cells, 0 ≤ σ ≤ κ on 𝒮, potentials μ kept | `run_qp_sweep.buyer_qp`, design matrix `run_qp_sweep.build_design` |
| 5 Fixed-price allocation | Algorithm 2′ at the estimated primitives, then the support check ψ_ia > 0 | `algorithm2prime_p2.solve_p2_fixed_price`; check in `run_qp_sweep.main` |
| 6 Clearing prices and wedge | τ\*, the wedge τ⁰ − τ\*, G(τ⁰), G(τ\*) and DWL | `algorithm2_p2.solve_p2`, called in `run_qp_sweep.main` |

`algorithm3.estimate_case` also runs the old cycle-projection least squares. `run_qp_sweep.py` uses only its steps 1–3 outputs (`data`, `S`, `c_hat`, `c_idle_hat`, `aS`) and discards that θ and σ.

Defaults used in the paper (Section 4.2.2): ϑ_σ = 1e-3 (`--lambda-sigma`), ε_b = 1e-5, ε = 1e-6, κ = 1 × max τ⁰ (`--sigma-cap`), Algorithm 2′ with 40 iterations, and N_clear = 20,000 L-BFGS-B iterations for the clearing solve.

## Ablation arms (`qp/run_ablation.bat`)

| Arm | Name in Table 3 of the draft | Flags | Output |
|---|---|---|---|
| Main | Algorithm 3 | (none) | `qp_sweep.csv` (from `run_qp_sweep.bat`) |
| A: no zero cells | No inactive | `--zero-cells none` | `ablation_nozero.csv` |
| B: **Projection (R only)** | Cycle projections | `--zero-set R` | `ablation_Ronly.csv` |
| C: no bound on σ | (not reported) | `--sigma-cap 0` | `ablation_nocap.csv` |
| D: σ = 0 | No σ | `--no-sigma` | `ablation_nosigma.csv` |
| E–G: squared hinge | (not reported) | `--zero-cells hinge --rho-v 0.1 / 1 / 10` | `robust_hinge*.csv` |

Arm B imposes only the zero cells that cycle projection can reach (the set R from `algorithm3_refit.reachable_mask`). With equalities only, the projection's QR form equals the potentials QP, so arm B is the cycle-projection estimator in QP form.

## The cycle-projection modules: the "Projection (R only)" arm

`code/algorithm3.py`, `algorithm3_blocks.py`, `algorithm3_inactive.py` and `algorithm3_refit.py` are the earlier implementation of Algorithm 3 (cycle projection with inactive-cell rows and a hinge refit). They are **not** the paper's Algorithm 3. They stay in the release for two reasons:

- Algorithm 3 imports its preprocessing, seller block and support step from them (table above).
- They are the original implementation of the "Projection (R only)" ablation arm.

| Old step | Module / function |
|---|---|
| Cycle projection: spanning forest, cycle rows **C** | `algorithm3_blocks.cycle_basis`, `build_projection` |
| Inactive cell rows **C⁰** | `algorithm3_inactive.mixed_rows`, `build_blocks` |
| Buyer refit with the ϑ_σ ridge and the ϑ_V squared hinge | `algorithm3_refit.refit` (per-rate ϑ_V in `SCHEDULE`) |
| Forward solve and support check | `run_steps_7_8.py` |
| Clearing solve, wedge, DWL | `run_step_9.py` |

## Modules that are not paper algorithms

- **`algorithm3_blocks.solve_io2`**: the superseded outer-loop estimator (σ damped by MSA across passes). Do not run it expecting Algorithm 3.
- **`p1_fixed_price.py`**: a fixed-price solver for P1, an earlier "Algorithm 1′" that the draft does not contain. `algorithm2prime_p2.py` imports two helpers from it.
- **`algorithm3_blocks.seller_block_links`**: the link-level seller QP (H = I, one supply scale), kept as the special case of `seller_block_svc`.

The old `algorithm2_io1.py` (an inverse solver for P1 with explicit node potentials) is not included: the draft has no such algorithm.

## Where code and draft do not match

Everything substantive on the toy network reproduces: `verify_toy.py` checks 13 reported quantities and all 13 agree. Two reported *iteration counts* do not:

| Quantity | Draft | Code |
|---|---|---|
| Algorithm 1, iterations at ε = 1e−7 | 19 | **18** |
| Algorithm 2′, iterations | 20 | **12** at tol 1e−7 |

Algorithm 1's looks like an off-by-one in the counting convention; every value at termination matches. Algorithm 2′'s converged point is identical (gross surplus agrees to four decimals), but no round tolerance reproduces exactly 20: 1e−8 gives 16 and 1e−9 gives 22.

Wall times in the draft are machine-dependent and are not checked here.

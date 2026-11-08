# AGORA: Aggregate Game Of Route Assignment

AGORA is the reference implementation for Chow (2026), *Inverse Aggregate Network Assignment Game*. In the game, travelers (buyers) assign their trips over a network, and operators (sellers) choose capacity and prices on it. AGORA has two codes:

| Code | Paper | What it does | Where |
|---|---|---|---|
| **AGORA-F** (Forward) | Algorithms 1, 2 and 2′ | Solves the game for flows, capacities and prices. Algorithm 1 is the L-BFGS-B dual solution of the path-flow formulation P1. Algorithm 2 is the L-BFGS-B dual solution of the PURC-based link formulation P2, which gives the clearing prices τ*. Algorithm 2′ solves P2 at fixed (observed) prices τ⁰. | `code/algorithm1_p1.py`, `code/algorithm2_p2.py`, `code/algorithm2prime_p2.py` |
| **AGORA-I** (Inverse) | Algorithm 3 | Solves the inverse problem IO1: estimates the model primitives from sampled flows, deployed capacities and observed fares with a seller QP and a buyer QP, then calls AGORA-F (Algorithm 2′ at τ⁰, and Algorithm 2 for τ* and the wedge τ⁰ − τ*). | `qp/run_qp_sweep.py`, which uses the sampling design and seller block in `code/algorithm3*.py` |

The LIRR instance now uses **service-based capacity**: LIRR seats are deployed on train services, and the instance has **two supply scales**. Every LIRR result must be rerun on this instance. It supersedes `angag_release_2026.zip` (the 1.6× link-capacity instance) and `angag_release.zip`.

```
code/       AGORA-F (Algorithms 1, 2, 2'); the seller block and the cycle-projection arm used by AGORA-I (algorithm3*.py);
            the LIRR harness (lirr_bench.py), verify_toy.py
qp/         AGORA-I (Algorithm 3, the single buyer QP), the ablation driver and the summaries; see qp/README.md
data/toy/   the three-node toy network of Section 4.1 (unchanged)
data/lirr/  the LIRR access network instance of Section 4.2 (rebuilt)
pipeline/   the instance-construction pipeline (see pipeline/README.md)
```

`MAPPING.md` maps the paper's algorithms and the steps of Algorithm 3 to modules.

Requirements: Python 3.10+, numpy, scipy, pandas, cvxpy and clarabel (`pip install -r requirements.txt`).

## What changed from `angag_release_2026`

### 1. Service-based capacity (model change; Section 3)

`P2Instance` takes an optional link–service incidence `H` (A_own × R), with each service's operator and cap. Capacity is chosen on services, and the capacity of link a is the sum of the seats of the services that use it:

    y_a = Σ_r H_ar Y_r

Each service's margin is the sum of its link margins: ω_r = Σ_a H_ar (τ_a − c_a − κ_a) − π_r. Shared links carry a track cap, (HY)_a ≤ C^max_a, with multiplier κ_a.

The service form is implemented throughout: in AGORA-F (Algorithm 2's dual, and Algorithm 2′'s seller step, with an inner dual over κ when there are track caps) and in AGORA-I's seller block (step 2 of Algorithm 3). With `H` omitted, H = I and the code is the link model. The toy network uses H = I, and `verify_toy.py` reproduces 13 of 13 values.

**Why.** In the link model the operator ranks links by their own fare margin. A short trunk link shared by every Main Line rider then earns a tiny link fare, so it is starved of capacity. At the timetabled budget this rationed the trunk and cut rail ridership to 40k. The 1.6× budget of `angag_release_2026` papered over this.

### 2. Two supply scales (model change)

α_S can be given per operator (α_S,m), and estimation groups operators that share a scale (`aS_group`).

| Operator | α_S | Estimation group | Role |
|---|---|---|---|
| LIRR | 0.25 | 0 | Fitted so that seats deployed by branch match the timetable |
| LOCAL | 4 | 1 | As in the previous instance |
| PARK | 4 | 1 | Shares LOCAL's scale |

- **Why one scale fails.** A single α_S cannot fit both rail and feeders. At α_S = 0.25, LOCAL spreads its budget almost evenly over 2,013 feeder links, and subway egress capacity falls from 25.6k to 2.1k. At α_S ≥ 2, the rail branch fit breaks down: Greenport and Montauk get no seats.
- **Why PARK shares LOCAL's scale.** 94 of PARK's 97 lots sit at their cap, so PARK carries almost no information about its own scale.

AGORA-I's seller block estimates one α_S per group and is still a convex QP. At full observation it recovers (0.25, 4.00) exactly when ρ_π = 1e-3, and (0.25, 3.98) at the default ρ_π = 1.

### 3. Supply data

| | `angag_release_2026` | this instance |
|---|---|---|
| Operators | LIRR (rail + parking), LOCAL | LIRR rail, LOCAL feeders, PARK lots (own budget) |
| LIRR capacity choice | per link | 481 services (origin station × terminal) + 3 single-link services |
| F_LIRR | 2,295,724 (1.6 × calibrated) | **179,520 seats per AM period**, from the timetable: Σ over branches of peak tph × cars × 110 × 3.2 h |
| F_PARK | (inside LIRR) | 0.995 × 62,640 spaces |
| F_LOCAL | 88,948 | 88,948 |
| Rail seat cost | $0.16 / seat-mi (assumed) | **$0.254 / seat-mi**, from NTD 2023 LIRR (ID 20100): $27.90 opex per vehicle revenue mile, 110 seats per car |
| Track caps | none | (HY)_a ≤ C^max_a on links shared by several services |
| α_S | 4 | 0.25 (LIRR); 4 (LOCAL, PARK) |
| c_idle | (0.27, 0.72) | (0.27, 0.72, 0.27), from the `c_idle` column of `operator_budgets.csv` |

New data files:
- `services.csv`: service id, operator, links and cap.
- `services_at_tau0.csv`: deployed seats per service at the ground truth.
- `operator_budgets.csv`: now carries the columns F, c_idle, alpha_S and aS_group.

### 4. Recalibrated station constants

The station constants are recalibrated from the base with three outer passes of `fit_station_ascs_sigma.py`, with σ taken from the latest Algorithm 2′ solve. Station-boarding wMAPE went 0.172 → 0.156 → 0.144 → 0.134.

### 5. Fixes carried over from `angag_release_2026`

- Fares are the published MTA peak one-way schedule (4 January 2026), applied as a two-part tariff: $8.6436 on terminal-entering links + $0.23340/mi.
- Welfare truth uses the generator's φ (`inst.phi`).
- `run_step_9.py` now keeps the instance's services when it builds the truth instance; it previously rebuilt a link-level instance.

## The instance at a glance

| | `angag_release_2026` | this instance |
|---|---|---|
| Rail boardings (target 72,130) | 71,463 | 71,930 |
| Rail share (target 28.35%) | 28.1% | 28.3% |
| Station fit wMAPE / correlation | 0.140 / 0.987 | 0.134 / 0.967 |
| Seats by branch vs timetable (correlation) | — | 0.90 |
| Cells with positive flow | 4,684 | 4,673 |
| Binding links (σ > 0) | 61 (16 transit, 44 park, 1 rail) | 59 (15 transit, 41 park, 3 rail) |
| max σ / max τ⁰ | 1.24 / 9.85 | 2.88 / 9.85 |
| Rail binding | — | Ronkonkoma terminal link (σ = 2.88), West Hempstead, Ronkonkoma S095→S094 |
| drive_constant (7-feature truth) | 5.876 | 5.536 |
| G(τ⁰) / G(τ*) / DWL, generator utility | 1,140,863 / 2,322,782 / 1,181,919 | 894,503 / 1,961,021 / 1,066,518 $/hr |

Unchanged: the network and demand (897 nodes, 8,320 links, 6 destination segments, 29,455 available cells, OD 254,398 AM trips), the fares and tariff, α_B = 1, and the LOCAL budget.

**Oracle clearing prices by fare zone** (ridership-weighted τ* along each station's rail path):

| Zone | Published | Model τ⁰ | Oracle τ* | vs published | Boardings |
|---|---|---|---|---|---|
| 4 | 13.50 | 13.55 | 5.21 | −61% | 31,778 |
| 7 | 15.25 | 15.51 | 8.25 | −46% | 19,296 |
| 9 | 18.25 | 17.94 | 11.06 | −39% | 14,008 |
| 10 | 21.50 | 21.16 | 16.24 | −24% | 6,485 |
| 12 | 25.50 | 25.20 | 16.30 | −36% | 275 |
| 14 | 33.00 | 32.02 | 20.11 | −39% | 88 |

τ* sits 24–61% below the published fares. Clearing fills the seats that run, so τ* is close to the route seat cost plus scarcity. At τ⁰ the model runs seats at 45% average occupancy, and fares recover 94% of seat cost in the AM peak inbound. The wedge τ⁰ − τ* is therefore the gap between average-cost fares and first-best marginal-cost prices.

## Verification

- **Toy.** `verify_toy.py`: 13 of 13 reported values are reproduced. The toy also reproduces with α_S given as an array (0.8, 0.8): τ is identical and W* = −10,038.82.
- **Two scales on the toy.** With α_S = (0.6, 2.5), AGORA-F's Algorithm 2 converges to a natural residual of 8e-10. The seller block at full observation returns (0.6000, 2.4999). Forcing one group returns 1.88.
- **Ground truth is an exact P2 point** (`pipeline/check_truth.py`):
  - **Buyer.** Re-solving the buyer at the stored y gives flows within 5e-8 (relative L1) and σ within 4e-9.
  - **Seller.** Re-solving the seller at the stored load factors gives Y within 8e-7.
  - **Feasibility.** The track caps hold to 5e-4.
- **Binding set.** 𝒮 = {f⁰ ≥ y⁰(1 − 1e-5)} has 59 links and equals {σ > 0}.
- **AGORA-I seller block** (`pipeline/check_seller.py`). At full observation with the c prior at the truth, it returns α_S = (0.25, 3.98) and c_idle exact at ρ_π = 1, and recovers everything exactly at ρ_π = 1e-3.

## Data provenance

| Layer | Status | Source |
|---|---|---|
| Tract centroids | observed | 671 Nassau and Suffolk tracts, 2025 Census Gazetteer |
| Trip productions | observed | LEHD LODES8 `ny_od_main_JT00_2022` |
| Station boarding profile | observed | MTA 2012–2014 LIRR Origin–Destination Report (spatial distribution only) |
| AM peak control total | observed | MTA 2025 Annual Ridership Report |
| Station names, branch topology, fare zones | real | 123 stations |
| LIRR fares | published | MTA peak one-way, effective 4 January 2026 |
| LIRR seat budget | **timetable** | peak tph × cars × 110 seats by branch |
| LIRR seat cost | **NTD 2023** | $27.90 opex per vehicle revenue mile (ID 20100) |
| Station parking prices | assumed | decaying in distance to Manhattan, $0.78–$6.26 |
| α_S (LIRR) | calibrated | seats deployed by branch vs the timetable |
| All flows | model output | not observations |

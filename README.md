Inverse Aggregate Network Assignment Game — code and data
Reference implementation for Chow (2026), Inverse Aggregate Network Assignment
Game, together with the two network instances used in Section 4.
Algorithm 1 — dual solution of the path-flow formulation P1
Algorithm 2 — dual solution of the PURC-based link formulation P2
Algorithm 2′ — P2 at fixed administered prices τ⁰
Algorithm 3 — inverse problem IO1 by cycle projection with inactive cell rows
`MAPPING.md` maps each algorithm in the paper to the module that implements it.
Read it first: Algorithm 3 is spread over four modules, and two modules in
`code/` are not paper algorithms.
Layout
```
code/                 the four algorithms, the LIRR harness, and verify_toy.py
data/toy/             the three-node toy network of Section 4.1
data/lirr/            the LIRR access network instance of Section 4.2
```
Requirements
Python 3.10+, `numpy`, `scipy`, `pandas`. No network access needed.
```
pip install numpy scipy pandas
```
Quick start — the toy network
```
cd code
python verify_toy.py
```
Runs Algorithms 1, 2 and 2′ on the three-node toy instance and checks the 13
quantities reported in Section 4.1: the dual objectives, the gross surpluses,
the operator marginal values, the deployment shares and the welfare gap at the
administered fares. Takes a few seconds.
The toy instance is defined in code (`algorithm1_p1.toy_instance`,
`algorithm2_p2.toy_instance_p2`); `data/toy/toynetwork.csv` is the same instance
as a documented sectioned CSV — scalars, operators, links, paths — and is the
file to read to see the inputs.
The LIRR instance
Every script takes the data directory explicitly and expects to run with it as
the working directory:
```
cd data/lirr
python ../../code/run_steps_7_8.py --data-dir . --out io2_support.csv
python ../../code/run_step_9.py    --data-dir . --in io2_support.csv --out io2_dwl.csv
```
Step 7–8 runs Algorithm 3 for each (seed, sampling rate) case and reports the
support check split two ways — on the reachable inactive cells R that step 6
imposes, and on the rest of the zero set V∖R, which is the genuine
specification check. Step 9 adds the clearing solve, τ*, the pricing wedge and
the deadweight loss. Both take `--resume` and append case by case; a full pass
of 25 cases is roughly 2.5 h plus 45 min on a recent laptop.
`code/lirr_bench.py` builds the instance and holds the sampling design and the
evaluation metrics.
Data provenance
Layer	Status	Source
Tract centroids	observed	671 Nassau and Suffolk tracts, 2025 Census Gazetteer
Trip productions	observed	LEHD LODES8 `ny_od_main_JT00_2022`
Station boarding profile	observed	MTA 2012–2014 LIRR Origin–Destination Report (spatial distribution only)
AM peak control total	observed	MTA 2025 Annual Ridership Report
Station names, branch topology, fare zones	real	123 stations
LIRR fares	stylized	see the note below
Station parking prices	assumed	decaying in distance to Manhattan, $0.78–$6.26
All flows	model output	not observations
The fares are stylized and are not a published MTA schedule for any year.
The peak one-way fare from an origin zone to the City Terminal Zone is set to
$5.75, $8.25, $11.25, $14.00, $16.50, $19.50, $23.00 and $29.00 for zone gaps 0
through 7. This reproduces the monotone step structure of the published LIRR
schedule at a uniformly lower level; it matches neither the schedule effective
20 August 2023 (9.25, 11.25, 13.00, 14.50, 17.50, 20.50, 24.50, 31.75 to zone 1)
nor the one effective 4 January 2026 (7.25, 7.25, 13.50, 15.25, 18.25, 21.50,
25.50, 33.00). Results that compare recovered clearing prices against "existing
fares" are comparisons against this stylized schedule, which is what generated
the instance's flows — not against the published table.
`data/lirr/links_p2.csv` carries the link-level administered price `tau_usd`.
On rail links it is that zone schedule projected onto links by nonnegative
least squares and then reallocated at a uniform $0.36907 per mile, a rate that
preserves the sum of the link tolls over total track miles. The reallocation
avoids the degenerate projection, in which the whole zone fare loads onto the
single link crossing a zone boundary and every other link prices at zero.
The instance-construction pipeline (geography, demand, the reference
equilibrium, the fare projection) is not part of this release; `data/lirr/` is
provided as data.
Files in `data/lirr/`
File	What it is
`links_p2.csv`	all 8,320 links: structure, owner, ℓ_a, φ_a, τ⁰_a, capacities, costs
`nodes.csv`	897 nodes — tract centroids, NYC zones, stations, parking
`segments.csv`, `b_node.csv`	the six destination segments and their net supply
`od_am.csv`, `paths.csv`	the AM OD matrix and the enumerated path set with flows
`supply_at_tau0.csv`	deployed capacity y⁰, observed rents, clearing gaps
`f_seg_link.csv`, `ground_truth_cells.csv`	segment-link cells; the latter is the P2 ground truth the sampler draws from
`ground_truth_links.csv`, `ground_truth_params.csv`	per-link truth and the 104 generating parameters
`sampling_plans.csv`	the nested PPS sampling plans
`operators.csv`, `station_boardings.csv`	operator budgets; boardings by station
`params_true_p2.csv`, `ground_truth_summary.json`	generating parameters and aggregates
Citation
> Chow, J.Y.J. (2026). Inverse Aggregate Network Assignment Game.

# Instance-construction pipeline (2026 fares, service-based capacity)

Run the steps in this folder, in order. Steps 1–6 write their outputs here, next to the scripts.

| Step | Command | Output |
|---|---|---|
| 1 | `python build_network.py` | stage-1 state: geography, access links, paths with published fares, gravity OD, MNL reference flows |
| 2 | `python export.py` | `nodes.csv`, `links.csv`, `od_am.csv`, `paths.csv`, `station_boardings.csv`, … |
| 3 | `python build_io2_package.py` | `links_p2.csv` (two-part rail tariff), `segments.csv`, `b_node.csv`, fare fit files |
| 4 | `python fit_station_ascs.py` | `theta_calibrated.npy`: buyer-only calibration to station boardings |
| 5 | `python apply_calibration.py` | φ in `links_p2.csv`, `params_true_p2.csv`, `link_features.npz` |
| 6 | copy the files below into a data folder with `operator_budgets.csv` | — |
| 7 | `python gen_truth.py <data> ../code <data>/truth.npz 1e-7 500` | Algorithm 2′ ground truth at τ⁰ (about 8–12 min) |
| 8 | `python make_gt_files.py <data> ../code <data>/truth.npz` | `ground_truth_*`, `supply_at_tau0.csv`, `f_seg_link.csv`, `station_boardings.csv` |
| 9 | outer passes | see below |

The files to copy in step 6 are: `links_p2`, `nodes`, `paths`, `od_am`, `segments`, `b_node`, `operators`, `station_boardings`, `params_true_p2`, the fare files and `theta_calibrated.npy`.

**Outer passes (step 9).** Run `SIGMA_FILE=<data>/supply_at_tau0.csv THETA_INIT=<previous theta.npy> python fit_station_ascs_sigma.py`, then repeat steps 5–8. The published instance used two passes after the initial fit. The station-boarding wMAPE went 0.444 → 0.167 → 0.140.

**Fares.** Fares are in `build_network.py` (`FARE_PEAK_2026`). The tariff fit is in `build_io2_package.py`. `data/` holds the tract and LODES OD inputs (`geo.py`, `li_tracts.csv`, `li_to_nyc_od.csv`).

**Services (service-based instance).** After step 5, run `python make_services.py <data>`. It writes `services.csv` with:
- one LIRR service per (origin station, terminal), along that pair's rail path in `paths.csv`;
- one-link services for rail links on no route;
- one service per parking lot (operator PARK) and per LOCAL link.

A service's cap is the smallest link capacity on its route. Rail seat costs in `links_p2.csv` are length × 27.90 / 110 (`lirr_cost_seat_mi` in `build_network.py`, NTD 2023).

**Budgets and scales.** `operator_budgets.csv` has the columns `operator, F, c_idle, alpha_S, aS_group`:

| operator | F | c_idle | alpha_S | aS_group |
|---|---|---|---|---|
| LIRR | 179,520 (timetable seats) | 0.27 | 0.25 | 0 |
| LOCAL | 88,948.18 | 0.72 | 4 | 1 |
| PARK | 62,326.8 (0.995 × spaces) | 0.27 | 4 | 1 |

`code/build_lirr.py` reads all five columns. The `ALPHA_S` environment variable used by `gen_truth.py` is ignored when `alpha_S` is present in the file.

**Choosing α_S for LIRR.** With a common α_S, the seats deployed by branch compare with the timetable as follows (`branch_fit.py`). The 1.0 run used the earlier $0.16/seat-mi cost.

| α_S | 0.1 | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| branch correlation | 0.897 | 0.894 | 0.888 | 0.80 | 0.825 |
| branch log-RMSE | 0.307 | 0.328 | 0.498 | 0.56 | 2.28 |

LOCAL needs a high α_S. At 0.25 its feeder and egress capacity is spread almost evenly over 2,013 links, so LOCAL and PARK keep 4 as their own group.

**Sandbox drivers.** These shell scripts have absolute paths; edit them before use.
- `svc2_run.sh <aLIRR> <aLOCAL> <tag>`: ground truth from a base folder.
- `svc_pass.sh <prev> <new> <a>`: one outer pass, i.e. the σ-aware station-constant fit, then apply, Algorithm 2′ and the ground-truth files.

The published instance ran from the base (pass-3 constants of the 2026 instance) through three passes. Station wMAPE went 0.172 → 0.156 → 0.144 → 0.134.

**Checks.**
- `check_truth.py <data> ../code` re-solves the buyer at the stored y and the seller at the stored load factors.
- `check_seller.py <data> ../code` runs the IO1 seller block at full observation.
- `clear_truth.py <data> ../code <out.npz>` is the oracle clearing solve.

**Superseded (link-capacity instance).** The 2026 link instance used F_LIRR = 1.6 × 1,434,827.52, because at 1.0× the trunk starved:

| F_LIRR scale | 1.0 | 1.25 | 1.4 | 1.6 | 2.0 |
|---|---|---|---|---|---|
| Rail boardings (before the outer passes) | 40,418 | 51,038 | 58,974 | 69,942 | 69,942 |

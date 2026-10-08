"""Write the release-format ground-truth files from an Algorithm 2' solution.

usage: python make_gt_files.py <data_dir> <code_dir> <truth.npz>

Writes into <data_dir>:
  ground_truth_cells.csv, f_seg_link.csv, supply_at_tau0.csv,
  ground_truth_links.csv, ground_truth_params.csv, ground_truth_summary.json,
  links_p2.csv (flow_true column), station_boardings.csv (boardings_ground_truth)
and prints the station-boarding fit against the calibration targets.
"""
import sys, os, json
import numpy as np, pandas as pd

data, code, npz = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(data); sys.path[:0] = [code, data]
import build_lirr

ALPHA_B, C_IDLE = 1.0, np.array([0.27, 0.72])
ALPHA_S = float(os.environ.get("ALPHA_S", "4.0"))
inst, links, segs, aidx, nidx, Z = build_lirr.build(ALPHA_B, ALPHA_S, C_IDLE, with_z=True)
OPS = list(inst.operator_names); C_IDLE = np.asarray(inst.c_idle, float)
AS_OP = np.asarray(inst.aS_op, float)   # per-operator scales (operator_budgets.csv)
z = np.load(npz)
f, y, sigma = z["f_seg"], z["y"], z["sigma"]
own = inst.owned_idx
tau0 = links.tau_usd.to_numpy(float)[own]
F_m = inst.F

# ---- cells
q, a = np.nonzero(f > 1e-9)
cells = pd.DataFrame(dict(segment=segs.segment.to_numpy()[q], link_id=links.link_id.to_numpy()[a],
                          f_true=np.round(f[q, a], 6), link_type=links.link_type.to_numpy()[a],
                          operator=links.operator.to_numpy()[a]))
cells.to_csv("ground_truth_cells.csv", index=False)
cells[["segment", "link_id", "f_true"]].to_csv("f_seg_link.csv", index=False)

# ---- link totals
fa = f.sum(0)
links["flow_true"] = np.round(fa, 5)
links.to_csv("links_p2.csv", index=False)
cap = links.capacity_period.to_numpy(float)[own]
lf_ = np.where(y > 0, fa[own] / np.maximum(y, 1e-12), np.nan)
sup = pd.DataFrame(dict(link_id=links.link_id.to_numpy()[own], link_type=links.link_type.to_numpy()[own],
                        operator=links.operator.to_numpy()[own], tau0=np.round(tau0, 6),
                        y_deployed=np.round(y, 6), sigma=np.round(sigma, 6), Cmax=cap,
                        flow=np.round(fa[own], 6), load_factor=np.round(lf_, 5)))
sup.to_csv("supply_at_tau0.csv", index=False)
if "Y" in z.files:
    Ysv = z["Y"]
    names = inst.svc_names or [f"svc{r}" for r in range(inst.n_svc)]
    svdf = pd.DataFrame(dict(service_id=names, operator=np.array(OPS)[inst.svc_owner],
                             Y_deployed=np.round(Ysv, 6), Cmax=inst.svc_Cmax))
    svdf.to_csv("services_at_tau0.csv", index=False)

g = pd.DataFrame(dict(link_id=links.link_id, from_node=links.from_node, to_node=links.to_node,
                      link_type=links.link_type, operator=links.operator, branch=links.branch,
                      length_mi=links.length_mi, ff_time_min=links.ff_time_min, ell_purc=links.ell,
                      Cmax=np.nan, prov_cost_per_pax=links.prov_cost_per_pax,
                      cap_cost_per_unit=links.cap_cost_per_unit, tau0_administered=links.tau_usd,
                      flow_true_at_tau0=np.round(fa, 5), y_deployed=np.nan, sigma_rationing=np.nan,
                      load_factor=np.nan))
g.loc[own, "Cmax"] = cap
g.loc[own, "y_deployed"] = np.round(y, 6)
g.loc[own, "sigma_rationing"] = np.round(sigma, 6)
g.loc[own, "load_factor"] = np.round(lf_, 5)
g["voc"] = np.round(g.flow_true_at_tau0 / g.Cmax, 5)
g.to_csv("ground_truth_links.csv", index=False)

# ---- parameters (the 7-feature summary the estimator targets, plus scales)
th = pd.read_csv("params_true_p2.csv").set_index("parameter").value
gp = [(f"time_{t}", th[f"time_{t}"], "time", "$/min") for t in
      ["RAIL", "ACCESS_DRIVE", "ACCESS_TRANSIT", "EGRESS", "OUTSIDE"]]
gp += [("nonfare_money", th["nonfare_money"], "money", "$"),
       ("drive_constant", th["drive_constant"], "modal_constant", "$"),
       ("alpha_B", ALPHA_B, "scale", ""), ("alpha_S", float(AS_OP[0]), "scale", ""),
       *[(f"alpha_S_{nm}", float(AS_OP[k]), "scale", "") for k, nm in enumerate(OPS)],
       *[(f"aS_group_{nm}", int(inst.aS_group[k]), "scale_group", "") for k, nm in enumerate(OPS)],
       *[(f"c_idle_{nm}", C_IDLE[k], "idle_cost", "$/unit") for k, nm in enumerate(OPS)],
       *[(f"F_{nm}", float(F_m[k]), "budget", "units") for k, nm in enumerate(OPS)]]
pd.DataFrame(gp, columns=["parameter", "value", "block", "units"]).to_csv("ground_truth_params.csv", index=False)

# ---- station boardings
lt = links.link_type.to_numpy()
ent = np.isin(lt, ["PARK", "ACCESS_TRANSIT"])
bd = pd.Series(fa[ent]).groupby(links.to_node.to_numpy()[ent]).sum()
sb = pd.read_csv("station_boardings.csv")
sb["boardings_ground_truth"] = sb.node_id.map(bd).fillna(0.0).round(2)
sb.to_csv("station_boardings.csv", index=False)
t, p = sb.target_boardings.to_numpy(float), sb.boardings_ground_truth.to_numpy(float)
fit = dict(wmape=float(np.abs(p - t).sum() / t.sum()), corr=float(np.corrcoef(p, t)[0, 1]),
           total=float(p.sum()), target=float(t.sum()))
by_zone = sb.groupby("fare_zone")[["target_boardings", "boardings_ground_truth"]].sum().round(0)

# ---- summary
y_idle = {}
for m, nm in enumerate(OPS):
    y_idle[nm] = round(float(np.atleast_1d(z["y_idle"])[m]) if np.size(z["y_idle"]) == len(F_m)
                       else float(F_m[m] - y[inst.link_owner[own] == m].sum()), 1)
outside = float(fa[lt == "OUTSIDE"].sum())
board = float(fa[ent].sum())
summ = dict(generator="Algorithm 2' (fixed price at tau^0, usage-based revenue)",
            fare_schedule=(str(pd.read_csv("fare_additivity_residual.csv").set_index("quantity").value.get("schedule", "")) if os.path.exists("fare_additivity_residual.csv") else ""),
            alpha_B=ALPHA_B, alpha_S={nm: float(AS_OP[k]) for k, nm in enumerate(OPS)}, c_idle={nm: float(C_IDLE[k]) for k, nm in enumerate(OPS)}, n_services=int(inst.n_svc),
            n_links=int(len(links)), n_owned=int(len(own)), n_segments=int(inst.n_seg),
            n_cells_available=int(inst.avail.sum()), n_cells_positive=int(len(cells)),
            sparsity=round(1 - len(cells) / inst.avail.sum(), 4),
            boardings=round(board, 1), outside=round(outside, 1),
            rail_share=round(board / (board + outside), 4),
            rail_share_target=round(t.sum() / inst.D.sum(), 4),
            station_fit_wmape=round(fit["wmape"], 4), station_fit_corr=round(fit["corr"], 4),
            n_sigma_positive=int((sigma > 1e-9).sum()), sigma_max=round(float(sigma.max()), 4),
            max_capacity_violation=round(float(np.max(fa[own] - y)), 6), idle_stock=y_idle,
            solver_residual=float(z["residual"]) if "residual" in z.files else None)
json.dump(summ, open("ground_truth_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print("\nboardings by fare zone (target vs ground truth):")
print(by_zone.to_string())

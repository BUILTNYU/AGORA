#!/usr/bin/env python3
"""Build the new Table 2 from the QP sweep (main arm) and the ablation arms.

    python summarize_ablation.py <qp_sweep folder>

Reads qp_sweep.csv (main: hard zero cells, kappa = 1) and every
ablation_*.csv / robust_*.csv present, averages over seeds per rate, and writes
  ablation_table.csv       Table 2 layout: R^2 and DWL error (generator truth)
  ablation_long.csv        every metric x arm x rate
Arms that are still running are included with however many seeds are done.
"""
import sys
from pathlib import Path
import pandas as pd

ARMS = [  # (file, label)
    ("qp_sweep.csv", "Algorithm 3 (QP)"),
    ("ablation_nozero.csv", "No zero-cell constraints"),
    ("ablation_Ronly.csv", "Zero cells in R only"),
    ("ablation_nocap.csv", "No bound on sigma"),
    ("ablation_nosigma.csv", "sigma = 0"),
    ("robust_hinge0.1.csv", "Hinge rho_V = 0.1"),
    ("robust_hinge1.csv", "Hinge rho_V = 1"),
    ("robust_hinge10.csv", "Hinge rho_V = 10"),
]
METRICS = [  # (column, label)
    ("r2_all", "R^2"),
    ("wmape_val", "wMAPE validation links"),
    ("DWL_relerr_gen", "DWL error (generator truth)"),
    ("GS_tau0_relerr_gen", "G(tau0) error (generator truth)"),
    ("GS_taustar_relerr_gen", "G(tau*) error (generator truth)"),
    ("wedge_absmean_err_gen", "mean |wedge| error, $ (generator truth)"),
    ("wedge_corr_gen", "wedge corr (generator truth)"),
    ("sigma_corr", "sigma corr"),
    ("sigma_relerr", "sigma relerr"),
    ("t_RAIL", "t_RAIL (true -0.367)"),
    ("money", "money (true -1.00)"),
    ("viol_notimposed", "support viol, cells not imposed"),
    ("DWL_relerr_paper", "DWL error (paper truth, for reference)"),
]


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    frames = []
    for f, lab in ARMS:
        p = d / f
        if p.exists():
            x = pd.read_csv(p)
            if len(x):
                x["arm"] = lab
                frames.append(x)
    if not frames:
        print("no results found in", d)
        return
    df = pd.concat(frames, ignore_index=True)
    df["GS_taustar_relerr_gen"] = (df.GS_taustar_hat - df.GS_taustar_true_gen) / df.GS_taustar_true_gen.abs()
    order = [lab for _, lab in ARMS if lab in set(df.arm)]
    g = df.groupby(["arm", "rate"])
    m = g.mean(numeric_only=True)
    n = g.size().rename("n_seeds")
    rates = sorted(df.rate.unique())

    rows = []
    for col, lab in METRICS:
        if col not in m.columns:
            continue
        for arm in order:
            r = {"metric": lab, "arm": arm}
            for rt in rates:
                r[f"{int(round(rt * 100))}%"] = m[col].get((arm, rt), float("nan"))
            rows.append(r)
    long = pd.DataFrame(rows)
    long.to_csv(d / "ablation_long.csv", index=False)

    # Table 2 layout
    t = []
    for rt in rates:
        r = {"rate": f"{int(round(rt * 100))}%"}
        for arm in order:
            r[f"R2 | {arm}"] = m["r2_all"].get((arm, rt), float("nan"))
        for arm in order:
            r[f"DWL err | {arm}"] = m["DWL_relerr_gen"].get((arm, rt), float("nan"))
        for arm in order:
            r[f"G(tau0) err | {arm}"] = m["GS_tau0_relerr_gen"].get((arm, rt), float("nan"))
        for arm in order:
            r[f"G(tau*) err | {arm}"] = m["GS_taustar_relerr_gen"].get((arm, rt), float("nan"))
        t.append(r)
    t = pd.DataFrame(t)
    t.to_csv(d / "ablation_table.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\nseeds finished per arm and rate:")
    print(n.unstack("rate").reindex(order).to_string())
    print()
    print(long.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(f"\nwritten: {d / 'ablation_table.csv'}\n         {d / 'ablation_long.csv'}")


if __name__ == "__main__":
    main()

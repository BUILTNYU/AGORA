#!/usr/bin/env python3
"""Per-rate means (and seed ranges) of the QP sweep on the service-based LIRR instance.

    python summarize_qp_sweep.py qp_sweep.csv

Welfare errors are against the generator truth (*_gen: the utility that
produced the flows, station constants included). Writes <input>_summary.csv.
"""
import sys
from pathlib import Path
import pandas as pd

ROWS = [
    ("t_RAIL", "t_RAIL (true -0.367)"),
    ("money", "money (true -1.00)"),
    ("drive", "drive (7-feature summary constant)"),
    ("alpha_S_LIRR", "alpha_S LIRR (true 0.25)"),
    ("alpha_S_LOCAL", "alpha_S LOCAL/PARK (true 4)"),
    ("sigma_corr", "sigma corr"),
    ("sigma_relerr", "sigma relerr"),
    ("r2_all", "R^2 (all cells)"),
    ("wmape_all", "wMAPE (all)"),
    ("wmape_val", "wMAPE (validation links)"),
    ("GS_tau0_relerr_gen", "G(tau0) relerr"),
    ("GS_taustar_relerr_gen", "G(tau*) relerr"),
    ("DWL_relerr_gen", "DWL relerr"),
    ("wedge_absmean_err_gen", "mean |wedge| error ($)"),
    ("wedge_corr_gen", "wedge corr"),
    ("viol_R", "support viol on R"),
    ("viol_VmR", "support viol on V\\R"),
    ("viol_notimposed", "support viol, cells not imposed"),
    ("fp_converged", "forward solve converged (share)"),
    ("qp_secs", "QP solve time (s)"),
]


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "qp_sweep.csv")
    df = pd.read_csv(path)
    if "GS_taustar_hat" in df and "GS_taustar_true_gen" in df:
        df["GS_taustar_relerr_gen"] = (df.GS_taustar_hat - df.GS_taustar_true_gen) / df.GS_taustar_true_gen.abs()
    g = df.groupby("rate")
    m, lo, hi = g.mean(numeric_only=True), g.min(numeric_only=True), g.max(numeric_only=True)
    out = []
    for col, lab in ROWS:
        if col not in m.columns:
            continue
        r = {"metric": lab}
        for rt in m.index:
            k = f"{int(round(rt * 100))}%"
            r[k] = m.loc[rt, col]
            r[k + " min"] = lo.loc[rt, col]
            r[k + " max"] = hi.loc[rt, col]
        out.append(r)
    res = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    show = res[["metric"] + [c for c in res.columns if c.endswith("%")]]
    print(f"\n{path.name}: seeds per rate {g.size().to_dict()}\n")
    print(show.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    dst = path.with_name(path.stem + "_summary.csv")
    res.to_csv(dst, index=False)
    print(f"\nwritten: {dst}")


if __name__ == "__main__":
    main()

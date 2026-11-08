# Algorithm 3: the single buyer QP (LIRR service instance)

This folder holds the paper's Algorithm 3 and its ablation arms. The scripts use the release root (the parent folder, with `code/` and `data/lirr/`) and write their output here.

Requirements: `pip install -r ../requirements.txt` (cvxpy with the CLARABEL solver is the only addition to numpy/scipy/pandas). `run_qp_sweep.bat` installs cvxpy and clarabel if they are missing. On Windows, put the full path of your Python interpreter in `pyexe.txt` here or in the release root if it is not on PATH.

1. **`run_qp_sweep.bat`** runs Algorithm 3 over 5 seeds × 10–50%.
   - Option [1] is the full grid with the clearing solve (step 6). Option [3] is a single test case: seed 0 at 30%.
   - Output is `qp_sweep.csv` plus `qp_sweep_summary.csv`.
2. **`run_ablation.bat`** runs the ablation arms A–D and the hinge arms E–G, each to its own CSV.
   - Arms can run in parallel windows, e.g. `run_ablation.bat A`.
   - Arm B (`--zero-set R`) is the "Projection (R only)" arm: the QP restricted to the zero cells that cycle projection can reach.
   - At the end, `summarize_ablation.py` builds the ablation table (`ablation_table.csv`, `ablation_long.csv`).

Without the batch files (any OS), from this folder:

```
python run_qp_sweep.py --release .. --out qp_sweep.csv --resume
python summarize_qp_sweep.py qp_sweep.csv
python run_qp_sweep.py --release .. --out ablation_Ronly.csv --zero-set R --resume
python summarize_ablation.py .
```

Every run is resumable: rerun with `--resume` and finished (seed, rate) cases are skipped. The two truth clearing solves are cached in `qp_sweep_truth_cache.npz` on the first run.

**Seller block.** It estimates one α_S per scale group. The output has the columns `alpha_S_LIRR`, `alpha_S_LOCAL` and `alpha_S_PARK`, and `alpha_S` is the LIRR value.

**Welfare truth.** Errors are reported against the generator's utility (the `*_gen` columns): G(τ⁰) = 894,503, G(τ*) = 1,961,021 and DWL = 1,066,518 $/hr. The `*_paper` columns use φ = Zθ_true (7 features) and are kept for comparison only.

**Reference case**: seed 0, 30% sampling.

| Metric | Value |
|---|---|
| t_RAIL / money | +0.003 / −1.119 |
| α_S LIRR / LOCAL-PARK (true 0.25 / 4) | 0.2504 / 3.977 |
| σ corr / relerr | 0.68 / 0.75 (59 links in 𝒮) |
| R² | 0.962 |
| wMAPE, validation links | 0.35 |
| G(τ⁰) error | −33.2% |
| G(τ*) error | −5.5% |
| DWL error | +17.7% |
| wedge corr | 0.72 |
| Run time | 6 min, plus about 18 min once for the two truth clearing solves (cached) |

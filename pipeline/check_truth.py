"""Exactness of a service ground truth: buyer KKT at the stored y, seller fixed
point at the stored load factors. usage: check_truth.py <data_dir> <code_dir>"""
import sys, os, numpy as np
D, C = map(os.path.abspath, sys.argv[1:3]); os.chdir(D); sys.path[:0] = [C, D]
import build_lirr, algorithm2prime_p2 as P
inst, links, *_ = build_lirr.build(1.0, 4.0, np.array([0.27, 0.72]), with_z=True)
z = np.load("truth.npz"); tau0 = z["tau0"]; y = z["y"]; Y = z["Y"]; f = z["f_seg"]; sg = z["sigma"]
u, s2, fp, fl, eta, _ = P.buyer_step(inst, tau0, y, u_init=z["u"], sigma_init=sg, max_iter=4000)
fs = np.zeros_like(f); fs[inst.p_seg, inst.p_link] = fp
print(f"buyer at stored y: rel L1 flow diff {np.abs(fs - f).sum() / f.sum():.2e}  "
      f"max |sigma diff| {np.abs(s2 - sg).max():.2e}  max conservation residual {np.abs(eta).max():.2e}")
fo = f.sum(0)[inst.owned_idx]
L = np.minimum(1.0, fo / np.maximum(y, 1e-300))
Ys, ys, *_ = P.seller_step_svc(inst, tau0, L)
print(f"seller at stored L: rel L1 Y diff {np.abs(Ys - Y).sum() / Y.sum():.2e}  max over-capacity {np.max(fo - y):.2e}")
print(f"track caps: max (HY - Cmax) on shared links {np.max((inst.Hs @ Y - inst.Cmax)[inst.lcap_idx]):.2e}")
S = fo >= y * (1 - 1e-5); print(f"binding set: |S| {S.sum()}  |sigma>0| {(sg > 1e-9).sum()}  S == (sigma>0): {np.array_equal(S, sg > 1e-9)}")

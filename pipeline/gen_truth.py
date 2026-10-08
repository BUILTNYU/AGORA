"""Ground truth at tau0 by Algorithm 2' (release code) at the generator's primitives.
usage: python gen_truth.py <data_dir> <code_dir> <out.npz> [tol] [max_iter]"""
import sys, os, time, numpy as np, pandas as pd
data, code, out = sys.argv[1], sys.argv[2], sys.argv[3]
tol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-7
mi = int(sys.argv[5]) if len(sys.argv) > 5 else 500
os.chdir(data); sys.path[:0] = [code, data]
import build_lirr
from algorithm2prime_p2 import solve_p2_fixed_price
ALPHA_S = float(os.environ.get("ALPHA_S", "4.0"))
inst, links, segs, aidx, nidx, Z = build_lirr.build(1.0, ALPHA_S, np.array([0.27, 0.72]), with_z=True)
print("alpha_S", inst.aS_op, "groups", inst.aS_group, "budgets", inst.F)
tau0 = links.tau_usd.to_numpy(float)[inst.owned_idx]
t0 = time.time()
fp = solve_p2_fixed_price(inst, tau0, tol=tol, max_iter=mi)
print("iters", fp.n_iter, "converged", fp.converged, "residual", fp.residual, f"{time.time()-t0:.0f}s")
np.savez(out, f_seg=fp.f_seg, y=fp.y, sigma=fp.sigma, u=fp.u, y_idle=getattr(fp, "y_idle", np.nan), tau0=tau0, Y=(fp.Y if fp.Y is not None else fp.y), residual=fp.residual)

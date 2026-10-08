"""Seller block at full observation recovers (alpha_S per group, c, c_idle).
usage: check_seller.py <data_dir> <code_dir> [rho_c] [rho_pi]"""
import sys, os, numpy as np
D, C = map(os.path.abspath, sys.argv[1:3]); os.chdir(D); sys.path[:0] = [C, D]
rc = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0; rp = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
import build_lirr, algorithm3_blocks as B
inst, links, *_ = build_lirr.build(1.0, 4.0, np.array([0.27, 0.72]), with_z=True)
z = np.load("truth.npz")
d = B.IO2Data(f_seg=z["f_seg"], y=z["y"], f_link=z["f_seg"].sum(0)[inst.owned_idx], tau0=z["tau0"],
              z=np.zeros((inst.n_seg, inst.n_links, 1)), c_prior=inst.c.copy(),
              c_idle_prior=inst.c_idle.copy(), Y=z["Y"])
c, ci, pi, aS, beta, obj = B.seller_block(inst, d, rc, rp, 1e-6)
print("true  alpha_S", inst.aS_op, " c_idle", inst.c_idle)
print("hat   alpha_S", np.round(aS, 4), " c_idle", np.round(ci, 4))
print(f"c: max |c_hat - c| {np.abs(c - inst.c).max():.3e}  rho_c {rc} rho_pi {rp}")

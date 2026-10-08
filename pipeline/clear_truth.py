"""Oracle clearing solve (Algorithm 2) at the generator primitives; writes tau* per owned link.
usage: ALPHA_S=.. python clear_truth.py <data_dir> <code_dir> <out.npz>"""
import sys, os, time, numpy as np
data, code, out = sys.argv[1:4]
os.chdir(data); sys.path[:0] = [os.path.abspath(code), data]
import build_lirr
from algorithm2_p2 import solve_p2
aS = float(os.environ.get("ALPHA_S", "4.0"))
inst, links, *_ = build_lirr.build(1.0, aS, np.array([0.27, 0.72]), with_z=True)
t0 = time.time(); clr = solve_p2(inst, tol=1e-4, max_iter=20000)
phi = np.where(inst.avail, inst.phi, 0.0)
gss = float((phi * clr.f_seg).sum() - inst.c @ clr.y)
print("clearing converged", clr.converged, f"{time.time()-t0:.0f}s  G(tau*)", round(gss))
np.savez(out, tau=clr.tau, y=clr.y, f_seg=clr.f_seg, GSs=gss)

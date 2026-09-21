#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproduce the toy-network numbers reported in Section 4.1 of the paper.

Runs Algorithm 1 on P1, Algorithm 2 on P2, and Algorithm 2' on P2 at the
administered fares, and checks each reported quantity against the draft.
Needs no data files: the toy instance is defined in code and mirrors
data/toy/toynetwork.csv.

    python verify_toy.py
"""
from __future__ import annotations

import numpy as np

import algorithm1_p1 as A1
import algorithm2_p2 as A2
import algorithm2prime_p2 as A2P

TOL = 1e-7


def check(label, got, want, atol):
    ok = abs(got - want) <= atol
    print(f"  {'OK ' if ok else 'XX '} {label:<44} {got:>12.4f}   paper "
          f"{want:>12.4f}")
    return ok


def main() -> int:
    inst = A1.toy_instance()
    s1 = A1.solve_p1(inst, tol=TOL, verbose=False)
    p2 = A2.toy_instance_p2()
    s2 = A2.solve_p2(p2, tol=TOL, verbose=False)
    s3 = A2P.solve_p2_fixed_price(p2, A2P.toy_tau0(), tol=TOL)

    share = [s1.y[inst.link_owner == m].sum() / inst.F[m] for m in range(2)]
    ok = []
    print("\n  Algorithm 1 on P1")
    ok += [check("dual objective W*", s1.dual_obj, -13391.852, 5e-3),
           check("gross surplus G_P1", s1.gross_surplus, 3454.16, 5e-3),
           check("v_BLUE", s1.v[0], -6.493, 1e-3),
           check("v_RED", s1.v[1], 0.763, 1e-3),
           check("v~_RED / v~_BLUE", s1.v_tilde[1] / s1.v_tilde[0], 3.24, 1e-2),
           check("BLUE deployment share", share[0], 0.850, 1e-3),
           check("RED deployment share", share[1], 0.998, 1e-3),
           check("u~ of OD(1,3)", s1.u_tilde[2], 0.86, 5e-3)]

    print("\n  Algorithm 2 on P2")
    ok += [check("dual objective W*", s2.dual_obj, -10038.82, 5e-2),
           check("iterations", float(s2.n_iter), 21, 0)]

    print("\n  Algorithm 2' on P2 at the administered fares")
    ok += [check("gross surplus G_P2(tau0)", s3.gross_surplus, 3141.93, 5e-3),
           check("welfare gap vs P2 optimum",
                 s2.gross_surplus - s3.gross_surplus, 330.51, 5e-2),
           check("gap as a share of the optimum",
                 (s2.gross_surplus - s3.gross_surplus) / s2.gross_surplus,
                 0.095, 1e-3)]

    print("\n  Iteration counts reported in the draft (see MAPPING.md)")
    print(f"      Algorithm 1  : {s1.n_iter} here, 19 in the draft")
    print(f"      Algorithm 2' : {s3.n_iter} here at tol {TOL:g}, "
          f"20 in the draft")

    n = sum(ok)
    print(f"\n  {n} of {len(ok)} reported values reproduced\n")
    return 0 if n == len(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

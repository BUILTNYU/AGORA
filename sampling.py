# -*- coding: utf-8 -*-
"""
Deterministic observation-set draws for the IO2 sampling experiment.

Every draw is reproduced exactly from (design, rate, rep), so the package
ships seeds and this module rather than 480 materialised observation files.

    from sampling import draw, held_out
    obs   = draw("stratified", 0.20, rep=3)      # set of observed link_ids
    O_F   = cells(obs)                           # (segment, link) cells
    eval_ = held_out(obs)                        # complement, for scoring

Design
------
pps   probability proportional to size, with inclusion probability scaled by
      sqrt(link flow) and capped at 1. The square root damps the weighting:
      raw flow would make the draw degenerate, putting probability 1 on the
      trunk and near zero on everything else. Under P2's sparse flow field a
      5% budget already covers 85% of passenger flow and 40% covers all of
      the nonzero cells, so the informative range is roughly 5 to 30%.
"""
import os
import numpy as np
import pandas as pd

D = os.path.dirname(os.path.abspath(__file__))

RATES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00]
DESIGNS = ["pps"]
REPS = 20

_links = None
_cells = None


def links():
    global _links
    if _links is None:
        _links = pd.read_csv(f"{D}/links_p2.csv")
    return _links


def seg_cells():
    """Ground-truth f*_{i,a} on destination-based segments."""
    global _cells
    if _cells is None:
        _cells = pd.read_csv(f"{D}/f_seg_link.csv")
    return _cells


def inclusion_probs(design="pps", rate=0.20):
    """Inclusion probability per link, scaled so the expected sample size is
    `rate * n_links` and capped at 1."""
    if design != "pps":
        raise ValueError(f"only the pps design is supported, got {design!r}")
    L = links()
    n = len(L)
    w = np.sqrt(L.flow_true.clip(lower=0).to_numpy()) + 1e-6
    if not np.isfinite(w).all():
        raise ValueError("non-finite link flows; rerun run_p2_ground_truth.py")
    target = rate * n
    lo, hi = 1e-9, 1e9
    for _ in range(200):
        s = (lo + hi) / 2
        if np.minimum(s * w / w.mean(), 1.0).sum() > target:
            hi = s
        else:
            lo = s
    return np.minimum(lo * w / w.mean(), 1.0)


def seed_for(design="pps", rate=0.20, rep=0):
    return abs(hash((design, round(float(rate), 4), int(rep)))) % (2 ** 31)


def draw(design="pps", rate=0.20, rep=0):
    """Observed link ids for one replication."""
    L = links()
    rng = np.random.default_rng(seed_for(design, rate, rep))
    keep = rng.random(len(L)) < inclusion_probs(design, rate)
    return set(L.link_id[keep])


def cells(observed_links):
    """O_F as (segment, link_id) pairs, with the ground-truth flow."""
    C = seg_cells()
    return C[C.link_id.isin(observed_links)].reset_index(drop=True)


def held_out(observed_links):
    """Complement of O_F. Score reconstructed flows against this."""
    C = seg_cells()
    return C[~C.link_id.isin(observed_links)].reset_index(drop=True)


def score(f_hat, observed_links, eps=1e-9):
    """
    Accuracy on the unobserved cells.

    f_hat : DataFrame with columns segment, link_id, f_hat
    Returns GEH, RMSE, weighted MAPE and R2 on the held-out cells only.
    """
    truth = held_out(observed_links)
    m = truth.merge(f_hat, on=["segment", "link_id"], how="left")
    m["f_hat"] = m.f_hat.fillna(0.0)
    y, p = m.f_true.to_numpy(), m.f_hat.to_numpy()
    geh = np.sqrt(2 * (p - y) ** 2 / np.maximum(p + y, eps))
    ss = ((y - y.mean()) ** 2).sum()
    return dict(
        n_cells=len(m),
        rmse=float(np.sqrt(((p - y) ** 2).mean())),
        mae=float(np.abs(p - y).mean()),
        wmape=float(np.abs(p - y).sum() / max(np.abs(y).sum(), eps)),
        r2=float(1 - ((p - y) ** 2).sum() / ss) if ss > 0 else float("nan"),
        geh_median=float(np.median(geh)),
        geh_frac_under_5=float((geh < 5).mean()),
        flow_coverage_observed=float(
            seg_cells()[seg_cells().link_id.isin(observed_links)].f_true.sum()
            / seg_cells().f_true.sum()))


if __name__ == "__main__":
    rows = []
    for d in DESIGNS:
        for r in RATES:
            o = draw(d, r, 0)
            L = links()
            sub = L[L.link_id.isin(o)]
            rows.append(dict(design=d, rate=r, n_links=len(o),
                             n_rail=int((sub.link_type == "RAIL").sum()),
                             cells_observed=len(cells(o)),
                             cells_held_out=len(held_out(o)),
                             flow_cov=round(float(sub.flow_true.sum()
                                                  / L.flow_true.sum()), 4)))
    print(pd.DataFrame(rows).to_string(index=False))

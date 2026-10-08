# -*- coding: utf-8 -*-
"""Nested bundle sensor design for the LIRR instance.

A link contributes to IO2 only by closing a cycle in the observed active
subgraph, and a cycle only adds information if it is independent of those
already present, so scattering sensors across unrelated links buys rows
without buying rank.  Instrumenting a whole tract -- its access links, the
parking links they feed, and its outside links -- keeps each alternative-route
comparison intact, which is the variation the rank condition of Algorithm 4
step 4 requires.

Tracts are drawn without replacement with probability proportional to
(access flow)^alpha.  alpha = 1 is strictly volume-proportional and
concentrates on the busiest catchments, making the sampled attributes
collinear; alpha = 0 is uniform.  alpha = 0.5 damps the concentration and is
the default.

`draw_nested` fixes one random ordering of tracts per seed and returns
prefixes of it, so the selection at each rate is a superset of the selection
at every smaller rate: the marginal value of added sensors is measured along a
single deployment path rather than confounded with variation between
independent draws.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

_links = pd.read_csv('links_p2.csv')
_lt = _links.link_type.to_numpy()
_frm = _links.from_node.to_numpy()
_to = _links.to_node.to_numpy()
_flow = _links.flow_true.to_numpy(float) if 'flow_true' in _links else None
if _flow is None:
    _gt = pd.read_csv('ground_truth_links.csv').set_index('link_id')
    _flow = _gt.flow_true_at_tau0.reindex(_links.link_id).to_numpy(float)

_acc = np.isin(_lt, ['ACCESS_DRIVE', 'ACCESS_TRANSIT'])
_out = _lt == 'OUTSIDE'
_park = _lt == 'PARK'
TRACTS = sorted(set(_frm[_acc]))
CENSUS = set(_links.link_id[np.isin(_lt, ['RAIL', 'EGRESS'])])

tract_flow = pd.Series(_flow[_acc]).groupby(_frm[_acc]).sum()

BUNDLE = {}
for _t in TRACTS:
    _ids = set(_links.link_id[(_acc | _out) & (_frm == _t)])
    _pk = set(_to[_acc & (_frm == _t)])
    _ids |= set(_links.link_id[_park & np.isin(_frm, list(_pk))])
    BUNDLE[_t] = _ids

RATES = (0.10, 0.20, 0.30, 0.40, 0.50)
SEEDS = (0, 1, 2, 3, 4)


def _order(seed: int, alpha: float = 0.5):
    rng = np.random.default_rng(abs(hash(("nested", int(seed)))) % 2**31)
    w = np.power(tract_flow.reindex(TRACTS).fillna(0.0).to_numpy(), alpha)
    w = np.maximum(w, w[w > 0].min() * 1e-3)
    rem, wr, out = list(TRACTS), w.copy(), []
    while rem:
        k = rng.choice(len(rem), p=wr / wr.sum())
        out.append(rem.pop(k)); wr = np.delete(wr, k)
    return out


def draw_nested(rates=RATES, seed: int = 0, alpha: float = 0.5,
                include_census: bool = True) -> dict:
    """Return {rate: set(link_id)}, nested across increasing rates."""
    order, sel, i, res = _order(seed, alpha), set(CENSUS) if include_census else set(), 0, {}
    for rate in sorted(rates):
        budget = int(round(rate * len(_links)))
        while i < len(order) and len(sel) < budget:
            sel |= BUNDLE[order[i]]; i += 1
        res[rate] = set(sel)
    return res

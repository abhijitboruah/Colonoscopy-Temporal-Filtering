"""Compare persistence durations with no persistence at equal false alerts per minute on REAL-Colon."""
import json

import numpy as np
import pandas as pd

D = pd.read_csv("results/realcolon/seconds_sweep.csv")


def curve(t):
    d = D[D.tau_s == t].sort_values("fa_per_min")
    return d.fa_per_min.values, d.lesion_sens.values, d.lesion_recall_3s.values


x0, s0, r0 = curve(0.0)
out = {}
for t in (0.2, 0.5, 1.0, 1.5, 2.0, 3.0):
    x, s, r = curve(t)
    lo, hi = max(x0.min(), x.min()), min(x0.max(), x.max(), 20)
    g = np.geomspace(lo, hi, 60)
    ds = np.interp(g, x, s) - np.interp(g, x0, s0)
    dr = np.interp(g, x, r) - np.interp(g, x0, r0)
    at5 = [np.interp(5, x, s) - np.interp(5, x0, s0), np.interp(5, x, r) - np.interp(5, x0, r0)]
    out[str(t)] = {"range": [round(float(lo), 2), round(float(hi), 2)],
                   "dsens": [round(float(ds.min()), 3), round(float(ds.max()), 3)],
                   "d3s": [round(float(dr.min()), 3), round(float(dr.max()), 3)],
                   "at5": [round(float(v), 3) for v in at5]}
json.dump(out, open("results/realcolon/seconds_compare.json", "w"), indent=1)
print(json.dumps(out))

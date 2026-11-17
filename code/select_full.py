"""Select the setting of every rule on all LDPolypVideo clips for transfer to REAL-Colon."""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from cv_unified import agg, configs

memo = pickle.load(open("results/ldpolypvideo/pervideo.pkl", "rb"))
held = {v[1] for v in json.load(open("data/split_manifest.json"))["test"]["videos"]}
names = sorted(next(iter(memo.values())).keys())
C = configs([3, 4, 5, 6, 8])
pv = lambda kind, p: memo[(kind, json.dumps(p, sort_keys=True))]
ceil = {cfg: max(agg(pv(k, p), names)["sens"] for p in g) for cfg, (k, g) in C.items()}
target = min(ceil.values())
sel = {}
for cfg, (kind, grid) in C.items():
    key, best = None, None
    for p in grid:
        s = agg(pv(kind, p), names)
        k = (-(s["sens"] >= target), s["fa_per_1k"], s["lat"] if not np.isnan(s["lat"]) else 1e9)
        if key is None or k < key:
            key, best, bs = k, p, s
    sel[cfg] = {"params": best, "fa_per_1k": bs["fa_per_1k"], "sens": bs["sens"], "lat": bs["lat"]}
    print(f"{cfg:34s} {json.dumps(best):55s} FA {bs['fa_per_1k']:.2f} sens {bs['sens']:.3f} lat {bs['lat']}")
print("videos", len(names), "target", round(target, 4), "ceilings", {k: round(v, 3) for k, v in ceil.items()})
json.dump({"target": target, "ceilings": ceil, "selected": sel}, open("results/ldpolypvideo/selected_full.json", "w"), indent=1)

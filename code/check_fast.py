"""Check that the fast engine reproduces the frame-by-frame reference implementation exactly."""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from core import load_cache
from fast import Video, per_video_all

memo = pickle.load(open("results/ldpolypvideo/pervideo.pkl", "rb"))
held = {v[1] for v in json.load(open("data/split_manifest.json"))["test"]["videos"]}
cache = load_cache("data/ldpolypvideo", exclude=held)
names = sorted(cache)[:: int(sys.argv[1]) if len(sys.argv) > 1 else 1]

KIND = {"Unfiltered detector": "none", "Fixed persistence, k=3": "fixed3", "Fixed persistence, k=6": "fixed6",
        "Fixed persistence (k selected)": "fixedsel", "Confidence-adaptive": "adaptive",
        "Sliding-window vote": "vote", "Hysteresis": "hysteresis", "Confidence accumulation": "evidence",
        "Motion-compensated persistence": "motion"}


def old_params(cfg, p):
    p = dict(p)
    if cfg in ("Fixed persistence, k=3", "Fixed persistence, k=6"):
        p.pop("k")
    elif cfg == "Fixed persistence (k selected)":
        p["k_max"] = p.pop("k")
    return json.dumps(p, sort_keys=True)


t0 = time.time()
n = bad = 0
for name in names:
    v = cache[name]
    V = Video(name, v["detections"], v["annotations"], v["n_frames"])
    res = per_video_all(V, [3, 4, 5, 6, 8])
    for (cfg, pj, _), got in res.items():
        key = (KIND[cfg], old_params(cfg, json.loads(pj)))
        if key not in memo:
            continue
        exp = memo[key][name]
        n += 1
        if tuple(got[:6]) != tuple(exp[:6]) or sorted(got[6]) != sorted(exp[6]):
            bad += 1
            if bad <= 10:
                print("MISMATCH", name, cfg, pj, got[:6], exp[:6])
print(f"{len(names)} videos, {n} settings compared, {bad} mismatches, {time.time()-t0:.0f} s")

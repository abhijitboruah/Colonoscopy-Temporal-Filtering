"""Five-fold grouped cross-validation on LDPolypVideo at matched per-episode sensitivity."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import evaluate_alerts, load_cache, make_folds, run_baseline, run_filter

THETA = 0.15
CT = [round(x, 3) for x in np.arange(0.05, 0.96, 0.025)]
CACHE = None


def configs(kgrid):
    G = [0.5, 1.0, 2.0, 3.0]
    return {
        "Unfiltered detector": ("none", [{"conf_thresh": c} for c in CT]),
        "Fixed persistence, k=3": ("fixed3", [{"conf_thresh": c} for c in CT]),
        "Fixed persistence, k=6": ("fixed6", [{"conf_thresh": c} for c in CT]),
        "Fixed persistence (k selected)": ("fixedsel", [{"conf_thresh": c, "k_max": k} for c, k in product(CT, kgrid)]),
        "Confidence-adaptive": ("adaptive", [{"conf_thresh": c, "k_max": k, "gamma": g} for c, k, g in product(CT, kgrid, G)]),
        "Sliding-window vote": ("vote", [{"conf_thresh": c, "window": w, "votes": v}
                                         for c, w, v in product(CT, [4, 5, 6, 8], [2, 3, 4, 5]) if v <= w]),
        "Hysteresis": ("hysteresis", [{"conf_thresh": c, "tau_high": t, "k_fix": k}
                                      for c, t, k in product(CT, [0.5, 0.6, 0.7, 0.8, 0.9],
                                                             sorted({3, 4, 6, 8} | {k for k in kgrid if k > 8}))]),
        "Confidence accumulation": ("evidence", [{"conf_thresh": c, "evidence": e, "min_len": m}
                                                 for c, e, m in product(CT, [0.5, 1.0, 1.5, 2.0, 3.0, 4.0], [1, 2, 3])]),
        "Motion-compensated persistence": ("motion", [{"conf_thresh": c, "k_fix": k, "iou_assoc": j}
                                                      for c, k, j in product(CT, kgrid, [0.2, 0.3, 0.4])]),
    }


def runner(kind):
    if kind == "none":
        return lambda d, **kw: run_filter(d, mode="none", k_max=1, gamma=0.0, theta=0.0, **kw)
    if kind == "fixed3":
        return lambda d, **kw: run_filter(d, mode="fixed", k_max=3, gamma=0.0, theta=THETA, **kw)
    if kind == "fixed6":
        return lambda d, **kw: run_filter(d, mode="fixed", k_max=6, gamma=0.0, theta=THETA, **kw)
    if kind == "fixedsel":
        return lambda d, **kw: run_filter(d, mode="fixed", gamma=0.0, theta=THETA, **kw)
    if kind == "adaptive":
        return lambda d, **kw: run_filter(d, mode="adaptive", theta=THETA, **kw)
    return lambda d, **kw: run_baseline(d, rule=kind, **kw)


def per_video(job):
    kind, params = job
    run = runner(kind)
    out = {}
    for name, v in CACHE.items():
        al = run(v["detections"], **params)
        m = evaluate_alerts(al, v["annotations"], v["n_frames"])
        out[name] = (m["n_false"], m["n_pure"], m["n_loc"], m["n_frames"],
                     m["n_detected"], m["n_episodes"], m["lat_list"])
    return out


def agg(pv, videos):
    fa = pu = lo = fr = hit = ep = 0
    lat = []
    for n in videos:
        a, b, c, d, e, f, g = pv[n]
        fa += a; pu += b; lo += c; fr += d; hit += e; ep += f; lat += g
    return {"fa_per_1k": 1000 * fa / fr, "pure_per_1k": 1000 * pu / fr, "loc_per_1k": 1000 * lo / fr,
            "sens": hit / ep, "lat": float(np.median(lat)) if lat else np.nan,
            "TP": hit, "FN": ep - hit, "FP": fa, "FP_pure": pu, "FP_loc": lo,
            "n_frames": fr, "n_episodes": ep, "lat_list": lat}


def init(cache):
    global CACHE
    CACHE = cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/ldpolypvideo")
    ap.add_argument("--manifest", default="data/split_manifest.json")
    ap.add_argument("--kgrid", default="3,4,5,6,8")
    ap.add_argument("--out", default="results/ldpolypvideo")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    kgrid = [int(x) for x in a.kgrid.split(",")]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    held = {v[1] for v in json.load(open(a.manifest))["test"]["videos"]}
    cache = load_cache(a.cache, exclude=held)
    names = sorted(cache)
    folds = make_folds(names)
    C = configs(kgrid)


    memo_p = out / "pervideo.pkl"
    memo = pickle.load(open(memo_p, "rb")) if memo_p.exists() else {}
    jobs = [(kind, p) for _, (kind, grid) in C.items() for p in grid
            if (kind, json.dumps(p, sort_keys=True)) not in memo]
    print(f"{len(jobs)} settings to evaluate", flush=True)
    if jobs:
        with Pool(a.procs, initializer=init, initargs=(cache,)) as pool:
            for i, res in enumerate(pool.imap(per_video, jobs, chunksize=8)):
                memo[(jobs[i][0], json.dumps(jobs[i][1], sort_keys=True))] = res
                if i % 200 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
                    pickle.dump(memo, open(memo_p, "wb"))
        pickle.dump(memo, open(memo_p, "wb"))

    def pv(kind, p):
        return memo[(kind, json.dumps(p, sort_keys=True))]


    targets, ceil_tab = [], []
    for i, test in enumerate(folds):
        train = [n for n in names if n not in test]
        ceil = {cfg: max(agg(pv(kind, p), train)["sens"] for p in grid) for cfg, (kind, grid) in C.items()}
        targets.append(min(ceil.values()))
        ceil_tab.append({"fold": i + 1, **ceil, "target": targets[-1]})
    pd.DataFrame(ceil_tab).to_csv(out / "ceilings.csv", index=False)

    rows, pooled_lat = [], {}
    for cfg, (kind, grid) in C.items():
        pooled_lat[cfg] = []
        for i, test in enumerate(folds):
            train = [n for n in names if n not in test]
            best, key = None, None
            for p in grid:
                s = agg(pv(kind, p), train)
                k = (-(s["sens"] >= targets[i]), s["fa_per_1k"], s["lat"] if not np.isnan(s["lat"]) else 1e9)
                if key is None or k < key:
                    key, best = k, p
            s = agg(pv(kind, best), sorted(test))
            pooled_lat[cfg] += s["lat_list"]
            rows.append({"configuration": cfg, "fold": i + 1, "target": targets[i],
                         **{k: v for k, v in s.items() if k != "lat_list"},
                         "params": json.dumps(best)})
    pf = pd.DataFrame(rows)
    pf.to_csv(out / "per_fold.csv", index=False)
    json.dump({k: v for k, v in pooled_lat.items()}, open(out / "pooled_latencies.json", "w"))
    json.dump({"folds": [sorted(f) for f in folds], "targets": targets, "kgrid": kgrid},
              open(out / "folds.json", "w"), indent=1)

    summ = pf.groupby("configuration", sort=False).agg(
        fa=("fa_per_1k", "mean"), fa_sd=("fa_per_1k", "std"), sens=("sens", "mean"),
        lat=("lat", "mean"), lat_sd=("lat", "std"), TP=("TP", "sum"), FN=("FN", "sum"),
        FP=("FP", "sum"), FP_pure=("FP_pure", "sum"), FP_loc=("FP_loc", "sum"))
    summ["lat_pooled"] = [float(np.median(pooled_lat[c])) for c in summ.index]
    summ.to_csv(out / "summary.csv")
    print(summ.round(3).to_string())
    print("targets", np.round(targets, 4))


if __name__ == "__main__":
    main()

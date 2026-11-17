"""REAL-Colon lesion-level evaluation with persistence expressed in seconds."""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fast import Tracks, Video, metrics
from realcolon_eval import KEYS, load

TAUS = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
CTS = [round(x, 3) for x in np.arange(0.05, 0.96, 0.05)]
SIGMA = 3.0
VI = pd.read_csv("data/realcolon/video_info.csv").set_index("unique_video_name")


def work(item):
    vid, v = item
    fps = float(VI.loc[vid, "fps"])
    V = Video(vid, v["dets"], v["ann"], v["n_frames"], keys=KEYS)
    out = {}
    for ct in CTS:
        tid, det = V.tracks(ct)
        T = Tracks(V, tid, det) if len(tid) else None
        for tau in TAUS:
            k = max(1, int(round(tau * fps)))
            al = T.length(k) if T is not None else np.zeros(0, np.int64)
            e = metrics(V, al, "episode")
            l = metrics(V, al, "lesion_id")
            out[(ct, tau)] = {"fa": e[0], "frames": e[3], "ep_hit": e[4], "ep_n": e[5],
                              "les_hit": l[4], "les_n": l[5], "les_lat_s": [x / fps for x in l[6]],
                              "ep_lat_s": [x / fps for x in e[6]], "minutes": e[3] / fps / 60}
    return vid, out


def main():
    vids = load("data/realcolon")
    items = sorted(vids.items(), key=lambda x: -x[1]["n_frames"])
    import pickle
    memo = Path("results/realcolon/seconds_pervideo.pkl")
    if memo.exists():
        res = pickle.load(open(memo, "rb"))
    else:
        with Pool(2) as pool:
            res = dict(pool.imap_unordered(work, items))
        pickle.dump(res, open(memo, "wb"))
    rows = []
    for ct in CTS:
        for tau in TAUS:
            r = [res[v][(ct, tau)] for v in res]
            fa = sum(x["fa"] for x in r); mins = sum(x["minutes"] for x in r)
            ll = [y for x in r for y in x["les_lat_s"]]
            les_n = sum(x["les_n"] for x in r)
            rows.append({"conf_thresh": ct, "tau_s": tau, "fa_per_min": fa / mins,
                         "fa_per_1k": 1000 * fa / sum(x["frames"] for x in r),
                         "ep_sens": sum(x["ep_hit"] for x in r) / sum(x["ep_n"] for x in r),
                         "lesion_sens": sum(x["les_hit"] for x in r) / les_n,
                         "lesion_recall_3s": sum(y <= SIGMA for y in ll) / les_n,
                         "lesion_lat_s_median": float(np.median(ll)) if ll else np.nan,
                         "ep_lat_s_median": float(np.median([y for x in r for y in x["ep_lat_s"]])) if fa else np.nan})
    D = pd.DataFrame(rows)
    out = Path("results/realcolon"); D.to_csv(out / "seconds_sweep.csv", index=False)
    pd.set_option("display.width", 200)
    print(D.round(3).to_string())


    def pool_(vs, ct, tau):
        r = [res[v][(ct, tau)] for v in vs]
        ll = [y for x in r for y in x["les_lat_s"]]
        n = sum(x["les_n"] for x in r)
        return {"fa_per_min": sum(x["fa"] for x in r) / sum(x["minutes"] for x in r),
                "lesion_sens": sum(x["les_hit"] for x in r) / n if n else np.nan,
                "lesion_recall_3s": sum(y <= SIGMA for y in ll) / n if n else np.nan,
                "les_hit": sum(x["les_hit"] for x in r), "les_n": n,
                "les_3s": sum(y <= SIGMA for y in ll), "lat": ll,
                "fa": sum(x["fa"] for x in r), "minutes": sum(x["minutes"] for x in r),
                "ep_hit": sum(x["ep_hit"] for x in r), "ep_n": sum(x["ep_n"] for x in r)}
    names = sorted(res)
    rows = []
    for LIMIT in (5.0, 10.0, 2.0):
        for family, taus in (("persistence", TAUS), ("threshold only", [0.0])):
            agg = {"fa": 0, "minutes": 0, "les_hit": 0, "les_n": 0, "les_3s": 0, "lat": [], "ep_hit": 0, "ep_n": 0}
            for c in sorted({v[:3] for v in names}):
                train = [v for v in names if v[:3] != c]; test = [v for v in names if v[:3] == c]
                cand = [(ct, t) for ct in CTS for t in taus]
                sc = {p: pool_(train, *p) for p in cand}
                okp = [p for p in cand if sc[p]["fa_per_min"] <= LIMIT]
                best = max(okp, key=lambda p: (sc[p]["lesion_sens"], sc[p]["lesion_recall_3s"], -sc[p]["fa_per_min"]))
                t = pool_(test, *best)
                for k in agg:
                    agg[k] += t[k]
                rows.append({"limit": LIMIT, "family": family, "centre": c, "conf_thresh": best[0], "tau_s": best[1],
                             "fa_per_min": t["fa_per_min"], "lesion_sens": t["lesion_sens"],
                             "lesion_recall_3s": t["lesion_recall_3s"]})
            rows.append({"limit": LIMIT, "family": family, "centre": "pooled",
                         "fa_per_min": agg["fa"] / agg["minutes"], "lesion_sens": agg["les_hit"] / agg["les_n"],
                         "lesion_recall_3s": agg["les_3s"] / agg["les_n"], "les_hit": agg["les_hit"], "les_n": agg["les_n"],
                         "lesion_lat_s_median": float(np.median(agg["lat"])), "ep_sens": agg["ep_hit"] / agg["ep_n"]})
    L = pd.DataFrame(rows)
    L.to_csv(out / "seconds_loco.csv", index=False)
    print(L.round(3).to_string())


if __name__ == "__main__":
    main()

"""Split the reduction in false alerts into one alert per tracklet and persistence."""
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from core import load_cache
from fast import CT, Tracks, Video, metrics

N = 10000


def onset_ldp(item):
    name, v = item
    V = Video(name, v["detections"], v["annotations"], v["n_frames"])
    out = {}
    for ct in CT:
        tid, det = V.tracks(ct)
        al = Tracks(V, tid, det).length(1) if len(tid) else np.zeros(0, np.int64)
        out[ct] = metrics(V, al)
    return name, out


def onset_rc(item):
    from realcolon_eval import KEYS
    vid, v = item
    V = Video(vid, v["dets"], v["ann"], v["n_frames"], keys=KEYS)
    out = {}
    for ct in CT:
        tid, det = V.tracks(ct)
        al = Tracks(V, tid, det).length(1) if len(tid) else np.zeros(0, np.int64)
        out[ct] = {k: metrics(V, al, k) for k in KEYS}
    return vid, out


def pool_(res, names, ct, key=None):
    r = [res[n][ct] if key is None else res[n][ct][key] for n in names]
    fa = sum(x[0] for x in r); fr = sum(x[3] for x in r); hit = sum(x[4] for x in r); ep = sum(x[5] for x in r)
    lat = [y for x in r for y in x[6]]
    return {"fa_per_1k": 1000 * fa / fr, "sens": hit / ep, "lat": float(np.median(lat)) if lat else np.nan}


def select(res, names, target, key=None):
    best = None
    for ct in CT:
        s = pool_(res, names, ct, key)
        k = (-(s["sens"] >= target), s["fa_per_1k"], s["lat"] if not np.isnan(s["lat"]) else 1e9)
        if best is None or k < best[0]:
            best = (k, ct)
    return best[1]


def boot(per_rec, ref_rec, rng):
    a, b = np.array(per_rec, float), np.array(ref_rec, float)
    n = len(a); idx = rng.integers(0, n, (N, n))
    sa, sb = a[idx].sum(1), b[idx].sum(1)
    fa, fb = 1000 * sa[:, 0] / sa[:, 1], 1000 * sb[:, 0] / sb[:, 1]
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    return {"fa": 1000 * a[:, 0].sum() / a[:, 1].sum(), "fa_ci": q(fa),
            "sens": a[:, 2].sum() / a[:, 3].sum(),
            "dfa_vs_ref": 1000 * a[:, 0].sum() / a[:, 1].sum() - 1000 * b[:, 0].sum() / b[:, 1].sum(),
            "dfa_ci": q(fa - fb)}


def main():
    rng = np.random.default_rng(0)
    out = {}

    held = {v[1] for v in json.load(open("data/split_manifest.json"))["test"]["videos"]}
    cache = load_cache("data/ldpolypvideo", exclude=held)
    with Pool(2) as p:
        L = dict(p.map(onset_ldp, sorted(cache.items())))
    F = json.load(open("results/ldpolypvideo/folds.json"))
    folds, targets = [set(f) for f in F["folds"]], F["targets"]
    names = sorted(cache)
    memo = pickle.load(open("results/ldpolypvideo/pervideo.pkl", "rb"))
    pf = pd.read_csv("results/ldpolypvideo/per_fold.csv")
    rec, ref, cts = [], [], []
    for i, test in enumerate(folds):
        train = [n for n in names if n not in test]
        ct = select(L, train, targets[i]); cts.append(ct)
        p = json.loads(pf[(pf.configuration == "Fixed persistence (k selected)") & (pf.fold == i + 1)].params.iloc[0])
        m = memo[("fixedsel", json.dumps(p, sort_keys=True))]
        for n in sorted(test):
            x = L[n][ct]; rec.append((x[0], x[3], x[4], x[5])); y = m[n]; ref.append((y[0], y[3], y[4], y[5]))
    lat = [y for i, test in enumerate(folds) for n in sorted(test) for y in L[n][cts[i]][6]]
    out["ldp"] = {**boot(rec, ref, rng), "thresholds": cts, "lat": float(np.median(lat))}
    perrec = {"ldp": [(L[n][cts[i]][0], L[n][cts[i]][3], L[n][cts[i]][4], L[n][cts[i]][5], L[n][cts[i]][6])
                      for i, test in enumerate(folds) for n in sorted(test)]}

    from realcolon_eval import load
    vids = load("data/realcolon")
    memo_p = Path("results/realcolon/onset_pervideo.pkl")
    if memo_p.exists():
        R = pickle.load(open(memo_p, "rb"))
    else:
        with Pool(2) as p:
            R = dict(p.imap_unordered(onset_rc, sorted(vids.items(), key=lambda x: -x[1]["n_frames"])))
        pickle.dump(R, open(memo_p, "wb"))
    rcn = sorted(R)
    res = pickle.load(open("results/realcolon/pervideo.pkl", "rb"))
    vi = pd.read_csv("data/realcolon/video_info.csv").set_index("unique_video_name")
    fps = {v: float(vi.loc[v, "fps"]) for v in rcn}

    S = json.load(open("results/ldpolypvideo/selected_full.json"))
    ct0 = select(L, names, S["target"])
    A = pd.read_csv("results/realcolon/zero_shot.csv").set_index("configuration")
    pk = A.loc["Fixed persistence (k selected)", "params"]
    rec = [(R[v][ct0]["episode"][0], R[v][ct0]["episode"][3], R[v][ct0]["episode"][4], R[v][ct0]["episode"][5]) for v in rcn]
    ref = [tuple(res[v][("Fixed persistence (k selected)", pk, "episode")][i] for i in (0, 3, 4, 5)) for v in rcn]
    perrec["zero_shot"] = [tuple(R[v][ct0]["episode"][i] for i in (0, 3, 4, 5, 6)) for v in rcn]
    out["zero_shot"] = {**boot(rec, ref, rng), "threshold": ct0,
                        "fa_per_min": sum(r[0] for r in rec) / sum(r[1] / fps[v] / 60 for r, v in zip(rec, rcn))}

    C = pd.read_csv("results/realcolon/loco_ceilings.csv")
    B = pd.read_csv("results/realcolon/loco_per_centre.csv")
    rec, ref, cts, lat_s = [], [], {}, []
    for c in sorted({v[:3] for v in rcn}):
        train = [v for v in rcn if v[:3] != c]; test = [v for v in rcn if v[:3] == c]
        tgt = float(C[C.centre == int(c)].target.iloc[0])
        ct = select(R, train, tgt, "episode"); cts[c] = ct
        pj = B[(B.configuration == "Fixed persistence (k selected)") & (B.centre == int(c))].params.iloc[0]
        for v in test:
            x = R[v][ct]["episode"]; rec.append((x[0], x[3], x[4], x[5])); lat_s += [y / fps[v] for y in x[6]]
            ref.append(tuple(res[v][("Fixed persistence (k selected)", pj, "episode")][i] for i in (0, 3, 4, 5)))
    out["loco"] = {**boot(rec, ref, rng), "thresholds": cts, "lat_s": float(np.median(lat_s)),
                   "fa_per_min": sum(r[0] for r in rec) / sum(r[1] / fps[v] / 60 for r, v in zip(rec, rcn))}
    perrec["loco"] = [tuple(R[v][cts[v[:3]]]["episode"][i] for i in (0, 3, 4, 5, 6)) for v in rcn]
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    pickle.dump(perrec, open("results/tables/onset_perrec.pkl", "wb"))
    json.dump(out, open("results/tables/onset.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

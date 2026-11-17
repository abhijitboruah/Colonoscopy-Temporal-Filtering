"""REAL-Colon evaluation: settings transferred from LDPolypVideo and leave-one-cohort-out."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fast import Video, add_episodes, grid, per_video_all

KGRID = [3, 4, 5, 6, 8]
KEYS = ("episode", "lesion_id")
CFGS = list(grid(KGRID))
RNG = np.random.default_rng(0)


def load(bundle_dir):
    vids = {}
    for p in sorted(Path(bundle_dir).glob("centre_*.npz")):
        z = np.load(p, allow_pickle=False)
        centre = p.stem.split("_")[1][:3]
        meta = json.loads(str(z["meta"]))
        for vid, m in meta.items():
            k = int(vid.split("-")[1])
            sel = z["vid"] == k
            dets = {}
            for f, c, b in zip(z["frame"][sel], z["conf"][sel], z["box"][sel]):
                dets.setdefault(int(f), []).append({"conf": c / 10000.0, "box": (b / 10.0).tolist()})
            asel = z["a_vid"] == k
            ann = {}
            for f, les, b in zip(z["a_frame"][asel], z["a_lesion"][asel], z["a_box"][asel]):
                ann.setdefault(int(f), []).append({"lesion_id": str(les), "box": (b / 10.0).tolist()})
            add_episodes(ann)
            vids[vid] = {"centre": centre, "n_frames": m["n_frames"], "dets": dets, "ann": ann}
    return vids


def _work(item):
    vid, v = item
    V = Video(vid, v["dets"], v["ann"], v["n_frames"], keys=KEYS)
    return vid, per_video_all(V, KGRID)


def pooled(res, vids, cfg, pj, key, fps):
    if not vids:
        return {"fa_per_1k": np.nan, "pure_per_1k": np.nan, "fa_per_min": np.nan, "sens": np.nan, "TP": 0,
                "N": 0, "FP": 0, "lat": np.nan, "lat_s": np.nan, "lat_list": []}
    fa = pu = fr = hit = ep = 0
    lat, lat_s, minutes = [], [], 0.0
    for v in vids:
        a, b, _, d, e, f, g = res[v][(cfg, pj, key)]
        fa += a; pu += b; fr += d; hit += e; ep += f
        lat += g; lat_s += [x / fps[v] for x in g]; minutes += d / fps[v] / 60
    return {"fa_per_1k": 1000 * fa / fr, "pure_per_1k": 1000 * pu / fr, "fa_per_min": fa / minutes,
            "sens": hit / ep if ep else np.nan, "TP": hit, "N": ep, "FP": fa,
            "lat": float(np.median(lat)) if lat else np.nan,
            "lat_s": float(np.median(lat_s)) if lat_s else np.nan, "lat_list": lat}


def select(res, vids, plist, cfg, target, fps):
    key = best = None
    for pj in plist:
        s = pooled(res, vids, cfg, pj, "episode", fps)
        k = (-(s["sens"] >= target), s["fa_per_1k"], s["lat"] if not np.isnan(s["lat"]) else 1e9)
        if key is None or k < key:
            key, best = k, pj
    return best


def boot(res, vids, choice, fps, n=10000):
    vids = list(vids)
    arr = {}
    for cfg, pj in choice.items():
        arr[cfg] = np.array([[res[v][(cfg, pj, "episode")][i] for i in (0, 3, 4, 5)] for v in vids], float)
    idx = RNG.integers(0, len(vids), (n, len(vids)))
    out = {}
    for cfg, A in arr.items():
        s = A[idx].sum(1)
        fa, se = 1000 * s[:, 0] / s[:, 1], s[:, 2] / s[:, 3]
        B = arr["Confidence-adaptive"][idx].sum(1)
        dfa = fa - 1000 * B[:, 0] / B[:, 1]
        dse = se - B[:, 2] / B[:, 3]
        q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
        out[cfg] = {"fa_ci": q(fa), "sens_ci": q(se), "dfa_vs_adaptive_ci": q(dfa), "dsens_vs_adaptive_ci": q(dse)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="data/realcolon")
    ap.add_argument("--out", default="results/realcolon")
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    vi = pd.read_csv(Path(a.bundle) / "video_info.csv").set_index("unique_video_name")
    vids = load(a.bundle)
    names = sorted(vids)
    fps = {v: float(vi.loc[v, "fps"]) for v in names}
    centres = sorted({vids[v]["centre"] for v in names})
    R = {"n_recordings": len(names)}


    desc = []
    for v in names:
        ann = vids[v]["ann"]
        eps = {a["episode"] for x in ann.values() for a in x}
        les = {a["lesion_id"] for x in ann.values() for a in x}
        pairs = {(a["episode"], a["lesion_id"]) for x in ann.values() for a in x}
        desc.append({"video": v, "centre": vids[v]["centre"], "fps": fps[v], "n_frames": vids[v]["n_frames"],
                     "annotated_frames": len(ann), "lesions": len(les), "episodes": len(eps),
                     "episodes_spanning_2plus_lesions": sum(len({l for e2, l in pairs if e2 == e}) > 1 for e in eps),
                     "n_dets": sum(len(x) for x in vids[v]["dets"].values()),
                     "lesions_listed": int(vi.loc[v, "num_lesions"])})
    D = pd.DataFrame(desc)
    D.to_csv(out / "recordings.csv", index=False)
    R["descriptives"] = {"frames": int(D.n_frames.sum()), "lesions": int(D.lesions.sum()),
                         "lesions_listed": int(D.lesions_listed.sum()), "episodes": int(D.episodes.sum()),
                         "lesion_free_recordings": int((D.lesions == 0).sum()),
                         "episodes_per_lesion_median": float((D.episodes / D.lesions.replace(0, np.nan)).median()),
                         "episodes_spanning_2plus_lesions": int(D.episodes_spanning_2plus_lesions.sum()),
                         "minutes": float((D.n_frames / D.fps).sum() / 60), "detections": int(D.n_dets.sum())}
    print(json.dumps(R["descriptives"], indent=1), flush=True)


    memo_p = out / "pervideo.pkl"
    res = pickle.load(open(memo_p, "rb")) if memo_p.exists() else {}
    todo = [(v, vids[v]) for v in names if v not in res]
    todo.sort(key=lambda x: -x[1]["n_frames"])
    if todo:
        with Pool(a.procs) as pool:
            for vid, r in pool.imap_unordered(_work, todo):
                res[vid] = r
                pickle.dump(res, open(memo_p, "wb"))
                print("  done", vid, flush=True)
    plists = {cfg: sorted({pj for (c, pj, k) in res[names[0]] if c == cfg}) for cfg in CFGS}


    S = json.load(open("results/ldpolypvideo/selected_full.json"))["selected"]
    choice = {}
    for cfg in CFGS:
        p = dict(S[cfg]["params"])
        if cfg == "Fixed persistence (k selected)":
            p["k"] = p.pop("k_max")
        elif cfg == "Fixed persistence, k=3":
            p["k"] = 3
        elif cfg == "Fixed persistence, k=6":
            p["k"] = 6
        choice[cfg] = json.dumps(p, sort_keys=True)
    rowsA = []
    for cfg in CFGS:
        e = pooled(res, names, cfg, choice[cfg], "episode", fps)
        l = pooled(res, names, cfg, choice[cfg], "lesion_id", fps)
        free = [v for v in names if vids[v]["ann"] == {}]
        fr = pooled(res, free, cfg, choice[cfg], "episode", fps)
        rowsA.append({"configuration": cfg, "params": choice[cfg], **{k: v for k, v in e.items() if k != "lat_list"},
                      "lesion_sens": l["sens"], "lesion_TP": l["TP"], "lesion_N": l["N"], "lesion_lat_s": l["lat_s"],
                      "fa_per_1k_lesion_free": fr["fa_per_1k"],
                      "LDP_fa": S[cfg]["fa_per_1k"], "LDP_sens": S[cfg]["sens"], "LDP_lat": S[cfg]["lat"]})
    A = pd.DataFrame(rowsA)
    A.to_csv(out / "zero_shot.csv", index=False)
    R["zero_shot_boot"] = boot(res, names, choice, fps)
    print("\nZERO-SHOT\n", A.drop(columns=["params"]).round(3).to_string(), flush=True)


    rowsB, lat_pool, ceil_tab = [], {c: [] for c in CFGS}, []
    held_choice = {}
    for c in centres:
        train = [v for v in names if vids[v]["centre"] != c]
        test = [v for v in names if vids[v]["centre"] == c]
        ceil = {cfg: max(pooled(res, train, cfg, pj, "episode", fps)["sens"] for pj in plists[cfg]) for cfg in CFGS}
        target = min(ceil.values())
        ceil_tab.append({"centre": c, **ceil, "target": target})
        for cfg in CFGS:
            pj = select(res, train, plists[cfg], cfg, target, fps)
            s = pooled(res, test, cfg, pj, "episode", fps)
            lat_pool[cfg] += s["lat_list"]
            for v in test:
                held_choice.setdefault(cfg, {})[v] = pj
            rowsB.append({"configuration": cfg, "centre": c, "target": target, "params": pj,
                          **{k: v for k, v in s.items() if k != "lat_list"}})
    B = pd.DataFrame(rowsB)
    B.to_csv(out / "loco_per_centre.csv", index=False)
    pd.DataFrame(ceil_tab).to_csv(out / "loco_ceilings.csv", index=False)
    SB = B.groupby("configuration", sort=False).agg(fa=("fa_per_1k", "mean"), fa_sd=("fa_per_1k", "std"),
                                                    sens=("sens", "mean"), sens_sd=("sens", "std"),
                                                    lat=("lat", "mean"), TP=("TP", "sum"), N=("N", "sum"),
                                                    FP=("FP", "sum"))
    SB["lat_pooled"] = [float(np.median(lat_pool[c])) if lat_pool[c] else np.nan for c in SB.index]
    SB.to_csv(out / "loco_summary.csv")
    print("\nLEAVE-ONE-COHORT-OUT\n", SB.round(3).to_string(), flush=True)


    idx = RNG.integers(0, len(names), (10000, len(names)))
    arr = {cfg: np.array([[res[v][(cfg, held_choice[cfg][v], "episode")][i] for i in (0, 3, 4, 5)]
                          for v in names], float) for cfg in CFGS}
    bootB = {}
    for cfg in CFGS:
        s, b = arr[cfg][idx].sum(1), arr["Confidence-adaptive"][idx].sum(1)
        fa, se = 1000 * s[:, 0] / s[:, 1], s[:, 2] / s[:, 3]
        q = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
        bootB[cfg] = {"fa_pooled": float(1000 * arr[cfg][:, 0].sum() / arr[cfg][:, 1].sum()),
                      "sens_pooled": float(arr[cfg][:, 2].sum() / arr[cfg][:, 3].sum()),
                      "fa_ci": q(fa), "sens_ci": q(se),
                      "dfa_vs_adaptive_ci": q(fa - 1000 * b[:, 0] / b[:, 1]),
                      "dsens_vs_adaptive_ci": q(se - b[:, 2] / b[:, 3])}
    R["loco_boot"] = bootB
    json.dump(R, open(out / "realcolon.json", "w"), indent=1)
    print(json.dumps(bootB, indent=1))


if __name__ == "__main__":
    main()

"""Bootstrap intervals for the held-out results of every rule (Tables 1 and 2)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
REF = "Fixed persistence (k selected)"
KIND = {"Unfiltered detector": "none", "Fixed persistence, k=3": "fixed3",
        "Fixed persistence, k=6": "fixed6", "Fixed persistence (k selected)": "fixedsel",
        "Confidence-adaptive": "adaptive", "Sliding-window vote": "vote",
        "Hysteresis": "hysteresis", "Confidence accumulation": "evidence",
        "Motion-compensated persistence": "motion"}
ONSET = "One alert per tracklet"
ORDER = ["Unfiltered detector", ONSET, "Fixed persistence, k=3", "Fixed persistence, k=6", REF,
         "Confidence-adaptive", "Sliding-window vote", "Hysteresis", "Confidence accumulation",
         "Motion-compensated persistence"]
N = 10000


def table(per_rec, fps=None):
    rng = np.random.default_rng(0)
    n = len(next(iter(per_rec.values())))
    idx = rng.integers(0, n, (N, n))
    A = {c: np.array([r[:4] for r in v], float) for c, v in per_rec.items()}
    rows = []
    for c in ORDER:
        a = A[c]
        s, b = a[idx].sum(1), A[REF][idx].sum(1)
        fa, se = 1000 * s[:, 0] / s[:, 1], s[:, 2] / s[:, 3]
        dfa = fa - 1000 * b[:, 0] / b[:, 1]
        dse = se - b[:, 2] / b[:, 3]
        lat = [x for r in per_rec[c] for x in r[4]]
        q = lambda x: np.nanpercentile(x, [2.5, 97.5])
        row = {"configuration": c,
               "fa_per_1k": 1000 * a[:, 0].sum() / a[:, 1].sum(), "fa_lo": q(fa)[0], "fa_hi": q(fa)[1],
               "sens": a[:, 2].sum() / a[:, 3].sum(), "sens_lo": q(se)[0], "sens_hi": q(se)[1],
               "lat_frames": float(np.median(lat)) if lat else np.nan,
               "dfa": 1000 * a[:, 0].sum() / a[:, 1].sum() - 1000 * A[REF][:, 0].sum() / A[REF][:, 1].sum(),
               "dfa_lo": q(dfa)[0], "dfa_hi": q(dfa)[1],
               "dsens": a[:, 2].sum() / a[:, 3].sum() - A[REF][:, 2].sum() / A[REF][:, 3].sum(),
               "dsens_lo": q(dse)[0], "dsens_hi": q(dse)[1],
               "TP": int(a[:, 2].sum()), "N": int(a[:, 3].sum()), "FP": int(a[:, 0].sum())}
        if fps is not None:
            lat_s = [x / fps[i] for i, r in enumerate(per_rec[c]) for x in r[4]]
            mins = sum(r[1] / fps[i] / 60 for i, r in enumerate(per_rec[c]))
            row["lat_s"] = float(np.median(lat_s)) if lat_s else np.nan
            row["fa_per_min"] = a[:, 0].sum() / mins
        rows.append(row)
    return pd.DataFrame(rows)


def ldp():
    memo = pickle.load(open("results/ldpolypvideo/pervideo.pkl", "rb"))
    folds = [set(f) for f in json.load(open("results/ldpolypvideo/folds.json"))["folds"]]
    pf = pd.read_csv("results/ldpolypvideo/per_fold.csv")
    per = {}
    for c in ORDER:
        if c == ONSET:
            continue
        recs = []
        for i, test in enumerate(folds):
            p = json.loads(pf[(pf.configuration == c) & (pf.fold == i + 1)].params.iloc[0])
            m = memo[(KIND[c], json.dumps(p, sort_keys=True))]
            for v in sorted(test):
                fa, _, _, fr, hit, ep, lat = m[v]
                recs.append((fa, fr, hit, ep, lat))
        per[c] = recs
    per[ONSET] = pickle.load(open("results/tables/onset_perrec.pkl", "rb"))["ldp"]
    return table(per)


def realcolon():
    res = pickle.load(open("results/realcolon/pervideo.pkl", "rb"))
    vi = pd.read_csv("data/realcolon/video_info.csv").set_index("unique_video_name")
    names = sorted(res)
    fps = [float(vi.loc[v, "fps"]) for v in names]
    out = {}

    A = pd.read_csv("results/realcolon/zero_shot.csv").set_index("configuration")
    ON = pickle.load(open("results/tables/onset_perrec.pkl", "rb"))
    per = {c: [res[v][(c, A.loc[c, "params"], "episode")][:1] + res[v][(c, A.loc[c, "params"], "episode")][3:]
               for v in names] for c in ORDER if c != ONSET}
    per[ONSET] = ON["zero_shot"]
    out["zero_shot"] = table({c: [(r[0], r[1], r[2], r[3], r[4]) for r in per[c]] for c in ORDER}, fps)

    B = pd.read_csv("results/realcolon/loco_per_centre.csv")
    per = {ONSET: ON["loco"]}
    for c in ORDER:
        if c == ONSET:
            continue
        recs = []
        for v in names:
            pj = B[(B.configuration == c) & (B.centre == int(v[:3]))].params.iloc[0]
            m = res[v][(c, pj, "episode")]
            recs.append((m[0], m[3], m[4], m[5], m[6]))
        per[c] = recs
    out["loco"] = table(per, fps)
    return out


def main():
    out = Path("results/tables"); out.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    L = ldp(); L.to_csv(out / "ldp_cv.csv", index=False)
    print("LDPolypVideo, pooled held-out folds\n", L.round(3).to_string())
    R = realcolon()
    for k, d in R.items():
        d.to_csv(out / f"realcolon_{k}.csv", index=False)
        print(f"\nREAL-Colon {k}\n", d.round(3).to_string())


if __name__ == "__main__":
    main()

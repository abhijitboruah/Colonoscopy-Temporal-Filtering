"""Fast engine for tracklets, filtering rules and alert-level measures."""
from __future__ import annotations

from itertools import product

import numpy as np

from core import iou

THETA = 0.15
CT = [round(x, 3) for x in np.arange(0.05, 0.96, 0.025)]


def add_episodes(ann, key="episode"):
    run, prev = -1, None
    for f in sorted(f for f in ann if ann[f]):
        if prev is None or f != prev + 1:
            run += 1
        prev = f
        for slot, a in enumerate(sorted(ann[f], key=lambda a: a["box"][0])):
            a[key] = (run, slot)
    return ann


class Video:

    def __init__(self, name, detections, annotations, n_frames, keys=("polyp_id",), meta=None):
        self.name, self.n_frames, self.keys, self.meta = name, n_frames, keys, meta or {}
        frames = sorted(detections)
        self.f, self.c, self.b = [], [], []
        for f in frames:
            for d in detections[f]:
                self.f.append(f); self.c.append(d["conf"]); self.b.append(d["box"])
        self.f = np.array(self.f, dtype=np.int64)
        self.c = np.array(self.c, dtype=np.float64)
        n = len(self.f)
        self.best = np.zeros(n)
        self.match = {k: np.full(n, -1, dtype=np.int64) for k in keys}
        self.ep_first, self.ep_ids = {k: {} for k in keys}, {k: {} for k in keys}
        for f, anns in annotations.items():
            for a in anns:
                for k in keys:
                    pid = self.ep_ids[k].setdefault(a[k], len(self.ep_ids[k]))
                    self.ep_first[k][pid] = min(self.ep_first[k].get(pid, f), f)
        for i in range(n):
            anns = annotations.get(int(self.f[i]), [])
            bi, m = 0.0, None
            for a in anns:
                v = iou(self.b[i], a["box"])
                if v > bi:
                    bi = v
                if v >= 0.5 and m is None:
                    m = a
            self.best[i] = bi
            if m is not None:
                for k in keys:
                    self.match[k][i] = self.ep_ids[k][m[k]]
        self.n_ep = {k: len(self.ep_ids[k]) for k in keys}


    def tracks(self, ct, iou_assoc=0.3, max_gap=2, predict=False):
        open_tracks, rows, ntid = [], [], 0
        idx = np.flatnonzero(self.c >= ct)
        if len(idx) == 0:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)

        fr = self.f[idx]
        cuts = np.flatnonzero(np.diff(fr)) + 1
        for grp in np.split(idx, cuts):
            f = int(self.f[grp[0]])
            open_tracks = [t for t in open_tracks if f - t[1][-1] <= max_gap]
            order = sorted(grp.tolist(), key=lambda i: -self.c[i])
            for i in order:
                box = self.b[i]
                best, best_iou = None, iou_assoc
                for t in open_tracks:
                    ref = t[2][-1]
                    if predict and len(t[2]) >= 2:
                        p, q = np.asarray(t[2][-2]), np.asarray(t[2][-1])
                        gap = max(1, t[1][-1] - t[1][-2])
                        ref = (q + (q - p) / gap * (f - t[1][-1])).tolist()
                    v = iou(box, ref)
                    if v >= best_iou and t[1][-1] != f:
                        best, best_iou = t, v
                if best is None:
                    best = [ntid, [], [], []]
                    ntid += 1
                    open_tracks.append(best)
                best[1].append(f); best[2].append(box); best[3].append(i)
                rows.append((best[0], i))
        rows.sort()
        r = np.array(rows, dtype=np.int64)
        return r[:, 0], r[:, 1]


class Tracks:

    def __init__(self, v: Video, tid, det):
        self.v, self.tid, self.det = v, tid, det
        n = len(tid)
        start = np.ones(n, bool)
        start[1:] = tid[1:] != tid[:-1]
        self.start = start
        grp = np.cumsum(start) - 1
        first = np.flatnonzero(start)
        self.L = np.arange(n) - first[grp] + 1
        c = v.c[det]
        cs = np.cumsum(c)
        base = np.where(first[grp] > 0, cs[first[grp] - 1], 0.0)
        self.sum = cs - base
        self.mean = self.sum / self.L
        cm = np.maximum.accumulate
        mx = c.copy()
        for s, e in zip(first, np.append(first[1:], n)):
            mx[s:e] = cm(c[s:e])
        self.max = mx
        self.conf = c
        self.frame = v.f[det]
        self.grp = grp

    def first_fire(self, fire):
        rows = np.flatnonzero(fire)
        if len(rows) == 0:
            return np.zeros(0, np.int64)
        _, first = np.unique(self.grp[rows], return_index=True)
        return self.det[rows[first]]


    def fixed(self, k, theta=THETA, agg="mean"):
        s = self.mean if agg == "mean" else self.max
        return self.first_fire((self.L >= k) & (s >= theta))

    def adaptive(self, k_max, gamma, theta=THETA, agg="mean"):
        s = self.mean if agg == "mean" else self.max
        k = 1 + np.round((k_max - 1) * (1.0 - s) ** gamma)
        return self.first_fire((self.L >= k) & (s >= theta))

    def vote(self, window, votes):
        key = self.grp * 10 ** 9 + self.frame
        cnt = np.arange(len(key)) + 1 - np.searchsorted(key, key - window, side="right")
        return self.first_fire(cnt >= votes)

    def hysteresis(self, tau_high, k_fix):
        return self.first_fire((self.conf >= tau_high) | (self.L >= k_fix))

    def evidence(self, evidence, min_len):
        return self.first_fire((self.sum >= evidence) & (self.L >= min_len))

    def length(self, k_fix):
        return self.first_fire(self.L >= k_fix)


def metrics(v: Video, alert_det, key="polyp_id"):
    m = v.match[key][alert_det]
    false = m < 0
    pure = false & (v.best[alert_det] <= 0.0)
    hit = {}
    for i, pid in zip(alert_det[~false], m[~false]):
        f = int(v.f[i])
        if pid not in hit or f < hit[pid]:
            hit[pid] = f
    lat = [hit[p] - v.ep_first[key][p] for p in hit]
    return (int(false.sum()), int(pure.sum()), int(false.sum() - pure.sum()),
            v.n_frames, len(hit), v.n_ep[key], lat)


def grid(kgrid):
    G = [0.5, 1.0, 2.0, 3.0]
    return {
        "Unfiltered detector": ("none", [{}]),
        "Fixed persistence, k=3": ("fixed", [{"k": 3}]),
        "Fixed persistence, k=6": ("fixed", [{"k": 6}]),
        "Fixed persistence (k selected)": ("fixed", [{"k": k} for k in kgrid]),
        "Confidence-adaptive": ("adaptive", [{"k_max": k, "gamma": g} for k, g in product(kgrid, G)]),
        "Sliding-window vote": ("vote", [{"window": w, "votes": n} for w, n in product([4, 5, 6, 8], [2, 3, 4, 5]) if n <= w]),
        "Hysteresis": ("hysteresis", [{"tau_high": t, "k_fix": k} for t, k in
                                      product([0.5, 0.6, 0.7, 0.8, 0.9], sorted({3, 4, 6, 8} | {k for k in kgrid if k > 8}))]),
        "Confidence accumulation": ("evidence", [{"evidence": e, "min_len": m} for e, m in
                                                 product([0.5, 1.0, 1.5, 2.0, 3.0, 4.0], [1, 2, 3])]),
        "Motion-compensated persistence": ("motion", [{"k_fix": k, "iou_assoc": j} for k, j in product(kgrid, [0.2, 0.3, 0.4])]),
    }


def per_video_all(v: Video, kgrid, extra=None):
    import json
    G = grid(kgrid)
    out = {}
    for ct in CT:
        tid, det = v.tracks(ct)
        T = Tracks(v, tid, det) if len(tid) else None
        mot = {}
        for cfg, (kind, plist) in G.items():
            for p in plist:
                if kind == "none":
                    al = np.flatnonzero(v.c >= ct)
                elif kind == "motion":
                    j = p["iou_assoc"]
                    if j not in mot:
                        t2, d2 = v.tracks(ct, iou_assoc=j, predict=True)
                        mot[j] = Tracks(v, t2, d2) if len(t2) else None
                    al = mot[j].length(p["k_fix"]) if mot[j] is not None else np.zeros(0, np.int64)
                elif T is None:
                    al = np.zeros(0, np.int64)
                else:
                    al = getattr(T, kind)(**p)
                pj = json.dumps({"conf_thresh": ct, **p}, sort_keys=True)
                for key in v.keys:
                    out[(cfg, pj, key)] = metrics(v, al, key)
        if extra:
            for name, fn in extra.items():
                for p, al in fn(v, T, ct):
                    pj = json.dumps({"conf_thresh": ct, **p}, sort_keys=True)
                    for key in v.keys:
                        out[(name, pj, key)] = metrics(v, al, key)
    return out

"""Reference implementation: tracklets, filtering rules and alert-level measures."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Tracklet:
    boxes: list = field(default_factory=list)
    confs: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    alerted: bool = False
    tid: int = -1

    def __len__(self):
        return len(self.frames)

    @property
    def mean_conf(self):
        return float(np.mean(self.confs))

    @property
    def max_conf(self):
        return float(np.max(self.confs))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def required_persistence(mean_conf, k_max, gamma):
    if gamma <= 0:
        return k_max
    return int(1 + round((k_max - 1) * (1.0 - mean_conf) ** gamma))


def run_filter(detections, *, mode, k_max, gamma, theta, conf_thresh,
               iou_assoc=0.3, max_gap=2, agg="mean", trace=None):
    open_tracks, alerts, next_tid = [], [], 0
    for f in sorted(detections):
        dets = [d for d in detections[f] if d["conf"] >= conf_thresh]
        if mode == "none":
            alerts.extend({"frame": f, "box": d["box"], "conf": d["conf"]} for d in dets)
            continue
        open_tracks = [t for t in open_tracks if f - t.frames[-1] <= max_gap]
        for d in sorted(dets, key=lambda x: -x["conf"]):
            best, best_iou = None, iou_assoc
            for t in open_tracks:
                v = iou(d["box"], t.boxes[-1])
                if v >= best_iou and t.frames[-1] != f:
                    best, best_iou = t, v
            if best is None:
                best = Tracklet(tid=next_tid)
                next_tid += 1
                open_tracks.append(best)
            best.boxes.append(d["box"]); best.confs.append(d["conf"]); best.frames.append(f)
        for t in open_tracks:
            if t.alerted or t.frames[-1] != f:
                continue
            s = {"mean": t.mean_conf, "max": t.max_conf, "last": t.confs[-1]}[agg]
            k = k_max if mode == "fixed" else required_persistence(s, k_max, gamma)
            fired = len(t) >= k and s >= theta
            if trace is not None:
                trace.append({"frame": f, "track": t.tid, "len": len(t), "conf": t.confs[-1],
                              "agg_conf": s, "required_k": k, "box": t.boxes[-1], "alerted": fired})
            if fired:
                t.alerted = True
                alerts.append({"frame": f, "box": t.boxes[-1], "conf": s, "len": len(t)})
    return alerts


def _associate(dets, open_tracks, f, iou_assoc, predict=False):
    for d in sorted(dets, key=lambda x: -x["conf"]):
        best, best_iou = None, iou_assoc
        for t in open_tracks:
            ref = t.boxes[-1]
            if predict and len(t.boxes) >= 2:
                p, q = np.asarray(t.boxes[-2]), np.asarray(t.boxes[-1])
                gap = max(1, t.frames[-1] - t.frames[-2])
                ref = (q + (q - p) / gap * (f - t.frames[-1])).tolist()
            v = iou(d["box"], ref)
            if v >= best_iou and t.frames[-1] != f:
                best, best_iou = t, v
        if best is None:
            best = Tracklet()
            open_tracks.append(best)
        best.boxes.append(d["box"]); best.confs.append(d["conf"]); best.frames.append(f)
    return open_tracks


def run_baseline(detections, *, rule, conf_thresh, iou_assoc=0.30, max_gap=2,
                 window=6, votes=3, tau_high=0.80, k_fix=6, evidence=2.0,
                 min_len=1, predict=False):
    open_tracks, alerts = [], []
    for f in sorted(detections):
        dets = [d for d in detections[f] if d["conf"] >= conf_thresh]
        open_tracks = [t for t in open_tracks if f - t.frames[-1] <= max_gap]
        open_tracks = _associate(dets, open_tracks, f, iou_assoc,
                                 predict=(rule == "motion") or predict)
        for t in open_tracks:
            if getattr(t, "alerted", False) or t.frames[-1] != f:
                continue
            if rule == "vote":
                fire = len([x for x in t.frames if x > f - window]) >= votes
            elif rule == "hysteresis":
                fire = (t.confs[-1] >= tau_high) or (len(t) >= k_fix)
            elif rule == "evidence":
                fire = (float(np.sum(t.confs)) >= evidence) and (len(t) >= min_len)
            elif rule == "motion":
                fire = len(t) >= k_fix
            else:
                raise ValueError(rule)
            if fire:
                t.alerted = True
                alerts.append({"frame": f, "box": t.boxes[-1],
                               "conf": float(np.mean(t.confs)), "len": len(t)})
    return alerts


def evaluate_alerts(alerts, annotations, n_frames, iou_hit=0.5):
    episode_first = {}
    for f, anns in annotations.items():
        for a in anns:
            pid = a["polyp_id"]
            episode_first[pid] = min(episode_first.get(pid, f), f)
    hit_frame = {}
    n_false = n_loc = n_pure = 0
    false_pos, true_pos = [], []
    for al in alerts:
        anns = annotations.get(al["frame"], [])
        best_iou, matched = 0.0, None
        for a in anns:
            v = iou(al["box"], a["box"])
            if v > best_iou:
                best_iou = v
            if v >= iou_hit and matched is None:
                matched = a["polyp_id"]
        if matched is None:
            n_false += 1
            if best_iou > 0.0:
                n_loc += 1
            else:
                n_pure += 1
            false_pos.append(al)
        else:
            true_pos.append(al)
            if matched not in hit_frame:
                hit_frame[matched] = al["frame"]
    lat = [hit_frame[p] - episode_first[p] for p in hit_frame]
    return {"n_false": n_false, "n_pure": n_pure, "n_loc": n_loc,
            "n_episodes": len(episode_first), "n_detected": len(hit_frame),
            "n_frames": n_frames, "lat_list": lat,
            "video_median_lat": float(np.median(lat)) if lat else float("nan"),
            "hit_frame": hit_frame, "episode_first": episode_first,
            "false_alerts": false_pos, "true_alerts": true_pos}


def load_cache(path, exclude=()):
    cache = {}
    for p in sorted(Path(path).glob("*.json")):
        if p.stem in exclude:
            continue
        d = json.load(open(p))
        if "annotations" not in d:
            continue
        cache[p.stem] = {"detections": {int(k): v for k, v in d["detections"].items()},
                         "annotations": {int(k): v for k, v in d["annotations"].items()},
                         "n_frames": d["n_frames"]}
    return cache


def make_folds(names, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    order = sorted(names)
    rng.shuffle(order)
    return [set(order[i::n_folds]) for i in range(n_folds)]


def score(cache, videos, runner, **kw):
    fa = pure = loc = frames = hit = eps = 0
    vid_med, ep_lat = [], []
    for name in videos:
        v = cache[name]
        al = runner(v["detections"], **kw)
        m = evaluate_alerts(al, v["annotations"], v["n_frames"])
        fa += m["n_false"]; pure += m["n_pure"]; loc += m["n_loc"]
        frames += m["n_frames"]; hit += m["n_detected"]; eps += m["n_episodes"]
        if m["lat_list"]:
            vid_med.append(m["video_median_lat"])
        ep_lat.extend(m["lat_list"])
    return {"fa_per_1k": fa / frames * 1000 if frames else np.nan,
            "pure_per_1k": pure / frames * 1000 if frames else np.nan,
            "sens": hit / eps if eps else np.nan,
            "lat_old": float(np.median(vid_med)) if vid_med else np.nan,
            "lat_ep": float(np.median(ep_lat)) if ep_lat else np.nan,
            "TP": hit, "FN": eps - hit, "FP": fa, "FP_pure": pure, "FP_loc": loc,
            "n_frames": frames, "n_episodes": eps, "ep_lat": ep_lat}

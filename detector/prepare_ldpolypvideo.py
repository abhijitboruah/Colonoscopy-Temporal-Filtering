"""Run the detector over LDPolypVideo, or attach the LDPolypVideo annotations to the released detections."""
import argparse
import json
from pathlib import Path


def load_raw_boxes(video_dir):
    boxes = {}
    for txt in sorted(video_dir.glob("*.txt")):
        lines = [ln.strip() for ln in txt.read_text().splitlines() if ln.strip()]
        n = int(lines[0])
        if len(lines) - 1 != n:
            raise ValueError(f"{txt}: header says {n} boxes but found {len(lines) - 1}")
        boxes[int(txt.stem) - 1] = [[float(x) for x in ln.split()] for ln in lines[1:]]
    return boxes


def episode_ids(frame_boxes):
    annotations, next_id, slots = {}, 0, []
    for f in sorted(frame_boxes):
        boxes = frame_boxes[f]
        if not boxes:
            slots = []
            continue
        ranked = sorted(boxes, key=lambda b: b[0])
        while len(slots) < len(ranked):
            slots.append(next_id)
            next_id += 1
        annotations[f] = [{"box": b, "polyp_id": slots[i]} for i, b in enumerate(ranked)]
    return annotations


def video_annotations(root, video_id):
    d = root / "Annotations" / video_id
    return episode_ids(load_raw_boxes(d)) if d.is_dir() else {}


def cmd_attach(a):
    root, n = Path(a.ldpolypvideo_root), 0
    for p in sorted(Path(a.detections).glob("*.json")):
        d = json.load(open(p))
        ann = video_annotations(root, p.stem)
        d["annotations"] = {str(k): v for k, v in ann.items()}
        json.dump(d, open(p, "w"))
        n += 1
    print(f"annotations attached to {n} clips")


def cmd_build(a):
    import cv2
    from ultralytics import YOLO

    root = Path(a.ldpolypvideo_root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    for img_dir in sorted(d for d in (root / "Images").iterdir() if d.is_dir()):
        frames = sorted(img_dir.glob("*.jpg"))
        if not frames:
            continue
        dets = {}
        for jpg in frames:
            r = model.predict(cv2.imread(str(jpg)), conf=0.01, imgsz=a.imgsz, verbose=False)[0]
            dets[str(int(jpg.stem) - 1)] = [{"box": b.xyxy[0].tolist(), "conf": float(b.conf)} for b in r.boxes]
        ann = video_annotations(root, img_dir.name)
        json.dump({"video": img_dir.name, "fps": None, "n_frames": len(frames), "detections": dets,
                   "annotations": {str(k): v for k, v in ann.items()}}, open(out / f"{img_dir.name}.json", "w"))
        print(img_dir.name, len(frames), "frames")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    at = sub.add_parser("attach")
    at.add_argument("--ldpolypvideo-root", required=True)
    at.add_argument("--detections", default="data/ldpolypvideo")
    at.set_defaults(func=cmd_attach)
    b = sub.add_parser("build")
    b.add_argument("--ldpolypvideo-root", required=True)
    b.add_argument("--weights", required=True)
    b.add_argument("--imgsz", type=int, default=320)
    b.add_argument("--out", default="data/ldpolypvideo")
    b.set_defaults(func=cmd_build)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()

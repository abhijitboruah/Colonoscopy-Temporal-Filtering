"""Convert the Kvasir-SEG and CVC-ClinicDB masks to YOLO boxes with a grouped training and validation split."""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label

MIN_AREA_PX = 50


def masks_to_boxes(mask: np.ndarray, min_area: int = MIN_AREA_PX) -> list[tuple[int, int, int, int]]:
    lbl, n = label(mask)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(xs) < min_area:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def provenance(name: str) -> str:
    return "cvc_clinicdb" if re.fullmatch(r"\d+\.png", name) else "kvasir_seg"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="dir containing NewTRimage/ and NewTRmask/")
    p.add_argument("--out", default="data/yolo_polyp_grouped")
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    raw = Path(a.raw)
    img_dir, mask_dir = raw / "NewTRimage", raw / "NewTRmask"
    names = sorted(f.name for f in img_dir.glob("*.png"))
    if not names:
        raise SystemExit(f"no images found under {img_dir}")

    out = Path(a.out)
    for split in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / split).mkdir(parents=True, exist_ok=True)

    by_source: dict[str, list[str]] = {"kvasir_seg": [], "cvc_clinicdb": []}
    for name in names:
        by_source[provenance(name)].append(name)

    rng = random.Random(a.seed)
    val_set: set[str] = set()


    kvasir_group = sorted(by_source["kvasir_seg"])
    rng.shuffle(kvasir_group)
    n_val_kvasir = round(len(kvasir_group) * a.val_frac)
    val_set.update(kvasir_group[:n_val_kvasir])


    cvc_ids = sorted(int(n.removesuffix(".png")) for n in by_source["cvc_clinicdb"])
    n_val_cvc = round(len(cvc_ids) * a.val_frac)
    start = rng.randrange(len(cvc_ids) - n_val_cvc + 1)
    val_set.update(f"{i}.png" for i in cvc_ids[start:start + n_val_cvc])

    n_boxes, n_skipped_empty = 0, 0
    counts = {"kvasir_seg": {"train": 0, "val": 0}, "cvc_clinicdb": {"train": 0, "val": 0}}
    for name in names:
        split_name = "val" if name in val_set else "train"
        source = provenance(name)

        mask = np.array(Image.open(mask_dir / name))
        img = Image.open(img_dir / name)
        w, h = img.size
        boxes = masks_to_boxes(mask)
        if not boxes:
            n_skipped_empty += 1
            continue

        n_boxes += len(boxes)
        counts[source][split_name] += 1
        stem = Path(name).stem
        (out / f"images/{split_name}/{stem}.png").write_bytes((img_dir / name).read_bytes())
        lines = []
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / f"labels/{split_name}/{stem}.txt").write_text("\n".join(lines) + "\n")

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: polyp\n"
    )

    for source, c in counts.items():
        print(f"{source}: {c['train']} train / {c['val']} val")
    print(f"total: {sum(c['train'] for c in counts.values())} train / "
          f"{sum(c['val'] for c in counts.values())} val, {n_boxes} boxes")
    if n_skipped_empty:
        print(f"WARNING: {n_skipped_empty} images had zero boxes after "
              f"min-area filtering and were skipped")
    print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()

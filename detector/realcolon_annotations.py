"""Add the REAL-Colon annotations to the released detection bundles in data/realcolon."""
import argparse
import json
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np


def open_archive(path):
    path = Path(path)
    if path.is_dir():
        for p in sorted(path.rglob("*.xml")):
            yield p.name, p.read_bytes()
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if not n.endswith("/"):
                    yield n, z.read(n)
    else:
        with tarfile.open(path, "r|*") as t:
            for m in t:
                if m.isfile():
                    yield m.name, t.extractfile(m).read()


def frame_index(name):
    m = re.search(r"_(\d+)(?:\.0+)?\.xml$", name, re.I)
    return int(m.group(1)) if m else None


def parse_xml(data):
    out = []
    for obj in ET.fromstring(data).iter("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        uid = obj.findtext("unique_id") or obj.findtext("name") or ""
        out.append((uid.strip(), [float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]))
    return out


def find_source(root, vid):
    hits = sorted(p for p in root.iterdir() if p.name.startswith(vid + "_annotation"))
    if not hits:
        raise FileNotFoundError(f"no annotation archive or folder for {vid} under {root}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realcolon-root", required=True)
    ap.add_argument("--bundle", default="data/realcolon")
    a = ap.parse_args()
    root, bundle = Path(a.realcolon_root), Path(a.bundle)
    if (root / "video_info.csv").exists():
        shutil.copy(root / "video_info.csv", bundle / "video_info.csv")
    for p in sorted(bundle.glob("centre_*.npz")):
        z = dict(np.load(p, allow_pickle=False))
        meta = json.loads(str(z["meta"]))
        AV, AF, AL, AB = [], [], [], []
        for vid in meta:
            k = int(vid.split("-")[1])
            for name, data in open_archive(find_source(root, vid)):
                f = frame_index(name)
                if f is None:
                    continue
                for uid, box in parse_xml(data):
                    AV.append(k); AF.append(f); AL.append(uid); AB.append(box)
        z.update(a_vid=np.array(AV, np.int16), a_frame=np.array(AF, np.int32), a_lesion=np.array(AL),
                 a_box=np.round(np.array(AB, np.float64).reshape(-1, 4) * 10).astype(np.int32))
        np.savez_compressed(p, **z)
        print(p.name, len(meta), "recordings,", len(AV), "annotated boxes")


if __name__ == "__main__":
    main()

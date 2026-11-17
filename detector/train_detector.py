"""Train the YOLOv8n polyp detector on the Kvasir-SEG and CVC-ClinicDB training set."""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo_polyp_grouped/dataset.yaml")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    YOLO("yolov8n.pt").train(data=a.data, imgsz=320, epochs=50, batch=16, seed=0, deterministic=True,
                             project=a.out, name="det_grouped")


if __name__ == "__main__":
    main()

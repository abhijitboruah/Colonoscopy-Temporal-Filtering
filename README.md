# Temporal filtering of false alerts in colonoscopy polyp detection

Code and detector outputs for the paper *Temporal filtering of false alerts in colonoscopy polyp detection* by Abhijit Boruah, Mrinal Gogoi, Shantanu Konwar and Rabinder Kumar Prasad.

The study compares rules that suppress false alerts of a real-time polyp detector by requiring a detection to persist across frames. One YOLOv8n detector is used throughout. Every rule is compared at matched per-episode sensitivity, with settings chosen on data not used to score them. The test data are 156 curated clips from LDPolypVideo and 60 full-length procedures from REAL-Colon.

## Contents

| Folder | What it holds |
| --- | --- |
| `detector/` | Training set construction, detector training, and detector inference on both test datasets |
| `code/` | Filtering rules, alert-level measures and evaluation protocol |
| `data/ldpolypvideo/` | Detector outputs for the 160 LDPolypVideo clips, one JSON file per clip |
| `data/realcolon/` | Detector outputs for the 60 REAL-Colon recordings, in eight compressed bundles |
| `data/split_manifest.json` | The four LDPolypVideo clips held out from all analyses |
| `results/` | Result files produced by `code/` |

The detector outputs hold every box with confidence of at least 0.01 (LDPolypVideo) or 0.05 (REAL-Colon). The dataset annotations are not redistributed. They are added from your own copy of each dataset, as described below.

## Datasets

| Dataset | Use | Source |
| --- | --- | --- |
| Kvasir-SEG | Detector training | Jha et al., MMM 2020 |
| CVC-ClinicDB | Detector training | Bernal et al., Comput Med Imaging Graph 2015 |
| LDPolypVideo | Test, five-fold cross-validation | Ma et al., MICCAI 2021 |
| REAL-Colon | Test, external | Biffi et al., Sci Data 2024, https://figshare.com/articles/media/REAL-colon_dataset/22202866 |

Kvasir-SEG and CVC-ClinicDB are used through the training split of PraNet (Fan et al., MICCAI 2020), the `NewTRimage` and `NewTRmask` folders.

## Installation

Python 3.12 was used.

```bash
pip install -r requirements.txt
```

## Reproducing the results from the released detector outputs

1. Add the LDPolypVideo annotations to the released detections. The root folder must contain `Annotations/<clip>/<frame>.txt`.

   ```bash
   python detector/prepare_ldpolypvideo.py attach --ldpolypvideo-root /path/to/LDPolypVideo
   ```

2. Add the REAL-Colon annotations. The root folder must contain the annotation archives `SSS-VVV_annotations.*` (or extracted folders of the same names) and `video_info.csv`.

   ```bash
   python detector/realcolon_annotations.py --realcolon-root /path/to/REAL-Colon
   ```

3. Run the evaluation from the repository root, in this order.

   ```bash
   python code/cv_unified.py
   python code/select_full.py
   python code/check_fast.py
   python code/realcolon_eval.py
   python code/realcolon_seconds.py
   python code/seconds_compare.py
   python code/decompose.py
   python code/boot_tables.py
   ```

The whole evaluation takes about 10 minutes on a two-core processor. `check_fast.py` confirms that the fast engine in `fast.py` gives the same results as the frame-by-frame reference implementation in `core.py`.

## Results and the paper

| Paper item | Produced by | File |
| --- | --- | --- |
| Table 1 | `cv_unified.py`, `decompose.py`, `boot_tables.py` | `results/tables/ldp_cv.csv` |
| Table 2 | `realcolon_eval.py`, `decompose.py`, `boot_tables.py` | `results/tables/realcolon_zero_shot.csv`, `results/tables/realcolon_loco.csv` |
| Table 3 | `realcolon_seconds.py` | `results/realcolon/seconds_loco.csv` |
| Persistence in seconds | `realcolon_seconds.py`, `seconds_compare.py` | `results/realcolon/seconds_sweep.csv`, `results/realcolon/seconds_compare.json` |
| One alert per tracklet against persistence | `decompose.py` | `results/tables/onset.json` |

## Reproducing the detector outputs

1. Build the training set with a grouped validation split. CVC-ClinicDB frames come from video sequences, so its validation images are held out as one contiguous block of frame numbers.

   ```bash
   python detector/build_dataset.py --raw /path/to/TrainDataset
   ```

2. Train the detector (YOLOv8n, 320 pixels, 50 epochs, seed 0).

   ```bash
   python detector/train_detector.py
   ```

3. Run the detector over LDPolypVideo.

   ```bash
   python detector/prepare_ldpolypvideo.py build --ldpolypvideo-root /path/to/LDPolypVideo --weights runs/det_grouped/weights/best.pt
   ```

4. Run the detector over REAL-Colon with `detector/realcolon_inference.ipynb` in Google Colab on a GPU runtime. The notebook downloads each recording from Figshare, runs the detector and deletes the frames. Its last section writes the eight bundles for `data/realcolon/`.

## Notes

LDPolypVideo provides no frame rate, so all LDPolypVideo results are given per frame. REAL-Colon results per minute and in seconds use the frame rate of each recording from `video_info.csv`.

## License

The code is released under the MIT License. The datasets remain under the terms set by their creators.

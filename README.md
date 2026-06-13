# HSV Split-and-Merge Fruit Segmentation

**Computer Vision - Assignment 2 - Deggendorf Institute of Technology - Summer 2026**

A classical (non-deep-learning) fruit segmentation pipeline built on the HSV colour
space. The core algorithm is a quadtree **split** followed by a greedy
Region-Adjacency-Graph **merge**, extended with Sobel-driven splitting, edge-aware
merging, circular hue statistics, z-scored nearest-neighbour classification with
rejection, and morphological post-processing -- all implemented in NumPy without
library segmentation functions.

---

## Results

| Configuration | Classes | Test images | Correct | Accuracy |
|---|---|---|---|---|
| 3-class | Cherry, Orange, Banana | 180 | 180 | **100.0%** |
| 5-class | + Avocado, Cucumber | 290 | 290 | **100.0%** |
| 10-class | full set | 590 | 586 | **99.3%** |

10-class per-class metrics (Fruits-360 Test):

| Class | Precision | Recall | F1 | Train images |
|---|---|---|---|---|
| Cherry | 1.000 | 1.000 | 1.000 | 200 |
| Orange | 1.000 | 1.000 | 1.000 | 200 |
| Banana | 1.000 | 1.000 | 1.000 | 200 |
| Avocado | 1.000 | 0.983 | 0.992 | 200 |
| Cucumber | 0.943 | 1.000 | 0.971 | 150 |
| Cherry Black | 1.000 | 1.000 | 1.000 | 200 |
| Cucumber 3 | 1.000 | 0.950 | 0.974 | 200 |
| Huckleberry | 0.984 | 1.000 | 0.992 | 200 |
| Raspberry | 1.000 | 1.000 | 1.000 | 200 |
| Lychee | 1.000 | 1.000 | 1.000 | 200 |

The 4 errors at 10-class all occur at known hue-overlap boundaries:
Avocado/Cucumber (yellow-green zone) and Cucumber 3 at the green-cyan boundary.

---

## Project structure

```
Computer-Vision-2/
|
+-- fruitseg/                    # importable library package
|   +-- __init__.py              # public API, version
|   +-- color_space.py           # (wl) RGB->HSV  (oc) guard_mask, circular_mean/variance
|   +-- preprocessing.py         # (oc) median_filter, gaussian_lowpass, sobel_edges
|   +-- split_merge.py           # (oc) split_quadtree, merge_regions  <- core algorithm
|   +-- features.py              # (oc) RegionStats, region_features, feature vectors
|   +-- classify.py              # (oc) build_references, classify_regions (z-scored NN)
|   +-- postprocess.py           # (oc) disk morphology, area_filter, connected components
|   +-- pipeline.py              # segment_image, SegmentationConfig, add_legend
|   +-- evaluation.py            # evaluate_classification, confusion matrix, F1
|
+-- scripts/
|   +-- run.py                   # evaluate on Test folder or segment a single image
|   +-- run_all.py               # produce all results, graphs, and overlays in one shot
|   +-- selftest.py              # synthetic end-to-end test (no dataset needed)
|   +-- make_slides.py           # generate results/presentation.pdf (15 slides)
|
+-- docs/
|   +-- ALGORITHM.md             # full algorithm description + IEEE references
|   +-- USAGE.md                 # parameter-tuning guide + advanced usage
|   +-- report.md                # research document (abstract to conclusion)
|
+-- results/
|   +-- eval_3fruit.csv          # confusion matrix + metrics, 3-class
|   +-- eval_5fruit.csv          # confusion matrix + metrics, 5-class
|   +-- eval_10fruit.csv         # confusion matrix + metrics, 10-class
|   +-- presentation.pdf         # generated slide deck (15 slides)
|   +-- graphs/                  # accuracy bars, confusion heatmaps, pipeline steps
|   +-- *.png                    # scene overlay images
|
+-- data/                        # datasets (not version-controlled)
|   +-- Fruits-360/
|   |   +-- fruits-360_100x100/fruits-360/Training/   # training images (100x100)
|   |   +-- fruits-360_100x100/fruits-360/Test/        # test images
|   |   +-- fruits-360_multi/test-multiple_fruits/     # multi-fruit scenes
|   +-- Fruits-262/Fruit-262/                          # natural-environment photos
|
+-- requirements.txt
+-- pyproject.toml               # pip install -e . installs the fruitseg package
+-- README.md
```

`(wl)` = library call allowed (NumPy / OpenCV / SciPy)
`(oc)` = own code -- only basic NumPy array operations, no library segmentation

---

## Algorithm pipeline

```
BGR image
  -> median filter (5x5, oc)
  -> Gaussian low-pass (sigma=1.5, oc)
  -> RGB->HSV (wl: cv2.cvtColor)
  -> guard mask: exclude dark/achromatic pixels where hue is undefined (oc)
  -> Sobel edge map (oc)
  -> SPLIT: quadtree, forced depth>=2 (>=16 start regions) (oc)
  -> MERGE: RAG, greedy most-similar-first, edge-aware veto (oc)
  -> feature vectors: 7-D [cos(H), sin(H), mean_S, circ_var_H, var_S, mean_V, 0] (oc)
  -> z-scored weighted NN classification with rejection (oc)
  -> per-class morphological opening + closing (disk SE, oc)
  -> 8-connected area filter (BFS, oc)
  -> colour overlay output
```

Key extensions beyond the minimal baseline:

- **Circular hue statistics** -- mean and variance computed via cos/sin encoding
  so the 0/360 degree wrap-around is handled correctly (Mardia & Jupp, 2000)
- **Sobel integration** in both split (edge criterion tau_E) and merge (edge veto)
  to align region boundaries with real object edges
- **Z-scored weighted NN** with per-dimension weights and a rejection threshold --
  regions too far from every class reference are labelled background (-1)
- **Own-code morphology** using disk structuring elements built with NumPy stride
  tricks; no cv2 morphology functions are used

---

## Class selection (10 classes)

Classes were chosen to maximise hue spread across the colour wheel while keeping
mean saturation above 0.40. Ramping strategy: 3 -> 5 -> 10 classes, each increment
adding classes that are separable from the existing set.

| # | Folder (Fruits-360) | Display name | Mean hue |
|---|---------------------|--------------|----------|
| 1 | Cherry 1 | Cherry | ~358 deg |
| 2 | Orange 1 | Orange | ~27 deg |
| 3 | Banana 1 | Banana | ~46 deg |
| 4 | Avocado 1 | Avocado | ~71 deg |
| 5 | Cucumber 1 | Cucumber | ~104 deg |
| 6 | Cherry Wax Black 1 | Cherry Black | ~314 deg |
| 7 | Cucumber 3 | Cucumber 3 | ~187 deg |
| 8 | Huckleberry 1 | Huckleberry | ~221 deg |
| 9 | Raspberry 1 | Raspberry | ~274 deg |
| 10 | Lychee 1 | Lychee | ~3 deg |

Edit `SPEC_10` in `scripts/run.py` to change the selection. Folder names must
match your local Fruits-360 copy exactly.

---

## Installation

```bash
# from the project root
pip install -r requirements.txt

# optional: install the fruitseg package for clean imports anywhere
pip install -e .

# make_slides.py also needs matplotlib
pip install matplotlib
```

Python 3.9+ required. Tested on Python 3.11 and 3.14.

---

## Quick self-test (no dataset needed)

Builds synthetic solid-colour fruit blobs, runs every pipeline stage, and prints
a small confusion matrix -- verifies the full code path without any real data:

```bash
python scripts/selftest.py
```

Expected output ends with: `SELF-TEST COMPLETE -- all modules executed.`

---

## Running on the real dataset

### Dataset layout

The Fruits-360 download unpacks to a nested directory. `run.py` auto-detects
the real Training/Test path from a short root, so both forms work:

```
data/Fruits-360                                             <- short root (pass this)
data/Fruits-360/fruits-360_100x100/fruits-360/Training     <- resolved Training path
data/Fruits-360/fruits-360_100x100/fruits-360/Test         <- resolved Test path
```

### 1. Evaluate on Fruits-360 Test (confusion matrix)

```bash
# 3-class
python scripts/run.py --train data/Fruits-360 --test data/Fruits-360 \
                      --evaluate --nfruits 3

# 5-class
python scripts/run.py --train data/Fruits-360 --test data/Fruits-360 \
                      --evaluate --nfruits 5

# 10-class
python scripts/run.py --train data/Fruits-360 --test data/Fruits-360 \
                      --evaluate --nfruits 10

# save a CSV report alongside the console output
python scripts/run.py --train data/Fruits-360 --test data/Fruits-360 \
                      --evaluate --nfruits 10 --save-report results/my_10class.csv
```

### 2. Segment a single image (scene overlay)

```bash
# multi-fruit scene (3-class)
python scripts/run.py --train data/Fruits-360 \
    --image "data/Fruits-360/fruits-360_multi/test-multiple_fruits/cherry_in_the_basket.jpg" \
    --out results/overlay.png --nfruits 3

# Fruits-262 cross-dataset demo
python scripts/run.py --train data/Fruits-360 \
    --image data/Fruits-262/Fruit-262/cherry/cherry_1.jpg \
    --out results/f262_overlay.png --nfruits 3
```

Single-image runs use a scene-tuned parameter preset (relaxed thresholds, mask
expansion). Pass `--no-scene-tuning` to use the base evaluation config instead.

### 3. Produce all results at once

```bash
python scripts/run_all.py
```

Generates in one run: confusion-matrix heatmaps, per-class bar charts, accuracy
comparison graph, scene overlay images, hue-distribution plot, region-count table,
and all CSV reports. Output goes to `results/` and `results/graphs/`.

### 4. Generate the presentation PDF

```bash
python scripts/make_slides.py --out results/presentation.pdf

# also export individual slides as PNGs
python scripts/make_slides.py --out results/presentation.pdf \
                              --png-dir results/slides/
```

---

## CLI reference -- scripts/run.py

| Flag | Default | Description |
|------|---------|-------------|
| `--train PATH` | required | Fruits-360 root or Training folder |
| `--test PATH` | -- | Fruits-360 root or Test folder (needed for `--evaluate`) |
| `--evaluate` | off | Run confusion-matrix evaluation on Test folder |
| `--nfruits N` | 3 | Number of classes: 3, 5, or 10 |
| `--image PATH` | -- | Segment a single image and write an overlay |
| `--out PATH` | `results/overlay.png` | Output path for `--image` runs |
| `--max-side N` | 480 | Resize longest image side to N before processing (0 = off) |
| `--no-scene-tuning` | off | Use base config instead of scene preset for `--image` |
| `--save-report PATH` | -- | Write CSV report (confusion matrix + metrics) |
| `--no-file-list` | off | Suppress per-file listing in console output |

---

## Key parameters -- SegmentationConfig

All thresholds live in `fruitseg/pipeline.py`. **Tune on the Training set only.**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `s_min`, `v_min` | 0.15, 0.15 | Guard mask -- exclude achromatic / dark pixels |
| `tau_h` | 0.05 | Split: max circular hue variance in a block |
| `tau_s` | 0.02 | Split: max saturation variance in a block |
| `tau_e` | 0.15 | Split: max mean Sobel energy in a block |
| `min_size` | 4 | Min block side length before stopping split |
| `min_start_depth` | 2 | Forced depth (2 -> >=16 start regions) |
| `hue_thresh` | 0.30 | Merge: max circular hue distance between regions |
| `sat_thresh` | 0.15 | Merge: max saturation mean distance between regions |
| `merged_var_h` | 0.08 | Merge: max hue variance of merged region |
| `merged_var_s` | 0.03 | Merge: max saturation variance of merged region |
| `edge_veto` | 0.35 | Merge: block if shared-boundary Sobel exceeds this |
| `reject_z` | 1.8 | Classification rejection radius (weighted z-score units) |
| `morph_radius` | 3 | Disk radius for morphological opening/closing |
| `min_area` | 150 | Drop connected components smaller than this (pixels) |
| `max_side` | 320 | Resize longest side before processing (0 = off) |

---

## Documentation

| File | Contents |
|------|----------|
| `docs/ALGORITHM.md` | Full algorithm description with IEEE references [1]-[10] |
| `docs/USAGE.md` | Parameter-tuning guide, reporting methodology, advanced usage |
| `docs/report.md` | Full research document (abstract, methodology, results, limitations) |
| `results/presentation.pdf` | 15-slide PDF: pipeline, results, limitations |

---

## Academic integrity

This is a reference implementation to **study and defend**. The assignment awards
0 points to both team members if any `(oc)` module cannot be explained from first
principles. Walk through `split_merge.py`, the circular statistics in
`color_space.py`, and the z-score logic in `classify.py` until you can derive
each step independently.

**Tuning rule**: Use only the Fruits-360 Training folder to set thresholds.
Using the Test set or Fruits-262 for parameter tuning is academic misconduct.

# Usage Guide

This package implements the complete HSV split-and-merge fruit segmentation
pipeline with all extensions. See **README.md** for the algorithm description and
the IEEE references; this file is the practical "how to run it" guide.

## Project map

```
fruit-segmentation/
├── README.md              # overview + quick start
├── pyproject.toml         # packaging (pip install -e .)
├── requirements.txt
├── fruitseg/        # the library (import package)
│   ├── __init__.py
│   ├── color_space.py     # (wl) RGB→HSV, (oc) guard mask + circular hue stats
│   ├── preprocessing.py   # (wl) median + Gaussian, (oc) Sobel edges
│   ├── split_merge.py     # (oc) quadtree SPLIT + RAG MERGE  ← core of the task
│   ├── postprocess.py     # (wl) morphology + area filter
│   ├── features.py        # (oc) per-region feature vectors
│   ├── classify.py        # references from Train + (oc) NN classifier w/ rejection
│   ├── pipeline.py        # orchestration + SegmentationConfig (all parameters)
│   └── evaluation.py      # confusion matrix + accuracy/recall/F1
├── scripts/               # CLI entry points
│   ├── run.py             # build references, segment, evaluate
│   └── selftest.py        # synthetic end-to-end test (no dataset needed)
├── notebooks/
│   └── demo.ipynb  # stage-by-stage visual demo
├── docs/                  # ALGORITHM.md (references) + USAGE.md (this file)
├── data/                  # datasets go here (gitignored)
└── results/               # generated overlays / matrices
```

`(wl)` = library allowed · `(oc)` = own code (basic numpy only).

## Requirements

```
pip install numpy opencv-python scipy
```

## Quick self-test (no dataset)

Verifies every module runs and writes two demo images:

```
python scripts/selftest.py
```

## 1. Validate on the Fruits-360 Test folder (confusion matrix)

```
python scripts/run.py --train /path/to/Fruits-360/Training \
              --test  /path/to/Fruits-360/Test \
              --evaluate --nfruits 3
```

Prints the confusion matrix and the per-class precision/recall/F1 table, and
counts correct vs. wrong files. For the bonus, repeat with `--nfruits 5` and
`--nfruits 10`.

## 2. Proof-of-concept on a multi-fruit / Fruits-262 image

```
python scripts/run.py --train /path/to/Fruits-360/Training \
              --image /path/to/scene.jpg \
              --out overlay.png --nfruits 3
```

Writes `overlay.png` with each detected fruit class tinted in its colour
(red, yellow, green, …), to resemble the assignment's target figure. For
`--image` runs, the CLI uses a stricter real-scene preset than the validation
run: higher `s_min`/`v_min`, lower merge tolerances, stronger edge veto,
smaller `reject_z`, and stronger cleanup. This keeps non-reference objects such
as grapes from being forced into the nearest selected fruit class. Use
`--no-scene-tuning` to disable this preset.

For the provided mixed bowl image
`apple_apricot_peach_peach(flat)_pear_plum_pomegranate_3.jpg`, use the
matching class preset instead of the generic assignment 10-fruit set:

```
python scripts/run.py --train data/Fruits-360/fruits-360_100x100/fruits-360/Training \
              --image "data/Fruits-360/fruits-360_multi/test-multiple_fruits/apple_apricot_peach_peach(flat)_pear_plum_pomegranate_3.jpg" \
              --out results/apple_apricot_peach_pear_plum_pomegranate_overlay.png \
              --preset mixed-bowl --max-side 640
```

This preset builds references for Apple, Apricot, Peach, Peach Flat, Pear, Plum,
and Pomegranate. The dark plum is difficult for HSV-only segmentation because
most of its usable colour lies in low-luminance pixels, so the pipeline rejects
uncertain plum pixels rather than forcing a large wrong label.

## 3. Adjusting the fruit classes

Edit `SPEC_10` in `run.py`. Each entry is
`("Fruits-360 folder name", "display name", (B, G, R) overlay colour)`.
The folder names must match your local Fruits-360 copy exactly. The list is
ranked by hue separation: the first three are the well-separated starter set; the
later ones deliberately share hues (mandarin/orange, strawberry/raspberry) so the
report can analyse them as the worst-performing classes.

## 4. Tuning parameters

All thresholds live in `SegmentationConfig` (`fruitseg/pipeline.py`). The ones you
will most likely tune on the Train set:

| parameter        | effect                                                    |
|------------------|-----------------------------------------------------------|
| `s_min`, `v_min`, `v_max` | guard mask — exclude shadow, desaturation, and very bright spill |
| `tau_h`, `tau_s` | split sensitivity — lower → finer splitting               |
| `tau_e`          | Sobel split trigger — lower → split on weaker edges        |
| `hue_thresh`, `sat_thresh` | merge tolerance — raise → more merging           |
| `merged_var_h`, `merged_var_s` | post-merge homogeneity ceiling             |
| `edge_veto`      | boundary edge above which a merge is blocked              |
| `reject_z`       | classification rejection radius — lower → more background  |
| `min_area`       | drop connected components smaller than this               |
| `refine_masks`, `refine_hue_tol` | trim classified regions back to class-matching pixels |

`scripts/run.py` keeps the base config for `--evaluate` so the confusion matrix
is measured on the intended clean Test images. It applies stricter guard,
merge, rejection, mask-refinement, and cleanup settings only for `--image`
scene overlays.

**Tune on Train only.** Using Test or Fruits-262 to tune parameters is flagged as
cheating in the assignment.

## 5. Producing debug images for the slides

`segment_image(..., return_debug=True)` additionally returns the guard mask, the
Sobel edge map, and the split/merged label images — useful for the
technical-background slides (show the quadtree split and the merge result).

## Important note on academic integrity

This is a reference implementation to **study and defend**. The assignment awards
0 points to both team members if a relevant part cannot be explained. Walk through
each `(oc)` module — especially `split_merge.py` and the circular statistics in
`color_space.py` — until you can derive them yourself. 
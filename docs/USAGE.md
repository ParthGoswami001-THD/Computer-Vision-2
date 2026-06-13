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
`--image` runs, the CLI uses the same final Train-derived algorithm with an
adaptive scene preset intended for qualitative transfer to cluttered images.
This scene path is **not** a separate evaluation protocol and should not be
reported as quantitative validation. Use `--no-scene-tuning` to disable this
preset and inspect the base config directly.


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

`scripts/run.py` keeps the base evaluation path for `--evaluate` so the
confusion matrix is measured on the intended clean Test images only. The
`--image` path reuses the same Train-derived logic but enables the adaptive
scene preset for qualitative real-scene overlays.

**Tune on Train only.** Using Test or Fruits-262 to tune parameters is flagged as
cheating in the assignment.

## 4.1 Reporting methodology

- Use **Fruits-360 Test** as the only quantitative validation set.
- Present `test-multiple_fruits` overlays as qualitative scene demonstrations.
- Present Fruits-262 as a **cross-dataset proof-of-concept / transfer test**,
  not as a tuning set and not as quantitative evaluation.
- The 7-D feature vector, adaptive rejection, and scene-cleanup steps are
  extensions beyond the minimal baseline and should be described as such in the
  report or defense.

## 5. Producing debug images for the slides

`segment_image(..., return_debug=True)` additionally returns the guard mask, the
Sobel edge map, and the split/merged label images — useful for the
technical-background slides (show the quadtree split and the merge result).

## Important note on academic integrity

This is a reference implementation to **study and defend**. The assignment awards
0 points to both team members if a relevant part cannot be explained. Walk through
each `(oc)` module — especially `split_merge.py` and the circular statistics in
`color_space.py` — until you can derive them yourself. 

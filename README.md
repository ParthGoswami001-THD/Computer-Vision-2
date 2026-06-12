# HSV Split-and-Merge Fruit Segmentation

Computer Vision, Assignment 2 — Deggendorf Institute of Technology, Summer 2026.

A classical (non-deep-learning) fruit-segmentation pipeline: RGB→HSV, a
quadtree **split** and Region-Adjacency-Graph **merge** on the homogeneity of
hue and saturation, then nearest-neighbour classification — with every assignment
extension implemented (median/Gaussian preprocessing, Sobel-driven splitting,
edge-aware merging, morphological cleanup, circular hue statistics, rejection).

The algorithm description and IEEE references are in **`docs/ALGORITHM.md`**;
practical how-to-run notes are in **`docs/USAGE.md`**.

## Project structure

```
fruit-segmentation/
├── README.md                 # this file — overview + quick start
├── pyproject.toml            # package metadata (pip install -e .)
├── requirements.txt          # dependencies
├── .gitignore
│
├── fruitseg/                   # ── the library (importable package) ──
│   ├── __init__.py           #    public API
│   ├── color_space.py        #    (wl) RGB→HSV · (oc) guard mask + circular hue stats
│   ├── preprocessing.py      #    (wl) median + Gaussian · (oc) Sobel edges
│   ├── split_merge.py        #    (oc) quadtree SPLIT + RAG MERGE   ← core of the task
│   ├── postprocess.py        #    (wl) morphology + area filter
│   ├── features.py           #    (oc) per-region feature vectors
│   ├── classify.py           #    Train references + (oc) NN classifier with rejection
│   ├── pipeline.py           #    orchestration + SegmentationConfig (all parameters)
│   └── evaluation.py         #    confusion matrix + accuracy/recall/F1
│
├── scripts/                  # ── command-line entry points ──
│   ├── run.py                #    build references, segment an image, or evaluate
│   └── selftest.py           #    synthetic end-to-end test (no dataset needed)
│
├── notebooks/                # ── interactive demo ──
│   └── demo.ipynb     #    stage-by-stage visualisation (slide material)
│
├── docs/                     # ── documentation ──
│   ├── ALGORITHM.md          #    full algorithm + IEEE references
│   └── USAGE.md              #    detailed usage + parameter-tuning guide
│
├── data/                     # ── datasets (not version-controlled) ──
│   └── .gitkeep              #    Fruits-360 / Fruits-262 go here
│
└── results/                  # ── generated overlays / matrices ──
    └── .gitkeep
```

`(wl)` = library allowed (numpy/scipy/opencv) · `(oc)` = own code (basic numpy only).

## Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt
#    (optional) install the package itself for clean imports:
pip install -e .

# 2. verify everything runs (no dataset required)
python scripts/selftest.py

# 3. open the demo notebook (runs on synthetic data out of the box)
jupyter notebook notebooks/demo.ipynb
```

## Running on the real data

Put the datasets under `data/` (see `data/.gitkeep`), then:

```bash
# validation confusion matrix on the Fruits-360 Test folder
python scripts/run.py --train data/Fruits-360/Training \
                      --test  data/Fruits-360/Test \
                      --evaluate --nfruits 3

# proof-of-concept overlay on a multi-fruit / Fruits-262 image
python scripts/run.py --train data/Fruits-360/Training \
                      --image data/fruits-262/scene.jpg \
                      --out results/overlay.png --nfruits 3
```

For the bonus, repeat with `--nfruits 5` and `--nfruits 10`. Edit the class
folder names in `scripts/run.py` (`SPEC_10`) to match your Fruits-360 copy, and
tune `SegmentationConfig` **on the Train set only**.

## Data flow (one image)

```
bgr ─► median+Gaussian ─► HSV + guard mask ─► Sobel edges
     ─► split_quadtree ─► merge_regions ─► region_features
     ─► classify_regions ─► per-class morphology + area filter ─► overlay
```

## Academic integrity

This is a reference implementation to **study and defend**. Be able to explain
every `(oc)` step — especially `split_merge.py` and the circular statistics in
`color_space.py`. Keep the AI-usage declaration on your closing slide.

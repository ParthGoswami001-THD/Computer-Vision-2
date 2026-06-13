# HSV Split-and-Merge Fruit Segmentation

Classical fruit segmentation and classification for **Computer Vision Assignment 2**
at Deggendorf Institute of Technology. The project uses an HSV-based split-and-merge
pipeline, nearest-neighbour classification from Fruits-360 training images, and
qualitative transfer tests on multi-fruit scenes and Fruits-262.

## Overview

This repository implements:

- RGB/BGR to HSV conversion with a luminance and saturation guard mask
- Own-code quadtree split phase with at least 16 initial regions
- Own-code region-adjacency merge phase
- Region classification from Fruits-360 training references
- Quantitative evaluation on the Fruits-360 Test split
- Qualitative overlays for `test-multiple_fruits` and Fruits-262

The main goal is to satisfy the assignment's classical-image-processing requirement
without relying on deep learning or library segmentation functions.

## Current Results

These figures come from the checked-in CSV reports in `results/`:

| Configuration | Classes | Accuracy |
|---|---|---:|
| 3-class | Cherry, Orange, Banana | 100.0% |
| 5-class | + Avocado, Cucumber | 100.0% |
| 10-class | full set | 99.3% |

Representative outputs:

- [Pipeline steps](/home/divine/Documents/Computer-Vision-2/results/graphs/pipeline_steps.png)
- [3-class Fruits-262 proof](/home/divine/Documents/Computer-Vision-2/results/fruits262_3class_proof.png)
- [3-class Fruits-262 composite scene](/home/divine/Documents/Computer-Vision-2/results/fruits262_3class_scene.png)
- [10-class multi-fruit overlay](/home/divine/Documents/Computer-Vision-2/results/multi_fruit_overlay.png)

## Selected Classes

The current class ranking in [scripts/run.py](/home/divine/Documents/Computer-Vision-2/scripts/run.py:53) is:

1. `Cherry 1` -> Cherry
2. `Orange 1` -> Orange
3. `Banana 1` -> Banana
4. `Avocado 1` -> Avocado
5. `Cucumber 1` -> Cucumber
6. `Cherry Wax Black 1` -> Cherry Black
7. `Cucumber 3` -> Cucumber 3
8. `Huckleberry 1` -> Huckleberry
9. `Raspberry 1` -> Raspberry
10. `Lychee 1` -> Lychee

The first 3 classes form the starter set, the first 5 extend into green hues, and
the full 10-class set covers most of the HSV colour wheel while intentionally keeping
some harder boundary cases for analysis.

## Repository Layout

```text
Computer-Vision-2/
├── fruitseg/
│   ├── color_space.py
│   ├── preprocessing.py
│   ├── split_merge.py
│   ├── features.py
│   ├── classify.py
│   ├── postprocess.py
│   ├── pipeline.py
│   └── evaluation.py
├── scripts/
│   ├── run.py
│   ├── run_all.py
│   ├── selftest.py
│   └── make_research_pdf.py
├── docs/
│   ├── ALGORITHM.md
│   ├── USAGE.md
│   └── report.md
├── results/
│   ├── eval_3fruit.csv
│   ├── eval_5fruit.csv
│   ├── eval_10fruit.csv
│   └── graphs/
└── data/
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Python `>=3.9` is required.

## Dataset Layout

The scripts accept either the direct `Training` / `Test` folders or the shorter
dataset root and resolve the common nested Fruits-360 structure automatically.

Expected layout:

```text
data/
├── Fruits-360/
│   ├── fruits-360_100x100/fruits-360/Training/
│   ├── fruits-360_100x100/fruits-360/Test/
│   └── fruits-360_multi/test-multiple_fruits/
└── Fruits-262/Fruit-262/
```

## How To Run

Quick self-test:

```bash
python scripts/selftest.py
```

3-class evaluation:

```bash
python scripts/run.py --train data/Fruits-360 \
                      --test data/Fruits-360 \
                      --evaluate --nfruits 3
```

5-class evaluation:

```bash
python scripts/run.py --train data/Fruits-360 \
                      --test data/Fruits-360 \
                      --evaluate --nfruits 5
```

10-class evaluation:

```bash
python scripts/run.py --train data/Fruits-360 \
                      --test data/Fruits-360 \
                      --evaluate --nfruits 10
```

Single-image overlay:

```bash
python scripts/run.py --train data/Fruits-360 \
                      --image "data/Fruits-360/fruits-360_multi/test-multiple_fruits/cherry_in_the_basket.jpg" \
                      --out results/overlay.png \
                      --nfruits 3
```

Generate all assignment outputs:

```bash
python scripts/run_all.py
```

Build the research PDF from the markdown report:

```bash
python scripts/make_research_pdf.py
```

## Method Summary

The processing chain is:

1. Median and Gaussian smoothing
2. HSV conversion
3. Guard-mask construction on saturation and value
4. Sobel edge extraction
5. Quadtree split using hue/saturation homogeneity
6. Region-adjacency merge with edge-aware veto
7. Feature extraction from merged regions
8. Z-scored nearest-neighbour classification with background rejection
9. Morphological cleanup and area filtering
10. Overlay rendering

The split-and-merge core and the region statistics are own-code. OpenCV is used
for image I/O and colour conversion only.

## Important Assignment Notes

- Tune using **Fruits-360 Training** only.
- Use **Fruits-360 Test** as the quantitative validation set.
- Treat `test-multiple_fruits` and **Fruits-262** as qualitative transfer tests.
- Do not present qualitative scene tuning as quantitative evaluation.

## Limitations

- Colour is the dominant cue, so hue overlap remains the main failure mode.
- Quadtree boundaries can look blocky on curved fruit edges.
- Thresholds that work well on clean Fruits-360 images transfer less reliably to
  cluttered real scenes.
- Very dark, reflective, or heavily occluded fruit regions can be rejected by the
  guard mask and classified as background.
- The system is closed-set: an unknown fruit may be mapped to the nearest known class.

## Documentation

- Algorithm details: [docs/ALGORITHM.md](/home/divine/Documents/Computer-Vision-2/docs/ALGORITHM.md)
- Usage guide: [docs/USAGE.md](/home/divine/Documents/Computer-Vision-2/docs/USAGE.md)
- Research report source: [docs/report.md](/home/divine/Documents/Computer-Vision-2/docs/report.md)
- Generated report PDF: [docs/research_report.pdf](/home/divine/Documents/Computer-Vision-2/docs/research_report.pdf)

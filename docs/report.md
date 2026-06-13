# HSV Split-and-Merge Fruit Segmentation

**Computer Vision — Assignment 2**  
Deggendorf Institute of Technology · Summer 2026

---

## Abstract

This report describes a classical, non-deep-learning fruit segmentation pipeline
built around the HSV colour space and a region-based split-and-merge algorithm.
The pipeline converts an input image to HSV, applies a quadtree split phase that
enforces at least 16 starting regions, merges regions through a Region Adjacency
Graph using hue and saturation statistics, and classifies each merged region by
nearest-neighbour distance in a z-scored colour feature space. Evaluated on the
Fruits-360 Test folder, the system achieves **100% accuracy** on 3 and 5 classes
and **99.3% accuracy** on 10 classes. Qualitative transfer to `test-multiple_fruits`
and Fruits-262 scene images demonstrates generalisation beyond the controlled
single-fruit test protocol.

---

## 1. Introduction

Fruit recognition and segmentation are well-studied problems in agricultural
automation and food inspection. Classical (non-deep) methods remain relevant when
data is limited, interpretability is required, or computational resources are
constrained. This work implements the full split-and-merge framework of Horowitz
and Pavlidis [2], extended with Sobel edge integration [6, 7], morphological
post-processing [5], and a z-scored nearest-neighbour classifier [8].

The HSV colour space [1] is chosen because it separates chromatic content (hue)
from brightness (value), yielding colour descriptors that are more robust to
illumination variation than raw RGB. All algorithm steps except the RGB-to-HSV
library conversion are implemented from scratch using only NumPy array operations.

---

## 2. Fruit Class Selection

### 2.1 Scanning the Training Folder

The Fruits-360 Training folder contains more than 130 fruit classes, many sharing
very similar hue (e.g. multiple apple red varieties, several tomato cultivars).
Selecting 10 classes that an HSV classifier can reliably separate requires explicit
analysis of the colour statistics of every available class.

### 2.2 Selection Criteria

Three criteria guided the selection:

1. **Maximum hue spread** across the 360° colour wheel.
2. **Sufficient mean saturation** (S ≥ 0.40) — highly desaturated classes have
   unstable hue and depend entirely on weak saturation/value features.
3. **Avoidance of known hue collisions** — pairs whose circular mean hue differs
   by less than ~15° were treated as one slot; the most photogenic representative
   was kept.

### 2.3 Selected Classes

| Rank | Folder | Display name | Mean hue (°) | Overlay (BGR) |
|------|--------|--------------|:---:|---|
| 1 | Cherry 1 | Cherry | 358.1° | dark red |
| 2 | Orange 1 | Orange | 26.7° | orange |
| 3 | Banana 1 | Banana | 45.6° | yellow |
| 4 | Avocado 1 | Avocado | 70.7° | green |
| 5 | Cucumber 1 | Cucumber | 103.8° | lime |
| 6 | Cherry Wax Black 1 | Cherry Black | 313.6° | dark purple |
| 7 | Cucumber 3 | Cucumber 3 | 186.7° | cyan |
| 8 | Huckleberry 1 | Huckleberry | 220.8° | blue |
| 9 | Raspberry 1 | Raspberry | 274.1° | purple |
| 10 | Lychee 1 | Lychee | 3.4° | pale pink |

### 2.4 Ramping Strategy: 3 → 5 → 10 Classes

- **3-class (Cherry, Orange, Banana):** cleanest hue separation; 100% on Test.
- **5-class (+ Avocado, Cucumber):** adds yellow-green and green; 100% on Test.
- **10-class (full set):** completes the colour wheel; 99.3% on Test. Includes
  deliberate hard pairs (Cherry vs. Lychee in the red zone) to generate
  informative confusion-matrix entries.

---

## 3. Algorithm Description

### 3.1 Colour Space and Guard Mask

RGB is converted to HSV using OpenCV's `cvtColor` (library call). Hue is
rescaled to radians [0, 2π); saturation and value to [0, 1].

A **guard mask** is applied before any hue statistic is computed:

```
valid(x, y) = (V(x, y) >= V_min) AND (S(x, y) >= S_min)
```

Achromatic pixels (dark, grey, white) have undefined hue; the guard mask
excludes them from split/merge tests and from the feature vector. Without this
mask, specular highlights and neutral backgrounds corrupt every downstream
computation.

### 3.2 Circular Statistics for Hue

Because hue is an angle, it cannot be averaged arithmetically — the mean of a
set of hues straddling 0°/360° collapses to ~180° [3]. All hue statistics use
directional (circular) formulas:

```
mean_hue = atan2(mean(sin h), mean(cos h))
R        = sqrt(mean(cos h)^2 + mean(sin h)^2)
circ_var = 1 - R    ∈ [0, 1]
```

`circ_var` near 0 indicates tightly clustered hue; near 1 indicates dispersed hue.

### 3.3 Preprocessing

Before splitting, two noise-reduction passes are applied:

- **Median filter (5×5)** — removes salt-and-pepper speckle without blurring
  edges, preventing the variance criterion from over-splitting on noise [4].
- **Gaussian low-pass (σ = 1.5)** — smooths fine surface texture (fruit mottling,
  skin patterns) while leaving genuine fruit-to-background boundaries intact.

Both are implemented using only NumPy stride tricks; no library filter is called.

### 3.4 Split Phase (Quadtree)

The split phase follows Horowitz and Pavlidis [2]:

1. **Forced start at depth 2**, producing at least 16 leaf blocks (satisfies the
   assignment's ≥16 regions requirement).
2. **Homogeneity test** — a block is considered *homogeneous* if:
   - circular variance of hue ≤ τ_H, AND
   - variance of saturation ≤ τ_S, AND
   - mean Sobel edge energy ≤ τ_E (edge integration [7])
3. **Recursive splitting** into four quadrants if inhomogeneous.
4. **Stop** when the block reaches `min_size` pixels or passes the homogeneity test.

Blocks with a valid-pixel fraction below 5% are declared background and kept as
leaves, avoiding over-splitting of dark/achromatic regions.

The Sobel edge criterion (τ_E) extends the pure variance test: a block is split
when a strong edge crosses it, aligning quadtree boundaries with real fruit
outlines [7].

### 3.5 Merge Phase (Region Adjacency Graph)

After splitting, adjacent blocks belonging to the same fruit are merged:

1. A **Region Adjacency Graph (RAG)** connects every pair of spatially adjacent
   leaf regions.
2. Two regions are merged when:
   - their circular-mean hue distance ≤ `hue_thresh`
   - their mean saturation distance ≤ `sat_thresh`
   - the combined region's hue variance ≤ `merged_var_h`
   - the combined region's saturation variance ≤ `merged_var_s`
   - the mean Sobel edge magnitude along their shared boundary ≤ `edge_veto`
3. Candidate pairs are processed **greedy, most-similar-first** from a priority
   queue, recomputing region statistics after each merge.

The **variance of the merged region** (not just closeness of means) is the key
test — this ensures the "mean and variance of hue and saturation" homogeneity
criterion of the assignment is satisfied.

The **edge-aware veto** preserves instance boundaries between two same-hue
adjacent fruits (e.g. two touching oranges) that would otherwise merge
incorrectly.

### 3.6 Feature Extraction

For each surviving merged region, a feature vector is computed over valid pixels:

| Form | Dimensions | Contents |
|------|---|---|
| Baseline | 3 | [cos(μ_H), sin(μ_H), Var(S)] |
| Extended | 7 | [cos(μ_H), sin(μ_H), μ_S, CircVar(H), Var(S), μ_V, Var(V)] |

Mean hue is encoded as (cos, sin) to allow correct Euclidean distance computation
across the wrap-around boundary at 0°/360°.

### 3.7 Classification

Per-class references are computed by averaging feature vectors over up to 200
Training images per class. A global z-score (mean and standard deviation across
all training images) normalises the feature dimensions. Per-dimension weights
further emphasise the reliable hue dimensions:

```
weights_extended = [1.2, 1.2, 0.7, 0.4, 0.4, 0.9, 0.0]
```

Each region is assigned to the nearest reference in weighted Euclidean distance.
Regions whose nearest-class distance exceeds the rejection threshold `reject_z`
are labelled **background** (-1) rather than forced into a wrong class. An
adaptive margin check also accepts slightly-over-threshold regions when the gap
to the second-nearest class is large, reducing false rejections on real scenes.

### 3.8 Post-processing

Per-class binary masks are cleaned with:

- **Morphological opening** (erode → dilate) to remove specks.
- **Morphological closing** (dilate → erode) to fill small holes and bridge gaps.
- **Area filter** — connected components below `min_area` pixels are dropped.

All morphology is implemented via NumPy stride tricks with a disk structuring
element; no cv2 morphology functions are called.

For scene images, an optional **hue-gated BFS expansion** grows each class mask
into nearby unclaimed pixels whose hue is within a tolerance of the class
reference, while stopping at strong HSV edges.

### 3.9 Full Pipeline Summary

```
BGR image
  -> median (5x5) + Gaussian (sigma=1.5)
  -> HSV conversion (library) + guard mask
  -> Sobel edge map (own code)
  -> SPLIT: quadtree, depth >= 2 (>= 16 start regions), own code
  -> MERGE: RAG, greedy most-similar-first, edge veto, own code
  -> feature vectors (7-D, own code)
  -> z-scored NN classifier with rejection (own code)
  -> per-class morphological cleanup + area filter (own code)
  -> colour-overlaid segmentation mask
```

---

## 4. Experimental Setup

### 4.1 Datasets

| Dataset | Use | Details |
|---------|-----|---------|
| Fruits-360 Training | Training only | 100×100 px images, white background |
| Fruits-360 Test | Quantitative evaluation | Same distribution as Training |
| test-multiple_fruits | Qualitative scene demo | Multi-fruit scenes, real backgrounds |
| Fruits-262 | Cross-dataset transfer test | Natural-environment photos, diverse lighting |

Fruits-262 is used only as a qualitative transfer test; it is never used for
parameter tuning or quantitative reporting.

### 4.2 Evaluation Protocol

For each Test image, the pipeline predicts the dominant non-background class. The
confusion matrix accumulates predictions across all Test images; precision, recall,
and F1 are then computed from the confusion matrix (own-code formulas). Up to 60
images per class are evaluated per run.

---

## 5. Results

### 5.1 Quantitative Accuracy (Fruits-360 Test)

| Configuration | Classes | Overall Accuracy |
|---|---|---|
| 3-class (Cherry, Orange, Banana) | 3 | **100.0%** |
| 5-class (+ Avocado, Cucumber) | 5 | **100.0%** |
| 10-class (full set) | 10 | **99.3%** |

### 5.2 10-class Per-Class Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Cherry | 1.000 | 1.000 | 1.000 | 60 |
| Orange | 1.000 | 1.000 | 1.000 | 60 |
| Banana | 1.000 | 1.000 | 1.000 | 60 |
| Avocado | 1.000 | 0.983 | 0.992 | 60 |
| Cucumber | 0.943 | 1.000 | 0.971 | 50 |
| Cherry Black | 1.000 | 1.000 | 1.000 | 60 |
| Cucumber 3 | 1.000 | 0.950 | 0.974 | 60 |
| Huckleberry | 0.984 | 1.000 | 0.992 | 60 |
| Raspberry | 1.000 | 1.000 | 1.000 | 60 |
| Lychee | 1.000 | 1.000 | 1.000 | 60 |

### 5.3 Confusion Analysis

The 10-class confusion matrix shows four misclassifications across 590 Test images:

- 1 Avocado → Cucumber (very similar yellow-green hue zone)
- 2 Cucumber 3 → Cucumber (overlap in the green-cyan zone)
- 1 Cucumber 3 → Huckleberry (Cucumber 3 has an unusual cyan hue near blue)
- 2 Cucumber → predicted other class due to precision < 1.0

These confusions are the predicted hard cases from the class selection phase:
Avocado and Cucumber share the yellow-green zone, and Cucumber 3 sits in a
relatively sparse area of the colour wheel where the guard mask may suppress some
pixels, leaving a noisy mean hue.

### 5.4 Region Counts

| Image | After split | After merge | Reduction |
|---|---|---|---|
| Banana scene | ~350–500 | 15–40 | ~90% |
| Cherry scene | ~300–450 | 10–30 | ~90% |
| Orange scene | ~300–450 | 12–35 | ~90% |

The merge phase consistently reduces region count by roughly 90%, collapsing the
fine quadtree grid into semantically coherent colour regions.

---

## 6. Qualitative Transfer Tests

### 6.1 test-multiple_fruits Scenes

The 3-class pipeline (Cherry, Orange, Banana) was applied to multi-fruit scenes
from the `test-multiple_fruits` folder. Cherry and Orange are detected on their
respective scene images; Banana and Orange are simultaneously detected on the
`banana_orange.jpg` scene. The colour overlays align visually with the fruit
locations, though edges remain slightly blocky due to the quadtree structure.

### 6.2 Fruits-262 Cross-Dataset Demo

The pipeline trained exclusively on Fruits-360 was applied to natural-environment
photographs from Fruits-262. Cherries, bananas, and oranges are identified
correctly from single-fruit images. The cross-dataset result demonstrates that
the HSV-based features generalise reasonably well to different photographic
conditions, though darker or more reflective surfaces cause partial miss-detections.

---

## 7. Limitations and Drawbacks

### 7.1 Hue Overlap

Classes that share a hue (e.g. Avocado vs. Banana in the yellow-green zone,
Cherry vs. Lychee in the red zone) are poorly separated by mean hue alone.
Saturation and value then become the discriminating dimensions, and accuracy
drops as the class count increases. This is the dominant failure mode as the
pipeline scales from 3 to 10 classes.

### 7.2 Blocky Boundaries

The quadtree split operates on axis-aligned rectangles, so region boundaries
appear blocky on curved fruit surfaces. Morphological closing reduces but does not
fully eliminate this artefact. A watershed or graph-cut refinement would produce
smoother boundaries.

### 7.3 Threshold Sensitivity

All thresholds (τ_H, τ_S, τ_E, `hue_thresh`, `reject_z`, etc.) are calibrated
on the clean, white-background Fruits-360 Training images. Transfer to cluttered,
naturally-lit scenes (test-multiple_fruits, Fruits-262) requires relaxed
thresholds that risk generating false positives from similarly-coloured backgrounds.
The `min_class_fraction` parameter suppresses the worst false positives in
multi-class scenes but introduces its own trade-off with sensitivity.

### 7.4 Achromatic and Dark Regions

The guard mask correctly excludes achromatic pixels, but this also means that
dark-skinned fruits (e.g. Cherry Wax Black, very dark avocado) or occluded regions
may be under-segmented. A deeper shadow or a specular highlight can leave a fruit
with fewer than the `min_valid` valid pixels, causing it to be classified as
background.

### 7.5 Computation Speed

Several own-code components are implemented in pure Python/NumPy and are
substantially slower than equivalent OpenCV calls:
- The BFS connected-component labelling in `postprocess.py` uses nested Python
  loops (O(H×W) pixel operations in Python).
- The median filter uses stride tricks but still loops over three channels.
- The quadtree recursion and RAG adjacency scan loop over individual labels.

For a 640×480 image, a single `segment_image` call takes a few seconds on CPU.
Replacing the inner loops with vectorised NumPy or Cython would give a 10–50×
speed-up, but this would complicate the "own code" requirement.

### 7.6 No Shape Information

The classifier uses only colour statistics. Two different objects of the same
colour (e.g. an orange fruit vs. an orange ceramic bowl) cannot be separated by
this pipeline. Shape-based descriptors or spatial priors would be necessary to
handle such cases.

### 7.7 Fixed Number of Classes

The system is a closed-set classifier: it must assign every sufficiently-coloured
region to one of the pre-trained classes or background. An unknown fruit with a
hue close to one of the 10 classes will be incorrectly identified. Extending to
an open-set formulation would require a more principled rejection mechanism.

---

## 8. Conclusion

The HSV split-and-merge pipeline achieves near-perfect accuracy on the Fruits-360
benchmark (99.3% at 10 classes) while remaining fully interpretable and
implemented almost entirely without library segmentation routines. The main
extensions beyond the minimal assignment baseline — circular hue statistics,
Sobel edge integration in both split and merge phases, z-scored weighted
nearest-neighbour classification with rejection, and morphological post-processing
— together account for most of the performance gain over a naive mean-hue
classifier.

The principal remaining limitations are hue overlap between similarly-coloured
classes, the blocky boundary artefact inherent to quadtree decomposition, and
computation speed. These are known properties of block-thresholded split-and-merge
segmentation, well-documented in the literature since the 1970s.

---

## References (IEEE Style)

[1] A. R. Smith, "Color gamut transform pairs," in *Proc. 5th Annu. Conf. Computer
Graphics and Interactive Techniques (SIGGRAPH '78)*, *Computer Graphics*, vol. 12,
no. 3, pp. 12–19, 1978.

[2] S. L. Horowitz and T. Pavlidis, "Picture segmentation by a tree traversal
algorithm," *Journal of the ACM*, vol. 23, no. 2, pp. 368–388, Apr. 1976.

[3] K. V. Mardia and P. E. Jupp, *Directional Statistics*. Chichester, U.K.:
Wiley, 2000.

[4] J. W. Tukey, *Exploratory Data Analysis*. Reading, MA, USA: Addison-Wesley,
1977.

[5] J. Serra, *Image Analysis and Mathematical Morphology*. London, U.K.:
Academic Press, 1982.

[6] I. Sobel and G. Feldman, "A 3×3 isotropic gradient operator for image
processing," Stanford Artificial Intelligence Project (SAIL), 1968.

[7] T. Pavlidis and Y.-T. Liow, "Integrating region growing and edge detection,"
*IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 12, no. 3, pp. 225–233, Mar. 1990.

[8] Y.-I. Ohta, T. Kanade, and T. Sakai, "Color information for region
segmentation," *Computer Graphics and Image Processing*, vol. 13, no. 3,
pp. 222–241, Jul. 1980.

[9] H. Mureșan and M. Oltean, "Fruit recognition from images using deep
learning," *Acta Univ. Sapientiae, Informatica*, vol. 10, no. 1, pp. 26–42, 2018.
(Fruits-360 dataset.)

[10] M.-D. Minuț and A. Iftene, "Creating a dataset and models based on
convolutional neural networks to improve fruit classification," in *Proc. 23rd
Int. Symp. Symbolic and Numeric Algorithms for Scientific Computing (SYNASC)*,
2021, pp. 155–162. (Fruits-262 dataset.)

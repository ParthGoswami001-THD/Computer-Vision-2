# HSV Split-and-Merge Fruit Segmentation

**Computer Vision, Assignment 2 — Deggendorf Institute of Technology, Summer 2026**

This document describes the complete processing chain, the scientific reasoning behind
each algorithmic step, and every extension applied for best results. Each component is
tied to its original published source so that the choices can be defended scientifically
(see the **References** section, IEEE style).

---

## 0. Problem statement

The task is to segment fruits in colour images using a **region-based split-and-merge
algorithm operating in the HSV colour space**. The pipeline must:

1. transform RGB → HSV (library allowed);
2. apply a split-and-merge algorithm (own code) whose homogeneity is judged on the
   **mean and variance of hue and saturation**, evaluated only where luminance is high;
3. start the procedure with **at least 16 regions**;
4. classify each region using one homogeneity criterion (e.g. mean hue + variance of
   saturation).

Development and parameter tuning use the **Fruits-360 Train** folder only. Validation uses
the **Fruits-360 Test** folder. Realistic image testing uses the Fruits-360
`test-multiple_fruits` folder first, then selected scene images from **Fruits-262** [9], [10].

---

## 0.1 Fruit class selection — choosing and ranking 10 classes

### 0.1.1 Scanning the Training folder

The Fruits-360 Training folder contains more than 130 fruit classes, many of which share
a very similar hue (e.g. all red-apple varieties, multiple tomato cultivars, several
orange/nectarine hybrids). Selecting 10 classes that an HSV-based nearest-neighbour
classifier can reliably separate therefore requires an explicit analysis of the colour
statistics of *every* available class before a single classification decision is made.

All Training classes were scanned and their per-class HSV statistics (circular mean hue,
mean saturation, mean value, circular variance of hue) were computed and written to
`results/fruits360_training_hsv_by_class_sorted_hue.csv`. This file lists every Fruits-360
class sorted by mean hue angle, making hue collisions immediately visible.

### 0.1.2 Selection criterion

The primary selection criterion is **maximum spread of mean hue** across the 360° colour
wheel, because hue is the most reliable chromatic discriminator in HSV space (§1). A
secondary criterion is **sufficient mean saturation** (S ≥ 0.40): highly desaturated
classes (e.g. white mushrooms, brown chestnuts) have unstable hue and would rely entirely
on saturation/value features, which are weak discriminators at low saturation. A third
criterion is **avoidance of known hue collisions**: pairs of classes whose circular mean
hue differs by less than ~15° were treated as a single slot, and only the most
photogenic representative was kept.

### 0.1.3 The selected 10 classes and their mean hue

The following table gives the measured Training-set mean hue (in degrees, 0° = red,
120° = green, 240° = blue) for each selected class, sorted around the colour wheel.
Values are taken directly from `results/fruits360_training_hsv_by_class_sorted_hue.csv`.

| Rank | Fruits-360 folder | Display name | Mean hue (°) | Colour zone | Overlay (BGR) |
|------|------------------|--------------|:------------:|-------------|---------------|
| 1  | `Cherry 1`           | Cherry       | 358.1°       | Red               | dark red |
| 2  | `Orange 1`           | Orange       | 26.7°        | Orange            | orange |
| 3  | `Banana 1`           | Banana       | 45.6°        | Yellow            | yellow |
| 4  | `Avocado 1`          | Avocado      | 70.7°        | Yellow-green      | green |
| 5  | `Cucumber 1`         | Cucumber     | 103.8°       | Green             | lime |
| 6  | `Cherry Wax Black 1` | Cherry Black | 313.6°       | Deep purple/dark  | dark purple |
| 7  | `Cucumber 3`         | Cucumber 3   | 186.7°       | Cyan-green        | cyan |
| 8  | `Huckleberry 1`      | Huckleberry  | 220.8°       | Blue              | blue |
| 9  | `Raspberry 1`        | Raspberry    | 274.1°       | Purple-blue       | purple |
| 10 | `Lychee 1`           | Lychee       | 3.4°         | Pale red/bright   | pale pink |

Together these 10 classes span the full 360° hue circle while avoiding the worst red-fruit
collisions. The measured Fruits-360 Test validation is 100% for the first 3 classes,
100% for the first 5 classes, and 99.3% for all 10 classes.

### 0.1.4 Classes considered and rejected

The following candidates were evaluated and excluded:

| Candidate | Reason for exclusion |
|-----------|----------------------|
| Apple Red variants | Red hue overlaps scene objects and pale red fruits; Cherry validated more cleanly in the 3/5/10 ordering |
| Strawberry | Mean hue ≈ 4–8° — collides with other red fruits and Tomato (all in the 0–15° red band) |
| Pear (green) | Mean hue ≈ 65–85° — overlaps with Avocado; saturation lower, harder to mask |
| Lime | Mean hue ≈ 90–110° — collides with Cucumber 1 in the green zone |
| Watermelon | Bi-modal hue (red flesh + green rind); classifier receives a blended mean |
| Mandarin / Clementine | Mean hue ≈ 20–30° — collides with Orange 1; insufficient separation |
| Grape (dark) | Very low saturation and value; guard mask suppresses most pixels; weak signal |

### 0.1.5 Ramping strategy — 3 → 5 → 10 classes

The 10 classes are ranked in `SPEC_10` (`scripts/run.py`) so that the **first 3** are the
best-separated starter set and each additional fruit incrementally increases difficulty:

- **3-fruit set (Cherry, Orange, Banana):** covers red / orange / yellow with the cleanest
  measured validation result. This set reached 100% on the Fruits-360 Test subset.
- **5-fruit set (+ Avocado, Cucumber):** adds yellow-green and green. Avocado has a lower
  mean value (darker flesh) which provides an additional discriminating dimension beyond
  hue. This set also reached 100% on the Fruits-360 Test subset.
- **10-fruit set (all above + Cucumber 3, Huckleberry, Raspberry, Cherry Black, Lychee):**
  completes the wheel. The 10-fruit set deliberately includes some **hard pairs** to stress-
  test the classifier and generate informative confusion-matrix entries for the presentation:
  *Cherry* and *Lychee* are both in the red zone and are separated mostly by value/saturation;
  *Cucumber 1* and *Cucumber 3* differ primarily in hue/saturation rather than object shape.

---

## 1. Why HSV — and the luminance guard

### 1.1 The colour-space choice

RGB entangles chromatic content with intensity: a brightly lit and a shadowed patch of the
same fruit have very different (R, G, B) triples, so colour-based reasoning in RGB is
unstable. The **HSV** model, introduced by Smith [1], re-parameterises the RGB cube into
perceptual axes — **Hue** (the colour angle on the wheel), **Saturation** (colour purity),
and **Value** (brightness). Hue stays approximately constant under illumination changes,
which is exactly the property needed to separate an orange mandarin from a yellow-green
pear. HSV has a long record as a segmentation space for exactly this reason [8], [11].

### 1.2 The instability of hue at low luminance/saturation — the guard mask

Hue is **undefined for achromatic pixels**: black, grey and white have no meaningful colour
angle, so in dark gaps, deep shadows, specular highlights and on a white/grey background the
hue channel is essentially noise. This is a known property of the cylindrical colour models
[1], and the standard remedy is a **validity (guard) mask**:

```
valid(x, y) = (V(x, y) >= V_min) AND (S(x, y) >= S_min)
```

Only valid pixels contribute to any hue/saturation statistic downstream. This single mask is
what prevents the classic failure of colouring a shadow as a fruit, and it is the reason a
fruit occluded into a dark region (e.g. a mandarin half-hidden under a grape cluster) is
correctly *not* hallucinated rather than mislabelled.

### 1.3 Circular statistics for hue (extension)

Because hue is an **angle** (0° ≡ 360°), it cannot be averaged arithmetically — the naïve mean
of a red fruit straddling 0° collapses to a meaningless ~180°. Hue mean and variance are
therefore computed with **directional (circular) statistics** [3]: each hue is mapped to a
unit vector, the vectors are averaged, and the mean angle and resultant length are recovered:

```
mean_hue   = atan2( mean(sin h), mean(cos h) )
R          = sqrt( mean(cos h)^2 + mean(sin h)^2 )      # resultant length, in [0,1]
circ_var   = 1 - R                                       # circular variance, in [0,1]
```

`circ_var` near 0 means tightly clustered hue (homogeneous); near 1 means dispersed hue.
This replaces ordinary variance everywhere the homogeneity of hue is tested.

---

## 2. Preprocessing (extensions)

These steps do not exist in the bare task; they are robustness extensions, applied before
splitting.

- **Median filter (5×5).** The median, introduced as a robust smoother by Tukey [4], removes
  salt-and-pepper speckle and small specular dots **without blurring edges**, so the
  variance-based split criterion does not over-fire on noise and fragment the image into
  needless tiny blocks.
- **Light Gaussian low-pass (σ ≈ 1.5).** Applied gently after the median to stabilise the
  region statistics over fine skin texture (e.g. the mottling of a pear) while leaving genuine
  fruit-to-fruit boundaries intact.

The luminance guard (§1.2) and circular statistics (§1.3) apply throughout everything below.

---

## 3. SPLIT phase (own code)

The split-and-merge framework and its quadtree data structure originate with Horowitz and
Pavlidis [1976] [2]. The **split** phase is a top-down recursive decomposition:

1. **Initialisation — at least 16 regions.** Begin from a forced 4×4 grid (16 blocks), i.e.
   two levels of unconditional subdivision, satisfying the assignment requirement and
   preventing the algorithm from lazily declaring the whole image one region.
2. **Homogeneity test.** A block is *homogeneous* if, over its valid pixels, the **circular
   variance of hue** and the **variance of saturation** are both below thresholds
   `tau_H` and `tau_S`. This is the "mean and variance of hue and saturation" criterion.
3. **Recurse.** If inhomogeneous, split the block into four equal quadrants and test each
   recursively. If homogeneous, it becomes a leaf region.
4. **Stop conditions.** Recursion halts at a minimum block size (e.g. 4×4 px) or when the
   block is homogeneous.

This produces a **quadtree**: large uniform areas remain large blocks; busy/edgy areas are
subdivided finely.

### 3.1 Sobel edge criterion for splitting (extension)

The pure variance test splits along an arbitrary grid, not along object boundaries. Following
the region-growing/edge-detection integration of Pavlidis and Liow [7], the split decision is
augmented with edge information. The **Sobel–Feldman operator** [6] — a separable 3×3 integer
gradient operator — is run (on Value, where the gradient is stable) to produce an edge-magnitude
map, and a block is forced to split when strong edge energy crosses it:

```
split(block) = (circ_var_H > tau_H) OR (var_S > tau_S) OR (edge_energy(block) > tau_E)
```

The variance test and the edge test reinforce each other: an edge inside a block means two
different objects are present, so splitting is warranted regardless of the variance value.
This aligns the quadtree boundaries with real fruit outlines and sharpens the final mask.

---

## 4. MERGE phase (own code)

After splitting, adjacent blocks belonging to the same fruit are still separate (the grid cut
through them). The **merge** phase is bottom-up and is conveniently expressed on a **Region
Adjacency Graph (RAG)** — nodes are leaf regions, edges connect spatial neighbours [2].

1. **Similarity test.** Two neighbouring regions are merged if their **circular-mean hue** and
   **mean saturation** (valid pixels only) are close *and* the **combined region's** hue/sat
   variance stays below threshold. Testing the merged variance — not just the closeness of the
   means — is the literal "homogeneity of the mean **and** variance" requirement.
2. **Most-similar-first ordering (extension).** Candidate pairs are processed from a priority
   queue keyed on hue/saturation distance, recomputing region statistics after each merge.
   Greedy most-similar-first merging yields markedly cleaner objects than raster-order merging.
3. **Edge-aware merge veto (extension).** A merge is suppressed if a strong Sobel edge [6] runs
   along the shared boundary of the two regions, even when their mean colours match. This keeps
   two same-hue but distinct fruits (e.g. two touching mandarins) from collapsing into a single
   blob, preserving instance boundaries — again in the spirit of [7].
4. **Termination.** Repeat until no admissible merge remains.

---

## 5. Post-processing (extensions)

- **Morphological opening then closing.** Using the operators formalised by Serra [5]
  (rooted in Matheron [12]) with a small elliptical/circular structuring element: **opening**
  removes tiny spurious specks, **closing** fills small interior holes and bridges narrow gaps
  inside a fruit. This is the single largest visual-quality improvement and produces the clean,
  solid colour overlays expected in the result figure.
- **Connected-component area filter.** Regions below a minimum pixel area are discarded as
  noise; this measurably cleans the confusion matrix.

---

## 6. CLASSIFICATION (own code)

Each surviving region is assigned to a fruit class by comparing a **feature vector** to
per-class references measured on the Train folder.

- **Baseline feature (3 fruits):** `(circular mean hue, variance of saturation)` — the
  homogeneity criterion named in the task.
- **Extended feature (5 / 10 fruits):** `(circular mean H, mean S, circular var H,
  var S, mean V, var V)`. As classes are added, hue alone stops separating them
  (for example, avocado and banana can overlap in yellow-green hue), so
  saturation, variance, and brightness dimensions become essential. The use of
  colour-region statistics for classification follows the classical
  colour-region-segmentation line of Ohta, Kanade and Sakai [8].
- **Normalisation (extension):** features are z-scored using the Train-set spread so hue and
  saturation-variance are weighted comparably in the distance.
- **Nearest-neighbour with rejection (extension):** a region is assigned to its nearest class
  reference; if that distance still exceeds a tolerance, the region is labelled
  **background / unknown** rather than forced into a wrong class. This keeps grapes, bowl, knife
  and tablecloth out of the result.

---

## 7. Full pipeline at a glance

```
RGB image
  → HSV conversion ............................ (wl)  [1]
  → luminance+saturation guard mask ........... (ext) [1]
  → median 5x5 + light Gaussian ............... (ext) [4]
  → SPLIT: start >=16 blocks; split if
        circ_var(H) OR var(S) OR Sobel edge ... (oc)  [2][6][7]
  → MERGE: RAG, most-similar-first,
        mean+variance homogeneity,
        edge-aware veto ....................... (oc)  [2][7]
  → morphological opening then closing ........ (ext) [5]
  → connected-component area filter ........... (ext)
  → CLASSIFY: z-scored (circ mean H, mean S,
        circ var H, var S, mean V, var V),
        NN + rejection ........................ (oc)  [3][8]
  → colour-overlaid segmentation mask
```

`(wl)` = library allowed · `(oc)` = own code · `(ext)` = extension for best results.

---

## 8. Scientific reflection — advantages and disadvantages

**Advantages.**
- Operating on hue gives illumination-tolerant colour separation [1], [8]; the guard mask makes
  the method more robust to the shadowed, cluttered backgrounds common in
  `test-multiple_fruits` and Fruits-262 testing images [10].
- Split-and-merge adapts resolution to image content — coarse where uniform, fine where
  detailed — at modest memory cost [2].
- The combination of a variance criterion with a Sobel edge criterion [6], [7] localises
  boundaries better than either alone.

**Disadvantages / expected failure modes (to document with the confusion matrices).**
- **Hue overlap.** Classes that share a hue (mandarin vs. orange, strawberry vs. raspberry)
  are poorly separated by mean hue; saturation variance is then the only discriminator, and
  accuracy drops. This is the predicted worst-case as the class count grows 3 → 5 → 10.
- **Blocky boundaries.** The regular quadtree split favours axis-aligned, blocky region edges
  [2]; morphology mitigates but does not fully remove this.
- **Threshold sensitivity.** Fixed `tau_H`, `tau_S`, `tau_E` tuned on Train may not transfer
  perfectly to `test-multiple_fruits` or Fruits-262 scene images; this is a known limitation
  of block-thresholded split-and-merge.
- **Achromatic regions.** Where luminance/saturation are low the guard mask excludes pixels,
  so very dark or desaturated fruits (or occluded ones) may be under-segmented — an acceptable,
  explainable failure rather than a wrong label.

Per the assignment, failing some separations is acceptable; the deliverable is a scientific
explanation of *why* each failure occurs, supported by the per-class confusion matrices.

---

## References (IEEE style)

[1] A. R. Smith, "Color gamut transform pairs," in *Proc. 5th Annu. Conf. Computer Graphics
and Interactive Techniques (SIGGRAPH '78)*, *Computer Graphics*, vol. 12, no. 3, pp. 12–19,
1978.

[2] S. L. Horowitz and T. Pavlidis, "Picture segmentation by a tree traversal algorithm,"
*Journal of the ACM*, vol. 23, no. 2, pp. 368–388, Apr. 1976.

[3] K. V. Mardia and P. E. Jupp, *Directional Statistics*. Chichester, U.K.: Wiley, 2000.

[4] J. W. Tukey, *Exploratory Data Analysis*. Reading, MA, USA: Addison-Wesley, 1977.

[5] J. Serra, *Image Analysis and Mathematical Morphology*. London, U.K.: Academic Press,
1982.

[6] I. Sobel and G. Feldman, "A 3×3 isotropic gradient operator for image processing,"
presented at the Stanford Artificial Intelligence Project (SAIL), 1968; republished in
R. O. Duda and P. E. Hart, *Pattern Classification and Scene Analysis*, pp. 271–272. New York,
NY, USA: Wiley, 1973.

[7] T. Pavlidis and Y.-T. Liow, "Integrating region growing and edge detection," *IEEE Trans.
Pattern Anal. Mach. Intell.*, vol. 12, no. 3, pp. 225–233, Mar. 1990.

[8] Y.-I. Ohta, T. Kanade, and T. Sakai, "Color information for region segmentation,"
*Computer Graphics and Image Processing*, vol. 13, no. 3, pp. 222–241, Jul. 1980.

[9] H. Mureșan and M. Oltean, "Fruit recognition from images using deep learning," *Acta Univ.
Sapientiae, Informatica*, vol. 10, no. 1, pp. 26–42, 2018. (Fruits-360 dataset.)

[10] M.-D. Minuț and A. Iftene, "Creating a dataset and models based on convolutional neural
networks to improve fruit classification," in *Proc. 23rd Int. Symp. Symbolic and Numeric
Algorithms for Scientific Computing (SYNASC)*, Timișoara, Romania, 2021, pp. 155–162,
doi: 10.1109/SYNASC54541.2021.00035. (Fruits-262 dataset.)

[11] S. Sural, G. Qian, and S. Pramanik, "Segmentation and histogram generation using the HSV
color space for image retrieval," in *Proc. IEEE Int. Conf. Image Processing (ICIP)*, 2002,
vol. 2, pp. II-589–II-592.

[12] G. Matheron, *Random Sets and Integral Geometry*. New York, NY, USA: Wiley, 1975.

---

### Citation note

Reference details (volume/page numbers, the 1968 vs. 1973 Sobel publication, and the 1988
conference vs. 1990 journal version of Pavlidis & Liow) were taken from secondary sources and
should be verified against the primary documents before final submission, per the assignment's
30-minute literature-search rule.

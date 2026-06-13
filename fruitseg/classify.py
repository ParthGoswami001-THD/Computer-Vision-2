"""!
@file classify.py
@brief Build per-class reference features from the Fruits-360 Training folder
       and classify regions with a z-scored nearest-neighbour rule plus a
       rejection threshold (own code for the distance and decision logic).

Training images (100x100 on a near-white background) isolate the fruit by
removing bright, weakly-saturated pixels. The resulting colour statistics are
averaged per class, then global z-score parameters are computed across all
training images so that hue and saturation variance contribute comparably in
the weighted Euclidean distance.
"""

import os
import glob
import numpy as np
import cv2
from dataclasses import dataclass

from .color_space import to_hsv_float, guard_mask, circular_mean, circular_variance
from .features import RegionStats


@dataclass
class ClassReference:
    """Reference descriptor for one fruit class, averaged over Training images."""
    name: str
    color_bgr: tuple
    mean_feature: np.ndarray = None
    std_feature: np.ndarray = None
    n_images: int = 0


def _fruit_mask(s, v, s_min=0.20, v_max=0.95):
    """!
    Isolate fruit pixels from the near-white Training background.

    Keep pixels that are well-saturated (fruit) and not blown-out highlights.
    White background is removed by the saturation threshold; specular highlights
    near V=1 are removed by the value ceiling.

    @param s      saturation array in [0, 1].
    @param v      value array in [0, 1].
    @param s_min  minimum saturation for a pixel to be considered fruit.
    @param v_max  maximum brightness; pixels above this are treated as highlights.
    @return boolean mask.
    """
    return (s >= s_min) & (v <= v_max)


def _image_feature(bgr, extended=True, s_min=0.20):
    """!Compute one Training image's feature vector over its fruit pixels."""
    h, s, v = to_hsv_float(bgr)
    fg = _fruit_mask(s, v, s_min=s_min)
    if fg.sum() < 50:
        return None
    hv, sv, vv = h[fg], s[fg], v[fg]
    mh = circular_mean(hv)
    cvh = circular_variance(hv)
    ms = float(sv.mean())
    vs = float(sv.var())
    mv = float(vv.mean())
    vv_var = float(vv.var())
    ch, sh = np.cos(mh), np.sin(mh)
    if extended:
        return np.array([ch, sh, ms, cvh, vs, mv, vv_var], dtype=np.float64)
    return np.array([ch, sh, vs], dtype=np.float64)


def build_references(train_dir, class_spec, extended=True, max_per_class=200):
    """!
    Build averaged reference features for each requested class (own code).

    Scans each class folder in train_dir, computes per-image feature vectors,
    and returns per-class mean/std along with global z-score parameters
    calculated across all training images.

    @param train_dir    path to the Fruits-360 Training folder.
    @param class_spec   list of (folder_name, display_name, overlay_bgr) tuples.
    @param extended     if True, use the richer 7-D feature form.
    @param max_per_class cap on images sampled per class.
    @return (references, norm_mean, norm_std) where references is list[ClassReference]
            and norm_* are global z-score parameters.
    """
    references = []
    all_feats = []
    for folder, name, color in class_spec:
        paths = sorted(glob.glob(os.path.join(train_dir, folder, "*.jpg")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(train_dir, folder, "*.png")))
        paths = paths[:max_per_class]
        feats = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            f = _image_feature(img, extended=extended)
            if f is not None:
                feats.append(f)
        if not feats:
            continue
        feats = np.vstack(feats)
        feat_std = feats.std(axis=0)
        feat_std[feat_std < 1e-6] = 0.0
        ref = ClassReference(name=name, color_bgr=color,
                             mean_feature=feats.mean(axis=0),
                             std_feature=feat_std,
                             n_images=len(feats))
        references.append(ref)
        all_feats.append(feats)

    if not all_feats:
        missing = [f for f, _n, _c in class_spec
                   if not glob.glob(os.path.join(train_dir, f, "*.jpg")) and
                      not glob.glob(os.path.join(train_dir, f, "*.png"))]
        raise RuntimeError(
            f"No training images found in '{train_dir}'.\n"
            f"  Folders not found: {missing or '(all folders exist but yielded no valid pixels)'}\n"
            f"  Check that --train points to the Fruits-360 *Training* directory, e.g.:\n"
            f"  data/Fruits-360/fruits-360_100x100/fruits-360/Training"
        )
    stacked = np.vstack(all_feats)
    norm_mean = stacked.mean(axis=0)
    norm_std = stacked.std(axis=0)
    norm_std[norm_std < 1e-6] = 1.0
    return references, norm_mean, norm_std


# Per-dimension weights applied after z-scoring. Hue (cos, sin) is the primary
# chromatic discriminator and is weighted highest; saturation and value statistics
# are supporting features. var_val is zeroed out because it adds noise at the
# 10-class scale without improving separation on this dataset.
#   baseline order: [cos_h, sin_h, var_s]
#   extended order: [cos_h, sin_h, mean_s, circ_var_h, var_s, mean_v, var_v]
_WEIGHTS_BASELINE = np.array([1.0, 1.0, 0.4])
_WEIGHTS_EXTENDED = np.array([1.2, 1.2, 0.7, 0.4, 0.4, 0.9, 0.0])


def classify_regions(region_stats, references, norm_mean, norm_std,
                     extended=True, reject_z=1.8, min_valid=30, weights=None):
    """!
    Assign each region to its nearest class reference, or background (own code).

    Features are z-scored with Training-set statistics, then multiplied by
    per-dimension weights so the reliable hue dimensions carry the most weight.
    A region whose nearest weighted distance exceeds reject_z is labelled
    background (-1). An adaptive margin check also accepts slightly-
    over-threshold regions when the nearest class is clearly closer than the
    second nearest.

    @param region_stats   list[RegionStats] from region_features.
    @param references     list[ClassReference].
    @param norm_mean, norm_std  z-score parameters from build_references.
    @param extended       feature form (must match references).
    @param reject_z       rejection radius in weighted normalised feature space.
    @param min_valid      regions with fewer valid pixels are background.
    @param weights        optional per-dimension weights (default chosen by form).
    @return list[int] class index per region (-1 == background).
    """
    if weights is None:
        weights = _WEIGHTS_EXTENDED if extended else _WEIGHTS_BASELINE
    ref_feats = np.vstack([
        ((r.mean_feature - norm_mean) / norm_std) * weights for r in references
    ])
    assignments = []
    for rs in region_stats:
        if rs.n_valid < min_valid:
            assignments.append(-1)
            continue
        feat = ((rs.feature_vector(extended=extended) - norm_mean) / norm_std) * weights
        d = np.linalg.norm(ref_feats - feat[None, :], axis=1)
        j = int(np.argmin(d))
        d1 = float(d[j])
        d2 = float(np.partition(d, 1)[1]) if len(d) > 1 else np.inf
        confident_margin = d2 - d1
        adaptive_limit = reject_z * 1.25
        accept = (d1 <= reject_z) or (d1 <= adaptive_limit and confident_margin >= 0.25)
        assignments.append(j if accept else -1)
    return assignments

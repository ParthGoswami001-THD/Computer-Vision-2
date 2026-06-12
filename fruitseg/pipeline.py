"""!
@file pipeline.py
@brief End-to-end orchestration of the segmentation pipeline.

segment_image() runs:
    median + Gaussian  ->  HSV + guard  ->  Sobel edges
    ->  split (quadtree)  ->  merge (RAG)
    ->  region features   ->  classify
    ->  per-class morphology + area filter  ->  colour overlay
"""

import numpy as np
import cv2
from dataclasses import dataclass, field

from .preprocessing import median_filter, gaussian_lowpass, sobel_edges
from .color_space import to_hsv_float, guard_mask
from .split_merge import split_quadtree, merge_regions
from .features import region_features
from .classify import classify_regions
from .postprocess import morphological_cleanup, area_filter


@dataclass
class SegmentationConfig:
    """!All tunable parameters in one place (tune on the Train set)."""
    # preprocessing
    median_ksize: int = 5
    gaussian_sigma: float = 1.5
    # guard mask
    s_min: float = 0.15
    v_min: float = 0.15
    # split thresholds
    tau_h: float = 0.05
    tau_s: float = 0.02
    tau_e: float = 0.15
    min_size: int = 4
    min_start_depth: int = 2          # >=16 starting regions
    # merge thresholds
    hue_thresh: float = 0.30
    sat_thresh: float = 0.15
    merged_var_h: float = 0.08
    merged_var_s: float = 0.03
    edge_veto: float = 0.35
    # classification
    extended_features: bool = True
    reject_z: float = 1.8
    min_valid: int = 30
    # post-processing
    morph_radius: int = 3
    min_area: int = 150
    # speed: longest side the image is resized to before processing (0 = no resize)
    max_side: int = 320


def _maybe_resize(bgr, max_side):
    if max_side and max(bgr.shape[:2]) > max_side:
        scale = max_side / max(bgr.shape[:2])
        new = (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale))
        return cv2.resize(bgr, new, interpolation=cv2.INTER_AREA)
    return bgr


def segment_image(bgr, references, norm_mean, norm_std, cfg=None, return_debug=False):
    """!
    Segment one image and return a colour overlay plus the per-pixel class map.

    @param bgr         uint8 BGR image (real-world / multi-fruit).
    @param references  list[ClassReference] from build_references.
    @param norm_mean,norm_std  z-score parameters from build_references.
    @param cfg         SegmentationConfig (defaults used if None).
    @param return_debug if True also return intermediate maps for slides.
    @return dict with keys: 'overlay', 'class_map', and (optionally) debug maps.
    """
    cfg = cfg or SegmentationConfig()
    bgr = _maybe_resize(bgr, cfg.max_side)

    # 1. preprocessing
    pre = median_filter(bgr, cfg.median_ksize)
    pre = gaussian_lowpass(pre, cfg.gaussian_sigma)

    # 2. HSV + guard mask + edges
    h, s, v = to_hsv_float(pre)
    valid = guard_mask(s, v, cfg.s_min, cfg.v_min)
    edges = sobel_edges(v)

    # 3. split
    split_lbl = split_quadtree(h, s, valid, edges,
                               tau_h=cfg.tau_h, tau_s=cfg.tau_s, tau_e=cfg.tau_e,
                               min_size=cfg.min_size, min_start_depth=cfg.min_start_depth)

    # 4. merge
    merged_lbl = merge_regions(split_lbl, h, s, valid, edges,
                               hue_thresh=cfg.hue_thresh, sat_thresh=cfg.sat_thresh,
                               merged_var_h=cfg.merged_var_h, merged_var_s=cfg.merged_var_s,
                               edge_veto=cfg.edge_veto)

    # 5. features + classification
    stats = region_features(merged_lbl, h, s, valid, min_valid=cfg.min_valid)
    assign = classify_regions(stats, references, norm_mean, norm_std,
                              extended=cfg.extended_features,
                              reject_z=cfg.reject_z, min_valid=cfg.min_valid)

    # 6. build per-class masks, clean them, compose class map + overlay
    H, W = merged_lbl.shape
    class_map = np.full((H, W), -1, dtype=np.int32)
    overlay = bgr.copy()
    n_classes = len(references)

    for cidx in range(n_classes):
        region_ids = [rs.label for rs, a in zip(stats, assign) if a == cidx]
        if not region_ids:
            continue
        mask = np.isin(merged_lbl, region_ids).astype(np.uint8)
        mask = morphological_cleanup(mask, cfg.morph_radius)
        mask = area_filter(mask, cfg.min_area)
        class_map[mask > 0] = cidx
        color = np.array(references[cidx].color_bgr, dtype=np.float32)
        sel = mask > 0
        overlay[sel] = (0.45 * overlay[sel] + 0.55 * color).astype(np.uint8)

    result = {"overlay": overlay, "class_map": class_map}
    if return_debug:
        result.update({
            "valid": valid, "edges": edges,
            "split_labels": split_lbl, "merged_labels": merged_lbl,
            "preprocessed": pre,
        })
    return result

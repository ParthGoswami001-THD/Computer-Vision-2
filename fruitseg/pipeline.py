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
from collections import deque
from dataclasses import dataclass, field

from .preprocessing import median_filter, gaussian_lowpass, sobel_edges
from .color_space import to_hsv_float, guard_mask
from .split_merge import split_quadtree, merge_regions
from .features import region_features
from .classify import classify_regions
from .postprocess import morphological_cleanup, area_filter, _connected_components_8


@dataclass
class SegmentationConfig:
    """!All tunable parameters in one place (tune on the Train set)."""
    # preprocessing
    median_ksize: int = 5
    gaussian_sigma: float = 1.5
    # guard mask
    s_min: float = 0.15
    v_min: float = 0.15
    v_max: float = 1.0
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
    use_hsv_edges: bool = False
    # classification
    extended_features: bool = True
    reject_z: float = 1.8
    min_valid: int = 30
    feature_weights: tuple = None
    # post-processing
    morph_radius: int = 3
    min_area: int = 150
    min_area_frac: float = 0.0
    refine_masks: bool = False
    refine_hue_tol: float = 0.45
    fill_components: bool = False
    expand_masks: bool = False
    expand_hue_tol: float = 0.38
    expand_s_min: float = 0.22
    expand_v_min: float = 0.25
    expand_min_seed_area: int = 5000
    expand_min_seed_frac: float = 0.0
    expand_edge_veto: float = 1.0
    # suppress classes that cover less than this fraction of all labeled pixels
    # (0 = no filtering; set to ~0.05 for scene images to kill false positives)
    min_class_fraction: float = 0.0
    # speed: longest side the image is resized to before processing (0 = no resize)
    max_side: int = 320


def _maybe_resize(bgr, max_side):
    if max_side and max(bgr.shape[:2]) > max_side:
        scale = max_side / max(bgr.shape[:2])
        new = (int(bgr.shape[1] * scale), int(bgr.shape[0] * scale))
        return cv2.resize(bgr, new, interpolation=cv2.INTER_AREA)
    return bgr


def _hsv_edge_map(h, s, v, valid):
    """Combine value, saturation and circular-hue edges for scene boundaries."""
    e_v = sobel_edges(v)
    e_s = sobel_edges(s)
    valid_f = valid.astype(np.float32)
    e_h_cos = sobel_edges((np.cos(h) * valid_f).astype(np.float32))
    e_h_sin = sobel_edges((np.sin(h) * valid_f).astype(np.float32))
    e_h = np.maximum(e_h_cos, e_h_sin)
    return np.clip(np.maximum(e_v, np.maximum(0.7 * e_s, 0.7 * e_h)),
                   0.0, 1.0).astype(np.float32)


def _fill_components(mask):
    """!
    Fill holes inside foreground regions by BFS-flooding from image borders (oc).

    Any background pixel reachable from the image border is truly external.
    Background pixels that are NOT reachable are interior holes; they are filled.
    No library contour or flood-fill function is called.
    """
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    external = np.zeros((H, W), dtype=bool)
    q = deque()
    for y in range(H):
        for x in (0, W - 1):
            if m[y, x] == 0 and not external[y, x]:
                external[y, x] = True
                q.append((y, x))
    for x in range(W):
        for y in (0, H - 1):
            if m[y, x] == 0 and not external[y, x]:
                external[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and m[ny, nx] == 0 and not external[ny, nx]:
                external[ny, nx] = True
                q.append((ny, nx))
    return (m | ~external).astype(np.uint8)


def add_legend(bgr, references, title="Legend"):
    """!Append a white legend panel with class colours and names.

    @param bgr        uint8 BGR overlay image.
    @param references list[ClassReference] with name and colour.
    @param title      legend title.
    @return uint8 BGR image with the legend panel appended on the right.
    """
    if not references:
        return bgr

    H, W = bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 1
    pad = 12
    swatch = 16
    row_h = 26
    title_h = 30

    text_widths = [
        cv2.getTextSize(r.name, font, scale, thickness)[0][0]
        for r in references
    ]
    title_w = cv2.getTextSize(title, font, scale, thickness)[0][0]
    panel_w = max(180, pad * 3 + swatch + max([title_w] + text_widths))
    needed_h = pad + title_h + row_h * len(references) + pad
    out_h = max(H, needed_h)

    out = np.full((out_h, W + panel_w, 3), 255, dtype=np.uint8)
    out[:H, :W] = bgr

    x0 = W
    cv2.line(out, (x0, 0), (x0, out_h - 1), (210, 210, 210), 1)
    cv2.putText(out, title, (x0 + pad, pad + 16), font, scale,
                (30, 30, 30), thickness, cv2.LINE_AA)

    y = pad + title_h
    for ref in references:
        y_mid = y + row_h // 2
        x_s = x0 + pad
        y_s = y_mid - swatch // 2
        color = tuple(int(c) for c in ref.color_bgr)
        cv2.rectangle(out, (x_s, y_s), (x_s + swatch, y_s + swatch),
                      color, -1)
        cv2.rectangle(out, (x_s, y_s), (x_s + swatch, y_s + swatch),
                      (80, 80, 80), 1)
        cv2.putText(out, ref.name, (x_s + swatch + pad, y_mid + 5),
                    font, scale, (30, 30, 30), thickness, cv2.LINE_AA)
        y += row_h

    return out


def _reference_hue(reference):
    """Recover the circular mean hue stored as cos/sin in a reference vector."""
    return float(np.arctan2(reference.mean_feature[1],
                            reference.mean_feature[0]) % (2 * np.pi))


def _expand_seed_masks(seed_masks, references, h, s, v, edges, cfg):
    """Expand confident class seeds to nearby pixels with matching hue.

    Only large seeds expand; small seeds keep their original classification.
    Expansion ONLY targets pixels not already claimed by any seed, so seeds
    from one class are never stolen by an adjacent class with a similar hue.
    """
    H, W = h.shape
    min_seed_area = cfg.expand_min_seed_area
    if cfg.expand_min_seed_frac > 0.0:
        min_seed_area = max(min_seed_area, int(round(H * W * cfg.expand_min_seed_frac)))

    active = [
        i for i, mask in enumerate(seed_masks)
        if int(mask.sum()) >= min_seed_area
    ]
    if not active:
        return seed_masks

    # Non-active classes preserve their seeds unchanged
    expanded = [m.copy() for m in seed_masks]

    # Build a mask of all pixels already claimed by ANY seed
    claimed = np.zeros(h.shape, dtype=bool)
    for m in seed_masks:
        claimed |= (m > 0)

    valid = guard_mask(s, v, cfg.expand_s_min, cfg.expand_v_min) & (v <= cfg.v_max)
    edge_ok = edges <= cfg.expand_edge_veto
    ref_hues = np.array([_reference_hue(references[i]) for i in active],
                        dtype=np.float32)
    distances = np.stack([
        np.abs(np.arctan2(np.sin(h - ref_h), np.cos(h - ref_h)))
        for ref_h in ref_hues
    ], axis=-1)
    nearest = np.argmin(distances, axis=-1)
    nearest_dist = np.take_along_axis(distances, nearest[..., None], axis=-1)[..., 0]
    # Only expand into UNCLAIMED valid pixels within hue tolerance, and do not
    # cross strong HSV edges.  This prevents a fruit seed from pulling in a
    # disconnected, similarly-coloured basket/background region.
    assignable = valid & edge_ok & ~claimed & (nearest_dist <= cfg.expand_hue_tol)

    for pos, cidx in enumerate(active):
        seed = (seed_masks[cidx] > 0)
        candidate = seed | ((nearest == pos) & assignable)
        labels, _areas = _connected_components_8(candidate.astype(np.uint8))
        seed_labels = np.unique(labels[seed])
        seed_labels = seed_labels[seed_labels > 0]
        if seed_labels.size:
            new_px = np.isin(labels, seed_labels).astype(np.uint8)
        else:
            new_px = seed.astype(np.uint8)
        new_px = morphological_cleanup(new_px, cfg.morph_radius)
        new_px = area_filter(new_px, cfg.min_area)
        if cfg.fill_components:
            new_px = _fill_components(new_px)
            new_px = area_filter(new_px, cfg.min_area)
        expanded[cidx] = new_px
    return expanded


def side_by_side(source_bgr, overlay_bgr, left_title=None,
                 right_title=None):
    """!Compose source and overlay images side by side with simple titles."""
    H, W = overlay_bgr.shape[:2]
    if source_bgr.shape[:2] != (H, W):
        source_bgr = cv2.resize(source_bgr, (W, H), interpolation=cv2.INTER_AREA)

    pad = 12
    gap = 10
    has_titles = bool(left_title or right_title)
    title_h = 34 if has_titles else 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1

    bottom_pad = pad if has_titles else 0
    out = np.full((H + title_h + bottom_pad, W * 2 + gap, 3),
                  255, dtype=np.uint8)
    y0 = title_h
    out[y0:y0 + H, :W] = source_bgr
    out[y0:y0 + H, W + gap:W * 2 + gap] = overlay_bgr
    if has_titles:
        if left_title:
            cv2.putText(out, left_title, (pad, 22), font, scale,
                        (30, 30, 30), thickness, cv2.LINE_AA)
        if right_title:
            cv2.putText(out, right_title, (W + gap + pad, 22), font, scale,
                        (30, 30, 30), thickness, cv2.LINE_AA)
    return out


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
    valid = guard_mask(s, v, cfg.s_min, cfg.v_min) & (v <= cfg.v_max)
    edges = _hsv_edge_map(h, s, v, valid) if cfg.use_hsv_edges else sobel_edges(v)

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
    stats = region_features(merged_lbl, h, s, valid, min_valid=cfg.min_valid,
                            value=v)
    assign = classify_regions(stats, references, norm_mean, norm_std,
                              extended=cfg.extended_features,
                              reject_z=cfg.reject_z, min_valid=cfg.min_valid,
                              weights=cfg.feature_weights)

    # 6. build per-class masks, clean them, compose class map + overlay
    H, W = merged_lbl.shape
    min_area = cfg.min_area
    if cfg.min_area_frac > 0.0:
        min_area = max(min_area, int(round(H * W * cfg.min_area_frac)))
    class_map = np.full((H, W), -1, dtype=np.int32)
    overlay = bgr.copy()
    n_classes = len(references)

    seed_masks = []
    for cidx in range(n_classes):
        region_ids = [rs.label for rs, a in zip(stats, assign) if a == cidx]
        if not region_ids:
            seed_masks.append(np.zeros((H, W), dtype=np.uint8))
            continue
        mask = np.isin(merged_lbl, region_ids)
        if cfg.refine_masks:
            ref_hue = _reference_hue(references[cidx])
            hue_delta = np.abs(np.arctan2(np.sin(h - ref_hue),
                                          np.cos(h - ref_hue)))
            mask = mask & valid & (hue_delta <= cfg.refine_hue_tol)
        mask = mask.astype(np.uint8)
        mask = morphological_cleanup(mask, cfg.morph_radius)
        mask = area_filter(mask, min_area)
        if cfg.fill_components:
            mask = _fill_components(mask)
            mask = area_filter(mask, min_area)
        seed_masks.append(mask)

    masks = _expand_seed_masks(seed_masks, references, h, s, v, edges, cfg) \
        if cfg.expand_masks else seed_masks

    # --- suppress minority classes (false positives from hue overlap) ----------
    if cfg.min_class_fraction > 0.0:
        total_labeled = sum(int(m.sum()) for m in masks)
        if total_labeled > 0:
            masks = [
                m if (int(m.sum()) / total_labeled >= cfg.min_class_fraction)
                else np.zeros_like(m)
                for m in masks
            ]

    for cidx, mask in enumerate(masks):
        if not mask.any():
            continue
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

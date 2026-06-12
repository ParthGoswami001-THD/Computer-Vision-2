"""!
@file postprocess.py
@brief Morphological opening/closing (wl) and a connected-component area filter
       applied to per-class masks to produce clean, solid segmentation regions.

Morphology origin: Serra, "Image Analysis and Mathematical Morphology",
                   Academic Press, 1982 (rooted in Matheron, 1975).
Opening removes small specks; closing fills small holes / bridges gaps.
"""

import numpy as np
import cv2


def morphological_cleanup(mask, radius=3):
    """!
    Opening then closing with an elliptical structuring element (wl).

    @param mask   boolean or uint8 mask for a single class.
    @param radius structuring-element radius in pixels.
    @return cleaned uint8 mask in {0,1}.
    """
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)    # remove specks
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)   # fill holes / bridge gaps
    return m


def area_filter(mask, min_area=150):
    """!
    Drop connected components smaller than `min_area` pixels (wl: connectedComponents).

    @param mask     uint8 mask in {0,1}.
    @param min_area minimum component area to keep.
    @return uint8 mask in {0,1} with small components removed.
    """
    m = (mask > 0).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):  # 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lbl == i] = 1
    return out

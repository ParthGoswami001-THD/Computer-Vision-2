"""!
@file postprocess.py
@brief Binary morphological opening/closing and 8-connectivity BFS
       connected-component area filtering (own code).

Opening removes small spurious blobs; closing fills small interior holes and
bridges narrow gaps inside a fruit region. Both use a disk structuring element
built via NumPy stride tricks -- no cv2 morphology calls are made.
"""

import numpy as np
from collections import deque


def _make_disk(radius):
    """Return a flat boolean mask selecting disk-shaped pixels from a k x k patch."""
    k = 2 * radius + 1
    yi, xi = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xi * xi + yi * yi <= radius * radius).ravel()


def _binary_dilate(mask, radius):
    """!
    Binary dilation with a disk structuring element (own code).

    Extracts every k x k neighbourhood via stride tricks, then checks whether
    any disk-shaped pixel in the neighbourhood is foreground. No cv2 morphology
    is called.

    @param mask   uint8 binary mask.
    @param radius disk radius in pixels.
    @return dilated uint8 mask.
    """
    r = radius
    k = 2 * r + 1
    disk = _make_disk(r)
    padded = np.pad(mask.astype(np.uint8), r, mode='constant', constant_values=0)
    H, W = mask.shape
    shape = (H, W, k, k)
    strd = padded.strides + padded.strides
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strd)
    flat = windows.reshape(H, W, k * k)[:, :, disk]
    return flat.any(axis=-1).astype(np.uint8)


def _binary_erode(mask, radius):
    """!
    Binary erosion with a disk structuring element (own code).

    A pixel stays foreground only if every disk-shaped neighbour is also
    foreground. No cv2 morphology is called.

    @param mask   uint8 binary mask.
    @param radius disk radius in pixels.
    @return eroded uint8 mask.
    """
    r = radius
    k = 2 * r + 1
    disk = _make_disk(r)
    padded = np.pad(mask.astype(np.uint8), r, mode='constant', constant_values=0)
    H, W = mask.shape
    shape = (H, W, k, k)
    strd = padded.strides + padded.strides
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strd)
    flat = windows.reshape(H, W, k * k)[:, :, disk]
    return flat.all(axis=-1).astype(np.uint8)


def morphological_cleanup(mask, radius=3):
    """!
    Opening (erode then dilate) followed by closing (dilate then erode) (own code).

    No cv2 morphology functions are called.

    @param mask   boolean or uint8 mask for a single class.
    @param radius structuring-element radius in pixels.
    @return cleaned uint8 mask in {0, 1}.
    """
    m = (mask.astype(np.uint8) > 0).astype(np.uint8)
    m = _binary_dilate(_binary_erode(m, radius), radius)
    m = _binary_erode(_binary_dilate(m, radius), radius)
    return m


def _connected_components_8(mask):
    """!
    8-connectivity BFS connected-component labelling (own code).

    Each foreground pixel seeds a BFS expanding into all 8-connected
    neighbours. Total work is O(H x W) since every pixel is enqueued once.

    @param mask  binary uint8 array (0 or 1).
    @return (labels int32 array, areas list[int]) where areas[i] is the pixel
            count of the component with label i+1 (label 0 = background).
    """
    H, W = mask.shape
    binary = mask > 0
    labels = np.zeros((H, W), dtype=np.int32)
    lbl = 0
    areas = []
    for y0 in range(H):
        for x0 in range(W):
            if binary[y0, x0] and labels[y0, x0] == 0:
                lbl += 1
                area = 0
                q = deque()
                q.append((y0, x0))
                labels[y0, x0] = lbl
                while q:
                    y, x = q.popleft()
                    area += 1
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < H and 0 <= nx < W
                                    and binary[ny, nx]
                                    and labels[ny, nx] == 0):
                                labels[ny, nx] = lbl
                                q.append((ny, nx))
                areas.append(area)
    return labels, areas


def area_filter(mask, min_area=150):
    """!
    Remove connected components smaller than min_area pixels (own code).

    Uses the BFS connected-component labelling above; no cv2.connectedComponents
    is called.

    @param mask     uint8 mask in {0, 1}.
    @param min_area minimum component area to keep.
    @return uint8 mask with small components zeroed out.
    """
    m = (mask > 0).astype(np.uint8)
    labels, areas = _connected_components_8(m)
    out = np.zeros_like(m)
    for i, area in enumerate(areas):
        if area >= min_area:
            out[labels == (i + 1)] = 1
    return out

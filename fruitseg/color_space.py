"""!
@file color_space.py
@brief RGB->HSV conversion (wl), the luminance/saturation guard mask, and
       circular (directional) statistics for the hue channel.

Why HSV:   Smith, "Color gamut transform pairs", SIGGRAPH '78. HSV separates
           chromatic content (hue) from intensity (value), giving illumination-
           tolerant colour reasoning.
Why guard: Hue is undefined for achromatic pixels (black/grey/white). In dark or
           desaturated regions the hue channel is noise, so those pixels must be
           excluded from every hue/saturation statistic.
Why circular stats:  Hue is an angle (0 == 2*pi); it cannot be averaged
           arithmetically.  Mardia & Jupp, "Directional Statistics", Wiley 2000.
"""

import numpy as np
import cv2


def to_hsv_float(bgr):
    """!
    Convert an 8-bit BGR image (OpenCV default) to float HSV with normalised ranges.

    OpenCV returns H in [0,179] (degrees/2), S,V in [0,255].  We re-scale to a
    convenient, interpretable domain:
        H -> radians in [0, 2*pi)
        S -> [0, 1]
        V -> [0, 1]

    @param bgr  uint8 array, shape (H, W, 3), channel order B,G,R (wl: cv2).
    @return (h_rad, s, v) three float32 arrays of shape (H, W).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)          # (wl) library conversion
    h = hsv[:, :, 0].astype(np.float32) * (np.pi / 90.0)  # [0,179] -> [0,2pi)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    return h, s, v


def guard_mask(s, v, s_min=0.15, v_min=0.15):
    """!
    Validity mask: True where the hue can be trusted.

    A pixel is valid only if it is bright enough AND saturated enough; otherwise
    its hue is meaningless (achromatic) and it is excluded downstream.

    @param s,v   saturation / value arrays in [0,1].
    @param s_min,v_min  thresholds (tune on the Train set).
    @return boolean array, same shape.
    """
    return (v >= v_min) & (s >= s_min)


def circular_mean(h_rad, weights=None):
    """!
    Circular mean of hue (own code).

    Each hue angle is mapped to a unit vector; the vectors are averaged and the
    mean angle is recovered with atan2.  This is the only correct "average" for a
    periodic quantity.

    @param h_rad   1-D array of hue values in radians.
    @param weights optional per-sample weights (e.g. a validity mask flattened).
    @return mean angle in [0, 2*pi); returns 0.0 if there are no samples.
    """
    if h_rad.size == 0:
        return 0.0
    c = np.cos(h_rad)
    s = np.sin(h_rad)
    if weights is None:
        cbar = c.mean()
        sbar = s.mean()
    else:
        wsum = weights.sum()
        if wsum <= 0:
            return 0.0
        cbar = (c * weights).sum() / wsum
        sbar = (s * weights).sum() / wsum
    ang = np.arctan2(sbar, cbar)
    return float(ang % (2.0 * np.pi))


def circular_variance(h_rad, weights=None):
    """!
    Circular variance of hue in [0,1] (own code).

    Defined as 1 - R, where R is the mean resultant length.  R near 1 means the
    angles are tightly clustered (variance ~0, homogeneous hue); R near 0 means
    they are spread around the circle (variance ~1).

    @param h_rad   1-D array of hue values in radians.
    @param weights optional per-sample weights.
    @return circular variance in [0,1]; returns 1.0 (maximally inhomogeneous)
            if there are no samples.
    """
    if h_rad.size == 0:
        return 1.0
    c = np.cos(h_rad)
    s = np.sin(h_rad)
    if weights is None:
        cbar = c.mean()
        sbar = s.mean()
    else:
        wsum = weights.sum()
        if wsum <= 0:
            return 1.0
        cbar = (c * weights).sum() / wsum
        sbar = (s * weights).sum() / wsum
    R = np.sqrt(cbar * cbar + sbar * sbar)
    return float(1.0 - R)


def circular_distance(a, b):
    """!
    Smallest absolute angular distance between two hue angles (own code).

    @param a,b  angles in radians.
    @return distance in [0, pi].
    """
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return float(abs(d))

"""!
@file color_space.py
@brief RGB->HSV conversion (library call), the saturation/luminance guard mask,
       and circular statistics for the hue channel (own code).

HSV separates chromatic content from intensity, making colour descriptions
more robust to illumination changes than raw RGB. Because hue is an angle
(0 == 2*pi), its mean and variance require directional statistics -- ordinary
arithmetic does not respect the circular topology.
"""

import numpy as np
import cv2


def to_hsv_float(bgr):
    """!
    Convert an 8-bit BGR image to float HSV with normalised ranges (library).

    OpenCV encodes H in [0, 179], S and V in [0, 255]. We rescale to:
        H -> radians in [0, 2*pi)
        S -> [0, 1]
        V -> [0, 1]

    @param bgr  uint8 array of shape (H, W, 3), channel order B, G, R.
    @return (h_rad, s, v) three float32 arrays of shape (H, W).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32) * (np.pi / 90.0)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    return h, s, v


def guard_mask(s, v, s_min=0.15, v_min=0.15):
    """!
    Return a boolean mask of pixels where hue is reliable (own code).

    In dark or desaturated regions the hue channel is essentially noise.
    Excluding those pixels prevents achromatic areas from corrupting any
    downstream hue or saturation statistic.

    @param s      saturation array in [0, 1].
    @param v      value array in [0, 1].
    @param s_min  minimum saturation threshold.
    @param v_min  minimum brightness threshold.
    @return boolean array of the same shape.
    """
    return (v >= v_min) & (s >= s_min)


def circular_mean(h_rad, weights=None):
    """!
    Compute the circular (directional) mean of hue angles (own code).

    Each angle is mapped to a unit vector on the complex unit circle; the
    vectors are averaged and the mean direction recovered with atan2. This is
    the only arithmetically correct average for a periodic quantity like hue.

    @param h_rad   1-D array of hue values in radians.
    @param weights optional per-sample weights.
    @return mean angle in [0, 2*pi); 0.0 if the input is empty.
    """
    if h_rad.size == 0:
        return 0.0
    c, s = np.cos(h_rad), np.sin(h_rad)
    if weights is None:
        cbar, sbar = c.mean(), s.mean()
    else:
        wsum = weights.sum()
        if wsum <= 0:
            return 0.0
        cbar = (c * weights).sum() / wsum
        sbar = (s * weights).sum() / wsum
    return float(np.arctan2(sbar, cbar) % (2.0 * np.pi))


def circular_variance(h_rad, weights=None):
    """!
    Compute the circular variance of hue angles in [0, 1] (own code).

    Defined as 1 - R where R is the mean resultant length. R close to 1
    indicates tightly clustered angles (low variance); R close to 0 indicates
    widely dispersed angles (high variance).

    @param h_rad   1-D array of hue values in radians.
    @param weights optional per-sample weights.
    @return circular variance in [0, 1]; 1.0 if input is empty.
    """
    if h_rad.size == 0:
        return 1.0
    c, s = np.cos(h_rad), np.sin(h_rad)
    if weights is None:
        cbar, sbar = c.mean(), s.mean()
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

    @param a, b  angles in radians.
    @return distance in [0, pi].
    """
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return float(abs(d))

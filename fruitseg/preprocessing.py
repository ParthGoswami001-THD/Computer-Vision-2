"""!
@file preprocessing.py
@brief Preprocessing extensions: median filtering, Gaussian low-pass (both wl),
       and an own-code Sobel-Feldman edge operator used by the split criterion.

Median filter:  Tukey, "Exploratory Data Analysis", 1977.  Robust to salt-and-
                pepper speckle, preserves edges -> prevents over-splitting.
Sobel operator: Sobel & Feldman, SAIL 1968.  Used here to drive the edge-based
                split criterion (Pavlidis & Liow, IEEE TPAMI 1990).
"""

import numpy as np
import cv2


def median_filter(bgr, ksize=5):
    """!
    5x5 median filter (wl: cv2.medianBlur).

    Removes impulse noise and small specular dots without blurring fruit
    boundaries, stabilising the variance-based split criterion.

    @param bgr   uint8 BGR image.
    @param ksize odd kernel size.
    @return filtered uint8 BGR image.
    """
    return cv2.medianBlur(bgr, ksize)


def gaussian_lowpass(bgr, sigma=1.5):
    """!
    Gentle Gaussian low-pass (wl: cv2.GaussianBlur).

    Smooths fine skin texture (e.g. pear mottling) so region statistics are
    stable; kept light so genuine boundaries survive.

    @param bgr   uint8 BGR image.
    @param sigma standard deviation in pixels.
    @return filtered uint8 BGR image.
    """
    if sigma <= 0:
        return bgr
    return cv2.GaussianBlur(bgr, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


# --- Sobel-Feldman 3x3 kernels (own code convolution below) --------------------
_KX = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=np.float32)
_KY = np.array([[-1, -2, -1],
                [0,  0,  0],
                [1,  2,  1]], dtype=np.float32)


def _convolve3x3(img, kernel):
    """!
    Own-code 3x3 convolution using shifted-array summation (basic numpy only).

    The image is zero-padded by one pixel; for each of the 9 kernel taps the
    correspondingly shifted view is accumulated.  No library convolution is used.

    @param img    float32 2-D array.
    @param kernel 3x3 float32 array.
    @return float32 array, same shape as img.
    """
    padded = np.pad(img, 1, mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    H, W = img.shape
    for dy in range(3):
        for dx in range(3):
            k = kernel[dy, dx]
            if k != 0.0:
                out += k * padded[dy:dy + H, dx:dx + W]
    return out


def sobel_edges(v):
    """!
    Own-code Sobel-Feldman gradient magnitude on the Value channel.

    Value is used (rather than hue) because the gradient is stable there even in
    weakly-coloured regions.  The magnitude is normalised to [0,1].

    @param v  value channel in [0,1].
    @return edge-magnitude map in [0,1], same shape.
    """
    gx = _convolve3x3(v, _KX)
    gy = _convolve3x3(v, _KY)
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    if m > 0:
        mag = mag / m
    return mag.astype(np.float32)

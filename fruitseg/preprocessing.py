"""!
@file preprocessing.py
@brief Preprocessing extensions: own-code median filtering (oc), own-code separable
       Gaussian low-pass (oc), and own-code Sobel-Feldman edge operator (oc).

Median filter:  Tukey, "Exploratory Data Analysis", 1977.  Robust to salt-and-
                pepper speckle, preserves edges -> prevents over-splitting.
Gaussian blur:  Marr & Hildreth, "Theory of edge detection", Proc. R. Soc. 1980.
                Separable implementation: G2D(x,y) = G(x) * G(y).
Sobel operator: Sobel & Feldman, SAIL 1968.  Used here to drive the edge-based
                split criterion (Pavlidis & Liow, IEEE TPAMI 1990).
"""

import numpy as np


# --- 1-D Gaussian helpers (own code) ------------------------------------------

def _gaussian_kernel_1d(sigma, radius):
    """Build a normalised 1-D Gaussian kernel of half-width `radius` (own code)."""
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-x * x / (2.0 * sigma * sigma))
    return k / k.sum()


def _convolve1d_h(channel, kernel):
    """Own-code 1-D convolution along image rows using shifted-array summation."""
    pad = len(kernel) // 2
    padded = np.pad(channel, ((0, 0), (pad, pad)), mode='reflect')
    H, W = channel.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i, k_val in enumerate(kernel):
        if k_val != 0.0:
            out += k_val * padded[:, i:i + W]
    return out


def _convolve1d_v(channel, kernel):
    """Own-code 1-D convolution along image columns using shifted-array summation."""
    pad = len(kernel) // 2
    padded = np.pad(channel, ((pad, pad), (0, 0)), mode='reflect')
    H, W = channel.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i, k_val in enumerate(kernel):
        if k_val != 0.0:
            out += k_val * padded[i:i + H, :]
    return out


def median_filter(bgr, ksize=5):
    """!
    Own-code median filter using numpy sliding-window extraction (oc).

    For each pixel a ksize×ksize neighbourhood is extracted via stride tricks,
    flattened and sorted; the middle element is the median.  Applied per channel.
    No library median or convolution is called.

    @param bgr   uint8 BGR image.
    @param ksize odd kernel size.
    @return filtered uint8 BGR image.
    """
    pad = ksize // 2
    mid = (ksize * ksize) // 2
    out = np.empty_like(bgr)
    for c in range(bgr.shape[2]):
        ch = bgr[:, :, c].astype(np.float32)
        padded = np.pad(ch, pad, mode='reflect')
        H, W = ch.shape
        shape = (H, W, ksize, ksize)
        strd = padded.strides + padded.strides
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strd)
        flat = windows.reshape(H, W, ksize * ksize)
        out[:, :, c] = np.sort(flat, axis=-1)[:, :, mid].astype(np.uint8)
    return out


def gaussian_lowpass(bgr, sigma=1.5):
    """!
    Own-code separable Gaussian low-pass filter (oc).

    A 1-D Gaussian kernel G(x) is convolved first along rows then along columns,
    exploiting Gaussian separability: G2D(x,y) = G(x) * G(y).  No library blur
    is called.

    @param bgr   uint8 BGR image.
    @param sigma standard deviation in pixels.
    @return filtered uint8 BGR image.
    """
    if sigma <= 0:
        return bgr
    radius = max(1, int(3.0 * sigma))
    k = _gaussian_kernel_1d(sigma, radius)
    out_channels = []
    for c in range(bgr.shape[2]):
        ch = bgr[:, :, c].astype(np.float32)
        ch = _convolve1d_h(ch, k)
        ch = _convolve1d_v(ch, k)
        out_channels.append(ch)
    return np.clip(np.stack(out_channels, axis=2), 0, 255).astype(np.uint8)


# --- Sobel-Feldman 3×3 kernels (own code convolution) -------------------------
_KX = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=np.float32)
_KY = np.array([[-1, -2, -1],
                [0,  0,  0],
                [1,  2,  1]], dtype=np.float32)


def _convolve3x3(img, kernel):
    """!
    Own-code 3×3 convolution using shifted-array summation (basic numpy only).

    The image is edge-padded by one pixel; for each of the 9 kernel taps the
    correspondingly shifted view is accumulated.  No library convolution is used.

    @param img    float32 2-D array.
    @param kernel 3×3 float32 array.
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
    Own-code Sobel-Feldman gradient magnitude on the Value channel (oc).

    Value is used rather than hue because the gradient is stable there even in
    weakly-coloured regions.  The magnitude is normalised to [0, 1].

    @param v  value channel in [0, 1].
    @return edge-magnitude map in [0, 1], same shape.
    """
    gx = _convolve3x3(v, _KX)
    gy = _convolve3x3(v, _KY)
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    if m > 0:
        mag = mag / m
    return mag.astype(np.float32)

"""!
@file preprocessing.py
@brief Median filter, separable Gaussian low-pass, and Sobel edge detection
       -- all implemented from scratch using only NumPy array operations.

The median pass removes salt-and-pepper noise without blurring edges, so the
variance-based split criterion does not over-fire on noise. The Gaussian pass
gently smooths fine surface texture. Sobel edges on the Value channel drive
the edge component of the split criterion.
"""

import numpy as np


def _gaussian_kernel_1d(sigma, radius):
    """Build a normalised 1-D Gaussian kernel of half-width radius."""
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-x * x / (2.0 * sigma * sigma))
    return k / k.sum()


def _convolve1d_h(channel, kernel):
    """1-D convolution along rows using shifted-array summation (own code)."""
    pad = len(kernel) // 2
    padded = np.pad(channel, ((0, 0), (pad, pad)), mode='reflect')
    H, W = channel.shape
    out = np.zeros((H, W), dtype=np.float32)
    for i, k_val in enumerate(kernel):
        if k_val != 0.0:
            out += k_val * padded[:, i:i + W]
    return out


def _convolve1d_v(channel, kernel):
    """1-D convolution along columns using shifted-array summation (own code)."""
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
    Per-channel median filter using NumPy stride-trick window extraction (own code).

    For each pixel a ksize x ksize neighbourhood is extracted via as_strided,
    flattened, sorted, and the middle element taken as the median. No library
    median or convolution is called.

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
    Separable Gaussian low-pass filter (own code).

    Exploits the separability G2D(x, y) = G(x) * G(y): a 1-D kernel is
    convolved first along rows, then along columns. No library blur is called.

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


_KX = np.array([[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]], dtype=np.float32)
_KY = np.array([[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]], dtype=np.float32)


def _convolve3x3(img, kernel):
    """!
    3x3 convolution via shifted-array summation (own code).

    The image is edge-padded by one pixel; for each of the nine kernel taps
    the corresponding shifted view is accumulated. No library convolution is
    called.

    @param img    float32 2-D array.
    @param kernel 3x3 float32 kernel.
    @return float32 result array, same shape as img.
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
    Sobel-Feldman gradient magnitude on the Value channel (own code).

    Value is used rather than hue because its gradient is stable even in
    weakly coloured or achromatic regions. The magnitude is normalised to
    [0, 1].

    @param v  value channel in [0, 1].
    @return edge-magnitude map in [0, 1], same shape as v.
    """
    gx = _convolve3x3(v, _KX)
    gy = _convolve3x3(v, _KY)
    mag = np.sqrt(gx * gx + gy * gy)
    m = mag.max()
    if m > 0:
        mag = mag / m
    return mag.astype(np.float32)

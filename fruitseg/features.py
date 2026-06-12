"""!
@file features.py
@brief Per-region feature extraction (own code).

The classification feature vector follows the homogeneity criteria named in the
task and is extended for the 5/10-fruit cases:

    baseline (3 fruits): (circular mean hue, variance of saturation)
    extended (5/10)    : (circular mean hue, mean saturation,
                          circular variance of hue, variance of saturation,
                          mean value, variance of value)

Mean hue is split into (cos, sin) components so the periodic quantity can be used
in a Euclidean nearest-neighbour classifier without wraparound artefacts.
"""

import numpy as np
from dataclasses import dataclass

from .color_space import circular_mean, circular_variance


@dataclass
class RegionStats:
    """!Container for one region's descriptors."""
    label: int
    n_valid: int
    n_pixels: int
    mean_hue: float       # radians
    mean_sat: float
    circ_var_hue: float
    var_sat: float
    mean_val: float = 0.0
    var_val: float = 0.0

    def feature_vector(self, extended=True):
        """!
        Build the numeric feature vector.

        Hue mean is encoded as (cos, sin) to respect circularity.

        @param extended  if True use the richer 5/10-fruit form.
        @return 1-D float numpy array.
        """
        ch, sh = np.cos(self.mean_hue), np.sin(self.mean_hue)
        if extended:
            return np.array([ch, sh, self.mean_sat,
                             self.circ_var_hue, self.var_sat,
                             self.mean_val, self.var_val], dtype=np.float64)
        return np.array([ch, sh, self.var_sat], dtype=np.float64)


def region_features(label_map, h, s, valid, min_valid=30, value=None):
    """!
    Compute RegionStats for every region in a (merged) label map (own code).

    Regions with too few valid pixels are reported with n_valid below `min_valid`
    so the caller can treat them as background.

    @param label_map int label image.
    @param h,s       hue (rad), saturation [0,1].
    @param valid     boolean guard mask.
    @param min_valid minimum valid pixels for a usable colour estimate.
    @param value     optional value/brightness channel [0,1].
    @return list[RegionStats], indexed implicitly by label.
    """
    n = int(label_map.max()) + 1
    out = []
    flat_lbl = label_map.ravel()
    flat_h = h.ravel()
    flat_s = s.ravel()
    flat_v = valid.ravel()
    flat_value = value.ravel() if value is not None else None

    order = np.argsort(flat_lbl, kind="stable")
    sorted_lbl = flat_lbl[order]
    bounds = np.searchsorted(sorted_lbl, np.arange(n + 1))

    for lbl in range(n):
        idx = order[bounds[lbl]:bounds[lbl + 1]]
        npix = idx.size
        vmask = flat_v[idx]
        nv = int(vmask.sum())
        if nv == 0:
            out.append(RegionStats(lbl, 0, npix, 0.0, 0.0, 1.0, 1.0))
            continue
        hv = flat_h[idx][vmask]
        sv = flat_s[idx][vmask]
        vv = flat_value[idx][vmask] if flat_value is not None else None
        out.append(RegionStats(
            label=lbl,
            n_valid=nv,
            n_pixels=npix,
            mean_hue=circular_mean(hv),
            mean_sat=float(sv.mean()),
            circ_var_hue=circular_variance(hv),
            var_sat=float(sv.var()),
            mean_val=float(vv.mean()) if vv is not None else 0.0,
            var_val=float(vv.var()) if vv is not None else 0.0,
        ))
    return out

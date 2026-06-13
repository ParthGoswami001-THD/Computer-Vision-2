"""!
@file features.py
@brief Per-region feature extraction (own code).

Feature vector layout:
    baseline (3 fruits):  [cos(mean_hue), sin(mean_hue), var_sat]
    extended (5 / 10):    [cos(mean_hue), sin(mean_hue), mean_sat,
                           circ_var_hue, var_sat, mean_val, var_val]

Mean hue is encoded as (cos, sin) so it can be used in a Euclidean
nearest-neighbour classifier without wraparound artefacts.
"""

import numpy as np
from dataclasses import dataclass

from .color_space import circular_mean, circular_variance


@dataclass
class RegionStats:
    """Container for descriptors of one merged region."""
    label: int
    n_valid: int
    n_pixels: int
    mean_hue: float
    mean_sat: float
    circ_var_hue: float
    var_sat: float
    mean_val: float = 0.0
    var_val: float = 0.0

    def feature_vector(self, extended=True):
        """!
        Build the numeric feature vector (own code).

        Hue mean is encoded as (cos, sin) to respect circularity.

        @param extended  if True, return the 7-element extended form; otherwise
                         return the 3-element baseline form.
        @return 1-D float64 numpy array.
        """
        ch, sh = np.cos(self.mean_hue), np.sin(self.mean_hue)
        if extended:
            return np.array([ch, sh, self.mean_sat,
                             self.circ_var_hue, self.var_sat,
                             self.mean_val, self.var_val], dtype=np.float64)
        return np.array([ch, sh, self.var_sat], dtype=np.float64)


def region_features(label_map, h, s, valid, min_valid=30, value=None):
    """!
    Compute RegionStats for every region in a merged label map (own code).

    Regions with fewer than min_valid valid pixels are reported with n_valid
    below that threshold so the classifier can treat them as background.

    @param label_map  int32 label image from merge_regions.
    @param h, s       hue (radians) and saturation [0, 1] arrays.
    @param valid      boolean guard mask.
    @param min_valid  minimum valid pixels for a usable colour estimate.
    @param value      optional Value channel array [0, 1].
    @return list[RegionStats], one entry per label.
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

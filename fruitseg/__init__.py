"""!
@package fruitseg
HSV split-and-merge fruit segmentation -- Computer Vision Assignment 2.

Pipeline overview (see docs/ALGORITHM.md for references):
    RGB -> HSV -> guard mask -> preprocessing -> SPLIT (quadtree, >= 16 regions)
    -> MERGE (Region Adjacency Graph, edge-aware) -> morphological cleanup
    -> feature extraction -> nearest-neighbour classification

Tags used in source files:
    (wl) = library allowed (numpy / opencv)
    (oc) = own code (basic numpy array operations only)
"""

from .color_space import (to_hsv_float, guard_mask,
                          circular_mean, circular_variance, circular_distance)
from .preprocessing import median_filter, gaussian_lowpass, sobel_edges
from .split_merge import split_quadtree, merge_regions
from .postprocess import morphological_cleanup, area_filter
from .features import region_features, RegionStats
from .classify import ClassReference, build_references, classify_regions
from .pipeline import segment_image, add_legend, side_by_side, SegmentationConfig
from .evaluation import (evaluate_classification, metrics_from_confusion,
                         print_report)

__version__ = "1.0.0"

__all__ = [
    "to_hsv_float", "guard_mask",
    "circular_mean", "circular_variance", "circular_distance",
    "median_filter", "gaussian_lowpass", "sobel_edges",
    "split_quadtree", "merge_regions",
    "morphological_cleanup", "area_filter",
    "region_features", "RegionStats",
    "ClassReference", "build_references", "classify_regions",
    "segment_image", "add_legend", "side_by_side", "SegmentationConfig",
    "evaluate_classification", "metrics_from_confusion", "print_report",
]

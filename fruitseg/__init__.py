"""!
@package fruitseg
HSV split-and-merge fruit segmentation (Computer Vision Assignment 2).

Pipeline (see docs/ALGORITHM.md for the scientific reasoning and IEEE references):

    RGB --(wl)--> HSV --> luminance/saturation guard mask
        --> preprocessing (median + Gaussian, Sobel edge map)
        --(oc)--> SPLIT (quadtree, >=16 start regions)
        --(oc)--> MERGE (Region Adjacency Graph, edge-aware)
        --(wl)--> morphological opening/closing + area filter
        --(oc)--> feature extraction + nearest-neighbour classification

Tags:  (wl) = library allowed (numpy/scipy/opencv)
       (oc) = own code  (only basic numpy matrix operations)

Module map:
    color_space    RGB->HSV (wl), guard mask + circular hue statistics (oc)
    preprocessing  median + Gaussian (wl), Sobel edges (oc)
    split_merge    quadtree split + RAG merge (oc)  <- core of the assignment
    postprocess    morphological cleanup + area filter (wl)
    features       per-region feature vectors (oc)
    classify       Train references + nearest-neighbour classifier (oc)
    pipeline       orchestration + SegmentationConfig (all parameters)
    evaluation     confusion matrix + accuracy/recall/F1
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

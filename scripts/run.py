"""!
@file scripts/run.py
@brief Main entry point: build class references from the Fruits-360 Training
       folder, segment a target image, or validate on the Test folder.

Run from the project root, e.g.:

    python scripts/run.py --train data/Fruits-360/fruits-360_100x100/fruits-360/Training \\
                          --image data/scene.jpg \\
                          --out results/overlay.png --nfruits 3

    python scripts/run.py --train data/Fruits-360/fruits-360_100x100/fruits-360/Training \\
                          --test  data/Fruits-360/fruits-360_100x100/fruits-360/Test \\
                          --evaluate --nfruits 3

Pass the shorthand root and the script resolves Training/Test automatically:

    python scripts/run.py --train data/Fruits-360 --test data/Fruits-360 \\
                          --evaluate --nfruits 3

Folder names in SPEC_10 must match your local Fruits-360 copy exactly.
"""

import argparse
import csv
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
from fruitseg import (build_references, segment_image, add_legend, side_by_side,
                      SegmentationConfig, evaluate_classification, print_report)


# Fruits-360 can be downloaded as a ZIP that unpacks to a nested directory.
# This helper tries the user-supplied path first, then a few common sub-paths
# so short-hands like  --train data/Fruits-360  still work.
_F360_SUBDIRS = [
    "fruits-360_100x100/fruits-360",
    "fruits-360",
]

def _resolve_fruits360(path, split):
    """Return the first existing directory that looks like a Fruits-360 split folder."""
    candidate = os.path.join(path, split)
    if os.path.isdir(candidate):
        return candidate
    for sub in _F360_SUBDIRS:
        candidate = os.path.join(path, sub, split)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(path, split)   # let the caller handle the missing-dir error


# Class specifications ranked by measured HSV separation from the Training set.
# The ordering keeps the 10-class validation path away from the worst hue
# collisions (red-fruit variants, pear/lime, watermelon/cucumber).
SPEC_10 = [
    ("Cherry 1",           "Cherry",       (0, 0, 180)),
    ("Orange 1",           "Orange",       (0, 140, 255)),
    ("Banana 1",           "Banana",       (0, 255, 255)),
    ("Avocado 1",          "Avocado",      (0, 200, 0)),
    ("Cucumber 1",         "Cucumber",     (80, 180, 40)),
    ("Cherry Wax Black 1", "Cherry Black", (80, 20, 80)),
    ("Cucumber 3",         "Cucumber 3",   (255, 180, 40)),
    ("Huckleberry 1",      "Huckleberry",  (255, 80, 0)),
    ("Raspberry 1",        "Raspberry",    (200, 60, 180)),
    ("Lychee 1",           "Lychee",       (180, 180, 255)),
]


def get_spec(nfruits):
    """!Return the class spec for 3, 5, or 10 fruits."""
    if nfruits == 3:
        return SPEC_10[:3]
    if nfruits == 5:
        return SPEC_10[:5]
    if nfruits == 10:
        return SPEC_10
    raise ValueError("nfruits must be 3, 5, or 10")


def build_references_for_run(train_dir, spec):
    """Build class references with a stable normalisation context.

    The 3-class run uses the 5-class z-score context so the starter classes
    are normalised consistently with the 5-fruit validation path.
    """
    if len(spec) == 3:
        refs, nmean, nstd = build_references(train_dir, SPEC_10[:5], extended=True)
        return refs[:3], nmean, nstd
    return build_references(train_dir, spec, extended=True)


def make_config(nfruits, max_side, scene_tuning=False):
    """!Build a SegmentationConfig for clean Test evaluation or messy scene overlays."""
    cfg = SegmentationConfig(extended_features=True, max_side=max_side)
    if scene_tuning:
        cfg.s_min = 0.15
        cfg.v_min = 0.15
        cfg.v_max = 1.0
        cfg.tau_e = 0.05
        cfg.hue_thresh = 0.22
        cfg.sat_thresh = 0.10
        cfg.merged_var_h = 0.05
        cfg.merged_var_s = 0.02
        cfg.edge_veto = 0.20
        cfg.use_hsv_edges = True
        cfg.reject_z = 1.55
        cfg.min_area = 80
        cfg.min_area_frac = 0.0008
        cfg.morph_radius = 4
        cfg.refine_masks = True
        cfg.refine_hue_tol = 0.35
        cfg.fill_components = False
        cfg.expand_masks = True
        cfg.expand_hue_tol = 0.25
        cfg.expand_s_min = 0.20
        cfg.expand_v_min = 0.20
        cfg.expand_min_seed_area = 180
        cfg.expand_min_seed_frac = 0.0045
        cfg.expand_edge_veto = 0.35
        cfg.min_class_fraction = 0.10
    return cfg


def _save_report(result, path, spec, refs):
    """Save confusion matrix, per-class metrics, and per-file outcomes to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    names = result["labels"]
    pred_names = result.get("pred_labels", names)
    conf = result["confusion"]
    m = result["metrics"]
    ref_by_name = {r.name: r for r in refs}

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["=== CONFUSION MATRIX (rows=true, cols=predicted) ==="])
        w.writerow([""] + pred_names)
        for i, row in enumerate(conf):
            w.writerow([names[i]] + list(row))
        w.writerow([])
        w.writerow(["=== PER-CLASS METRICS ==="])
        w.writerow(["class", "n_train", "precision", "recall", "f1", "support"])
        for name, d in m["per_class"].items():
            n_train = ref_by_name[name].n_images if name in ref_by_name else ""
            w.writerow([name, n_train,
                        f"{d['precision']:.4f}", f"{d['recall']:.4f}",
                        f"{d['f1']:.4f}", d["support"]])
        w.writerow(["overall_accuracy", "", f"{m['overall_accuracy']:.4f}", "", "", ""])
        w.writerow([])
        w.writerow(["=== CORRECT CLASSIFICATIONS ==="])
        w.writerow(["file", "true_class"])
        for path_f, tn in result["correct_files"]:
            w.writerow([path_f, tn])
        w.writerow([])
        w.writerow(["=== WRONG / REJECTED CLASSIFICATIONS ==="])
        w.writerow(["file", "true_class", "predicted_class"])
        for entry in result["wrong_files"]:
            if len(entry) == 3:
                w.writerow(list(entry))
            else:
                w.writerow([entry[0], entry[1], "background"])


def main():
    ap = argparse.ArgumentParser(
        description="HSV split-and-merge fruit segmentation")
    ap.add_argument("--train", required=True, help="Fruits-360 Training folder")
    ap.add_argument("--image", help="image to segment (multi-fruit scene or Fruits-262)")
    ap.add_argument("--out", default="results/overlay.png", help="output overlay path")
    ap.add_argument("--test", help="Fruits-360 Test folder (requires --evaluate)")
    ap.add_argument("--evaluate", action="store_true", help="run Test-folder evaluation")
    ap.add_argument("--nfruits", type=int, default=3, choices=[3, 5, 10])
    ap.add_argument("--max-side", type=int, default=480)
    ap.add_argument("--no-scene-tuning", action="store_true",
                    help="use the base config instead of the scene preset for --image")
    ap.add_argument("--save-report", metavar="PATH",
                    help="save evaluation results to a CSV at PATH")
    ap.add_argument("--no-file-list", action="store_true",
                    help="suppress per-file listing in console output")
    args = ap.parse_args()

    # Accept both the short root (data/Fruits-360) and the full path
    # (data/Fruits-360/fruits-360_100x100/fruits-360/Training).
    train_dir = _resolve_fruits360(args.train, "Training") \
                if not os.path.basename(args.train).lower().startswith("train") \
                else args.train
    test_dir = _resolve_fruits360(args.test, "Test") \
               if args.test and not os.path.basename(args.test).lower().startswith("test") \
               else args.test

    spec = get_spec(args.nfruits)
    cfg = make_config(len(spec), args.max_side, scene_tuning=False)
    cfg.extended_features = True

    print(f"Building references for {len(spec)} fruits ...")
    print(f"  training dir: {train_dir}")
    refs, nmean, nstd = build_references_for_run(train_dir, spec)
    print("  built:", ", ".join(f"{r.name}({r.n_images})" for r in refs))

    if args.evaluate:
        if not args.test:
            ap.error("--evaluate requires --test")
        print(f"  test dir:     {test_dir}")
        result = evaluate_classification(test_dir, spec, refs, nmean, nstd, cfg)
        print_report(result, show_files=not args.no_file_list)
        if args.save_report:
            _save_report(result, args.save_report, spec, refs)
            print(f"Report saved -> {args.save_report}")
        return

    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img is None:
            ap.error(f"could not read image: {args.image}")
        scene_cfg = make_config(len(spec), args.max_side,
                                scene_tuning=not args.no_scene_tuning)
        scene_cfg.extended_features = True
        res = segment_image(img, refs, nmean, nstd, scene_cfg)
        detected_refs = [ref for i, ref in enumerate(refs)
                         if (res["class_map"] == i).any()]
        comparison = side_by_side(img, res["overlay"])
        overlay = add_legend(comparison, detected_refs, title="Detected")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, overlay)
        print(f"Saved overlay -> {args.out}")


if __name__ == "__main__":
    main()

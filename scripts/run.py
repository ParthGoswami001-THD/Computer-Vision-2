"""!
@file scripts/run.py
@brief Main entry point.  Builds class references from the Fruits-360 Train folder,
       segments a target image (test-multiple_fruits or Fruits-262), and saves the overlay;
       or validates on the Test folder and prints the confusion matrix.

Run from the project root, e.g.:

    python scripts/run.py --train data/Fruits-360/Training \
                          --image data/scene.jpg \
                          --out results/overlay.png --nfruits 3

    python scripts/run.py --train data/Fruits-360/Training \
                          --test  data/Fruits-360/Test \
                          --evaluate --nfruits 3

Class folder names in SPEC_10 must match your local Fruits-360 copy.
Overlay colours are (B, G, R): red pears, blue mandarins, etc.
"""

import argparse
import csv
import os
import sys

# --- make the fruitseg package importable when run from anywhere ----------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
from fruitseg import (build_references, segment_image, add_legend, side_by_side,
                    SegmentationConfig, evaluate_classification, print_report)

# -----------------------------------------------------------------------------
# Class specifications: (folder_name, display_name, overlay_BGR)
# Ranked by measured HSV separation from the Fruits-360 Training set. This set
# keeps the 10-class validation path away from the worst hue collisions
# (strawberry/pomegranate, pear/lime, watermelon/cucumber).
# -----------------------------------------------------------------------------
SPEC_10 = [
    ("Apple Red 1",        "Apple",        (0, 0, 180)),       # red, dark
    ("Orange 1",           "Orange",       (0, 140, 255)),     # orange
    ("Banana 1",           "Banana",       (0, 255, 255)),     # yellow
    ("Avocado 1",          "Avocado",      (0, 200, 0)),       # yellow-green/dark
    ("Cucumber 1",         "Cucumber",     (80, 180, 40)),     # green
    ("Cherry Wax Black 1", "Cherry Black", (80, 20, 80)),      # near-purple/dark
    ("Cucumber 3",         "Cucumber 3",   (255, 180, 40)),    # cyan-green
    ("Huckleberry 1",      "Huckleberry",  (255, 80, 0)),      # blue
    ("Raspberry 1",        "Raspberry",    (200, 60, 180)),    # purple
    ("Lychee 1",           "Lychee",       (180, 180, 255)),   # pale red/bright
]

MIXED_BOWL_SPEC = [
    ("Apple Red 1",   "Apple",       (0, 0, 255)),
    ("Apricot 1",     "Apricot",     (0, 150, 255)),
    ("Peach 1",       "Peach",       (40, 170, 255)),
    ("Peach Flat 1",  "Peach Flat",  (80, 190, 255)),
    ("Pear 1",        "Pear",        (120, 220, 180)),
    ("Plum 1",        "Plum",        (180, 60, 180)),
    ("Pomegranate 1", "Pomegranate", (30, 30, 220)),
]


def get_spec(nfruits, preset="assignment"):
    """!Return the class spec for 3, 5, or 10 fruits."""
    if preset == "mixed-bowl":
        return MIXED_BOWL_SPEC
    if nfruits == 3:
        return SPEC_10[:3]
    if nfruits == 5:
        return SPEC_10[:5]
    if nfruits == 10:
        return SPEC_10
    raise ValueError("nfruits must be 3, 5, or 10")


def make_config(nfruits, max_side, scene_tuning=False):
    """!Build a config for either clean Test evaluation or messy scene overlays."""
    cfg = SegmentationConfig(extended_features=nfruits >= 5, max_side=max_side)
    if scene_tuning:
        cfg.s_min = 0.30
        cfg.v_min = 0.35
        cfg.v_max = 1.0
        cfg.tau_e = 0.05
        cfg.hue_thresh = 0.22
        cfg.sat_thresh = 0.10
        cfg.merged_var_h = 0.05
        cfg.merged_var_s = 0.02
        cfg.edge_veto = 0.20
        cfg.reject_z = 1.4
        cfg.min_area = 800
        cfg.morph_radius = 5
        cfg.refine_masks = True
        cfg.refine_hue_tol = 0.45
        cfg.fill_components = True
        cfg.expand_masks = True
        cfg.expand_hue_tol = 0.38
        cfg.expand_s_min = 0.22
        cfg.expand_v_min = 0.25
        cfg.expand_min_seed_area = 5000
    return cfg


def _save_report(result, path, spec, refs):
    """Save confusion matrix, per-class metrics, and per-file outcomes to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    names = result["labels"]
    conf = result["confusion"]
    m = result["metrics"]
    ref_by_name = {r.name: r for r in refs}

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)

        # --- confusion matrix -------------------------------------------------
        w.writerow(["=== CONFUSION MATRIX (rows=true, cols=predicted) ==="])
        w.writerow([""] + names)
        for i, row in enumerate(conf):
            w.writerow([names[i]] + list(row))
        w.writerow([])

        # --- per-class metrics ------------------------------------------------
        w.writerow(["=== PER-CLASS METRICS ==="])
        w.writerow(["class", "n_train", "precision", "recall", "f1", "support"])
        for name, d in m["per_class"].items():
            n_train = ref_by_name[name].n_images if name in ref_by_name else ""
            w.writerow([name, n_train,
                        f"{d['precision']:.4f}", f"{d['recall']:.4f}",
                        f"{d['f1']:.4f}", d["support"]])
        w.writerow(["overall_accuracy", "", f"{m['overall_accuracy']:.4f}", "", "", ""])
        w.writerow([])

        # --- correct file names -----------------------------------------------
        w.writerow(["=== CORRECT CLASSIFICATIONS ==="])
        w.writerow(["file", "true_class"])
        for path_f, true_name in result["correct_files"]:
            w.writerow([path_f, true_name])
        w.writerow([])

        # --- wrong / rejected file names --------------------------------------
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
    ap.add_argument("--image", help="image to segment (test-multiple_fruits or Fruits-262)")
    ap.add_argument("--out", default="results/overlay.png", help="output overlay path")
    ap.add_argument("--test", help="Fruits-360 Test folder (for --evaluate)")
    ap.add_argument("--evaluate", action="store_true", help="run Test-folder evaluation")
    ap.add_argument("--nfruits", type=int, default=3, choices=[3, 5, 10])
    ap.add_argument("--preset", default="assignment",
                    choices=["assignment", "mixed-bowl"],
                    help="class set to use; mixed-bowl matches the apple/apricot/peach/pear/plum/pomegranate sample")
    ap.add_argument("--max-side", type=int, default=480)
    ap.add_argument("--no-scene-tuning", action="store_true",
                    help="use the base Train/Test config for --image overlays")
    ap.add_argument("--save-report", metavar="PATH",
                    help="save evaluation results (metrics + file names) to a CSV at PATH")
    ap.add_argument("--no-file-list", action="store_true",
                    help="suppress per-file correct/wrong listing in the console output")
    args = ap.parse_args()

    spec = get_spec(args.nfruits, args.preset)
    extended = len(spec) >= 5
    cfg = make_config(len(spec), args.max_side, scene_tuning=False)
    cfg.extended_features = extended
    if args.preset == "mixed-bowl":
        cfg.refine_hue_tol = 0.65
        cfg.reject_z = 1.8
        cfg.min_area = 500

    print(f"Building references for {len(spec)} fruits ...")
    refs, nmean, nstd = build_references(args.train, spec, extended=extended)
    print("  built:", ", ".join(f"{r.name}({r.n_images})" for r in refs))

    if args.evaluate:
        if not args.test:
            ap.error("--evaluate requires --test")
        result = evaluate_classification(args.test, spec, refs, nmean, nstd, cfg)
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
        scene_cfg.extended_features = extended
        if args.preset == "mixed-bowl":
            scene_cfg.refine_hue_tol = 0.65
            scene_cfg.reject_z = 1.8
            scene_cfg.min_area = 500
        res = segment_image(img, refs, nmean, nstd, scene_cfg)
        detected_refs = [
            ref for i, ref in enumerate(refs)
            if (res["class_map"] == i).any()
        ]
        comparison = side_by_side(img, res["overlay"])
        overlay = add_legend(comparison, detected_refs, title="Detected")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, overlay)
        print(f"Saved overlay -> {args.out}")


if __name__ == "__main__":
    main()

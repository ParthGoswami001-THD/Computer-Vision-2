"""!
@file scripts/run.py
@brief Main entry point.  Builds class references from the Fruits-360 Train folder,
       segments a target image (multi-fruit / Fruits-262), and saves the overlay;
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
import os
import sys

# --- make the fruitseg package importable when run from anywhere ----------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
from fruitseg import (build_references, segment_image, SegmentationConfig,
                    evaluate_classification, print_report)

# -----------------------------------------------------------------------------
# Class specifications: (folder_name, display_name, overlay_BGR)
# Ranked by hue separation. First 3 = well-separated starter set; 4-5 add the
# first confusions; 6-10 add genuinely hard, hue-overlapping classes that the
# report should analyse as worst-performers.  Edit folder names to match your copy.
# -----------------------------------------------------------------------------
SPEC_10 = [
    ("Strawberry", "Strawberry", (0, 0, 255)),     # red
    ("Banana",     "Banana",     (0, 255, 255)),   # yellow
    ("Avocado",    "Avocado",    (0, 200, 0)),     # green
    ("Orange",     "Orange",     (0, 140, 255)),   # orange
    ("Plum",       "Plum",       (255, 0, 0)),     # blue/purple
    ("Apricot",    "Apricot",    (0, 170, 255)),   # ~orange  (hard)
    ("Pineapple",  "Pineapple",  (0, 230, 230)),   # ~yellow  (hard)
    ("Kiwi",       "Kiwi",       (60, 160, 60)),   # green-brown (hard)
    ("Raspberry",  "Raspberry",  (40, 40, 220)),   # ~red     (hard)
    ("Pear",       "Pear",       (120, 220, 180)), # yellow-green (hard)
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


def main():
    ap = argparse.ArgumentParser(
        description="HSV split-and-merge fruit segmentation")
    ap.add_argument("--train", required=True, help="Fruits-360 Training folder")
    ap.add_argument("--image", help="image to segment (multi-fruit / Fruits-262)")
    ap.add_argument("--out", default="results/overlay.png", help="output overlay path")
    ap.add_argument("--test", help="Fruits-360 Test folder (for --evaluate)")
    ap.add_argument("--evaluate", action="store_true", help="run Test-folder evaluation")
    ap.add_argument("--nfruits", type=int, default=3, choices=[3, 5, 10])
    ap.add_argument("--max-side", type=int, default=320)
    args = ap.parse_args()

    spec = get_spec(args.nfruits)
    extended = args.nfruits >= 5
    cfg = SegmentationConfig(extended_features=extended, max_side=args.max_side)

    print(f"Building references for {args.nfruits} fruits ...")
    refs, nmean, nstd = build_references(args.train, spec, extended=extended)
    print("  built:", ", ".join(f"{r.name}({r.n_images})" for r in refs))

    if args.evaluate:
        if not args.test:
            ap.error("--evaluate requires --test")
        result = evaluate_classification(args.test, spec, refs, nmean, nstd, cfg)
        print_report(result)
        return

    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img is None:
            ap.error(f"could not read image: {args.image}")
        res = segment_image(img, refs, nmean, nstd, cfg)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, res["overlay"])
        print(f"Saved overlay -> {args.out}")


if __name__ == "__main__":
    main()

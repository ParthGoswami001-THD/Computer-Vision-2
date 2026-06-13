"""!
@file scripts/selftest.py
@brief Synthetic end-to-end test -- no real dataset required.

Builds tiny synthetic Training/Test folders of solid-colour fruit blobs and a
multi-fruit scene, then runs the full pipeline and evaluation to verify that
every module executes correctly and data flows through without error.

Run from the project root:
    python scripts/selftest.py
"""

import os
import sys
import tempfile

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import cv2
from fruitseg import (build_references, segment_image, SegmentationConfig,
                      evaluate_classification, print_report)


def _solid_fruit(color_bgr, size=100, radius=38, jitter=10):
    img = np.full((size, size, 3), 255, np.uint8)
    cx = size // 2 + np.random.randint(-6, 6)
    cy = size // 2 + np.random.randint(-6, 6)
    col = np.clip(np.array(color_bgr) + np.random.randint(-jitter, jitter, 3), 0, 255)
    cv2.circle(img, (cx, cy), radius, col.tolist(), -1)
    return img


def _make_dataset(root, specs, n_train=12, n_test=6):
    for split, count in (("Training", n_train), ("Test", n_test)):
        for folder, _name, color in specs:
            d = os.path.join(root, split, folder)
            os.makedirs(d, exist_ok=True)
            for i in range(count):
                cv2.imwrite(os.path.join(d, f"{i}.jpg"), _solid_fruit(color))


def _make_multi_scene(specs):
    H, W = 300, 360
    scene = np.full((H, W, 3), 210, np.uint8)
    scene = np.clip(scene.astype(int) + np.random.randint(-12, 12, (H, W, 3)),
                    0, 255).astype(np.uint8)
    for (cy, cx), (_f, _n, color) in zip([(90, 80), (90, 200), (210, 130)], specs):
        col = np.clip(np.array(color) + np.random.randint(-8, 8, 3), 0, 255)
        cv2.circle(scene, (cx, cy), 45, col.tolist(), -1)
    return scene


def main():
    np.random.seed(0)
    specs = [
        ("Strawberry", "Strawberry", (0, 0, 255)),
        ("Banana",     "Banana",     (0, 255, 255)),
        ("Avocado",    "Avocado",    (0, 200, 0)),
    ]
    with tempfile.TemporaryDirectory() as root:
        print("1) building synthetic dataset ...")
        _make_dataset(root, specs)

        print("2) building references ...")
        refs, nmean, nstd = build_references(os.path.join(root, "Training"),
                                             specs, extended=False)
        for r in refs:
            print(f"   {r.name:<12} n={r.n_images} feat={np.round(r.mean_feature, 3)}")

        print("3) segmenting a synthetic multi-fruit scene ...")
        cfg = SegmentationConfig(extended_features=False, max_side=300)
        scene = _make_multi_scene(specs)
        res = segment_image(scene, refs, nmean, nstd, cfg, return_debug=True)
        cmap = res["class_map"]
        print("   split regions :", int(res["split_labels"].max()) + 1)
        print("   merged regions:", int(res["merged_labels"].max()) + 1)
        present = sorted(set(cmap[cmap >= 0].ravel().tolist()))
        print("   classes found in scene:", [refs[i].name for i in present])

        print("4) evaluating on synthetic Test folder ...")
        result = evaluate_classification(os.path.join(root, "Test"),
                                         specs, refs, nmean, nstd, cfg, max_per_class=6)
        print_report(result)

    print("\nSELF-TEST COMPLETE -- all modules executed.")


if __name__ == "__main__":
    main()

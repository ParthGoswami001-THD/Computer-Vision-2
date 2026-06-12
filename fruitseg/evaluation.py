"""!
@file evaluation.py
@brief Evaluation utilities: confusion matrix and per-class accuracy/recall/F1,
       with filename logging as required by the assignment.

Single-fruit validation on the Fruits-360 Test folder: each Test image contains
one fruit on a (near) white background, so the predicted class is the dominant
non-background region produced by the pipeline.
"""

import os
import glob
import numpy as np
import cv2

from .pipeline import segment_image, SegmentationConfig


def _dominant_class(class_map, n_classes):
    """!Return the most frequent non-background class in a class map (-1 if none)."""
    if (class_map >= 0).any():
        counts = np.bincount(class_map[class_map >= 0].ravel(), minlength=n_classes)
    else:
        counts = np.zeros(n_classes, dtype=int)
    if counts.sum() == 0:
        return -1
    return int(np.argmax(counts))


def evaluate_classification(test_dir, class_spec, references, norm_mean, norm_std,
                            cfg=None, max_per_class=60):
    """!
    Validate on single-fruit Test images and build a confusion matrix.

    @param test_dir   Fruits-360 'Test' folder.
    @param class_spec list of (folder, name, color).
    @param references,norm_mean,norm_std  classifier parameters.
    @param cfg        SegmentationConfig.
    @param max_per_class images sampled per class.
    @return dict: confusion (n x n), labels, metrics, and filename logs.
    """
    cfg = cfg or SegmentationConfig()
    names = [r.name for r in references]
    n = len(references)
    confusion = np.zeros((n, n), dtype=int)
    correct_files, wrong_files = [], []

    for true_idx, (folder, name, _color) in enumerate(class_spec):
        paths = sorted(glob.glob(os.path.join(test_dir, folder, "*.jpg")))
        if not paths:
            paths = sorted(glob.glob(os.path.join(test_dir, folder, "*.png")))
        for p in paths[:max_per_class]:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            res = segment_image(img, references, norm_mean, norm_std, cfg)
            pred = _dominant_class(res["class_map"], n)
            if pred < 0:
                wrong_files.append((p, name, "background"))
                continue
            confusion[true_idx, pred] += 1
            if pred == true_idx:
                correct_files.append((p, name))
            else:
                wrong_files.append((p, name, names[pred]))

    metrics = metrics_from_confusion(confusion, names)
    return {
        "confusion": confusion,
        "labels": names,
        "metrics": metrics,
        "correct_files": correct_files,
        "wrong_files": wrong_files,
    }


def metrics_from_confusion(confusion, names):
    """!Compute per-class precision, recall, F1 and overall accuracy (own code)."""
    n = confusion.shape[0]
    out = {}
    total = confusion.sum()
    diag = np.trace(confusion)
    out["overall_accuracy"] = float(diag / total) if total else 0.0
    per_class = {}
    for i in range(n):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class[names[i]] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(confusion[i, :].sum()),
        }
    out["per_class"] = per_class
    return out


def print_report(result, show_files=True):
    """!Pretty-print the confusion matrix, metrics, and per-file results.

    @param result      dict returned by evaluate_classification.
    @param show_files  if True, list every wrongly-classified file and
                       summarise correctly-classified ones by class
                       (satisfies the assignment's "file names and results"
                       documentation requirement).
    """
    names = result["labels"]
    conf = result["confusion"]
    w = max(len(s) for s in names) + 1
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(" " * (w + 1) + " ".join(f"{s[:6]:>7}" for s in names))
    for i, row in enumerate(conf):
        print(f"{names[i]:<{w}} " + " ".join(f"{v:>7d}" for v in row))
    m = result["metrics"]
    print(f"\nOverall accuracy: {m['overall_accuracy']:.3f}")
    print(f"\n{'class':<{w}} {'prec':>6} {'recall':>7} {'f1':>6} {'n':>5}")
    for name, d in m["per_class"].items():
        print(f"{name:<{w}} {d['precision']:>6.3f} {d['recall']:>7.3f} "
              f"{d['f1']:>6.3f} {d['support']:>5d}")

    correct = result["correct_files"]
    wrong = result["wrong_files"]
    print(f"\nCorrect: {len(correct)}   Wrong/rejected: {len(wrong)}")

    if not show_files:
        return

    # --- correctly-classified files grouped by class --------------------------
    if correct:
        print("\n--- CORRECT classifications ---")
        by_class = {}
        for path, true_name in correct:
            by_class.setdefault(true_name, []).append(path)
        for cls, paths in sorted(by_class.items()):
            print(f"  {cls} ({len(paths)} files):")
            for p in paths:
                print(f"    {p}")

    # --- wrongly-classified / rejected files ----------------------------------
    if wrong:
        print("\n--- WRONG / REJECTED classifications ---")
        for entry in wrong:
            if len(entry) == 3:
                path, true_name, pred_name = entry
            else:
                path, true_name, pred_name = entry[0], entry[1], "background"
            print(f"  {path}")
            print(f"    true={true_name}  predicted={pred_name}")

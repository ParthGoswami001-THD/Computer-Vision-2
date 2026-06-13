"""!
@file scripts/run_all.py
@brief Complete project runner.  Produces every result and graph required by
       the Computer Vision Assignment 2 specification:

    1.  Self-test  (synthetic pipeline validation)
    2.  Pipeline-step visualisation (intermediate maps on one scene image)
    3.  Quantitative evaluation on Fruits-360 Test  (3 / 5 / 10 fruits)
        -> confusion-matrix heatmaps, per-class bar charts, accuracy comparison
    4.  Scene overlay demos  (single-class and multi-fruit scenes)
    5.  Hue-distribution plot across all 10 selected classes
    6.  Region-count table (split vs merge)

Run from project root:
    python3 scripts/run_all.py
"""

import os
import sys
import csv
import subprocess
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from fruitseg import (
    build_references, segment_image, add_legend, side_by_side,
    SegmentationConfig, evaluate_classification, print_report,
    to_hsv_float, guard_mask, median_filter, gaussian_lowpass, sobel_edges,
    split_quadtree, merge_regions,
)

# ── paths ────────────────────────────────────────────────────────────────────
TRAIN  = os.path.join(_ROOT, "data/Fruits-360/fruits-360_100x100/fruits-360/Training")
TEST   = os.path.join(_ROOT, "data/Fruits-360/fruits-360_100x100/fruits-360/Test")
MULTI  = os.path.join(_ROOT, "data/Fruits-360/fruits-360_multi/test-multiple_fruits")
F262   = os.path.join(_ROOT, "data/Fruits-262/Fruit-262")
RES    = os.path.join(_ROOT, "results")
GRAPHS = os.path.join(RES, "graphs")
os.makedirs(RES, exist_ok=True)
os.makedirs(GRAPHS, exist_ok=True)

# ── class specifications ──────────────────────────────────────────────────────
SPEC_10 = [
    ("Cherry 1",           "Cherry",       (0,   0, 180), "#B22222"),
    ("Orange 1",           "Orange",       (0, 140, 255), "#FF8C00"),
    ("Banana 1",           "Banana",       (0, 255, 255), "#FFD700"),
    ("Avocado 1",          "Avocado",      (0, 200,   0), "#228B22"),
    ("Cucumber 1",         "Cucumber",     (80, 180,  40), "#7CFC00"),
    ("Cherry Wax Black 1", "Cherry Black", (80,  20,  80), "#8B008B"),
    ("Cucumber 3",         "Cucumber 3",   (255,180,  40), "#00CED1"),
    ("Huckleberry 1",      "Huckleberry",  (255,  80,   0), "#0000CD"),
    ("Raspberry 1",        "Raspberry",    (200,  60, 180), "#8A2BE2"),
    ("Lychee 1",           "Lychee",       (180, 180, 255), "#FFB6C1"),
]

SPEC_3  = SPEC_10[:3]
SPEC_5  = SPEC_10[:5]

PLT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#333333",
    "text.color":       "#222222",
    "axes.labelcolor":  "#222222",
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "grid.color":       "#CCCCCC",
    "grid.alpha":       0.6,
    "axes.titlecolor":  "#111111",
    "axes.spines.top":  False,
    "axes.spines.right":False,
}


def _drop_hex(spec): return [(f, n, tuple(int(c) for c in col)) for f,n,col,_ in spec]


# ── config helpers ────────────────────────────────────────────────────────────

def make_eval_cfg(nfruits):
    return SegmentationConfig(extended_features=True, max_side=320)


def make_scene_cfg(nfruits=10):
    cfg = SegmentationConfig(extended_features=True, max_side=640)
    cfg.s_min = 0.15; cfg.v_min = 0.15; cfg.v_max = 1.0
    cfg.tau_e = 0.05; cfg.hue_thresh = 0.22; cfg.sat_thresh = 0.10
    cfg.merged_var_h = 0.05; cfg.merged_var_s = 0.02; cfg.edge_veto = 0.20
    cfg.use_hsv_edges = True
    cfg.reject_z = 1.55; cfg.min_area = 80; cfg.min_area_frac = 0.0008; cfg.morph_radius = 4
    cfg.refine_masks = True; cfg.refine_hue_tol = 0.35; cfg.fill_components = False
    cfg.expand_masks = True; cfg.expand_hue_tol = 0.25
    cfg.expand_s_min = 0.20; cfg.expand_v_min = 0.20
    cfg.expand_min_seed_area = 180; cfg.expand_min_seed_frac = 0.0045
    cfg.min_class_fraction = 0.10
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — TRAINING FOLDER SCAN
# ══════════════════════════════════════════════════════════════════════════════

def run_training_scan():
    """Assignment step 2: scan Training folder → CSV + bar chart of all 260 classes."""
    print("\n" + "═"*70)
    print("STEP 2.  TRAINING FOLDER SCAN")
    print("═"*70)

    rows = []
    for cls in sorted(os.listdir(TRAIN)):
        p = os.path.join(TRAIN, cls)
        if not os.path.isdir(p):
            continue
        imgs = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".png"))]
        rows.append((cls, len(imgs)))

    rows.sort(key=lambda x: -x[1])
    total_imgs = sum(r[1] for r in rows)
    print(f"  Total classes : {len(rows)}")
    print(f"  Total images  : {total_imgs:,}")
    print(f"  Avg per class : {total_imgs/len(rows):.0f}")

    # Save CSV
    csv_path = os.path.join(RES, "training_scan.csv")
    with open(csv_path, "w", newline="") as fh:
        import csv as _csv
        w = _csv.writer(fh)
        w.writerow(["class", "n_images"])
        w.writerows(rows)
    print(f"  CSV → {csv_path}")

    # Bar chart — top 30 + highlight our 10 selected
    our10 = {s[1] for s in SPEC_10}
    top   = rows[:30]
    names = [r[0][:18] for r in top]
    counts = [r[1] for r in top]
    bar_cols = ["#2166AC" if r[0] in our10 else "#AACCE8" for r in top]

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(18, 6), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(range(len(top)), counts, color=bar_cols, zorder=3)
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Images in Training folder", fontsize=12)
        ax.set_title(
            f"Fruits-360 Training folder  —  Top 30 of {len(rows)} classes  "
            f"(dark blue = our 10 selected)",
            fontsize=13, color="#111111", fontweight="bold")
        ax.grid(axis="y", zorder=0)
        out = os.path.join(GRAPHS, "training_scan.png")
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Plot → {out}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3+4 — DATASET EXPLORATION  (Test folder · test-multiple_fruits · F262)
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset_exploration():
    """Assignment steps 3+4: explore Test, test-multiple_fruits, and Fruits-262."""
    print("\n" + "═"*70)
    print("STEP 3+4.  DATASET EXPLORATION")
    print("═"*70)

    # ── Step 3a: Test folder ────────────────────────────────────────────────
    test_classes = sorted(os.listdir(TEST))
    test_rows = []
    for cls in test_classes:
        p = os.path.join(TEST, cls)
        if os.path.isdir(p):
            n = len([f for f in os.listdir(p) if f.lower().endswith((".jpg",".png"))])
            test_rows.append((cls, n))
    print(f"  Test folder: {len(test_rows)} classes, "
          f"{sum(r[1] for r in test_rows):,} images")

    # ── Step 3b: test-multiple_fruits ───────────────────────────────────────
    multi_imgs = sorted([f for f in os.listdir(MULTI)
                         if f.lower().endswith((".jpg",".png"))])
    print(f"  test-multiple_fruits: {len(multi_imgs)} scene images")

    # ── Step 4: Fruits-262 suitability ─────────────────────────────────────
    f262_classes = sorted(os.listdir(F262))
    f262_matching = []
    our_names_lower = {s[1].lower() for s in SPEC_10}
    # Also map common synonyms
    synonym = {"cherry": "Cherry 1", "avocado": "Avocado",
               "orange": "Orange", "banana": "Banana", "raspberry": "Raspberry",
               "lychee": "Lychee"}
    for cls in f262_classes:
        if os.path.isdir(os.path.join(F262, cls)):
            n = len([f for f in os.listdir(os.path.join(F262, cls))
                     if f.lower().endswith((".jpg",".png"))])
            mapped = synonym.get(cls.lower(), cls.title())
            match = mapped in {s[1] for s in SPEC_10}
            f262_matching.append((cls, n, mapped, match))

    matched = [(c,n,m) for c,n,m,ok in f262_matching if ok]
    print(f"  Fruits-262: {len(f262_classes)} classes total, "
          f"{len(matched)} match our 10-class spec")
    for cls, n, mapped in matched:
        print(f"    {cls:12s} → {mapped:14s}  {n:4d} images")

    # ── Figure: grid of sample images from Fruits-262 matching classes ───────
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, len(matched), figsize=(len(matched)*2.5, 6),
                                 facecolor="white")
        if len(matched) == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        fig.suptitle(
            "Fruits-262  —  Classes Matching Our Spec  (natural-environment images)",
            fontsize=13, color="#111111", fontweight="bold", y=1.01)

        for col, (cls, n, mapped) in enumerate(matched):
            folder_p = os.path.join(F262, cls)
            imgs_f = sorted([f for f in os.listdir(folder_p)
                             if f.lower().endswith((".jpg",".png"))],
                            key=lambda f: int(f.split(".")[0])
                            if f.split(".")[0].isdigit() else 0)
            # top row: image 0 (typical natural photo)
            for row, idx in enumerate([0, min(5, len(imgs_f)-1)]):
                ax = axes[row, col]
                bgr = cv2.imread(os.path.join(folder_p, imgs_f[idx]))
                if bgr is not None:
                    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                ax.set_title(mapped if row == 0 else f"{n} images",
                             fontsize=9, color="#111111", fontweight="bold", pad=3)
                ax.axis("off"); ax.set_facecolor("white")

        axes[0, 0].set_ylabel("Sample 1", fontsize=9)
        axes[1, 0].set_ylabel("Sample 2", fontsize=9)
        plt.tight_layout()
        out = os.path.join(GRAPHS, "fruits262_selection.png")
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Plot → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_selftest():
    print("\n" + "═"*70)
    print("1.  SELF-TEST  (synthetic pipeline validation)")
    print("═"*70)
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "scripts", "selftest.py")],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("SELFTEST STDERR:", result.stderr[:500])
    ok = result.returncode == 0
    print("  → SELF-TEST:", "PASSED ✓" if ok else "FAILED ✗")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 2. PIPELINE STEP VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _colorise_labels(label_map):
    """Map integer region labels to a vivid colour image."""
    n = int(label_map.max()) + 1
    np.random.seed(42)
    palette = (np.random.randint(60, 240, (n + 1, 3))).astype(np.uint8)
    palette[0] = [30, 30, 30]
    out = palette[(label_map + 1).clip(0, n)]
    return out.astype(np.uint8)


def run_pipeline_visualisation(refs, nmean, nstd):
    print("\n" + "═"*70)
    print("2.  PIPELINE-STEP VISUALISATION")
    print("═"*70)

    img_path = os.path.join(MULTI, "banana_orange.jpg")
    bgr = cv2.imread(img_path)
    bgr = cv2.resize(bgr, (640, int(bgr.shape[0] * 640 / bgr.shape[1])))

    # --- run each step keeping intermediates ---
    pre = median_filter(bgr, 5)
    pre = gaussian_lowpass(pre, 1.5)
    h, s, v = to_hsv_float(pre)
    valid = guard_mask(s, v, 0.15, 0.15)
    edges = sobel_edges(v)

    cfg = make_scene_cfg()
    split_lbl = split_quadtree(h, s, valid, edges,
                               tau_h=cfg.tau_h, tau_s=cfg.tau_s, tau_e=cfg.tau_e,
                               min_size=cfg.min_size, min_start_depth=cfg.min_start_depth)
    merged_lbl = merge_regions(split_lbl, h, s, valid, edges,
                               hue_thresh=cfg.hue_thresh, sat_thresh=cfg.sat_thresh,
                               merged_var_h=cfg.merged_var_h, merged_var_s=cfg.merged_var_s,
                               edge_veto=cfg.edge_veto)
    res = segment_image(bgr, refs, nmean, nstd, cfg)

    n_split  = int(split_lbl.max()) + 1
    n_merged = int(merged_lbl.max()) + 1
    print(f"  Split regions:  {n_split}")
    print(f"  Merged regions: {n_merged}")

    # ── matplotlib figure ──────────────────────────────────────────────────
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, 4, figsize=(22, 11), facecolor="white")
        fig.suptitle("HSV Split-and-Merge  —  Pipeline Steps",
                     fontsize=18, color="#111111", fontweight="bold", y=0.98)

        panels = [
            (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
             "① Original BGR image", None),
            (cv2.cvtColor(pre, cv2.COLOR_BGR2RGB),
             "② Median + Gaussian blur ", None),
            ((v * 255).astype(np.uint8),
             "③ HSV — Value channel ", "gray"),
            ((valid * 255).astype(np.uint8),
             f"④ Guard mask  (S≥0.15, V≥0.15)\n{valid.sum():,} valid px", "gray"),
            ((edges * 255).astype(np.uint8),
             "⑤ Sobel edge magnitude ", "Greys_r"),
            (_colorise_labels(split_lbl),
             f"⑥ Quadtree split  [{n_split} regions] ", None),
            (_colorise_labels(merged_lbl),
             f"⑦ RAG merge  [{n_merged} regions] ", None),
            (cv2.cvtColor(res["overlay"], cv2.COLOR_BGR2RGB),
             "⑧ Final classification overlay", None),
        ]

        for ax, (img, title, cmap) in zip(axes.flat, panels):
            ax.set_facecolor("white")
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                ax.imshow(img, cmap=cmap or "gray", aspect="auto")
            else:
                ax.imshow(img, aspect="auto")
            ax.set_title(title, fontsize=11, color="#222222", pad=6,
                         fontweight="bold")
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out = os.path.join(GRAPHS, "pipeline_steps.png")
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved → {out}")
    return n_split, n_merged


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUANTITATIVE EVALUATION  (3 / 5 / 10 fruits)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_confusion_heatmap(confusion, true_labels, pred_labels, title, outpath):
    n_rows = len(true_labels)
    n_cols = len(pred_labels)
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.1), max(5, n_rows * 0.95)),
                               facecolor="white")
        ax.set_facecolor("white")

        im = ax.imshow(confusion, cmap="Blues", aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n_cols)); ax.set_xticklabels(pred_labels, rotation=40,
                                                         ha="right", fontsize=11)
        ax.set_yticks(range(n_rows)); ax.set_yticklabels(true_labels, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=13, labelpad=8)
        ax.set_ylabel("True", fontsize=13, labelpad=8)
        ax.set_title(title, fontsize=15, color="#111111", fontweight="bold", pad=12)

        vmax = confusion.max() if confusion.max() > 0 else 1
        for i in range(n_rows):
            for j in range(n_cols):
                v = confusion[i, j]
                txt_col = "white" if v > 0.55 * vmax else "#111111"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=12, color=txt_col, fontweight=weight)

        plt.tight_layout()
        fig.savefig(outpath, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _draw_per_class_bars(metrics, labels, title, outpath, accent="#2166AC"):
    prec   = [metrics["per_class"][n]["precision"] for n in labels]
    recall = [metrics["per_class"][n]["recall"]    for n in labels]
    f1     = [metrics["per_class"][n]["f1"]        for n in labels]
    x = np.arange(len(labels))
    bw = 0.26

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5.5),
                               facecolor="white")
        ax.set_facecolor("white")
        ax.bar(x - bw, prec,   bw, label="Precision", color="#4393C3", zorder=3)
        ax.bar(x,       recall, bw, label="Recall",    color="#92C5DE", zorder=3)
        ax.bar(x + bw,  f1,     bw, label="F1-score",  color=accent,    zorder=3)

        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right",
                                              fontsize=11)
        ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=13)
        ax.set_title(title, fontsize=14, color="#111111", fontweight="bold", pad=10)
        ax.axhline(1.0, color="#666666", lw=0.8, ls="--", zorder=2)
        ax.grid(axis="y", zorder=0)
        ax.legend(fontsize=11)

        for xi, f in zip(x, f1):
            ax.text(xi + bw, f + 0.02, f"{f:.2f}", ha="center", va="bottom",
                    fontsize=9, color="#111111")

        plt.tight_layout()
        fig.savefig(outpath, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def run_evaluation():
    print("\n" + "═"*70)
    print("3.  QUANTITATIVE EVALUATION  (Fruits-360 Test folder)")
    print("═"*70)

    results_store = {}
    for spec, tag in [(SPEC_3, "3fruit"), (SPEC_5, "5fruit"), (SPEC_10, "10fruit")]:
        n = len(spec)
        print(f"\n  ── {n}-class evaluation ──")
        sp3 = _drop_hex(spec)
        refs, nmean, nstd = build_references(TRAIN, sp3, extended=True)
        cfg = make_eval_cfg(n)
        t0 = time.time()
        result = evaluate_classification(TEST, sp3, refs, nmean, nstd, cfg,
                                         max_per_class=60)
        elapsed = time.time() - t0
        acc = result["metrics"]["overall_accuracy"]
        print_report(result, show_files=False)
        print(f"  Elapsed: {elapsed:.1f}s")
        results_store[tag] = (result, [s[1] for s in spec], acc)

        # save CSV
        csv_path = os.path.join(RES, f"eval_{tag}.csv")
        _save_csv(result, csv_path, refs)
        print(f"  CSV  → {csv_path}")

        # confusion heatmap
        labels = [s[1] for s in spec]
        pred_labels = result.get("pred_labels", labels)
        _draw_confusion_heatmap(
            result["confusion"], labels, pred_labels,
            f"Confusion Matrix — {n}-class  (accuracy = {acc*100:.1f}%)",
            os.path.join(GRAPHS, f"confusion_{tag}.png")
        )
        print(f"  Plot → {GRAPHS}/confusion_{tag}.png")

        # per-class bars
        _draw_per_class_bars(
            result["metrics"], labels,
            f"Precision / Recall / F1 per class — {n}-class",
            os.path.join(GRAPHS, f"metrics_{tag}.png")
        )
        print(f"  Plot → {GRAPHS}/metrics_{tag}.png")

    return results_store


def _save_csv(result, path, refs):
    names = result["labels"]
    pred_names = result.get("pred_labels", names)
    conf  = result["confusion"]
    m     = result["metrics"]
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
            n_tr = ref_by_name[name].n_images if name in ref_by_name else ""
            w.writerow([name, n_tr,
                        f"{d['precision']:.4f}", f"{d['recall']:.4f}",
                        f"{d['f1']:.4f}", d["support"]])
        w.writerow(["overall_accuracy", "", f"{m['overall_accuracy']:.4f}", "", "", ""])
        w.writerow([])
        w.writerow(["=== CORRECT CLASSIFICATIONS ==="])
        w.writerow(["file", "true_class"])
        for pf, tn in result["correct_files"]:
            w.writerow([pf, tn])
        w.writerow([])
        w.writerow(["=== WRONG / REJECTED CLASSIFICATIONS ==="])
        w.writerow(["file", "true_class", "predicted_class"])
        for entry in result["wrong_files"]:
            w.writerow(list(entry) if len(entry) == 3 else [entry[0], entry[1], "background"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. ACCURACY COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════════════

def draw_accuracy_comparison(results_store):
    tags   = ["3fruit", "5fruit", "10fruit"]
    labels = ["3-class", "5-class", "10-class"]
    accs   = [results_store[t][2] * 100 for t in tags]
    colors = ["#4393C3", "#2166AC", "#053061"]

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
        ax.set_facecolor("white")
        bars = ax.bar(labels, accs, color=colors, width=0.55, zorder=3)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Overall Accuracy (%)", fontsize=13)
        ax.set_title("Classification Accuracy  —  3 / 5 / 10 Fruits\n"
                     "(Fruits-360 Test folder, 60 images per class)",
                     fontsize=13, color="#111111", fontweight="bold", pad=10)
        ax.axhline(100, color="#555555", lw=0.8, ls="--", zorder=2)
        ax.grid(axis="y", zorder=0)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, f"{acc:.1f}%",
                    ha="center", va="bottom", fontsize=15,
                    fontweight="bold", color="#111111")
        plt.tight_layout()
        out = os.path.join(GRAPHS, "accuracy_comparison.png")
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"\n  Saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5. HUE DISTRIBUTION PLOT  (training-set distributions of each class)
# ══════════════════════════════════════════════════════════════════════════════

def draw_hue_distribution():
    print("\n" + "═"*70)
    print("5.  HUE DISTRIBUTION across 10 selected classes")
    print("═"*70)

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, 5, figsize=(22, 8), facecolor="white")
        fig.suptitle("HSV Hue Distribution per Class  (Fruits-360 Training set)",
                     fontsize=16, color="#111111", fontweight="bold", y=1.01)

        for ax, (folder, name, _, hex_color) in zip(axes.flat, SPEC_10):
            ax.set_facecolor("white")
            folder_path = os.path.join(TRAIN, folder)
            imgs = sorted([f for f in os.listdir(folder_path)
                           if f.lower().endswith((".jpg", ".png"))])[:30]
            all_hues = []
            for fname in imgs:
                bgr = cv2.imread(os.path.join(folder_path, fname))
                if bgr is None:
                    continue
                h_arr, s_arr, v_arr = to_hsv_float(bgr)
                valid = guard_mask(s_arr, v_arr, 0.15, 0.15)
                if valid.sum() > 0:
                    hue_deg = np.degrees(h_arr[valid])
                    all_hues.append(hue_deg)
            if all_hues:
                all_hues = np.concatenate(all_hues)
                bins = np.linspace(0, 360, 73)
                ax.hist(all_hues, bins=bins, color=hex_color, alpha=0.80,
                        edgecolor="white", linewidth=0.3, zorder=3)
            ax.set_xlim(0, 360)
            ax.set_title(name, fontsize=12, color="#111111", fontweight="bold")
            ax.set_xlabel("Hue (°)", fontsize=9)
            ax.set_ylabel("Pixel count", fontsize=9)
            ax.grid(axis="y", zorder=0, alpha=0.5)
            ax.set_xticks([0, 90, 180, 270, 360])

        plt.tight_layout()
        out = os.path.join(GRAPHS, "hue_distribution.png")
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6. REGION-COUNT CHART  (split vs merge for multiple images)
# ══════════════════════════════════════════════════════════════════════════════

def draw_region_count_chart(n_split_main, n_merged_main):
    print("\n" + "═"*70)
    print("6.  REGION COUNT  (split → merge reduction)")
    print("═"*70)

    # Run on a few images to get a spread
    test_images = [
        ("banana_1.jpg",     "Banana scene"),
        ("cherry(wax)_1.jpg","Cherry scene"),
        ("orange_1.jpg",     "Orange scene"),
    ]
    cfg = make_scene_cfg()
    splits_all, merges_all, img_labels = [], [], []

    # Use the 3-fruit refs for speed
    sp3 = _drop_hex(SPEC_3)
    refs3, nmean3, nstd3 = build_references(TRAIN, sp3, extended=True)

    for fname, label in test_images:
        p = os.path.join(MULTI, fname)
        if not os.path.exists(p):
            continue
        bgr = cv2.imread(p)
        pre = gaussian_lowpass(median_filter(bgr, 5), 1.5)
        h, s, v = to_hsv_float(pre)
        valid = guard_mask(s, v, cfg.s_min, cfg.v_min)
        edges = sobel_edges(v)
        sl = split_quadtree(h, s, valid, edges,
                             tau_h=cfg.tau_h, tau_s=cfg.tau_s, tau_e=cfg.tau_e,
                             min_size=cfg.min_size, min_start_depth=cfg.min_start_depth)
        ml = merge_regions(sl, h, s, valid, edges,
                            hue_thresh=cfg.hue_thresh, sat_thresh=cfg.sat_thresh,
                            merged_var_h=cfg.merged_var_h, merged_var_s=cfg.merged_var_s,
                            edge_veto=cfg.edge_veto)
        ns, nm = int(sl.max()) + 1, int(ml.max()) + 1
        splits_all.append(ns); merges_all.append(nm); img_labels.append(label)
        reduction = 100 * (1 - nm / ns)
        print(f"  {label:18s}  split={ns:4d}  merge={nm:4d}  reduction={reduction:.0f}%")

    if not splits_all:
        return

    x = np.arange(len(img_labels))
    bw = 0.38
    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
        ax.set_facecolor("white")
        ax.bar(x - bw/2, splits_all, bw, label="After split", color="#4393C3", zorder=3)
        ax.bar(x + bw/2, merges_all, bw, label="After merge", color="#D6604D", zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(img_labels, fontsize=12)
        ax.set_ylabel("Number of regions", fontsize=13)
        ax.set_title("Region count: Split (quadtree) vs. Merge (RAG) [oc]",
                     fontsize=13, color="#111111", fontweight="bold", pad=10)
        ax.grid(axis="y", zorder=0)
        ax.legend(fontsize=12)
        for xi, (s, m) in enumerate(zip(splits_all, merges_all)):
            ax.text(xi - bw/2, s + max(splits_all)*0.01, str(s),
                    ha="center", va="bottom", fontsize=10, color="#2166AC")
            ax.text(xi + bw/2, m + max(splits_all)*0.01, str(m),
                    ha="center", va="bottom", fontsize=10, color="#B2182B")
        plt.tight_layout()
        out = os.path.join(GRAPHS, "region_count.png")
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SCENE OVERLAY DEMOS
# ══════════════════════════════════════════════════════════════════════════════

def _refine_overlay(bgr, class_map, refs, hue_tol=0.65, search_r=14):
    """
    Rebuild the overlay by applying hue-gated BFS expansion to every detected class.

    For each class that appears in class_map:
      1. Keep only the largest connected component (drops stray false pixels).
      2. Dilate the seed by `search_r` pixels to define the spatial search window.
      3. Within that window, mark pixels whose hue is within `hue_tol` rad of the
         class reference AND saturation/value > 0.04 as eligible.
      4. BFS flood-fill from the seed into eligible 8-connected neighbours.
      5. Morphological closing (r=3) to smooth jagged quadtree edges.
      6. Keep the expanded mask without global hole-fill to avoid bowl/rim floods.
    Classes are merged in a single pass: each pixel goes to whichever class's
    seed is closest (first-come-first-served in class-index order).
    """
    from collections import deque
    from fruitseg.postprocess import (_binary_dilate, _binary_erode,
                                      _connected_components_8)
    from fruitseg.pipeline import _fill_components as _fc, _reference_hue

    h_map, s_map, v_map = to_hsv_float(bgr)
    H, W   = bgr.shape[:2]
    n      = len(refs)
    new_cm = np.full((H, W), -1, dtype=np.int32)
    overlay = bgr.copy().astype(np.float32)

    for cidx in range(n):
        raw = (class_map == cidx).astype(np.uint8)
        if not raw.any():
            continue

        # Largest CC
        lbls, areas = _connected_components_8(raw)
        if not areas:
            continue
        seed = (lbls == (int(np.argmax(areas)) + 1)).astype(np.uint8)

        # Spatial search window + hue gate
        ref_hue   = _reference_hue(refs[cidx])
        hue_dist  = np.abs(np.arctan2(np.sin(h_map - ref_hue),
                                      np.cos(h_map - ref_hue)))
        search    = _binary_dilate(seed, search_r)
        eligible  = ((search > 0)
                     & (hue_dist <= hue_tol)
                     & (s_map > 0.04)
                     & (v_map > 0.04)
                     & (new_cm == -1))     # don't steal already-claimed pixels

        # BFS from seed into eligible neighbours
        result = seed.copy()
        q = deque(map(tuple, np.argwhere(seed > 0)))
        while q:
            y, x = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < H and 0 <= nx < W
                            and result[ny, nx] == 0
                            and eligible[ny, nx]):
                        result[ny, nx] = 1
                        eligible[ny, nx] = False   # mark consumed
                        q.append((ny, nx))

        # Smooth jagged edges without global hole-fill.
        result = _binary_erode(_binary_dilate(result, 3), 3)

        sel = result > 0
        new_cm[sel] = cidx
        col = np.array(refs[cidx].color_bgr, dtype=np.float32)
        overlay[sel] = 0.35 * overlay[sel] + 0.65 * col

    return new_cm, np.clip(overlay, 0, 255).astype(np.uint8)


def _smooth_class_overlay(bgr, class_map, refs):
    """
    Build a clean overlay for multi-fruit scene images using the pipeline's
    own class_map, with light closing to smooth the blocky quadtree boundaries.
    No BFS expansion — each class stays within its detected region.
    """
    from fruitseg.postprocess import _binary_dilate, _binary_erode

    cH, cW  = class_map.shape[:2]
    if bgr.shape[:2] != (cH, cW):
        bgr = cv2.resize(bgr, (cW, cH), interpolation=cv2.INTER_AREA)
    H, W    = bgr.shape[:2]
    overlay = bgr.copy().astype(np.float32)
    # Process classes in reverse frequency order so dominant class wins ties
    class_counts = [(int((class_map == i).sum()), i) for i in range(len(refs))]
    for _, cidx in sorted(class_counts):
        mask = (class_map == cidx).astype(np.uint8)
        if not mask.any():
            continue
        # Closing smooths blocky quadtree edges without filling a whole enclosed
        # bowl/rim region as foreground.
        mask = _binary_erode(_binary_dilate(mask, 5), 5)
        col = np.array(refs[cidx].color_bgr, dtype=np.float32)
        sel = mask > 0
        overlay[sel] = 0.35 * overlay[sel] + 0.65 * col

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _save_overlay(bgr_src, res, refs, out_path, detected_only=True):
    H, W = res["overlay"].shape[:2]
    src_resized = cv2.resize(bgr_src, (W, H))

    refined_ov = _smooth_class_overlay(src_resized, res["class_map"], refs)

    if detected_only:
        shown = [ref for i, ref in enumerate(refs)
                 if (res["class_map"] == i).any()]
    else:
        shown = refs
    comp = side_by_side(src_resized, refined_ov,
                        left_title="Original", right_title="Segmented")
    comp = add_legend(comp, shown, title="Detected")
    cv2.imwrite(out_path, comp)


# ── Step 9 helper: 3-class multi-fruit scene from test-multiple_fruits ────────

def run_3class_scene_demo(refs3, nmean3, nstd3):
    """
    Assignment step 9: run the 3-class algorithm on test-multiple_fruits images
    that contain Cherry, Orange or Banana to satisfy the multi-fruit scene
    requirement for split-and-merge.
    """
    print("\n" + "═"*70)
    print("STEP 9.  3-CLASS MULTI-FRUIT SCENE  (test-multiple_fruits)")
    print("═"*70)

    scene_files = [
        ("cherry(wax)_1.jpg",        "Cherry scene"),
        ("orange_5.jpg",        "Orange scene"),
        ("banana_orange.jpg",   "Banana + Orange scene"),
    ]
    cfg = make_scene_cfg(3)

    for fname, desc in scene_files:
        p = os.path.join(MULTI, fname)
        if not os.path.exists(p):
            print(f"  SKIP {fname}")
            continue
        bgr = cv2.imread(p)
        res = segment_image(bgr, refs3, nmean3, nstd3, cfg)
        detected = [refs3[i].name for i in range(len(refs3))
                    if (res["class_map"] == i).any()]
        out_name = fname.replace(".jpg", "_3class_overlay.png")
        _save_overlay(bgr, res, refs3, os.path.join(RES, out_name))
        print(f"  {desc}: detected {detected}")
        print(f"  Saved → results/{out_name}")


def run_scene_demos(refs10, nmean10, nstd10):
    print("\n" + "═"*70)
    print("7.  SCENE OVERLAY DEMOS")
    print("═"*70)

    cfg = make_scene_cfg(10)
    sp3 = _drop_hex(SPEC_10)

    demos = [
        (os.path.join(MULTI, "cherry_17.jpg"),
         10, SPEC_10, refs10, nmean10, nstd10,
         "cherry_scene_overlay.png", "Cherry scene  (10-class)"),
        (os.path.join(MULTI, "banana_1.jpg"),
         10, SPEC_10, refs10, nmean10, nstd10,
         "banana_scene_overlay.png", "Banana scene  (10-class)"),
        (os.path.join(MULTI, "huckleberry.jpg"),
         10, SPEC_10, refs10, nmean10, nstd10,
         "huckleberry_scene_overlay.png", "Huckleberry scene  (10-class)"),
    ]

    for img_p, nf, spec, refs, nmean, nstd, out_name, desc in demos:
        if not os.path.exists(img_p):
            print(f"  SKIP  {desc}  (image not found)")
            continue
        bgr = cv2.imread(img_p)
        sc_cfg = make_scene_cfg(nf)
        res = segment_image(bgr, refs, nmean, nstd, sc_cfg)
        out_path = os.path.join(RES, out_name)
        _save_overlay(bgr, res, refs, out_path)
        classes_detected = [refs[i].name for i in range(len(refs))
                            if (res["class_map"] == i).any()]
        print(f"  {desc}: {classes_detected}")
        print(f"  Saved → results/{out_name}")

    print(f"\n  Building composite 10-class scene from Test images ...")
    tile_size = 200
    tiles = []
    for folder, name, col, _ in SPEC_10:
        cls_dir = os.path.join(TEST, folder)
        if not os.path.isdir(cls_dir):
            continue
        imgs = sorted([f for f in os.listdir(cls_dir)
                       if f.lower().endswith((".jpg", ".png"))])
        if not imgs:
            continue
        img_path = os.path.join(cls_dir, imgs[len(imgs) // 2])
        tile = cv2.imread(img_path)
        if tile is None:
            continue
        tiles.append(cv2.resize(tile, (tile_size, tile_size)))

    if tiles:
        cols, gap, bg = 5, 6, 230
        rows_list = []
        for r in range(0, len(tiles), cols):
            row = tiles[r:r + cols]
            while len(row) < cols:
                row.append(np.full((tile_size, tile_size, 3), bg, np.uint8))
            row_img = row[0]
            for t in row[1:]:
                row_img = np.hstack([row_img,
                                     np.full((tile_size, gap, 3), bg, np.uint8), t])
            rows_list.append(row_img)
        h_bar = np.full((gap, rows_list[0].shape[1], 3), bg, np.uint8)
        composite = rows_list[0]
        for ri in rows_list[1:]:
            composite = np.vstack([composite, h_bar, ri])

        comp_cfg = SegmentationConfig(extended_features=True, max_side=640)
        comp_cfg.s_min = 0.10; comp_cfg.v_min = 0.10; comp_cfg.v_max = 0.97
        comp_cfg.tau_h = 0.06; comp_cfg.tau_s = 0.025; comp_cfg.tau_e = 0.12
        comp_cfg.hue_thresh = 0.28; comp_cfg.sat_thresh = 0.14
        comp_cfg.merged_var_h = 0.07; comp_cfg.merged_var_s = 0.03
        comp_cfg.edge_veto = 0.30; comp_cfg.reject_z = 2.5
        comp_cfg.min_area = 200; comp_cfg.morph_radius = 3
        comp_cfg.refine_masks = True; comp_cfg.refine_hue_tol = 0.55
        comp_cfg.fill_components = True; comp_cfg.expand_masks = False
        comp_cfg.min_class_fraction = 0.0
        res = segment_image(composite, refs10, nmean10, nstd10, comp_cfg)
        _save_overlay(composite, res, refs10,
                      os.path.join(RES, "multi_fruit_overlay.png"))
        classes_detected = [refs10[i].name for i in range(len(refs10))
                            if (res["class_map"] == i).any()]
        print(f"  10-class composite: detected {classes_detected}")
        print(f"  Saved → results/multi_fruit_overlay.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY FIGURE  (all results in one poster)
# ══════════════════════════════════════════════════════════════════════════════

def draw_summary_poster(results_store):
    print("\n" + "═"*70)
    print("8.  SUMMARY POSTER")
    print("═"*70)

    def _load(path):
        if not os.path.exists(path):
            return np.zeros((100, 200, 3), np.uint8)
        bgr = cv2.imread(path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    with plt.rc_context(PLT_STYLE):
        fig = plt.figure(figsize=(26, 18), facecolor="white")
        fig.suptitle("HSV Split-and-Merge Fruit Segmentation  —  Complete Results\n"
                     "Deggendorf Institute of Technology  •  Computer Vision Assignment 2  •  Summer 2026",
                     fontsize=16, color="#111111", fontweight="bold", y=0.995)

        gs = fig.add_gridspec(3, 4, hspace=0.38, wspace=0.28,
                              left=0.04, right=0.97, top=0.95, bottom=0.03)

        def img_ax(row, col, colspan=1, rowspan=1):
            ax = fig.add_subplot(gs[row:row+rowspan, col:col+colspan])
            ax.set_facecolor("white")
            ax.axis("off")
            return ax

        ax = img_ax(0, 0, colspan=4)
        ax.imshow(_load(os.path.join(GRAPHS, "pipeline_steps.png")), aspect="auto")
        ax.set_title("Pipeline Steps  (Original → Preprocessed → Guard Mask → Sobel → Split → Merge → Overlay)",
                     fontsize=11, color="#111111", fontweight="bold", pad=5)

        for ci, tag in enumerate(["3fruit", "5fruit", "10fruit"]):
            ax = img_ax(1, ci)
            ax.imshow(_load(os.path.join(GRAPHS, f"confusion_{tag}.png")), aspect="auto")
            n = tag.replace("fruit","")
            acc = results_store[tag][2]
            ax.set_title(f"{n}-class Confusion Matrix  (acc = {acc*100:.1f}%)",
                         fontsize=11, color="#111111", fontweight="bold", pad=5)

        ax = img_ax(1, 3)
        ax.imshow(_load(os.path.join(GRAPHS, "accuracy_comparison.png")), aspect="auto")
        ax.set_title("Accuracy Comparison  (3 / 5 / 10 classes)",
                     fontsize=11, color="#111111", fontweight="bold", pad=5)

        for ci, (fname, title) in enumerate([
            ("cherry_scene_overlay.png",   "Cherry scene"),
            ("banana_scene_overlay.png",   "Banana scene"),
            ("metrics_10fruit.png",        "10-class Precision / Recall / F1"),
        ]):
            ax = img_ax(2, ci)
            ax.imshow(_load(os.path.join(RES if "overlay" in fname else GRAPHS, fname)),
                      aspect="auto")
            ax.set_title(title, fontsize=11, color="#111111", fontweight="bold", pad=5)

        out = os.path.join(RES, "summary_poster.png")
        fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 9a. FRUITS-262  3-CLASS PROOF-OF-CONCEPT  (assignment step 11)
# ══════════════════════════════════════════════════════════════════════════════

def run_fruits262_3class_proof():
    """
    Assignment step 11: test the 3-class algorithm on Fruits-262 and save
    a proof-of-concept image.  We stitch the best Cherry / Orange / Banana
    images from Fruits-262 side-by-side to create a single multi-fruit scene,
    then run the 3-class pipeline on each individually and render the overlay
    side-by-side so the colour separation is clearly visible.
    """
    print("\n" + "═"*70)
    print("9a. FRUITS-262  3-CLASS PROOF-OF-CONCEPT  (assignment step 11)")
    print("═"*70)

    # 3-class spec — same ordering as SPEC_3 so refs index matches
    # Use 5-class normalization context for consistency with assignment spec
    spec5 = _drop_hex(SPEC_5)
    refs_full, nm3, ns3 = build_references(TRAIN, spec5, extended=True)
    refs3 = refs_full[:3]  # Take only first 3 classes

    cfg = make_scene_cfg(3)
    cfg.max_side = 480
    cfg.s_min = 0.12; cfg.v_min = 0.12
    cfg.hue_thresh = 0.20; cfg.sat_thresh = 0.08
    # Orange-Banana hue gap is ~0.33 rad.  Tolerances must stay strictly below
    # that so the two classes never share hue-compatible pixels after
    # refinement / expansion.
    cfg.refine_hue_tol = 0.22
    cfg.expand_hue_tol = 0.20
    # Suppress any class that covers less than 12 % of all labelled pixels.
    # In a single-fruit image one class dominates (~80 %+); false-positive
    # patches of a different class are always <12 % and get zeroed out.
    # In the three-fruit composite each class covers ~25-35 %, so all three
    # survive the threshold.
    cfg.min_class_fraction = 0.12
    # Amplify hue dimensions so orange (27 deg) stays firmly separated from
    # banana (46 deg) even when natural-light photos shift the measured hue
    # toward the golden-yellow range.
    cfg.feature_weights = (2.5, 2.5, 0.7, 0.4, 0.4, 0.9, 0.0)

    candidates = [
        ("cherry", "1.jpg", 0, "Cherry"),
        ("orange", "1.jpg", 1, "Orange"),
        ("banana", "1.jpg", 2, "Banana"),
    ]

    target_h = 300
    orig_panels, ov_panels, labels, colors = [], [], [], []

    for folder, fname, cidx, name in candidates:
        folder_path = os.path.join(F262, folder)
        if not os.path.isdir(folder_path):
            print(f"  SKIP {name} — folder not found")
            continue
        p = os.path.join(folder_path, fname)
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"  SKIP {name} — image not found ({fname})")
            continue
        scale = target_h / bgr.shape[0]
        bgr = cv2.resize(bgr, (int(bgr.shape[1] * scale), target_h))
        chosen_file = fname
        res = segment_image(bgr, refs3, nm3, ns3, cfg)
        pred_idx = _dominant_idx(res["class_map"], len(refs3))
        pred_name = refs3[pred_idx].name if pred_idx >= 0 else "background"
        ov = _smooth_class_overlay(bgr, res["class_map"], refs3)
        ok = name.lower() == pred_name.lower()
        print(f"  {name:8s}  predicted: {pred_name:8s}  {'✓' if ok else '✗'}  ({chosen_file})")

        orig_panels.append(bgr)
        ov_panels.append(ov)
        labels.append((name, pred_name, ok))
        colors.append(tuple(int(c) for c in refs3[cidx].color_bgr))

    if not orig_panels:
        print("  No images found — skipping.")
        return

    # ── matplotlib figure: 2 rows × N cols ────────────────────────────────────
    nc = len(orig_panels)
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, nc, figsize=(nc * 3.5, 8), facecolor="white")
        if nc == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        fig.suptitle(
            "Fruits-262  —  3-Class Proof-of-Concept  (Cherry / Orange / Banana)\n"
            "Classifier trained on Fruits-360 only — tested on real-world Fruits-262 images",
            fontsize=13, color="#111111", fontweight="bold", y=1.01)

        hex_map = {"Cherry": "#B22222", "Orange": "#E07000", "Banana": "#C8A000"}

        for col, (orig, ov, (true_name, pred_name, ok)) in enumerate(
                zip(orig_panels, ov_panels, labels)):
            ax_top = axes[0, col]
            ax_bot = axes[1, col]

            ax_top.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
            ax_top.set_title(true_name, fontsize=13,
                             color=hex_map.get(true_name, "#333333"),
                             fontweight="bold", pad=5)
            ax_top.axis("off"); ax_top.set_facecolor("white")

            ax_bot.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
            c = "#1a7a1a" if ok else "#cc2200"
            ax_bot.set_xlabel(f"{'✓' if ok else '✗'} Predicted: {pred_name}",
                              fontsize=11, color=c, fontweight="bold", labelpad=5)
            ax_bot.set_xticks([]); ax_bot.set_yticks([])
            ax_bot.set_facecolor("white")
            for sp in ax_bot.spines.values():
                sp.set_visible(False)

        axes[0, 0].set_ylabel("Original", fontsize=11, color="#333333")
        axes[1, 0].set_ylabel("Overlay",  fontsize=11, color="#333333")

        # Legend patches
        patches = [mpatches.Patch(color=hex_map[n], label=n)
               for n in ["Cherry", "Orange", "Banana"]]
        fig.legend(handles=patches, loc="lower center", ncol=3,
                   fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        out = os.path.join(RES, "fruits262_3class_proof.png")
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved → {out}")

    # ── Composite scene: stitch 3 F262 images into one scene image and run
    #    the 3-class pipeline on it — this produces the "similar to the above
    #    figure" result required by step 11.
    if len(orig_panels) == 3:
        # Keep the composite large enough that the small cherry panel survives
        # the second resize used to build the stitched scene.
        target_h2 = 400
        panels_resized = []
        for bgr_p in orig_panels:
            sc = target_h2 / bgr_p.shape[0]
            panels_resized.append(
                cv2.resize(bgr_p, (int(bgr_p.shape[1]*sc), target_h2)))
        # Pad all to same height, then stack horizontally with a thin divider
        gap = np.full((target_h2, 6, 3), 200, dtype=np.uint8)
        scene = np.hstack([panels_resized[0], gap,
                           panels_resized[1], gap,
                           panels_resized[2]])

        scene_res = segment_image(scene, refs3, nm3, ns3, cfg)
        scene_ov  = _smooth_class_overlay(scene, scene_res["class_map"], refs3)

        detected = [refs3[i].name for i in range(len(refs3))
                    if (scene_res["class_map"] == i).any()]
        print(f"  Composite scene detected: {detected}")

        comp = side_by_side(scene, scene_ov,
                            left_title="Fruits-262 composite (Original)",
                            right_title="3-class segmentation overlay")
        shown_refs = [refs3[i] for i in range(len(refs3))
                      if (scene_res["class_map"] == i).any()]
        comp = add_legend(comp, shown_refs, title="Detected")
        scene_out = os.path.join(RES, "fruits262_3class_scene.png")
        cv2.imwrite(scene_out, comp)
        print(f"  Composite scene → {scene_out}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. FRUITS-262 CROSS-DATASET DEMO  (different source, same trained refs)
# ══════════════════════════════════════════════════════════════════════════════

# Fruits-262 folder name → (display label, title hex, overlay BGR)
# Overlay BGR colours are vivid and maximally distinct from each other so
# every class is immediately recognisable as a different shade in the demo.
F262_CLASSES = [
    ("orange",    "Orange",       "#E07000", (0,  130, 255)),   # vivid orange
    ("banana",    "Banana",       "#C8A000", (0,  230, 255)),   # vivid yellow
    ("avocado",   "Avocado",      "#1A7A1A", (0,  200,   0)),   # vivid green
    ("raspberry", "Raspberry",    "#7B00CC", (200, 0,  200)),   # vivid magenta
    ("lychee",    "Lychee",       "#CC4488", (255, 80, 180)),   # vivid pink
    ("cherry",    "Cherry",       "#B22222", (0,   0, 180)),     # vivid red
]


def _make_f262_cfg():
    """Config for Fruits-262 natural photos: relaxed thresholds for better detection."""
    cfg = make_scene_cfg(10)
    cfg.max_side = 480
    cfg.s_min = 0.12; cfg.v_min = 0.12; cfg.v_max = 1.0
    cfg.tau_e = 0.10; cfg.hue_thresh = 0.22; cfg.sat_thresh = 0.10
    cfg.merged_var_h = 0.08; cfg.merged_var_s = 0.03
    cfg.refine_hue_tol = 0.40
    cfg.expand_hue_tol = 0.28
    cfg.min_class_fraction = 0.0
    return cfg


def _dominant_idx(class_map, n):
    """Index of the most-frequent non-background class (-1 if none)."""
    if (class_map >= 0).any():
        counts = np.bincount(class_map[class_map >= 0].ravel(), minlength=n)
        return int(np.argmax(counts))
    return -1


def _filled_class_overlay(bgr, class_map, cidx, color_bgr, ref_hue,
                          h_map, s_map, v_map):
    """
    Build a fruit-shaped overlay using hue-gated BFS expansion from seed pixels.

    Steps:
    1. Keep the largest connected component of the classified mask (the seed).
    2. Build a spatial search region by dilating the seed by 12 px — this
       defines how far the fill is allowed to reach from the original detection.
    3. Mark every pixel in the search region whose hue is within 0.70 rad of
       the class reference hue AND saturation > 0.04 as "compatible".
       (Low S/V threshold so highlights and shadows are included; the hue
       constraint stops the fill from bleeding into differently-coloured
       background or neighbouring fruits.)
    4. BFS-flood from seed pixels expanding only into compatible neighbours,
       keeping 8-connectivity — prevents jumping across gaps to unrelated regions.
    5. Morphological closing (r=3) to smooth jagged edges.
    6. Blend at 60 % opacity.
    """
    from collections import deque
    from fruitseg.postprocess import (_binary_dilate, _binary_erode,
                                      _connected_components_8)

    mask = (class_map == cidx).astype(np.uint8)
    if not mask.any():
        return bgr.copy()

    # Step 1: largest connected component
    lbls, areas = _connected_components_8(mask)
    if areas:
        biggest = int(np.argmax(areas)) + 1
        seed = (lbls == biggest).astype(np.uint8)
    else:
        seed = mask

    search = _binary_dilate(seed, 10)
    hue_dist   = np.abs(np.arctan2(np.sin(h_map - ref_hue),
                                   np.cos(h_map - ref_hue)))
    compatible = (search > 0) & (hue_dist <= 0.65) & (s_map > 0.06) & (v_map > 0.06)

    H, W   = seed.shape
    result = seed.copy()
    q      = deque(map(tuple, np.argwhere(seed > 0)))
    while q:
        y, x = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (0 <= ny < H and 0 <= nx < W
                        and result[ny, nx] == 0
                        and compatible[ny, nx]):
                    result[ny, nx] = 1
                    compatible[ny, nx] = False
                    q.append((ny, nx))

    result = _binary_erode(_binary_dilate(result, 3), 3)

    out = bgr.copy().astype(np.float32)
    col = np.array(color_bgr, dtype=np.float32)
    out[result > 0] = 0.30 * out[result > 0] + 0.70 * col
    return np.clip(out, 0, 255).astype(np.uint8)


def run_fruits262_demo(refs10, nmean10, nstd10):
    print("\n" + "═"*70)
    print("9.  FRUITS-262 CROSS-DATASET DEMO")
    print("═"*70)

    cfg = _make_f262_cfg()
    n_refs = len(refs10)
    results = []   # (label, hex_col, orig_rgb, overlay_rgb, pred_name)

    for folder, label, hex_col, overlay_bgr_col in F262_CLASSES:
        folder_path = os.path.join(F262, folder)
        if not os.path.isdir(folder_path):
            print(f"  SKIP {label} — folder not found")
            continue
        all_imgs = [f for f in os.listdir(folder_path)
                    if f.lower().endswith((".jpg", ".png"))]
        imgs = sorted(all_imgs, key=lambda f: int(os.path.splitext(f)[0])
                      if os.path.splitext(f)[0].isdigit() else 0)
        if not imgs:
            print(f"  SKIP {label} — no images found")
            continue

        # Choose a visually strong natural photo without peeking at the label
        # outcome. This keeps the demo deterministic and avoids selecting
        # samples only because the current classifier already got them right.
        bgr = None
        pick = imgs[0]
        best_score = -1.0

        for candidate in imgs[:15]:
            tmp = cv2.imread(os.path.join(folder_path, candidate))
            if tmp is None:
                continue
            _h_c, s_c, v_c = to_hsv_float(tmp)
            valid = (s_c >= cfg.s_min) & (v_c >= cfg.v_min)
            score = float(valid.sum()) * float(s_c[valid].mean() if valid.any() else 0.0)
            if score > best_score:
                best_score = score
                bgr, pick = tmp, candidate
        if bgr is None:
            print(f"  SKIP {label} — no readable images")
            continue
        res = segment_image(bgr, refs10, nmean10, nstd10, cfg)
        h_map, s_map, v_map = to_hsv_float(bgr)

        cm = res["class_map"]
        didx = _dominant_idx(cm, n_refs)
        pred_str = refs10[didx].name if didx >= 0 else "background"
        is_correct = (pred_str and (label.lower() in pred_str.lower()
                                    or pred_str.lower() in label.lower()))
        tick = "✓" if is_correct else "✗"
        print(f"  {label:14s}  predicted: {pred_str:14s} {tick}  ({pick})")

        # Hue-gated BFS fill using the class's vivid distinct overlay colour.
        if didx >= 0:
            from fruitseg.pipeline import _reference_hue
            ref_hue_val = _reference_hue(refs10[didx])
            overlay_bgr = _filled_class_overlay(bgr, cm, didx,
                                                overlay_bgr_col,
                                                ref_hue_val,
                                                h_map, s_map, v_map)
        else:
            overlay_bgr = bgr.copy()

        orig_rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        results.append((label, hex_col, orig_rgb, overlay_rgb, pred_str, is_correct))

    if not results:
        print("  No Fruits-262 images found — skipping demo.")
        return

    # ── 2-row grid: top = originals, bottom = dominant-class overlays ─────────
    ncols = len(results)
    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, ncols,
                                 figsize=(ncols * 3.0, 7),
                                 facecolor="white")
        if ncols == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        fig.suptitle("Fruits-262  —  Cross-Dataset Demo  (classifier trained on Fruits-360 only)",
                     fontsize=14, color="#111111", fontweight="bold", y=1.01)

        for col, (label, hex_col, orig, overlay, pred, ok) in enumerate(results):
            ax_top = axes[0, col]
            ax_bot = axes[1, col]

            ax_top.imshow(orig)
            ax_top.set_title(label, fontsize=11, color=hex_col,
                             fontweight="bold", pad=4)
            ax_top.axis("off"); ax_top.set_facecolor("white")

            ax_bot.imshow(overlay)
            col_txt = "#1a7a1a" if ok else "#cc2200"
            marker   = "✓" if ok else "✗"
            ax_bot.set_xlabel(f"{marker} {pred}", fontsize=10, color=col_txt,
                              fontweight="bold", labelpad=4)
            ax_bot.set_xticks([]); ax_bot.set_yticks([])
            ax_bot.set_facecolor("white")
            for spine in ax_bot.spines.values():
                spine.set_visible(False)

        axes[0, 0].set_ylabel("Original", fontsize=10, color="#333333")
        axes[1, 0].set_ylabel("Overlay",  fontsize=10, color="#333333")

        plt.tight_layout()
        out = os.path.join(RES, "fruits262_demo.png")
        fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    print("\n" + "█"*70)
    print("█  HSV SPLIT-AND-MERGE FRUIT SEGMENTATION  —  COMPLETE PROJECT RUN")
    print("█"*70)

    # ── Assignment step 1: self-test ─────────────────────────────────────────
    run_selftest()

    # ── Assignment step 2: Training folder scan ──────────────────────────────
    run_training_scan()

    # ── Assignment steps 3+4: dataset exploration (Test + F262) ─────────────
    run_dataset_exploration()

    # ── Build references (10-class, reused for most steps) ───────────────────
    print("\n  Building 10-class references …")
    sp10 = _drop_hex(SPEC_10)
    refs10, nmean10, nstd10 = build_references(TRAIN, sp10, extended=True)
    print("  " + ", ".join(f"{r.name}({r.n_images})" for r in refs10))

    sp3_refs = _drop_hex(SPEC_3)
    refs3, nmean3, nstd3 = build_references(TRAIN, sp3_refs, extended=True)

    # ── Pipeline visualisation (step 5 doc support) ───────────────────────────
    n_split, n_merged = run_pipeline_visualisation(refs10, nmean10, nstd10)

    # ── Hue distribution (step 6 doc support) ────────────────────────────────
    draw_hue_distribution()

    # ── Assignment steps 8–10: evaluation on Test folder (3 / 5 / 10 class) ──
    results_store = run_evaluation()

    # ── Accuracy comparison chart ─────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  ACCURACY COMPARISON CHART")
    print("═"*70)
    draw_accuracy_comparison(results_store)

    # ── Region count chart ────────────────────────────────────────────────────
    draw_region_count_chart(n_split, n_merged)

    # ── Assignment step 9: 3-class multi-fruit scene (test-multiple_fruits) ───
    run_3class_scene_demo(refs3, nmean3, nstd3)

    # ── Assignment step 7: 10-class scene overlay demos ──────────────────────
    run_scene_demos(refs10, nmean10, nstd10)

    # ── Summary poster ────────────────────────────────────────────────────────
    draw_summary_poster(results_store)

    # ── Assignment step 11: Fruits-262 3-class proof-of-concept ─────────────
    run_fruits262_3class_proof()

    # ── Assignment step 12: Fruits-262 full cross-dataset demo ───────────────
    run_fruits262_demo(refs10, nmean10, nstd10)

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "█"*70)
    print(f"█  ALL DONE in {elapsed:.1f}s")
    print("█"*70)
    print(f"\n  Results directory: {RES}")
    print("  Graphs:")
    for f in sorted(os.listdir(GRAPHS)):
        print(f"    graphs/{f}")
    print("  Overlays + CSVs:")
    for f in sorted(os.listdir(RES)):
        if os.path.isfile(os.path.join(RES, f)):
            print(f"    {f}")

    accs = {tag: results_store[tag][2] for tag in ["3fruit","5fruit","10fruit"]}
    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │  3-class accuracy:  {accs['3fruit']*100:5.1f}%  │")
    print(f"  │  5-class accuracy:  {accs['5fruit']*100:5.1f}%  │")
    print(f"  │ 10-class accuracy:  {accs['10fruit']*100:5.1f}%  │")
    print(f"  └─────────────────────────────┘")


if __name__ == "__main__":
    main()

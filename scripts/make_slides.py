"""!
@file scripts/make_slides.py
@brief Generate a self-contained PDF presentation with all results, pipeline
       diagrams, and limitation slides.

Produces one A4-landscape figure per slide and saves them to a single PDF.
No external dataset is required -- all accuracy numbers are hard-coded from
the evaluation CSVs (results/eval_3fruit.csv, eval_5fruit.csv, eval_10fruit.csv).

Run from the project root:
    python3 scripts/make_slides.py --out results/presentation.pdf
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# Colour palette
_BG   = "#0d1117"
_TEXT = "#e6edf3"
_BLUE = "#4fa3e0"
_GRN  = "#3fb950"
_RED  = "#f85149"
_YEL  = "#d29922"
_GREY = "#8b949e"

_FIG_W, _FIG_H = 16, 9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _new_fig(title_text="", subtitle=""):
    """Create a dark-theme figure with optional title/subtitle."""
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    fig.patch.set_facecolor(_BG)
    if title_text:
        fig.text(0.5, 0.93, title_text, ha="center", va="top",
                 fontsize=24, fontweight="bold", color=_TEXT)
    if subtitle:
        fig.text(0.5, 0.87, subtitle, ha="center", va="top",
                 fontsize=13, color=_GREY)
    return fig


def _text_block(fig, lines, x=0.05, y=0.80, dy=0.075,
                bold_first=False, fontsize=15, color=_TEXT):
    """Render a list of text lines on the figure."""
    for i, line in enumerate(lines):
        w = "bold" if (bold_first and i == 0) else "normal"
        fig.text(x, y - i * dy, line, fontsize=fontsize,
                 color=color, fontweight=w, va="top")


# ---------------------------------------------------------------------------
# slide definitions
# ---------------------------------------------------------------------------

def slide_title():
    """Title slide."""
    fig = plt.figure(figsize=(_FIG_W, _FIG_H))
    fig.patch.set_facecolor(_BG)
    fig.text(0.5, 0.60, "HSV Split-and-Merge", ha="center",
             fontsize=40, fontweight="bold", color=_BLUE)
    fig.text(0.5, 0.50, "Fruit Segmentation Pipeline", ha="center",
             fontsize=32, fontweight="bold", color=_TEXT)
    fig.text(0.5, 0.38, "Computer Vision  --  Assignment 2", ha="center",
             fontsize=18, color=_GREY)
    fig.text(0.5, 0.30, "Deggendorf Institute of Technology  --  Summer 2026",
             ha="center", fontsize=14, color=_GREY)
    return fig


def slide_outline():
    """Table of contents."""
    fig = _new_fig("Presentation Outline")
    items = [
        "1.  Motivation and problem statement",
        "2.  Dataset and class selection (3 -> 5 -> 10)",
        "3.  Pipeline overview",
        "4.  Step-by-step algorithm walk-through",
        "5.  Quantitative results",
        "6.  Confusion matrix analysis",
        "7.  Qualitative scene demos",
        "8.  Limitations and drawbacks",
        "9.  Conclusion",
    ]
    _text_block(fig, items, x=0.18, y=0.80, dy=0.075, fontsize=16)
    return fig


def slide_motivation():
    """Why HSV? Colour-space motivation."""
    fig = _new_fig("Why HSV?",
                   subtitle="Separating chromatic content from illumination")

    fig.text(0.05, 0.78, "RGB", fontsize=20, fontweight="bold", color=_RED)
    rgb_lines = [
        "- R, G, B values change with illumination",
        "- Same hue -> different (R, G, B) under different lighting",
        "- No single channel captures 'colour' cleanly",
    ]
    _text_block(fig, rgb_lines, x=0.05, y=0.71, dy=0.07, fontsize=14)

    fig.text(0.52, 0.78, "HSV", fontsize=20, fontweight="bold", color=_GRN)
    hsv_lines = [
        "- Hue (H) -- what colour?  (illumination-tolerant)",
        "- Saturation (S) -- how vivid?",
        "- Value (V) -- how bright?",
        "- H is stable across many lighting changes",
    ]
    _text_block(fig, hsv_lines, x=0.52, y=0.71, dy=0.07, fontsize=14)

    fig.text(0.5, 0.24,
             "Guard mask excludes achromatic pixels (dark / grey / white) "
             "where hue is undefined",
             ha="center", fontsize=14, color=_YEL)
    fig.text(0.5, 0.15,
             "valid(x,y) = [V >= V_min]  AND  [S >= S_min]",
             ha="center", fontsize=15, color=_BLUE,
             fontfamily="monospace")
    return fig


def slide_circular_stats():
    """Why circular statistics for hue."""
    fig = _new_fig("Circular Statistics for Hue",
                   subtitle="Hue is an angle -- arithmetic mean fails at 0/360 degrees")

    theta = np.linspace(0, 2 * np.pi, 360)
    ax = fig.add_axes([0.04, 0.10, 0.35, 0.72], polar=True)
    ax.set_facecolor(_BG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(colors=_GREY, labelsize=8)
    colours = [matplotlib.colors.hsv_to_rgb((t / (2 * np.pi), 0.85, 0.90))
               for t in theta]
    for i in range(len(theta) - 1):
        ax.plot(theta[i:i+2], [1, 1], color=colours[i], linewidth=3)
    ax.set_yticks([])
    ax.set_title("Hue wheel", color=_TEXT, fontsize=11, pad=10)

    # Two hues near 0/360 boundary
    h1, h2 = np.deg2rad(350), np.deg2rad(10)
    ax.annotate("", xy=(h1, 0.92), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=_BLUE, lw=2))
    ax.annotate("", xy=(h2, 0.92), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=_GRN, lw=2))

    lines = [
        "Problem: mean(350, 10) = 180 degrees  -- WRONG!",
        "",
        "Fix -- circular mean:",
        "  map hue -> unit vector on circle",
        "  mean_H = atan2(mean sin h, mean cos h)",
        "",
        "Circular variance:",
        "  R = ||mean(cos h, sin h)||",
        "  circ_var = 1 - R  in [0, 1]",
        "",
        "  circ_var near 0  ->  tight hue cluster",
        "  circ_var near 1  ->  spread hue (inhomogeneous)",
    ]
    _text_block(fig, lines, x=0.44, y=0.82, dy=0.063, fontsize=14)
    return fig


def slide_class_selection():
    """Class selection -- colour wheel visualisation."""
    fig = _new_fig("Class Selection: 10 Fruits on the Colour Wheel",
                   subtitle="Maximum hue spread  |  S >= 0.40  |  avoid hue collisions")

    classes = [
        ("Cherry",       358.1, (0.90, 0.65, 0.60)),
        ("Orange",        26.7, (1.00, 0.60, 0.00)),
        ("Banana",        45.6, (1.00, 0.95, 0.20)),
        ("Avocado",       70.7, (0.40, 0.80, 0.10)),
        ("Cucumber",     103.8, (0.45, 0.90, 0.20)),
        ("Cucumber 3",   186.7, (0.20, 0.90, 0.90)),
        ("Huckleberry",  220.8, (0.20, 0.40, 1.00)),
        ("Raspberry",    274.1, (0.70, 0.20, 0.85)),
        ("Cherry Black", 313.6, (0.50, 0.10, 0.50)),
        ("Lychee",         3.4, (1.00, 0.80, 0.80)),
    ]

    ax = fig.add_axes([0.12, 0.08, 0.48, 0.76], polar=True)
    ax.set_facecolor(_BG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    ax.tick_params(colors=_GREY, labelsize=8)

    theta = np.linspace(0, 2 * np.pi, 360)
    ring_colours = [matplotlib.colors.hsv_to_rgb((t / (2 * np.pi), 0.75, 0.88))
                    for t in theta]
    for i in range(len(theta) - 1):
        ax.plot(theta[i:i+2], [0.7, 0.7], color=ring_colours[i], linewidth=6)

    for name, deg, col in classes:
        rad = np.deg2rad(deg)
        ax.plot(rad, 0.7, "o", color=col, ms=14, zorder=5,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(name, xy=(rad, 0.7), xytext=(rad, 1.08),
                    ha="center", va="center", fontsize=8, color=_TEXT,
                    arrowprops=dict(arrowstyle="-", color=_GREY, lw=0.8))

    fig.text(0.64, 0.72, "3-class:  Cherry, Orange, Banana",
             color=_GRN, fontsize=13)
    fig.text(0.64, 0.63, "5-class:  + Avocado, Cucumber",
             color=_YEL, fontsize=13)
    fig.text(0.64, 0.54, "10-class: all 10 listed",
             color=_BLUE, fontsize=13)
    fig.text(0.64, 0.38, "Rejected classes:", color=_GREY, fontsize=12)
    fig.text(0.64, 0.30, "- Apple Red <-> Cherry (hue diff < 8 degrees)",
             color=_GREY, fontsize=11)
    fig.text(0.64, 0.23, "- Pear <-> Banana (hue diff < 12 degrees)",
             color=_GREY, fontsize=11)
    fig.text(0.64, 0.16, "- Watermelon <-> Cucumber (hue diff < 10 degrees)",
             color=_GREY, fontsize=11)
    return fig


def slide_pipeline_overview():
    """Box-and-arrow pipeline diagram."""
    fig = _new_fig("Pipeline Overview")

    stages = [
        ("BGR Image", _GREY),
        ("Median + Gaussian\nPreprocessing", _BLUE),
        ("HSV Conversion\n+ Guard Mask", _BLUE),
        ("Sobel\nEdge Map", _BLUE),
        ("Quadtree\nSPLIT", _GRN),
        ("RAG\nMERGE", _GRN),
        ("Feature\nExtraction", _YEL),
        ("Z-scored NN\nClassification", _YEL),
        ("Morphological\nCleanup", _RED),
        ("Colour\nOverlay", _GREY),
    ]

    n = len(stages)
    box_w = 0.077
    box_h = 0.14
    gap = (1.0 - 0.04 - n * box_w) / (n - 1)
    y_box = 0.38

    for i, (label, col) in enumerate(stages):
        x0 = 0.02 + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x0, y_box), box_w, box_h,
            boxstyle="round,pad=0.01",
            facecolor=col + "33", edgecolor=col, linewidth=1.8,
            transform=fig.transFigure, figure=fig, clip_on=False)
        fig.add_artist(rect)
        fig.text(x0 + box_w / 2, y_box + box_h / 2, label,
                 ha="center", va="center", fontsize=8.5,
                 color=_TEXT, fontweight="bold")
        if i < n - 1:
            fig.text(x0 + box_w + gap / 2, y_box + box_h / 2 + 0.005,
                     "->", ha="center", va="center", fontsize=14, color=_GREY)

    annotations = [
        (0.02 + 0 * (box_w + gap) + box_w / 2, "Input"),
        (0.02 + 1 * (box_w + gap) + box_w / 2, "(oc) NumPy\nstride tricks"),
        (0.02 + 2 * (box_w + gap) + box_w / 2, "(wl) cv2\nguard (oc)"),
        (0.02 + 3 * (box_w + gap) + box_w / 2, "(oc) own code"),
        (0.02 + 4 * (box_w + gap) + box_w / 2, "(oc) >=16 start\nregions"),
        (0.02 + 5 * (box_w + gap) + box_w / 2, "(oc) priority\nqueue + edge veto"),
        (0.02 + 6 * (box_w + gap) + box_w / 2, "(oc) 7-D vector"),
        (0.02 + 7 * (box_w + gap) + box_w / 2, "(oc) z-score +\nrejection"),
        (0.02 + 8 * (box_w + gap) + box_w / 2, "(oc) disk SE\n+ area filter"),
        (0.02 + 9 * (box_w + gap) + box_w / 2, "Output"),
    ]
    for x, txt in annotations:
        fig.text(x, y_box - 0.09, txt, ha="center", va="top",
                 fontsize=8, color=_GREY)

    fig.text(0.5, 0.18, "(oc) = own code (NumPy only)    (wl) = library call allowed",
             ha="center", fontsize=12, color=_GREY)
    return fig


def slide_split():
    """Quadtree split explanation."""
    fig = _new_fig("Step 1 -- Quadtree Split",
                   subtitle="Recursive subdivision  |  forced depth >= 2  ->  >= 16 start regions")

    ax = fig.add_axes([0.04, 0.10, 0.42, 0.72])
    ax.set_facecolor(_BG)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Quadtree decomposition (schematic)", color=_TEXT, fontsize=10)

    def draw_rect(x0, y0, x1, y1, col=_GREY, lw=1.0):
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                     fill=False, edgecolor=col, linewidth=lw))

    # Depth-1 split
    draw_rect(0, 0, 4, 4, _BLUE, 2.0)
    draw_rect(0, 2, 2, 4, _BLUE, 1.8)
    draw_rect(2, 2, 4, 4, _BLUE, 1.8)
    draw_rect(0, 0, 2, 2, _BLUE, 1.8)
    draw_rect(2, 0, 4, 2, _BLUE, 1.8)

    # Depth-2 split on top-left quad (inhomogeneous)
    draw_rect(0, 2, 1, 3, _GRN, 1.4)
    draw_rect(1, 2, 2, 3, _GRN, 1.4)
    draw_rect(0, 3, 1, 4, _GRN, 1.4)
    draw_rect(1, 3, 2, 4, _GRN, 1.4)

    # Depth-3 on one sub-quad
    draw_rect(0, 3, 0.5, 3.5, _YEL, 1.0)
    draw_rect(0.5, 3, 1, 3.5, _YEL, 1.0)
    draw_rect(0, 3.5, 0.5, 4, _YEL, 1.0)
    draw_rect(0.5, 3.5, 1, 4, _YEL, 1.0)

    ax.text(2.0, 3.0, "homogeneous\n(leaf)", ha="center", va="center",
            color=_TEXT, fontsize=8)
    ax.text(1.0, 1.0, "homogeneous\n(leaf)", ha="center", va="center",
            color=_TEXT, fontsize=8)
    ax.text(3.0, 1.0, "homogeneous\n(leaf)", ha="center", va="center",
            color=_TEXT, fontsize=8)

    lines = [
        "Homogeneity test (ALL must hold):",
        "  1. circ_var(H) <= tau_H",
        "  2. var(S)       <= tau_S",
        "  3. mean Sobel   <= tau_E  <- edge integration",
        "",
        "Split if ANY criterion violated.",
        "",
        "Forced depth 2:",
        "  -> always at least 16 leaf regions",
        "  (assignment requirement satisfied)",
        "",
        "Min block size: 8 x 8 pixels",
        "Achromatic blocks (few valid pixels)",
        "  declared background immediately",
    ]
    _text_block(fig, lines, x=0.50, y=0.82, dy=0.063, fontsize=14)
    return fig


def slide_merge():
    """RAG merge explanation."""
    fig = _new_fig("Step 2 -- Region Adjacency Graph Merge",
                   subtitle="Greedy most-similar-first  |  edge-aware veto")

    lines = [
        "After split:  ~300-500 small blocks per image",
        "",
        "Build RAG:  connect every pair of adjacent leaf blocks",
        "",
        "Priority queue (min-heap) keyed by hue distance:",
        "  pop most-similar pair (r1, r2)",
        "  |",
        "  check merge criteria:",
        "    - circ-dist(mean_H1, mean_H2) <= hue_thresh",
        "    - |mean_S1 - mean_S2|         <= sat_thresh",
        "    - combined circ_var(H)         <= merged_var_h",
        "    - combined var(S)              <= merged_var_s",
        "    - mean Sobel on shared edge    <= edge_veto",
        "  |",
        "  merge (or skip) -> update adjacency -> push new pairs",
        "",
        "After merge:  ~10-40 regions per image  (~90% reduction)",
    ]
    _text_block(fig, lines, x=0.07, y=0.82, dy=0.058, fontsize=13)

    fig.text(0.5, 0.07,
             "Edge veto prevents two same-hue adjacent fruits (e.g. two oranges) "
             "from merging across a real object boundary.",
             ha="center", fontsize=12, color=_YEL)
    return fig


def slide_features_classifier():
    """Feature vector and classifier."""
    fig = _new_fig("Feature Extraction & Classification",
                   subtitle="7-D z-scored weighted nearest-neighbour with rejection")

    fig.text(0.05, 0.79, "Feature vector (extended, 7-D):", fontsize=15,
             fontweight="bold", color=_BLUE)
    fv = [
        "[  cos(mu_H),  sin(mu_H),    <- hue encoded as unit vector",
        "   mu_S,   circ_var(H),      <- saturation mean + hue spread",
        "   var(S),   mu_V,  0.0  ]   <- var_V weight = 0 (noise at 10 classes)",
    ]
    _text_block(fig, fv, x=0.07, y=0.72, dy=0.065, fontsize=13,
                color="#b0d8ff")

    fig.text(0.05, 0.50, "Classification:", fontsize=15,
             fontweight="bold", color=_GRN)
    cl = [
        "1. Build per-class reference = mean feature over <=200 Training images",
        "2. Compute global z-score (mu, sigma) across all training images",
        "3. For each region:  z = (feat - mu) / sigma",
        "4. d_i = sum_k  w_k * (z_k - ref_i_k)^2   (weighted Euclidean)",
        "5. class = argmin d_i",
        "6. If d_min > reject_z  ->  label as background (-1)",
        "7. Adaptive margin: accept slightly-over-threshold region if gap to",
        "   2nd-nearest class is large",
    ]
    _text_block(fig, cl, x=0.07, y=0.44, dy=0.058, fontsize=13)

    fig.text(0.05, 0.06,
             "weights = [1.2, 1.2, 0.7, 0.4, 0.4, 0.9, 0.0]",
             fontsize=13, color=_YEL, fontfamily="monospace")
    return fig


def slide_results_bar():
    """Accuracy bar chart -- 3 / 5 / 10 class."""
    fig = _new_fig("Quantitative Results -- Fruits-360 Test Set")

    ax = fig.add_axes([0.10, 0.18, 0.38, 0.62])
    ax.set_facecolor(_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GREY)
    ax.tick_params(colors=_TEXT)

    configs = ["3-class", "5-class", "10-class"]
    accuracies = [100.0, 100.0, 99.32]
    errors = [0, 0, 4]
    colours = [_GRN, _GRN, _BLUE]

    bars = ax.bar(configs, accuracies, color=colours, edgecolor=_GREY, linewidth=0.8,
                  width=0.5)
    ax.set_ylim(97, 100.5)
    ax.set_ylabel("Accuracy (%)", color=_TEXT, fontsize=12)
    ax.set_title("Overall classification accuracy", color=_TEXT, fontsize=11)
    for bar, acc, err in zip(bars, accuracies, errors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{acc:.2f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=_TEXT)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.20,
                f"{err} error{'s' if err != 1 else ''}",
                ha="center", va="top", fontsize=10, color=_GREY)
    ax.yaxis.label.set_color(_TEXT)

    # Table of per-class F1 for 10-class
    col_labels = ["Class", "Prec", "Rec", "F1"]
    rows = [
        ["Cherry",       "1.000", "1.000", "1.000"],
        ["Orange",       "1.000", "1.000", "1.000"],
        ["Banana",       "1.000", "1.000", "1.000"],
        ["Avocado",      "1.000", "0.983", "0.992"],
        ["Cucumber",     "0.943", "1.000", "0.971"],
        ["Cherry Black", "1.000", "1.000", "1.000"],
        ["Cucumber 3",   "1.000", "0.950", "0.974"],
        ["Huckleberry",  "0.984", "1.000", "0.992"],
        ["Raspberry",    "1.000", "1.000", "1.000"],
        ["Lychee",       "1.000", "1.000", "1.000"],
    ]
    ax2 = fig.add_axes([0.56, 0.12, 0.40, 0.74])
    ax2.axis("off")
    ax2.set_title("10-class per-class metrics", color=_TEXT,
                  fontsize=12, pad=6)
    tbl = ax2.table(cellText=rows, colLabels=col_labels, loc="center",
                    cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(_GREY)
        if r == 0:
            cell.set_facecolor(_BLUE + "55")
            cell.set_text_props(color=_TEXT, fontweight="bold")
        else:
            cell.set_facecolor(_BG)
            f1_val = rows[r - 1][3]
            cell.set_text_props(
                color=_GRN if float(f1_val) >= 0.999 else _YEL)
    return fig


def slide_confusion():
    """10-class confusion matrix heat-map."""
    fig = _new_fig("10-class Confusion Matrix (Fruits-360 Test)")

    names_short = ["Cherry", "Orange", "Banana", "Avocado", "Cucumber",
                   "Ch.Black", "Cuc.3", "Huckle.", "Rasp.", "Lychee", "BG"]

    # Confusion matrix from eval_10fruit.csv
    C = np.array([
        [60,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # Cherry
        [ 0, 60,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # Orange
        [ 0,  0, 60,  0,  0,  0,  0,  0,  0,  0,  0],  # Banana
        [ 0,  0,  0, 59,  1,  0,  0,  0,  0,  0,  0],  # Avocado
        [ 0,  0,  0,  0, 50,  0,  0,  0,  0,  0,  0],  # Cucumber
        [ 0,  0,  0,  0,  0, 60,  0,  0,  0,  0,  0],  # Cherry Black
        [ 0,  0,  0,  0,  2,  0, 57,  1,  0,  0,  0],  # Cucumber 3
        [ 0,  0,  0,  0,  0,  0,  0, 60,  0,  0,  0],  # Huckleberry
        [ 0,  0,  0,  0,  0,  0,  0,  0, 60,  0,  0],  # Raspberry
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0, 60,  0],  # Lychee
    ], dtype=float)

    ax = fig.add_axes([0.08, 0.10, 0.60, 0.72])
    ax.set_facecolor(_BG)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "dark_blue", [_BG, _BLUE])
    ax.imshow(C, cmap=cmap, aspect="auto", vmin=0)

    n_rows, n_cols = C.shape
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(names_short, rotation=35, ha="right",
                       fontsize=9, color=_TEXT)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(names_short[:n_rows], fontsize=9, color=_TEXT)
    ax.set_xlabel("Predicted", color=_TEXT, fontsize=11)
    ax.set_ylabel("True", color=_TEXT, fontsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GREY)

    for i in range(n_rows):
        for j in range(n_cols):
            v = int(C[i, j])
            if v == 0:
                continue
            col = _TEXT if v < 30 else _BG
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold")

    fig.text(0.72, 0.78, "Error analysis:", fontsize=14, fontweight="bold",
             color=_TEXT)
    fig.text(0.72, 0.70, "Avocado -> Cucumber: 1", fontsize=12, color=_RED)
    fig.text(0.72, 0.63, "(yellow-green hue collision)", fontsize=11, color=_GREY)
    fig.text(0.72, 0.55, "Cucumber 3 -> Cucumber: 2", fontsize=12, color=_RED)
    fig.text(0.72, 0.48, "(green-cyan overlap)", fontsize=11, color=_GREY)
    fig.text(0.72, 0.40, "Cucumber 3 -> Huckleberry: 1", fontsize=12, color=_RED)
    fig.text(0.72, 0.33, "(cyan-blue boundary)", fontsize=11, color=_GREY)
    fig.text(0.72, 0.22, "All other classes:", fontsize=12, color=_GRN)
    fig.text(0.72, 0.14, "zero errors", fontsize=14, color=_GRN, fontweight="bold")
    return fig


def slide_qualitative():
    """Qualitative / scene demo slide."""
    fig = _new_fig("Qualitative Scene Results",
                   subtitle="test-multiple_fruits  --  Fruits-262 transfer")

    panels = [
        ("Banana scene", "Banana and Orange detected\nvia HSV hue+saturation", _YEL),
        ("Cherry scene", "Cherry blob isolated;\nwhite background rejected", _RED),
        ("Fruits-262\n(cross-dataset)", "Correct class on natural\nphotographs (no retraining)", _BLUE),
    ]
    xs = [0.04, 0.36, 0.68]
    for (title, desc, col), x in zip(panels, xs):
        rect = mpatches.FancyBboxPatch(
            (x, 0.12), 0.29, 0.68,
            boxstyle="round,pad=0.01",
            facecolor=col + "15", edgecolor=col, linewidth=1.8,
            transform=fig.transFigure, figure=fig, clip_on=False)
        fig.add_artist(rect)
        fig.text(x + 0.145, 0.76, title, ha="center", fontsize=13,
                 fontweight="bold", color=col)
        fig.text(x + 0.145, 0.60, "[overlay image]", ha="center",
                 fontsize=18, color=_GREY)
        fig.text(x + 0.145, 0.24, desc, ha="center", fontsize=11,
                 color=_TEXT, multialignment="center")

    fig.text(0.5, 0.04,
             "See results/scene_*.png for actual saved overlays",
             ha="center", fontsize=11, color=_GREY)
    return fig


def slide_limitations():
    """Limitations and drawbacks."""
    fig = _new_fig("Limitations and Drawbacks")

    items = [
        ("Hue overlap",
         "Classes sharing a hue zone (Avocado/Banana, Cherry/Lychee) cannot\n"
         "be separated by mean hue alone; accuracy degrades as class count grows."),
        ("Blocky boundaries",
         "Quadtree operates on axis-aligned rectangles -> blocky edges on\n"
         "curved fruit surfaces; closing reduces but does not eliminate this."),
        ("Threshold sensitivity",
         "All tau values calibrated on white-background Fruits-360 images;\n"
         "scene transfer requires manual relaxation and risks false positives."),
        ("Achromatic/dark regions",
         "Guard mask excludes dark pixels -> dark-skinned fruits (Cherry Black)\n"
         "or heavy shadows reduce valid pixel count and risk background misclass."),
        ("Computation speed",
         "Pure NumPy BFS and quadtree recursion: a 640x480 image takes\n"
         "several seconds; vectorised/compiled code would give 10-50x speed-up."),
        ("No shape information",
         "Pipeline uses only colour statistics -> cannot distinguish a fruit\n"
         "from a same-coloured non-fruit object in the scene."),
        ("Closed-set classifier",
         "An unknown fruit with hue near a known class is mis-identified;\n"
         "open-set rejection needs a more principled distance threshold."),
    ]

    x_left = 0.05
    y_start = 0.82
    dy = 0.105
    for i, (title, detail) in enumerate(items):
        y = y_start - i * dy
        fig.text(x_left, y, f">> {title}:", fontsize=12,
                 fontweight="bold", color=_YEL, va="top")
        fig.text(x_left + 0.015, y - 0.030, detail, fontsize=10,
                 color=_TEXT, va="top")
    return fig


def slide_conclusion():
    """Conclusion slide."""
    fig = _new_fig("Conclusion")

    points = [
        "[OK]  Pure HSV split-and-merge pipeline (almost entirely own-code NumPy)",
        "[OK]  99.3% accuracy on 10 classes (Fruits-360 Test, 590 images)",
        "[OK]  100% accuracy on 3-class and 5-class subsets",
        "[OK]  Circular hue statistics + Sobel edge integration extend classical S&M",
        "[OK]  Z-scored weighted NN with rejection handles background robustly",
        "[OK]  Morphological own-code cleanup produces clean per-class masks",
        "",
        "Main limitations: hue overlap at 10 classes, blocky boundaries,",
        "threshold sensitivity on natural-environment images.",
        "",
        "Extension directions: contour-snapping post-processing, shape features,",
        "data-driven threshold optimisation, Cython/C speed-up.",
    ]
    _text_block(fig, points, x=0.10, y=0.82, dy=0.070, fontsize=15)
    return fig


def slide_references():
    """References slide."""
    fig = _new_fig("References")

    refs = [
        "[1]  A. R. Smith, 'Color gamut transform pairs,' SIGGRAPH 1978.",
        "[2]  S. L. Horowitz & T. Pavlidis, 'Picture segmentation by a tree traversal,"
        "      algorithm,' J. ACM, 1976.",
        "[3]  K. V. Mardia & P. E. Jupp, Directional Statistics. Wiley, 2000.",
        "[4]  J. W. Tukey, Exploratory Data Analysis. Addison-Wesley, 1977.",
        "[5]  J. Serra, Image Analysis and Mathematical Morphology. Academic, 1982.",
        "[6]  I. Sobel & G. Feldman, 'A 3x3 isotropic gradient operator,' SAIL, 1968.",
        "[7]  T. Pavlidis & Y.-T. Liow, 'Integrating region growing and edge"
        "      detection,' IEEE TPAMI, 1990.",
        "[8]  Y.-I. Ohta, T. Kanade & T. Sakai, 'Color information for region"
        "      segmentation,' CGIP, 1980.",
        "[9]  H. Muresan & M. Oltean, 'Fruit recognition using deep learning,'"
        "      Acta Univ. Sapientiae, 2018.  (Fruits-360 dataset)",
        "[10] M.-D. Minut & A. Iftene, 'Creating a dataset ... Fruits-262,'"
        "      SYNASC, 2021.",
    ]
    _text_block(fig, refs, x=0.05, y=0.82, dy=0.075, fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

SLIDES = [
    slide_title,
    slide_outline,
    slide_motivation,
    slide_circular_stats,
    slide_class_selection,
    slide_pipeline_overview,
    slide_split,
    slide_merge,
    slide_features_classifier,
    slide_results_bar,
    slide_confusion,
    slide_qualitative,
    slide_limitations,
    slide_conclusion,
    slide_references,
]


def main():
    ap = argparse.ArgumentParser(description="Generate presentation PDF")
    ap.add_argument("--out", default="results/presentation.pdf",
                    help="output PDF path (default: results/presentation.pdf)")
    ap.add_argument("--png-dir", metavar="DIR", default=None,
                    help="also export each slide as a PNG into DIR")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)

    print(f"Generating {len(SLIDES)} slides -> {args.out}")
    with PdfPages(args.out) as pdf:
        for i, fn in enumerate(SLIDES):
            fig = fn()
            pdf.savefig(fig, bbox_inches="tight", facecolor=_BG)
            if args.png_dir:
                p = os.path.join(args.png_dir, f"slide_{i+1:02d}_{fn.__name__}.png")
                fig.savefig(p, bbox_inches="tight", facecolor=_BG, dpi=150)
            plt.close(fig)
            print(f"  [{i+1:2d}/{len(SLIDES)}] {fn.__name__}")

    print("Done.")


if __name__ == "__main__":
    main()

"""!
@file split_merge.py
@brief The own-code heart of : quadtree SPLIT and Region-Adjacency-Graph
       MERGE, both using the homogeneity of the mean and variance of hue and
       saturation evaluated only over guard-valid (bright, saturated) pixels.

Algorithm origin:  Horowitz & Pavlidis, "Picture segmentation by a tree traversal
                   algorithm", J. ACM 23(2):368-388, 1976.
Edge integration:  Pavlidis & Liow, "Integrating region growing and edge
                   detection", IEEE TPAMI 12(3):225-233, 1990.

Only basic numpy array operations are used (no library segmentation routines).
"""

import heapq
import numpy as np

from .color_space import circular_mean, circular_variance, circular_distance


# =============================================================================
# Block statistics helper
# =============================================================================
def _block_stats(h, s, valid, y0, x0, y1, x1):
    """!
    Compute homogeneity statistics over the valid pixels of a rectangular block.

    @return dict with circular hue variance, saturation variance, valid count.
    """
    hb = h[y0:y1, x0:x1].ravel()
    sb = s[y0:y1, x0:x1].ravel()
    vb = valid[y0:y1, x0:x1].ravel()
    if vb.sum() == 0:
        return {"circ_var_h": 1.0, "var_s": 1.0, "n_valid": 0}
    hv = hb[vb]
    sv = sb[vb]
    return {
        "circ_var_h": circular_variance(hv),
        "var_s": float(sv.var()),
        "n_valid": int(vb.sum()),
    }


# =============================================================================
# SPLIT phase  (top-down quadtree)
# =============================================================================
def split_quadtree(h, s, valid, edges,
                   tau_h=0.05, tau_s=0.02, tau_e=0.15,
                   min_size=4, min_start_depth=2, min_valid_frac=0.05):
    """!
    Recursively split the image into homogeneous square blocks (own code).

    A block is split into four quadrants if it is inhomogeneous, where
    inhomogeneous means ANY of:
        * circular variance of hue   > tau_h          (mean/variance of hue)
        * variance of saturation     > tau_s          (mean/variance of saturation)
        * mean Sobel edge energy     > tau_e          (edge-based split extension)
    Splitting is FORCED for the first `min_start_depth` levels so the procedure
    begins with at least 4**min_start_depth regions (>=16 for depth 2), satisfying
    the assignment requirement.  Recursion stops at `min_size` pixels.

    Blocks that are almost entirely invalid (background / dark) are not split
    further; they are kept as leaves and later classified as background.

    @param h,s    hue (rad) and saturation [0,1] arrays.
    @param valid  boolean guard mask.
    @param edges  Sobel edge-magnitude map in [0,1].
    @param tau_h,tau_s,tau_e  homogeneity thresholds.
    @param min_size           smallest allowed block edge (pixels).
    @param min_start_depth    forced split depth (>=2 guarantees >=16 regions).
    @param min_valid_frac     below this fraction of valid pixels a block is a leaf.
    @return label_map int32 array (each leaf block has a unique label >= 0).
    """
    H, W = h.shape
    label_map = np.full((H, W), -1, dtype=np.int32)
    next_label = [0]
    leaves = []  # bookkeeping (#leaves)

    def _homogeneous(y0, x0, y1, x1):
        st = _block_stats(h, s, valid, y0, x0, y1, x1)
        frac = st["n_valid"] / max(1, (y1 - y0) * (x1 - x0))
        if frac < min_valid_frac:
            return True  # mostly background -> stop splitting (leaf)
        e = float(edges[y0:y1, x0:x1].mean())
        return (st["circ_var_h"] <= tau_h
                and st["var_s"] <= tau_s
                and e <= tau_e)

    def _recurse(y0, x0, y1, x1, depth):
        bh, bw = y1 - y0, x1 - x0
        can_split = (bh >= 2 * min_size) and (bw >= 2 * min_size)
        force = depth < min_start_depth
        if can_split and (force or not _homogeneous(y0, x0, y1, x1)):
            my = y0 + bh // 2
            mx = x0 + bw // 2
            _recurse(y0, x0, my, mx, depth + 1)   # top-left
            _recurse(y0, mx, my, x1, depth + 1)   # top-right
            _recurse(my, x0, y1, mx, depth + 1)   # bottom-left
            _recurse(my, mx, y1, x1, depth + 1)   # bottom-right
        else:
            lbl = next_label[0]
            next_label[0] += 1
            label_map[y0:y1, x0:x1] = lbl
            leaves.append((y0, x0, y1, x1))

    _recurse(0, 0, H, W, 0)
    return label_map


# =============================================================================
# Region statistics accumulators (for O(1) merges)
# =============================================================================
class _RegionAcc:
    """!
    Sufficient statistics for a region so merging is O(1).

    Stores summed cos/sin of hue, sums of saturation and saturation^2, and the
    valid-pixel count.  Mean hue, hue variance and saturation variance are derived
    on demand from these sums.
    """
    __slots__ = ("sum_cos", "sum_sin", "sum_s", "sum_s2", "n", "npix")

    def __init__(self):
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.sum_s = 0.0
        self.sum_s2 = 0.0
        self.n = 0       # valid pixel count
        self.npix = 0    # total pixel count (incl. invalid)

    def add_pixels(self, h_vals, s_vals):
        self.sum_cos += float(np.cos(h_vals).sum())
        self.sum_sin += float(np.sin(h_vals).sum())
        self.sum_s += float(s_vals.sum())
        self.sum_s2 += float((s_vals * s_vals).sum())
        self.n += int(h_vals.size)

    def merge_from(self, other):
        self.sum_cos += other.sum_cos
        self.sum_sin += other.sum_sin
        self.sum_s += other.sum_s
        self.sum_s2 += other.sum_s2
        self.n += other.n
        self.npix += other.npix

    @property
    def mean_hue(self):
        if self.n == 0:
            return 0.0
        return float(np.arctan2(self.sum_sin / self.n, self.sum_cos / self.n) % (2 * np.pi))

    @property
    def circ_var_h(self):
        if self.n == 0:
            return 1.0
        cbar = self.sum_cos / self.n
        sbar = self.sum_sin / self.n
        return float(1.0 - np.sqrt(cbar * cbar + sbar * sbar))

    @property
    def mean_s(self):
        return 0.0 if self.n == 0 else self.sum_s / self.n

    @property
    def var_s(self):
        if self.n == 0:
            return 1.0
        m = self.sum_s / self.n
        return float(max(0.0, self.sum_s2 / self.n - m * m))


# =============================================================================
# Union-Find for region labels
# =============================================================================
class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return self.find(a)


# =============================================================================
# MERGE phase  (Region Adjacency Graph, greedy most-similar-first)
# =============================================================================
def merge_regions(label_map, h, s, valid, edges,
                  hue_thresh=0.30, sat_thresh=0.15,
                  merged_var_h=0.08, merged_var_s=0.03,
                  edge_veto=0.35):
    """!
    Merge adjacent leaf regions whose mean hue and mean saturation are similar
    and whose *combined* hue/saturation variance stays low (own code).

    Extensions implemented here:
      * Region Adjacency Graph + greedy most-similar-first ordering (priority queue).
      * Combined mean AND variance homogeneity test (not just closeness of means).
      * Edge-aware merge veto: a merge is rejected if a strong Sobel edge runs
        along the shared boundary, preserving instance boundaries between two
        same-hue but distinct fruits.

    @param label_map  int32 leaf labels from split_quadtree.
    @param h,s,valid,edges  as before.
    @param hue_thresh   max circular hue distance (rad) to consider merging.
    @param sat_thresh   max mean-saturation distance to consider merging.
    @param merged_var_h max circular hue variance allowed AFTER merge.
    @param merged_var_s max saturation variance allowed AFTER merge.
    @param edge_veto    mean boundary edge magnitude above which a merge is vetoed.
    @return new int32 label_map with merged (relabelled, contiguous) regions.
    """
    H, W = label_map.shape
    n_labels = int(label_map.max()) + 1
    if n_labels <= 1:
        return label_map.copy()

    # --- accumulate per-region statistics --------------------------------------
    accs = [_RegionAcc() for _ in range(n_labels)]
    flat_lbl = label_map.ravel()
    flat_h = h.ravel()
    flat_s = s.ravel()
    flat_v = valid.ravel()
    # total pixel counts
    counts = np.bincount(flat_lbl, minlength=n_labels)
    for i in range(n_labels):
        accs[i].npix = int(counts[i])
    # valid-pixel sums, grouped by label (vectorised per label via masks)
    order = np.argsort(flat_lbl, kind="stable")
    sorted_lbl = flat_lbl[order]
    boundaries = np.searchsorted(sorted_lbl, np.arange(n_labels + 1))
    for i in range(n_labels):
        idx = order[boundaries[i]:boundaries[i + 1]]
        vi = flat_v[idx]
        if vi.any():
            accs[i].add_pixels(flat_h[idx][vi], flat_s[idx][vi])

    # --- build adjacency + per-edge mean boundary edge magnitude ---------------
    adj = {}  # frozenset({a,b}) -> [sum_edge, count]

    def _add_edge(a, b, e):
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        rec = adj.get(key)
        if rec is None:
            adj[key] = [e, 1]
        else:
            rec[0] += e
            rec[1] += 1

    # horizontal neighbours
    la = label_map[:, :-1]
    lb = label_map[:, 1:]
    eh = 0.5 * (edges[:, :-1] + edges[:, 1:])
    diff = la != lb
    for a, b, e in zip(la[diff], lb[diff], eh[diff]):
        _add_edge(int(a), int(b), float(e))
    # vertical neighbours
    la = label_map[:-1, :]
    lb = label_map[1:, :]
    ev = 0.5 * (edges[:-1, :] + edges[1:, :])
    diff = la != lb
    for a, b, e in zip(la[diff], lb[diff], ev[diff]):
        _add_edge(int(a), int(b), float(e))

    # --- priority queue keyed on colour distance -------------------------------
    uf = _UnionFind(n_labels)

    def _distance(a, b):
        ha, hb = accs[a].mean_hue, accs[b].mean_hue
        dh = circular_distance(ha, hb)
        ds = abs(accs[a].mean_s - accs[b].mean_s)
        # normalise hue distance (0..pi) to roughly the same scale as sat (0..1)
        return (dh / np.pi) + ds

    heap = []
    for (a, b), (esum, ecnt) in adj.items():
        heapq.heappush(heap, (_distance(a, b), a, b))

    alive = [True] * n_labels  # a label is dead once merged into another root

    def _mergeable(a, b):
        ha, hb = accs[a].mean_hue, accs[b].mean_hue
        if circular_distance(ha, hb) > hue_thresh:
            return False
        if abs(accs[a].mean_s - accs[b].mean_s) > sat_thresh:
            return False
        # edge-aware veto along the shared boundary
        key = (a, b) if a < b else (b, a)
        rec = adj.get(key)
        if rec is not None and rec[1] > 0 and (rec[0] / rec[1]) > edge_veto:
            return False
        # combined-variance homogeneity test
        tmp = _RegionAcc()
        tmp.merge_from(accs[a])
        tmp.merge_from(accs[b])
        if tmp.circ_var_h > merged_var_h or tmp.var_s > merged_var_s:
            return False
        return True

    # --- greedy most-similar-first merging -------------------------------------
    while heap:
        d, a, b = heapq.heappop(heap)
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb or not alive[ra] or not alive[rb]:
            continue  # stale entry
        if not _mergeable(ra, rb):
            continue
        # merge rb into ra
        accs[ra].merge_from(accs[rb])
        alive[rb] = False
        # rebuild adjacency edges of rb onto ra and push fresh distances
        new_edges = []
        for (x, y), rec in list(adj.items()):
            if x == rb or y == rb:
                other = y if x == rb else x
                ro = uf.find(other)
                if ro != ra and alive[ro]:
                    new_edges.append((ro, rec))
                del adj[(x, y)]
        uf.union(ra, rb)
        for ro, rec in new_edges:
            key = (ra, ro) if ra < ro else (ro, ra)
            old = adj.get(key)
            if old is None:
                adj[key] = [rec[0], rec[1]]
            else:
                old[0] += rec[0]
                old[1] += rec[1]
            heapq.heappush(heap, (_distance(ra, ro), ra, ro))

    # --- produce contiguous relabelled map -------------------------------------
    roots = {}
    out = np.empty_like(label_map)
    flat_in = label_map.ravel()
    flat_out = out.ravel()
    for i in range(flat_in.size):
        r = uf.find(int(flat_in[i]))
        if r not in roots:
            roots[r] = len(roots)
        flat_out[i] = roots[r]
    return out

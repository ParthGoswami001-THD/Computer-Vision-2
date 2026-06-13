"""!
@file split_merge.py
@brief Quadtree SPLIT and Region-Adjacency-Graph MERGE, both in own code.

Homogeneity is tested on the circular variance of hue and the variance of
saturation, evaluated only over guard-valid pixels. The merge phase also
checks edge strength along shared boundaries so that two same-coloured but
distinct fruits are not collapsed into a single blob.

References:
    Horowitz & Pavlidis, J. ACM 23(2):368-388, 1976
    Pavlidis & Liow, IEEE TPAMI 12(3):225-233, 1990
"""

import heapq
import numpy as np

from .color_space import circular_mean, circular_variance, circular_distance


def _block_stats(h, s, valid, y0, x0, y1, x1):
    """!
    Homogeneity statistics over the valid pixels of a rectangular block.

    @return dict with keys circ_var_h, var_s, n_valid.
    """
    hb = h[y0:y1, x0:x1].ravel()
    sb = s[y0:y1, x0:x1].ravel()
    vb = valid[y0:y1, x0:x1].ravel()
    if vb.sum() == 0:
        return {"circ_var_h": 1.0, "var_s": 1.0, "n_valid": 0}
    hv, sv = hb[vb], sb[vb]
    return {
        "circ_var_h": circular_variance(hv),
        "var_s": float(sv.var()),
        "n_valid": int(vb.sum()),
    }


def split_quadtree(h, s, valid, edges,
                   tau_h=0.05, tau_s=0.02, tau_e=0.15,
                   min_size=4, min_start_depth=2, min_valid_frac=0.05):
    """!
    Recursively split the image into homogeneous square blocks (own code).

    A block is split if ANY of the following holds:
        circular variance of hue  > tau_h
        variance of saturation    > tau_s
        mean Sobel edge energy    > tau_e

    The first min_start_depth levels are split unconditionally, guaranteeing
    at least 4**min_start_depth leaf regions (>= 16 for depth 2). Blocks
    whose valid-pixel fraction falls below min_valid_frac are treated as
    uniform background and kept as leaves.

    @param h, s         hue (radians) and saturation [0, 1] arrays.
    @param valid         boolean guard mask.
    @param edges         Sobel edge-magnitude map in [0, 1].
    @param tau_h, tau_s, tau_e  homogeneity thresholds.
    @param min_size      smallest allowed block side in pixels.
    @param min_start_depth  forced split depth (>= 2 gives >= 16 start regions).
    @param min_valid_frac   blocks below this valid fraction are leaves.
    @return int32 label map; each leaf block has a unique label >= 0.
    """
    H, W = h.shape
    label_map = np.full((H, W), -1, dtype=np.int32)
    next_label = [0]

    def _homogeneous(y0, x0, y1, x1):
        st = _block_stats(h, s, valid, y0, x0, y1, x1)
        frac = st["n_valid"] / max(1, (y1 - y0) * (x1 - x0))
        if frac < min_valid_frac:
            return True
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
            _recurse(y0, x0, my, mx, depth + 1)
            _recurse(y0, mx, my, x1, depth + 1)
            _recurse(my, x0, y1, mx, depth + 1)
            _recurse(my, mx, y1, x1, depth + 1)
        else:
            lbl = next_label[0]
            next_label[0] += 1
            label_map[y0:y1, x0:x1] = lbl

    _recurse(0, 0, H, W, 0)
    return label_map


class _RegionAcc:
    """Sufficient statistics for a region enabling O(1) merges.

    Stores summed cos/sin of hue and saturation moments; mean hue, hue
    variance, and saturation variance are derived on demand from these sums.
    """
    __slots__ = ("sum_cos", "sum_sin", "sum_s", "sum_s2", "n", "npix")

    def __init__(self):
        self.sum_cos = 0.0
        self.sum_sin = 0.0
        self.sum_s = 0.0
        self.sum_s2 = 0.0
        self.n = 0
        self.npix = 0

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
        return float(np.arctan2(self.sum_sin / self.n,
                                self.sum_cos / self.n) % (2 * np.pi))

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


def merge_regions(label_map, h, s, valid, edges,
                  hue_thresh=0.30, sat_thresh=0.15,
                  merged_var_h=0.08, merged_var_s=0.03,
                  edge_veto=0.35):
    """!
    Merge adjacent leaf regions whose colour statistics are similar (own code).

    Two regions are merged when:
      - their mean hue distance is within hue_thresh
      - their mean saturation distance is within sat_thresh
      - the combined region's hue/saturation variance stays below merged_var_h / merged_var_s
      - the mean edge magnitude along their shared boundary does not exceed edge_veto

    Pairs are processed greedily from a priority queue keyed on colour distance
    so the most similar neighbours always merge first. Region statistics are
    maintained as accumulated sums, so each merge is O(1).

    @param label_map   int32 leaf labels from split_quadtree.
    @param h, s, valid, edges  as returned by the pipeline preprocessing steps.
    @param hue_thresh   maximum circular hue distance (rad) for merging.
    @param sat_thresh   maximum mean-saturation distance for merging.
    @param merged_var_h maximum circular hue variance allowed after merging.
    @param merged_var_s maximum saturation variance allowed after merging.
    @param edge_veto    mean boundary edge magnitude above which a merge is blocked.
    @return new int32 label map with contiguous relabelled regions.
    """
    H, W = label_map.shape
    n_labels = int(label_map.max()) + 1
    if n_labels <= 1:
        return label_map.copy()

    accs = [_RegionAcc() for _ in range(n_labels)]
    flat_lbl = label_map.ravel()
    flat_h = h.ravel()
    flat_s = s.ravel()
    flat_v = valid.ravel()
    counts = np.bincount(flat_lbl, minlength=n_labels)
    for i in range(n_labels):
        accs[i].npix = int(counts[i])
    order = np.argsort(flat_lbl, kind="stable")
    sorted_lbl = flat_lbl[order]
    boundaries = np.searchsorted(sorted_lbl, np.arange(n_labels + 1))
    for i in range(n_labels):
        idx = order[boundaries[i]:boundaries[i + 1]]
        vi = flat_v[idx]
        if vi.any():
            accs[i].add_pixels(flat_h[idx][vi], flat_s[idx][vi])

    # Build adjacency dict: (a, b) -> [sum_edge, count]
    adj = {}

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

    la = label_map[:, :-1]
    lb = label_map[:, 1:]
    eh = 0.5 * (edges[:, :-1] + edges[:, 1:])
    diff = la != lb
    for a, b, e in zip(la[diff], lb[diff], eh[diff]):
        _add_edge(int(a), int(b), float(e))
    la = label_map[:-1, :]
    lb = label_map[1:, :]
    ev = 0.5 * (edges[:-1, :] + edges[1:, :])
    diff = la != lb
    for a, b, e in zip(la[diff], lb[diff], ev[diff]):
        _add_edge(int(a), int(b), float(e))

    uf = _UnionFind(n_labels)

    def _distance(a, b):
        dh = circular_distance(accs[a].mean_hue, accs[b].mean_hue)
        ds = abs(accs[a].mean_s - accs[b].mean_s)
        return (dh / np.pi) + ds

    heap = []
    for (a, b) in adj:
        heapq.heappush(heap, (_distance(a, b), a, b))

    alive = [True] * n_labels

    def _mergeable(a, b):
        if circular_distance(accs[a].mean_hue, accs[b].mean_hue) > hue_thresh:
            return False
        if abs(accs[a].mean_s - accs[b].mean_s) > sat_thresh:
            return False
        key = (a, b) if a < b else (b, a)
        rec = adj.get(key)
        if rec is not None and rec[1] > 0 and (rec[0] / rec[1]) > edge_veto:
            return False
        tmp = _RegionAcc()
        tmp.merge_from(accs[a])
        tmp.merge_from(accs[b])
        if tmp.circ_var_h > merged_var_h or tmp.var_s > merged_var_s:
            return False
        return True

    while heap:
        d, a, b = heapq.heappop(heap)
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb or not alive[ra] or not alive[rb]:
            continue
        if not _mergeable(ra, rb):
            continue
        accs[ra].merge_from(accs[rb])
        alive[rb] = False
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

    # Relabel roots to contiguous integers
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

import heapq
import random


def tanimoto_distance(a, b):
    """1 - |A&B| / |A|B|. Both empty -> 0.0. A true metric."""
    union = (a | b).bit_count()
    if union == 0:
        return 0.0
    return 1.0 - (a & b).bit_count() / union


class _Node:
    __slots__ = ("vp", "left", "right", "l_lo", "l_hi", "r_lo", "r_hi", "bucket")

    def __init__(self):
        self.vp = None
        self.left = None
        self.right = None
        self.l_lo = self.l_hi = 0.0
        self.r_lo = self.r_hi = 0.0
        self.bucket = None


class VPTree:
    def __init__(self, fingerprints, dist=tanimoto_distance, leaf_size=16,
                 n_candidates=8, sample_size=24, seed=0):
        self.fps = fingerprints
        self.dist = dist
        self.leaf_size = max(1, leaf_size)
        self.n_candidates = n_candidates
        self.sample_size = sample_size
        self._rng = random.Random(seed)
        self.root = self._build(list(range(len(fingerprints)))) if fingerprints else None

    def _select_vantage(self, ids):
        if len(ids) <= 2:
            return ids[0]
        cands = self._rng.sample(ids, min(self.n_candidates, len(ids)))
        sample = self._rng.sample(ids, min(self.sample_size, len(ids)))
        best, best_var = cands[0], -1.0
        for c in cands:
            ds = [self.dist(self.fps[c], self.fps[s]) for s in sample]
            mean = sum(ds) / len(ds)
            var = sum((d - mean) ** 2 for d in ds) / len(ds)
            if var > best_var:
                best, best_var = c, var
        return best

    def _build(self, ids):
        node = _Node()
        if len(ids) <= self.leaf_size:
            node.bucket = ids
            return node

        vp = self._select_vantage(ids)
        vp_fp = self.fps[vp]
        scored = sorted((self.dist(vp_fp, self.fps[i]), i) for i in ids if i != vp)

        mid = len(scored) // 2
        left, right = scored[:mid], scored[mid:]

        node.vp = vp
        node.l_lo, node.l_hi = left[0][0], left[-1][0]
        node.r_lo, node.r_hi = right[0][0], right[-1][0]
        node.left = self._build([i for _, i in left])
        node.right = self._build([i for _, i in right])
        return node

    def range_search(self, query_fp, r):
        out = []
        if self.root is None:
            return out
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.bucket is not None:
                for i in n.bucket:
                    d = self.dist(query_fp, self.fps[i])
                    if d <= r:
                        out.append((d, i))
                continue
            d = self.dist(query_fp, self.fps[n.vp])
            if d <= r:
                out.append((d, n.vp))
            if d + r >= n.l_lo and d - r <= n.l_hi:
                stack.append(n.left)
            if d + r >= n.r_lo and d - r <= n.r_hi:
                stack.append(n.right)
        out.sort()
        return out

    def knn(self, query_fp, k):
        if k <= 0 or self.root is None:
            return []
        heap = []

        def tau():
            return -heap[0][0] if len(heap) == k else float("inf")

        def offer(d, i):
            item = (-d, -i)
            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

        def visit(n):
            if n.bucket is not None:
                for i in n.bucket:
                    offer(self.dist(query_fp, self.fps[i]), i)
                return
            d = self.dist(query_fp, self.fps[n.vp])
            offer(d, n.vp)
            sides = [(n.left, n.l_lo, n.l_hi), (n.right, n.r_lo, n.r_hi)]
            sides.sort(key=lambda s: max(s[1] - d, d - s[2], 0.0))
            for child, lo, hi in sides:
                t = tau()
                if d + t >= lo and d - t <= hi:
                    visit(child)

        visit(self.root)
        return sorted((-nd, -ni) for nd, ni in heap)
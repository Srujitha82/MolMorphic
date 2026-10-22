"""Brute-force baseline and correctness checker for the VP-tree.

Interface the VP-tree (Members 2 and 3) must provide so the checker can test it:

    tree.knn(query_fp, k)           -> list of (distance, index), nearest first
    tree.range_search(query_fp, r)  -> list of (distance, index), any order

`index` is the position of the molecule in the `fingerprints` list that the
tree was built from.
"""
import heapq
import random
from typing import List, Sequence, Tuple

from .fingerprint import tanimoto_distance

Result = Tuple[float, int]
EPS = 1e-12


def knn_bruteforce(fingerprints: Sequence[int], query: int, k: int) -> List[Result]:
    """k nearest molecules by scanning everything. O(N log k)."""
    if k <= 0:
        return []
    scored = ((tanimoto_distance(query, fp), i) for i, fp in enumerate(fingerprints))
    return heapq.nsmallest(k, scored)  # sorted by (distance, index)


def range_bruteforce(fingerprints: Sequence[int], query: int, r: float) -> List[Result]:
    """All molecules with distance <= r, sorted by (distance, index). O(N)."""
    out = []
    for i, fp in enumerate(fingerprints):
        d = tanimoto_distance(query, fp)
        if d <= r:
            out.append((d, i))
    out.sort()
    return out


def count_distance_computations(fingerprints: Sequence[int]) -> int:
    """Brute force always computes N distances per query (baseline for benchmarks)."""
    return len(fingerprints)


def check_correctness(tree, fingerprints: Sequence[int], queries: Sequence[int],
                      ks: Sequence[int] = (1, 5, 10), radii: Sequence[float] = (0.0, 0.3, 0.6),
                      verbose: bool = True) -> bool:
    """Compare the tree against brute force for every query.

    kNN: compare the sorted DISTANCE lists (ties may legitimately return different
    molecules at equal distance, so ids are not compared).
    Range: compare the exact SET of indices.
    Returns True only if everything matches.
    """
    ok = True
    for qi, q in enumerate(queries):
        for k in ks:
            expected = [d for d, _ in knn_bruteforce(fingerprints, q, k)]
            got = [d for d, _ in tree.knn(q, k)]
            if len(expected) != len(got) or any(abs(a - b) > EPS for a, b in zip(expected, got)):
                ok = False
                if verbose:
                    print(f"[MISMATCH] kNN query#{qi} k={k}\n  expected={expected}\n  got     ={got}")
        for r in radii:
            expected_ids = {i for _, i in range_bruteforce(fingerprints, q, r)}
            got_ids = {i for _, i in tree.range_search(q, r)}
            if expected_ids != got_ids:
                ok = False
                if verbose:
                    print(f"[MISMATCH] range query#{qi} r={r}\n"
                          f"  missing={sorted(expected_ids - got_ids)} extra={sorted(got_ids - expected_ids)}")
    if verbose:
        print("All results match brute force." if ok else "Correctness check FAILED.")
    return ok


def sample_queries(fingerprints: Sequence[int], n: int = 20, seed: int = 0) -> List[int]:
    """Pick n random library fingerprints as queries (reproducible with seed)."""
    rng = random.Random(seed)
    return [rng.choice(fingerprints) for _ in range(n)]


def random_bit_fingerprints(n: int, bits: int = 2048, density: float = 0.03, seed: int = 0) -> List[int]:
    """Synthetic fingerprints for scale testing when you do not have a big dataset.

    Bits are drawn around a few random 'family centres' so the data is clustered,
    like real chemical libraries (pure random bits make every distance almost equal).
    """
    rng = random.Random(seed)
    centres = []
    for _ in range(max(1, n // 200)):
        c = 0
        for b in range(bits):
            if rng.random() < density:
                c |= 1 << b
        centres.append(c)
    out = []
    for _ in range(n):
        fp = rng.choice(centres)
        for b in rng.sample(range(bits), k=max(1, int(bits * density * 0.3))):
            fp ^= 1 << b  # flip a few bits so family members are similar, not identical
        out.append(fp)
    return out

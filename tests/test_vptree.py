import random
from src.vptree import VPTree, tanimoto_distance


def make_fps(n, bits=256, seed=3):
    rng = random.Random(seed)
    centers = [rng.getrandbits(bits) for _ in range(10)]
    out = []
    for _ in range(n):
        fp = rng.choice(centers)
        for b in range(bits):
            if rng.random() < 0.05:
                fp ^= 1 << b
        out.append(fp)
    return out


def test_knn_matches_bruteforce_distances():
    fps = make_fps(1000)
    tree = VPTree(fps, leaf_size=8)
    for q in fps[:20]:
        truth = sorted(tanimoto_distance(q, f) for f in fps)[:7]
        got = [d for d, _ in tree.knn(q, 7)]
        assert got == truth


def test_range_matches_bruteforce_indices():
    fps = make_fps(1000)
    tree = VPTree(fps, leaf_size=8)
    for q in fps[:20]:
        expected = {i for i, f in enumerate(fps) if tanimoto_distance(q, f) <= 0.4}
        assert {i for _, i in tree.range_search(q, 0.4)} == expected


def test_duplicates_and_tiny_inputs():
    fps = [5] * 50
    tree = VPTree(fps, leaf_size=4)
    assert len(tree.range_search(5, 0.0)) == 50
    assert VPTree([], leaf_size=4).knn(1, 3) == []
    assert VPTree([7], leaf_size=4).knn(7, 3) == [(0.0, 0)]
import os

from src.bruteforce import (check_correctness, knn_bruteforce, random_bit_fingerprints,
                            range_bruteforce, sample_queries)
from src.fingerprint import load_library, tanimoto_distance

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_molecules.csv")


class BruteForceTree:
    """Stand-in 'tree' that is correct by construction. Replace with the real VP-tree."""
    def __init__(self, fps):
        self.fps = fps

    def knn(self, q, k):
        return knn_bruteforce(self.fps, q, k)

    def range_search(self, q, r):
        return range_bruteforce(self.fps, q, r)


class BrokenKnnTree(BruteForceTree):
    def knn(self, q, k):
        return knn_bruteforce(self.fps, q, k)[1:]  # drops the nearest neighbour


class BrokenRangeTree(BruteForceTree):
    def range_search(self, q, r):
        return range_bruteforce(self.fps, q, r)[:-1]  # misses one result


def test_knn_returns_sorted_k_results_and_self_first():
    fps = random_bit_fingerprints(100, seed=3)
    res = knn_bruteforce(fps, fps[10], 5)
    assert len(res) == 5
    assert [d for d, _ in res] == sorted(d for d, _ in res)
    assert res[0][0] == 0.0  # the query is in the library


def test_knn_edge_cases():
    fps = random_bit_fingerprints(10, seed=4)
    assert knn_bruteforce(fps, fps[0], 0) == []
    assert len(knn_bruteforce(fps, fps[0], 50)) == 10  # k > N returns all
    assert knn_bruteforce([], 0b1, 3) == []            # empty library


def test_range_edge_cases():
    fps = random_bit_fingerprints(50, seed=5)
    q = fps[0]
    assert {i for _, i in range_bruteforce(fps, q, 1.0)} == set(range(50))  # r=1 -> everything
    zero = range_bruteforce(fps, q, 0.0)
    assert all(d == 0.0 for d, _ in zero) and any(i == 0 for _, i in zero)
    assert range_bruteforce([], q, 0.5) == []


def test_range_results_all_within_radius():
    fps, _ = load_library(SAMPLE)
    q = fps[0]
    for d, i in range_bruteforce(fps, q, 0.7):
        assert d <= 0.7 and d == tanimoto_distance(q, fps[i])


def test_duplicates_are_all_returned():
    fps = [0b1010, 0b1010, 0b1010, 0b0101]
    assert len(range_bruteforce(fps, 0b1010, 0.0)) == 3


def test_checker_passes_on_correct_tree():
    fps, _ = load_library(SAMPLE)
    assert check_correctness(BruteForceTree(fps), fps, sample_queries(fps, 10), verbose=False)


def test_checker_detects_broken_knn():
    fps, _ = load_library(SAMPLE)
    assert not check_correctness(BrokenKnnTree(fps), fps, sample_queries(fps, 5), verbose=False)


def test_checker_detects_broken_range():
    fps, _ = load_library(SAMPLE)
    assert not check_correctness(BrokenRangeTree(fps), fps, sample_queries(fps, 5), verbose=False)


def test_sample_queries_reproducible():
    fps = random_bit_fingerprints(50, seed=6)
    assert sample_queries(fps, 10, seed=1) == sample_queries(fps, 10, seed=1)


def test_synthetic_generator_size():
    fps = random_bit_fingerprints(300, seed=7)
    assert len(fps) == 300

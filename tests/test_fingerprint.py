import itertools
import os
import random

import pytest

from src.fingerprint import (FP_BITS, load_library, popcount, smiles_to_fingerprint,
                             tanimoto_distance, tanimoto_similarity)

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_molecules.csv")
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
SALICYLIC = "OC(=O)C1=CC=CC=C1O"
ETHANOL = "CCO"


# ---------- fingerprint conversion ----------
def test_fingerprint_is_int_within_bit_range():
    fp = smiles_to_fingerprint(ASPIRIN)
    assert isinstance(fp, int)
    assert 0 < fp < (1 << FP_BITS)


def test_same_molecule_different_smiles_gives_same_fingerprint():
    # two valid SMILES spellings of ethanol
    assert smiles_to_fingerprint("CCO") == smiles_to_fingerprint("OCC")


@pytest.mark.parametrize("bad", ["", "   ", "not_a_smiles", "C1CC", "C(C)(C)(C)(C)C", None, 123])
def test_invalid_smiles_raises_value_error(bad):
    with pytest.raises(ValueError):
        smiles_to_fingerprint(bad)


def test_whitespace_around_smiles_is_ignored():
    assert smiles_to_fingerprint("  CCO  ") == smiles_to_fingerprint("CCO")


# ---------- popcount ----------
def test_popcount():
    assert popcount(0) == 0
    assert popcount(0b1011) == 3
    assert popcount((1 << 2048) - 1) == 2048


# ---------- Tanimoto distance ----------
def test_hand_computed_example():
    a, b = 0b11110000, 0b11000011  # |A&B|=2, |A|B|=6 -> T=1/3, d=2/3
    assert tanimoto_similarity(a, b) == pytest.approx(1 / 3)
    assert tanimoto_distance(a, b) == pytest.approx(2 / 3)


def test_identical_distance_zero():
    fp = smiles_to_fingerprint(ASPIRIN)
    assert tanimoto_distance(fp, fp) == 0.0


def test_disjoint_distance_one():
    assert tanimoto_distance(0b1100, 0b0011) == 1.0


def test_both_empty_distance_zero_no_division_error():
    assert tanimoto_distance(0, 0) == 0.0


def test_one_empty_distance_one():
    assert tanimoto_distance(0, 0b101) == 1.0
    assert tanimoto_distance(0b101, 0) == 1.0


def test_symmetry_and_range():
    rng = random.Random(1)
    for _ in range(200):
        a, b = rng.getrandbits(64), rng.getrandbits(64)
        d = tanimoto_distance(a, b)
        assert d == tanimoto_distance(b, a)
        assert 0.0 <= d <= 1.0


def test_triangle_inequality_random_bits():
    rng = random.Random(2)
    for _ in range(2000):
        a, b, c = (rng.getrandbits(32) for _ in range(3))
        assert tanimoto_distance(a, c) <= tanimoto_distance(a, b) + tanimoto_distance(b, c) + 1e-12


def test_triangle_inequality_real_molecules():
    fps, _ = load_library(SAMPLE)
    for a, b, c in itertools.combinations(fps[:15], 3):
        assert tanimoto_distance(a, c) <= tanimoto_distance(a, b) + tanimoto_distance(b, c) + 1e-12
        assert tanimoto_distance(a, b) <= tanimoto_distance(a, c) + tanimoto_distance(c, b) + 1e-12


def test_chemically_similar_molecules_are_closer():
    asp, sal, eth = (smiles_to_fingerprint(s) for s in (ASPIRIN, SALICYLIC, ETHANOL))
    assert tanimoto_distance(asp, sal) < tanimoto_distance(asp, eth)


# ---------- library loading ----------
def test_load_sample_library():
    fps, records = load_library(SAMPLE)
    assert len(fps) == len(records) == 31
    assert records[0]["name"] == "aspirin"


def test_load_library_skips_invalid_rows(tmp_path, capsys):
    p = tmp_path / "lib.csv"
    p.write_text("id,name,smiles\n1,ethanol,CCO\n2,broken,xxx\n3,methanol,CO\n")
    fps, records = load_library(str(p))
    assert len(fps) == 2
    assert [r["name"] for r in records] == ["ethanol", "methanol"]
    assert "skipped 1" in capsys.readouterr().out

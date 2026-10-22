"""Fingerprint generation and Tanimoto distance.

A fingerprint is stored as a Python int used as a bit-vector (bit i set means
substructure i is present). Python ints give us fast AND / OR and popcount, and
they are hashable and compact, so the VP-tree code can treat them as opaque
values and only ever call `tanimoto_distance`.
"""
import csv
from typing import List, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

FP_RADIUS = 2
FP_BITS = 2048

RDLogger.DisableLog("rdApp.*")  # we handle invalid SMILES ourselves
_generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def popcount(x: int) -> int:
    """Number of set bits in x."""
    try:
        return x.bit_count()  # Python 3.10+
    except AttributeError:
        return bin(x).count("1")


def smiles_to_fingerprint(smiles: str) -> int:
    """Convert a SMILES string to a 2048-bit Morgan (ECFP4-like) fingerprint.

    Raises ValueError if the SMILES cannot be parsed.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    bitvect = _generator.GetFingerprint(mol)
    return int(bitvect.ToBitString(), 2)


def tanimoto_similarity(a: int, b: int) -> float:
    """T(A,B) = |A and B| / |A or B|. Two empty fingerprints count as identical."""
    union = popcount(a | b)
    if union == 0:
        return 1.0
    return popcount(a & b) / union


def tanimoto_distance(a: int, b: int) -> float:
    """d(A,B) = 1 - T(A,B). This (Jaccard) distance is a true metric:
    non-negative, d=0 iff A==B, symmetric, and satisfies the triangle inequality.
    """
    return 1.0 - tanimoto_similarity(a, b)


def load_library(path: str) -> Tuple[List[int], List[dict]]:
    """Load a CSV with columns id,name,smiles.

    Returns (fingerprints, records). Rows with invalid SMILES are skipped and
    reported on stdout. fingerprints[i] belongs to records[i].
    """
    fingerprints: List[int] = []
    records: List[dict] = []
    skipped = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                fp = smiles_to_fingerprint(row["smiles"])
            except ValueError:
                skipped += 1
                continue
            fingerprints.append(fp)
            records.append(row)
    if skipped:
        print(f"[load_library] skipped {skipped} row(s) with invalid SMILES")
    return fingerprints, records

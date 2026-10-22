# MolMorphic
# VP-Tree Molecule Similarity Search

Fast, exact similarity search over drug-like molecules using a Vantage-Point Tree (VP-tree) with Tanimoto distance.

## Problem

In virtual screening for drug discovery, chemists start from one molecule that works against a disease target and need to find structurally similar molecules in libraries of millions (ChEMBL, ZINC, PubChem). Comparing the query against every molecule is too slow.

Given a library of N molecules and a query molecule, this project returns:

- the **k most similar** molecules (kNN search), and
- all molecules within **Tanimoto distance r** (range search),

without comparing the query against every molecule.

## Approach

1. Each molecule (SMILES string) is converted to a 2048-bit Morgan fingerprint using RDKit. Each bit marks the presence of a circular substructure.
2. Distance between two fingerprints A and B:

   `d(A, B) = 1 - |A ∩ B| / |A ∪ B|` (Tanimoto / Jaccard distance)

   This is a true metric (non-negative, symmetric, `d = 0` only for identical fingerprints, triangle inequality holds). Tanimoto *similarity* alone is not a metric, so the tree is built on the distance.
3. A VP-tree splits the library at the median distance from a vantage point. During search, the triangle inequality lets us discard whole subtrees without computing distances to them.

### Why a VP-tree?

| Alternative | Why it is weaker here |
|---|---|
| Linear scan | O(N) distance computations per query |
| Hash table | Exact match only |
| k-d tree / R-tree | Need coordinate axes; degrade in high dimensions |
| BK-tree | Needs integer distances; Tanimoto distance is real-valued |
| LSH (MinHash) | Approximate; can miss true neighbours |
| **VP-tree** | Needs only a metric, handles real-valued distances, balanced by median split, exact results |

## Complexity

| Operation | Time |
|---|---|
| Distance (F-bit fingerprint, w-bit words) | O(F / w), effectively constant |
| Build | O(N log N) distance computations |
| Query (average) | O(log N) distance computations |
| Query (worst case) | O(N) |
| Space | O(N) |
| Brute-force kNN / range (baseline) | O(N log k) / O(N) |

## Project structure

```
vptree-molecule-search/
├── src/
│   ├── fingerprint.py     # SMILES -> fingerprint, Tanimoto distance, CSV loader
│   ├── bruteforce.py      # brute-force baseline + correctness checker
│   └── vptree.py          # VP-tree build, kNN and range search
├── tests/
│   ├── test_fingerprint.py
│   └── test_bruteforce.py
├── data/
│   └── sample_molecules.csv
├── brute_demo.py          # CLI: brute-force nearest neighbours
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.9+.

```bash
git clone https://github.com/<org-or-user>/vptree-molecule-search.git
cd vptree-molecule-search
pip install -r requirements.txt
```

## Usage

Find the 5 nearest molecules to aspirin (brute force):

```bash
python brute_demo.py "CC(=O)OC1=CC=CC=C1C(=O)O" --k 5
```

Example output:

```
0.000  aspirin              CC(=O)OC1=CC=CC=C1C(=O)O
0.552  salicylic acid       OC(=O)C1=CC=CC=C1O
0.643  benzoic acid         OC(=O)C1=CC=CC=C1
0.647  methyl salicylate    COC(=O)C1=CC=CC=C1O
0.750  toluene              CC1=CC=CC=C1
```

### From Python

```python
from src.fingerprint import smiles_to_fingerprint, tanimoto_distance, load_library
from src.bruteforce import knn_bruteforce, range_bruteforce

fps, records = load_library("data/sample_molecules.csv")
q = smiles_to_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")

knn_bruteforce(fps, q, k=5)        # [(distance, index), ...]
range_bruteforce(fps, q, r=0.6)    # all molecules within distance 0.6
```

## Tree interface

The VP-tree must provide the following so the correctness checker can verify it against brute force:

```python
tree.knn(query_fp, k)            # -> [(distance, index), ...], nearest first
tree.range_search(query_fp, r)   # -> [(distance, index), ...]
```

`index` is the position of the molecule in the fingerprint list the tree was built from.

## Testing

```bash
python -m pytest
```

The tests cover fingerprint conversion (including invalid SMILES), the metric properties of Tanimoto distance (symmetry, range, triangle inequality), edge cases (empty fingerprints, empty library, `k > N`, `r = 0`, `r = 1`, duplicates), and the correctness checker itself.

To verify the VP-tree against brute force:

```python
from src.bruteforce import check_correctness, sample_queries

check_correctness(tree, fps, sample_queries(fps, 50))
```

For larger scale tests without a big dataset, `random_bit_fingerprints(100000)` generates clustered synthetic fingerprints.

## Edge cases handled

- Query already in the library (distance 0)
- No molecule within radius r
- `k` larger than the library size, empty library, single molecule
- Duplicate molecules and ties at equal distance
- Two empty fingerprints (distance defined as 0, avoids division by zero)
- Invalid SMILES (raises `ValueError`; skipped with a count when loading a CSV)

## Team

| Member | GitHub | Contribution |
|---|---|---|
| Member 1 | [@username](https://github.com/username) | SMILES to fingerprint conversion, Tanimoto distance, brute-force baseline, correctness checker, unit tests |
| Member 2 | [@username](https://github.com/username) | VP-tree build, vantage point selection, leaf buckets |
| Member 3 | [@username](https://github.com/username) | kNN and range search, benchmarks, CLI demo, dataset loader |

## Data

`data/sample_molecules.csv` is a small sample of well-known molecules for demos and tests. For larger experiments, download a library from [ChEMBL](https://www.ebi.ac.uk/chembl/) or [ZINC](https://zinc.docking.org/) as a CSV with columns `id,name,smiles`, and keep it out of Git (see `.gitignore`).

## License

MIT

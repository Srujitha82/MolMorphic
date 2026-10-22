"""Member 1 demo: brute-force nearest neighbours for a SMILES query.
Usage: python brute_demo.py "CC(=O)OC1=CC=CC=C1C(=O)O" --k 5
"""
import argparse

from src.bruteforce import knn_bruteforce
from src.fingerprint import load_library, smiles_to_fingerprint

p = argparse.ArgumentParser()
p.add_argument("smiles")
p.add_argument("--k", type=int, default=5)
p.add_argument("--data", default="data/sample_molecules.csv")
args = p.parse_args()

fps, records = load_library(args.data)
try:
    q = smiles_to_fingerprint(args.smiles)
except ValueError as e:
    raise SystemExit(f"Error: {e}")
for d, i in knn_bruteforce(fps, q, args.k):
    print(f"{d:.3f}  {records[i]['name']:<20} {records[i]['smiles']}")

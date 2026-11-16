"""Concatenate network ensembles into one weights file: merge_nets.py OUT.npz A.npz B.npz ..."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solution import seqnet  # noqa: E402

if __name__ == "__main__":
    nets = [n for path in sys.argv[2:] for n in seqnet.load(path)]
    seqnet.save(nets, sys.argv[1])
    print("wrote", sys.argv[1], "with", len(nets), "networks")

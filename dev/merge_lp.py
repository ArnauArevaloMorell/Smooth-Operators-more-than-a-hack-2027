"""Save the probability-average of several saved held-out outputs under a new tag (for SEQ_TAG).

usage: merge_lp.py OUT_TAG TAG1+TAG2+...
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz

out, tags = sys.argv[1], sys.argv[2].split("+")
n = 0
for case in hz.CASES:
    paths = [os.path.join(hz.CACHE_DIR, "seqlp%s_%d_%s_%d.npy" % ((t,) + case)) for t in tags]
    if not all(os.path.isfile(p) for p in paths):
        continue
    lp = np.log(np.mean([np.exp(np.load(p).astype(np.float64)) for p in paths], axis=0))
    np.save(os.path.join(hz.CACHE_DIR, "seqlp%s_%d_%s_%d.npy" % ((out,) + case)), lp.astype(np.float32))
    n += 1
print("merged", n, "cases into", out)

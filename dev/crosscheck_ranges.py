"""REFERENCE ranges for solution/crosscheck.py from the medium/heavy development cases."""
import json, os, sys
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import crosscheck, pipeline, tagging

def one(case):
    p = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
    labels, pred = hz.run_case(case, p)
    F, _, blocks, _ = hz.attach_logits(case, hz.logo_models()[case[0]])
    return crosscheck.stats(F, pred, blocks, hz.simulate_phases(labels, blocks, case[2]))

if __name__ == "__main__":
    cases = [c for c in hz.CASES if c[1] in ("medium", "heavy")]
    with mp.Pool(3) as pool:
        rows = pool.map(one, cases)
    keys = sorted(set(k for r in rows for k in r))
    ref = {k: (round(min(r[k] for r in rows if k in r), 3), round(max(r[k] for r in rows if k in r), 3)) for k in keys}
    print(json.dumps(ref, indent=1))

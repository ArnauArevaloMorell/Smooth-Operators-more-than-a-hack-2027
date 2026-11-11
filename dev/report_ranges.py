"""Compute solution/report.py's REFERENCE ranges from the sample-game cases."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import report
p = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
rows = []
for case in hz.CASES:
    labels, pred = hz.run_case(case, p)
    F, _, blocks, _ = hz.attach_logits(case, hz.logo_models()[case[0]])
    rows.append(report.stats(F, pred, blocks, 25))
ref = {k: (round(float(min(r[k] for r in rows)), 3), round(float(max(r[k] for r in rows)), 3)) for k in rows[0]}
print(json.dumps(ref, indent=1))

"""How much does the Smart-Tagging phase weight risk if real phases are worse than simulated?

Re-scores the tuned parameters with phases simulated at lower accuracy, for several weights.
    python dev/phase_robustness.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
from solution import pipeline, tagging  # noqa: E402

p0 = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
for p_correct, p_tag in ((0.85, 0.6), (0.7, 0.6), (0.6, 0.4)):
    row = []
    for w in (0.0, 0.2, 0.4, p0.get("w_phase", 0.0)):
        p = dict(p0, w_phase=w)
        s = []
        for case in hz.CASES:
            F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
            T = len(labels)
            prior = pipeline.tag_prior_from_setpieces(tags, F, p, T, 25)
            ph = hz.simulate_phases(labels, blocks, case[2], p_tag=p_tag, p_correct=p_correct)
            prior = prior + tagging.phase_prior(ph, T, w)
            s.append(hz.score(labels, pipeline.predict(F, blocks, p, prior))["spell_f1_mean"])
        row.append("w {:.3f}: {:.4f}".format(w, np.mean(s)))
    print("phases correct {:.0%}, coverage {:.0%} | ".format(p_correct, p_tag) + " | ".join(row),
          flush=True)

"""Stress-test the set-piece prior weights against worse Smart Tagging than simulated."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import pipeline, tagging
p0 = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
variants = {"as simulated": dict(), "team wrong 20%, 20% false tags": dict(p_wrong=0.2, fp_rate=0.2),
            "team wrong 35%, 40% false tags": dict(p_wrong=0.35, fp_rate=0.4)}
for name, kw in variants.items():
    row = []
    for wt, wtt in ((0.0, 0.0), (1.5, 0.6), (p0["w_tag"], p0["w_tag_team"])):
        p = dict(p0, w_tag=wt, w_tag_team=wtt)
        s = []
        for case in hz.CASES:
            F, labels, blocks, _ = hz.attach_logits(case, hz.logo_models()[case[0]])
            tags = hz.simulate_tags(labels, blocks, case[2], **kw)
            prior = pipeline.tag_prior_from_setpieces(tags, F, p, len(labels), 25)
            prior = prior + tagging.phase_prior(hz.simulate_phases(labels, blocks, case[2]), len(labels), p["w_phase"])
            s.append(hz.score(labels, pipeline.predict(F, blocks, p, prior))["spell_f1_mean"])
        row.append("w_tag %.2f/%.2f: %.4f" % (wt, wtt, np.mean(s)))
    print("%-32s | %s" % (name, " | ".join(row)), flush=True)

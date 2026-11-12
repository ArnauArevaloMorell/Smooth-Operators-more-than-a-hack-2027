"""Held-out score of averaging several networks' (test-time swapped) outputs on the Metrica folds.

usage: exp_ens.py NAME=TAG1+TAG2+... [NAME=...]   (TAGs as saved by dev/exp_tta.py, e.g. FXBMT)
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import pipeline

P = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
specs = [a.split("=", 1) for a in sys.argv[1:]]
rows = {name: [] for name, _ in specs}
for case in [c for c in hz.CASES if c[0] in hz.METRICA_GAMES]:
    F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
    for name, tagstr in specs:
        lps = [np.load(os.path.join(hz.CACHE_DIR, "seqlp%s_%d_%s_%d.npy" % ((t,) + case)))
               for t in tagstr.split("+")]
        G = dict(F)
        G["seq_lp"] = np.log(np.mean([np.exp(l.astype(np.float64)) for l in lps], axis=0))
        prior = pipeline.tag_prior_from_setpieces(tags, G, P, len(labels), 25)
        rows[name].append((case, hz.score(labels, pipeline.predict(G, blocks, P, prior))["spell_f1_mean"]))
for name, _ in specs:
    r = rows[name]
    print("%-10s " % name + "  ".join("%s %.4f" % (s, np.mean([x[1] for x in r if x[0][1] == s]))
                                     for s in ("light", "medium", "heavy"))
          + "  ALL %.4f" % np.mean([x[1] for x in r]), flush=True)

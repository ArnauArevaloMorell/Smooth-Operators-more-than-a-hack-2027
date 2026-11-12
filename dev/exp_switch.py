"""Decoder scan on held-out ensemble outputs (Metrica folds): per-case spellF1 for each setting."""
import json, os, sys, itertools, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import pipeline

P0 = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
TAGS = os.environ.get("ENS", "FXBM2T+FXBM2s1T").split("+")
grid = json.loads(sys.argv[1])     # {"switch_mult": [...], "w_seq": [...], ...}
out = sys.argv[2]
keys = sorted(grid)
res = {}
GROUP = hz.DFL_GAMES if os.environ.get("GROUP") == "dfl" else hz.METRICA_GAMES
for case in [c for c in hz.CASES if c[0] in GROUP]:
    F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
    lps = [np.load(os.path.join(hz.CACHE_DIR, "seqlp%s_%d_%s_%d.npy" % ((t,) + case))) for t in TAGS]
    G = dict(F); G["seq_lp"] = np.log(np.mean([np.exp(l.astype(np.float64)) for l in lps], axis=0))
    for vals in itertools.product(*[grid[k] for k in keys]):
        p = dict(P0)
        for k, v in zip(keys, vals):
            if k.endswith("_mult"):
                p[k[:-5]] = P0[k[:-5]] * v
            else:
                p[k] = v
        prior = pipeline.tag_prior_from_setpieces(tags, G, p, len(labels), 25)
        pred = pipeline.predict(G, blocks, p, prior)
        r = hz.score(labels, pred)
        res[(case, vals)] = (r["spell_f1_mean"], r["n_pred_spells"], r["n_gt_spells"])
    print(case, flush=True)
pickle.dump((keys, grid, res), open(out, "wb"))

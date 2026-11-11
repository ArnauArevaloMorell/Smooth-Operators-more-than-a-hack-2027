"""Score sequence-network training variants through the full decoder on held-out Metrica cases.

    python dev/compare_seq_variants.py "" F X P      # tags given to train_seqnet.py --tag
"""
import json, os, sys
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import pipeline

P = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))


def one(args):
    case, tag = args
    F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
    F = dict(F)
    path = os.path.join(hz.CACHE_DIR, "seqlp%s_%d_%s_%d.npy" % ((tag,) + case))
    if not os.path.isfile(path):
        return case, tag, None, None, None
    F["seq_lp"] = np.load(path).astype(np.float64)
    prior = pipeline.tag_prior_from_setpieces(tags, F, P, len(labels), 25)
    pred = pipeline.predict(F, blocks, P, prior)
    lp = F["seq_lp"]
    al = labels < 2
    g = labels != 3
    team = (np.where(lp[:, 0] > lp[:, 1], 0, 1)[al] == labels[al]).mean()
    dead = ((np.argmax(lp, 1) == 2)[g] == (labels[g] == 2)).mean()
    return case, tag, hz.score(labels, pred)["spell_f1_mean"], team, dead


if __name__ == "__main__":
    tags = sys.argv[1:] or [""]
    cases = [c for c in hz.CASES if c[0] in hz.METRICA_GAMES]
    with mp.Pool(4) as pool:
        res = pool.map(one, [(c, t) for c in cases for t in tags])
    for t in tags:
        r = [x for x in res if x[1] == t and x[2] is not None]
        if not r:
            print("%-6s (no outputs yet)" % (t or "base"))
            continue
        line = " | ".join("%s %.4f" % (s, np.mean([x[2] for x in r if x[0][1] == s])) for s in ("light", "medium", "heavy"))
        print("%-6s spellF1 %s | mean %.4f | team acc %.4f | dead acc %.4f  (n=%d)" % (
            t or "base", line, np.mean([x[2] for x in r]), np.mean([x[3] for x in r]), np.mean([x[4] for x in r]), len(r)))

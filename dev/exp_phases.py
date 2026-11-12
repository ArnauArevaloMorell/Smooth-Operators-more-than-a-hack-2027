"""Multi-phase averaging: run the network at all 5 stride offsets (and on the team-swapped view),
average probabilities per frame. Held out on the Metrica folds with a variant's fold networks.

    python dev/exp_phases.py FXB
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz, sample_games, degrade
from solution import features, pipeline, seqnet

P = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
TAG = sys.argv[1] if len(sys.argv) > 1 else "FXB"


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


def frame_probs(net, X, blocks, T, offsets, swap):
    acc = np.zeros((T, 3))
    n = 0
    for o in offsets:
        for s, e in blocks:
            idx = np.arange(s + o, e, seqnet.STRIDE)
            p = softmax(seqnet.forward(net, X[idx]))
            if swap:
                p = p[:, [1, 0, 2]]
            for k in range(3):
                acc[s:e, k] += np.interp(np.arange(s, e), idx, p[:, k])
        n += 1
    return acc / n


rows = []
for case in [c for c in hz.CASES if c[0] in hz.METRICA_GAMES]:
    net = dict(np.load(os.path.join(hz.CACHE_DIR, "seqnet%s_fold_%s.npz" % (TAG, hz.fold_of(case[0])))))
    F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
    _, Xd, _, _, _, _, Xo = hz.case_data(*case)
    Xh, _ = hz.hazard_data(case)
    X = seqnet.design(Xd, Xo, Xh)
    g = sample_games.load_game(case[0])
    ball, xy = degrade.degrade(g, case[1], case[2])
    X_sw = pipeline.swapped_design(ball, xy, g.team, g.pitch, g.fps, blocks)
    T = len(labels)
    res = []
    for offsets in ((2,), (0, 1, 2, 3, 4)):
        pr = (frame_probs(net, X, blocks, T, offsets, False) + frame_probs(net, X_sw, blocks, T, offsets, True)) / 2
        G = dict(F)
        G["seq_lp"] = np.log(np.maximum(pr, 1e-12))
        G["seq_lp"][labels == 3] = np.nan
        prior = pipeline.tag_prior_from_setpieces(tags, G, P, T, 25)
        res.append(hz.score(labels, pipeline.predict(G, blocks, P, prior))["spell_f1_mean"])
    rows.append((case, res[0], res[1]))
    print(case, "1 phase %.4f | 5 phases %.4f" % tuple(res), flush=True)
for s in ("light", "medium", "heavy"):
    r = [x for x in rows if x[0][1] == s]
    print("%-6s 1 phase %.4f | 5 phases %.4f" % (s, np.mean([x[1] for x in r]), np.mean([x[2] for x in r])))
print("ALL    1 phase %.4f | 5 phases %.4f" % (np.mean([x[1] for x in rows]), np.mean([x[2] for x in rows])))

"""Test-time team-swap averaging for the sequence network, held out (base LOGO networks)."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
import train_seqnet as t
from solution import pipeline, seqnet

P = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
GROUP = hz.DFL_GAMES if (len(sys.argv) > 2 and sys.argv[2] == "dfl") else hz.METRICA_GAMES


def frame_lp(step_probs, blocks, T):
    out = np.full((T, 3), np.nan)
    for (s, e), pr in zip(blocks, step_probs):
        idx = np.arange(s + seqnet.STRIDE // 2, e, seqnet.STRIDE)
        lp = np.log(pr)
        for k in range(3):
            out[s:e, k] = np.interp(np.arange(s, e), idx, lp[:, k])
    return out


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)


rows = []
for case in [c for c in hz.CASES if c[0] in GROUP]:
    net = dict(np.load(os.path.join(hz.CACHE_DIR, "seqnet%s_fold_%s.npz" % (TAG, hz.fold_of(case[0])))))
    F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
    seqs = t.case_seq(case)
    seqs_f = t.seq_direct(*case, flip=True)
    normal = [softmax(seqnet.forward(net, x)) for x, _ in seqs]
    flipped = [softmax(seqnet.forward(net, x))[:, [1, 0, 2]] for x, _ in seqs_f]
    res = []
    for name, probs in (("plain", normal), ("tta", [(a + b) / 2 for a, b in zip(normal, flipped)])):
        G = dict(F)
        G["seq_lp"] = frame_lp(probs, blocks, len(labels))
        prior = pipeline.tag_prior_from_setpieces(tags, G, P, len(labels), 25)
        res.append(hz.score(labels, pipeline.predict(G, blocks, P, prior))["spell_f1_mean"])
    rows.append((case, res[0], res[1]))
    np.save(os.path.join(hz.CACHE_DIR, "seqlp%sT_%d_%s_%d.npy" % ((TAG,) + case)),
            frame_lp([(a + b) / 2 for a, b in zip(normal, flipped)], blocks, len(labels)).astype(np.float32))
    print(case, "plain %.4f  tta %.4f" % (res[0], res[1]), flush=True)
for s in ("light", "medium", "heavy"):
    r = [x for x in rows if x[0][1] == s]
    print("%-6s plain %.4f | tta %.4f" % (s, np.mean([x[1] for x in r]), np.mean([x[2] for x in r])))
print("ALL    plain %.4f | tta %.4f" % (np.mean([x[1] for x in rows]), np.mean([x[2] for x in rows])))

"""How far off are the boundaries of near-miss spells? (held-out Metrica, ensemble outputs)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
import possession_metrics as pm
from solution import pipeline

P = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
TAGS = os.environ.get("ENS", "FXBM2T+FXBM2s1T").split("+")
offs = []   # (iou, start_offset_s, end_offset_s, gt_len_s)
for case in [c for c in hz.CASES if c[0] in hz.METRICA_GAMES]:
    F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
    lps = [np.load(os.path.join(hz.CACHE_DIR, "seqlp%s_%d_%s_%d.npy" % ((t,) + case))) for t in TAGS]
    G = dict(F); G["seq_lp"] = np.log(np.mean([np.exp(l.astype(np.float64)) for l in lps], axis=0))
    prior = pipeline.tag_prior_from_setpieces(tags, G, P, len(labels), 25)
    pred = pipeline.predict(G, blocks, P, prior)
    runs = pm.valid_runs(labels)
    gs, ps = pm.possession_spells(labels, runs), pm.possession_spells(pred, runs)
    best = {}
    for iou, i, j in pm.spell_overlaps(gs, ps):
        if i not in best:
            best[i] = (iou, j)
    for i, (iou, j) in best.items():
        offs.append((iou, (ps[j][0] - gs[i][0]) / 25, (ps[j][1] - gs[i][1]) / 25, (gs[i][1] - gs[i][0]) / 25))
o = np.array(offs)
for lo, hi in ((0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
    m = (o[:, 0] >= lo) & (o[:, 0] < hi)
    a = np.abs(o[m, 1:3])
    print("IoU %.1f-%.1f n=%4d | |start off| med %.2fs p75 %.2fs | |end off| med %.2fs p75 %.2fs | both<1s %.2f | gt len med %.1fs" % (
        lo, hi, m.sum(), np.median(a[:, 0]), np.percentile(a[:, 0], 75), np.median(a[:, 1]),
        np.percentile(a[:, 1], 75), np.mean((a < 1).all(axis=1)), np.median(o[m, 3])))

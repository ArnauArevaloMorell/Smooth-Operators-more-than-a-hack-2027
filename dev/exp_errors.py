"""Where do held-out spell errors come from? (Metrica folds, an ensemble of saved network outputs.)"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
import possession_metrics as pm
from solution import pipeline

P = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
tags_ = sys.argv[1].split("+") if len(sys.argv) > 1 else ["FXBM2T", "FXBM2s1T"]
agg = {"gt": 0, "pred": 0, "best_iou": [], "pred_best_iou": [], "gt_len": [], "pred_len": []}
for case in [c for c in hz.CASES if c[0] in hz.METRICA_GAMES]:
    F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
    lps = [np.load(os.path.join(hz.CACHE_DIR, "seqlp%s_%d_%s_%d.npy" % ((t,) + case))) for t in tags_]
    G = dict(F); G["seq_lp"] = np.log(np.mean([np.exp(l.astype(np.float64)) for l in lps], axis=0))
    prior = pipeline.tag_prior_from_setpieces(tags, G, P, len(labels), 25)
    pred = pipeline.predict(G, blocks, P, prior)
    runs = pm.valid_runs(labels)
    gs, ps = pm.possession_spells(labels, runs), pm.possession_spells(pred, runs)
    ov = pm.spell_overlaps(gs, ps)
    bg = np.zeros(len(gs)); bp = np.zeros(len(ps))
    for iou, i, j in ov:
        bg[i] = max(bg[i], iou); bp[j] = max(bp[j], iou)
    agg["gt"] += len(gs); agg["pred"] += len(ps)
    agg["best_iou"] += list(bg); agg["pred_best_iou"] += list(bp)
    agg["gt_len"] += [(b - a) / 25 for a, b, _ in gs]; agg["pred_len"] += [(b - a) / 25 for a, b, _ in ps]
bg = np.array(agg["best_iou"]); bp = np.array(agg["pred_best_iou"]); gl = np.array(agg["gt_len"]); pl = np.array(agg["pred_len"])
print("true spells %d  predicted %d" % (agg["gt"], agg["pred"]))
print("true spells by best IoU: none %.3f  <0.3 %.3f  0.3-0.5 %.3f  0.5-0.7 %.3f  >=0.7 %.3f" % (
    np.mean(bg == 0), np.mean((bg > 0) & (bg < .3)), np.mean((bg >= .3) & (bg < .5)), np.mean((bg >= .5) & (bg < .7)), np.mean(bg >= .7)))
print("pred spells by best IoU: none %.3f  <0.3 %.3f  0.3-0.5 %.3f  0.5-0.7 %.3f  >=0.7 %.3f" % (
    np.mean(bp == 0), np.mean((bp > 0) & (bp < .3)), np.mean((bp >= .3) & (bp < .5)), np.mean((bp >= .5) & (bp < .7)), np.mean(bp >= .7)))
for lo, hi in ((0, 2), (2, 4), (4, 8), (8, 15), (15, 999)):
    m = (gl >= lo) & (gl < hi); n = (pl >= lo) & (pl < hi)
    print("len %3d-%3ds: true n=%4d recall@0.5 %.3f | pred n=%4d precision@0.5 %.3f" % (
        lo, hi, m.sum(), np.mean(bg[m] >= .5) if m.any() else 0, n.sum(), np.mean(bp[n] >= .5) if n.any() else 0))

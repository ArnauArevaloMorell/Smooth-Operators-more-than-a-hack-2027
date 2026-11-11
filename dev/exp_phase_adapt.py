"""Phase accuracy vs the label-free agreement statistic, and the best phase weight for each.

For simulated phase accuracies p_correct (and coverage p_tag), measure (a) the agreement of
phases with clear control episodes -- computable on BAR-ATH without labels -- and (b) spellF1 for
a range of phase weights. Output: the mapping agreement -> best weight.
"""
import json, os, sys
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import features, pipeline, tagging

WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.9, 1.3)

def one(args):
    case, p_correct, p_tag = args
    p = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
    F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
    T = len(labels)
    phases = hz.simulate_phases(labels, blocks, case[2], p_tag=p_tag, p_correct=p_correct)
    ep = features.control_episodes(F, blocks)
    agr, n = tagging.phase_agreement(phases, ep, T)
    base = pipeline.tag_prior_from_setpieces(tags, F, p, T, 25)
    scores = []
    for w in WEIGHTS:
        prior = base + tagging.phase_prior(phases, T, w)
        scores.append(hz.score(labels, pipeline.predict(F, blocks, dict(p, w_phase=w), prior))["spell_f1_mean"])
    return case, p_correct, p_tag, agr, scores

if __name__ == "__main__":
    hz.logo_models()
    jobs = [(c, pc, pt) for pc in (0.6, 0.75, 0.85, 0.95) for pt in (0.6, 0.95) for c in hz.CASES if c[1] != "light"]
    with mp.Pool(4) as pool:
        res = pool.map(one, jobs)
    for pc in (0.6, 0.75, 0.85, 0.95):
        for pt in (0.6, 0.95):
            r = [x for x in res if x[1] == pc and x[2] == pt]
            s = np.mean([x[4] for x in r], axis=0)
            print("p_correct %.2f coverage %.2f | agreement %.3f | %s | best w %.1f" % (
                pc, pt, np.mean([x[3] for x in r]), " ".join("w%.1f:%.4f" % (w, v) for w, v in zip(WEIGHTS, s)), WEIGHTS[int(np.argmax(s))]), flush=True)

"""Experiment: learned switch hazard. LOGO-trained hazard MLP, scan its decoder weight."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import features, mlp, pipeline
pm = hz.pm
RADIUS = 5

def hazard_data(case):
    path = os.path.join(hz.CACHE_DIR, "haz_%d_%s_%d.npy" % case)
    F, Xd, Xs, labels, blocks, tags, Xo = hz.case_data(*case)
    if os.path.isfile(path):
        Xh = np.load(path)
    else:
        Xh = features.hazard_design(F, 25, blocks).astype(np.float32); np.save(path, Xh)
    y = np.zeros(len(labels))
    for f, a, b in pm.transitions(labels, pm.valid_runs(labels)):
        if a < 2 and b < 2:
            y[max(0, f - RADIUS):f + RADIUS + 1] = 1
    return Xh, y, labels

def logo_hazard():
    path = os.path.join(hz.CACHE_DIR, "logo_hazard.json")
    if os.path.isfile(path):
        return {int(k): v for k, v in json.load(open(path)).items()}
    D = {c: hazard_data(c) for c in hz.CASES}
    out = {}
    for g in hz.GAMES:
        X, Y = [], []
        for c in hz.CASES:
            if c[0] == g: continue
            Xh, y, labels = D[c]
            idx = np.flatnonzero(labels < 2)[::2]
            X.append(Xh[idx]); Y.append(y[idx])
        out[g] = mlp.fit(np.vstack(X), np.concatenate(Y), hidden=32, epochs=6, seed=0)
    json.dump({str(k): v for k, v in out.items()}, open(path, "w"))
    return out

if __name__ == "__main__":
    H = logo_hazard()
    p0 = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
    cache = {}
    for case in hz.CASES:
        F, labels, blocks, tags = hz.attach_logits(case, hz.logo_models()[case[0]])
        F = dict(F); F["hazard_logit"] = mlp.logit(H[case[0]], hazard_data(case)[0])
        cache[case] = (F, labels, blocks, tags)
    def run(p):
        s = []
        for case, (F, labels, blocks, tags) in cache.items():
            prior = pipeline.tag_prior_from_setpieces(tags, F, p, len(labels), 25)
            s.append(hz.score(labels, pipeline.predict(F, blocks, p, prior))["spell_f1_mean"])
        return np.mean(s), s
    base, _ = run(p0); print("constant switch cost: %.4f" % base, flush=True)
    for wh in (0.2, 0.4, 0.7):
        for hb in (-2.0, 0.0, 2.0):
            for sw in (1.0, 2.0):
                q = dict(p0, w_hazard=wh, hazard_bias=hb, switch=p0["switch"] * sw)
                m, s = run(q)
                print("w_h %.1f bias %+.0f switch x%.0f: %.4f  (light/med/heavy g1 %s)" % (wh, hb, sw, m, " ".join("%.3f" % x for x in s[:3])), flush=True)

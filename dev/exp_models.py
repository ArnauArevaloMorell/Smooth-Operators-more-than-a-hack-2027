"""LOGO comparison of per-frame models: logistic vs MLP, dead-ball and ownership."""
import sys, os, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
from solution import features, logreg, mlp

def ctrl_cols(F, blocks, T):
    cH, cA = features.control_kernel(F, 1.5)
    m = np.maximum(cH, cA)
    return np.column_stack([features._block_apply(lambda v: features.window_mean_dir(v, a, b), m, blocks, T)
                            for a, b in ((25, 25), (50, 0), (0, 50), (125, 125))])

def data(case, sub=3):
    F, Xd, Xs, labels, blocks, _ = hz.case_data(*case)
    T = len(labels)
    Xd2 = np.column_stack([Xd, np.nan_to_num(ctrl_cols(F, blocks, T))])
    Xo = features.owner_design(F, 25, blocks)
    g = np.flatnonzero(labels != 3)[::sub]
    a = np.flatnonzero((labels < 2))[::sub]
    return (Xd2[g], (labels[g] == 2).astype(float), Xs[a], Xo[a], (labels[a] == 0).astype(float))

def ll(z, y):
    p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)), np.mean((p > .5) == y)

D = {c: data(c) for c in hz.CASES}
for g in hz.GAMES:
    tr = [D[c] for c in hz.CASES if c[0] != g]; te = [D[c] for c in hz.CASES if c[0] == g]
    cat = lambda L, i: np.concatenate([x[i] for x in L])
    t = time.time()
    res = []
    nd = tr[0][0].shape[1] - 4
    m = logreg.fit(cat(tr, 0)[:, :nd], cat(tr, 1), l2=10); res.append(("dead LR", ll(logreg.logit(m, cat(te, 0)[:, :nd]), cat(te, 1))))
    m = mlp.fit(cat(tr, 0), cat(tr, 1)); res.append(("dead MLP+ctrl", ll(mlp.logit(m, cat(te, 0)), cat(te, 1))))
    m = logreg.fit(cat(tr, 2), cat(tr, 4), l2=10); res.append(("shape LR", ll(logreg.logit(m, cat(te, 2)), cat(te, 4))))
    m = logreg.fit(cat(tr, 3), cat(tr, 4), l2=10); res.append(("owner LR", ll(logreg.logit(m, cat(te, 3)), cat(te, 4))))
    m = mlp.fit(cat(tr, 3), cat(tr, 4)); res.append(("owner MLP", ll(mlp.logit(m, cat(te, 3)), cat(te, 4))))
    print("held-out game", g, " ".join("| %s ll %.3f acc %.3f" % (n, a, b) for n, (a, b) in res), "%.0fs" % (time.time() - t), flush=True)

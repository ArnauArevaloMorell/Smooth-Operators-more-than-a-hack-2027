"""Richer ball-free shape features for ownership during ball gaps: LOGO accuracy on gap frames."""
import os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz, sample_games, degrade
from solution import features, logreg, mlp

def shape2_raw(ball, xy, team, pitch, fps, blocks):
    L, W = pitch
    P = xy.astype(np.float64) * [L, W]
    T = len(P)
    home = np.asarray(team) == 0
    signs = features.attack_sign(P, team, blocks)
    cols = {}
    def put(name, s, e, v):
        cols.setdefault(name, np.full(T, np.nan))[s:e] = v
    import warnings
    warnings.simplefilter("ignore")
    for (s, e), sign in zip(blocks, signs):
        Pb = P[s:e]
        V = features._velocity(Pb, fps)
        for k, m, sg in (("H", home, sign), ("A", ~home, -sign)):
            x = Pb[:, m, 0] * sg + (L if sg < 0 else 0)      # x' toward this team's attacking goal
            y = Pb[:, m, 1]
            vx = V[:, m, 0] * sg
            cx = np.nanmean(x, 1); cy = np.nanmean(y, 1)
            put("cx" + k, s, e, cx / L)
            put("vx" + k, s, e, np.nanmean(vx, 1))
            put("sx" + k, s, e, np.nanstd(x, 1))
            put("sy" + k, s, e, np.nanstd(y, 1))
            put("comp" + k, s, e, np.nanmean(np.hypot(x - cx[:, None], y - cy[:, None]), 1))
            xs = np.sort(np.where(np.isfinite(x), x, np.nan), 1)
            put("max" + k, s, e, np.nanmax(x, 1) / L)
            put("deep" + k, s, e, np.nanpercentile(x, 15, axis=1) / L)
            put("opp" + k, s, e, np.nansum(x > L / 2, 1))
            put("fwdv" + k, s, e, np.nanmean(np.where(x > np.nanmedian(x, 1)[:, None], vx, np.nan), 1))
    names = sorted(set(n[:-1] for n in cols))
    out = []
    for n in names:
        d = cols[n + "H"] - cols[n + "A"]
        out += [d, cols[n + "H"], cols[n + "A"]]
    return out

def design(raw, fps, blocks, T):
    cols = []
    for v in raw:
        for b, a in ((25, 25), (100, 100), (50, 0), (0, 50), (250, 250)):
            cols.append(features._block_apply(lambda u: features.window_mean_dir(u, b, a), v, blocks, T))
    return np.nan_to_num(np.column_stack(cols))

def data(case):
    path = os.path.join(hz.CACHE_DIR, "shape2_%d_%s_%d.npy" % case)
    F, Xd, Xs, labels, blocks, tags, Xo = hz.case_data(*case)
    if os.path.isfile(path):
        X2 = np.load(path)
    else:
        g = sample_games.load_game(case[0]); ball, xy = degrade.degrade(g, case[1], case[2])
        X2 = design(shape2_raw(ball, xy, g.team, g.pitch, 25, blocks), 25, blocks, len(labels)).astype(np.float32)
        np.save(path, X2)
    gap = (labels < 2) & ~np.nan_to_num(F["vis"]).astype(bool)
    alive = labels < 2
    return Xs, X2, labels, gap, alive

D = {c: data(c) for c in hz.CASES}
def acc(z, y): return np.mean((z > 0) == y)
for g in hz.GAMES:
    tr = [D[c] for c in hz.CASES if c[0] != g]; te = [D[c] for c in hz.CASES if c[0] == g]
    def stack(L, i, sub=4):
        return np.vstack([x[i][np.flatnonzero(x[4])[::sub]] for x in L]).astype(np.float64), \
               np.concatenate([(x[2][np.flatnonzero(x[4])[::sub]] == 0) for x in L]).astype(float)
    Xs_tr, y_tr = stack(tr, 0); X2_tr, _ = stack(tr, 1)
    gaps = lambda L, i: np.vstack([x[i][x[3]] for x in L]).astype(np.float64)
    y_te = np.concatenate([(x[2][x[3]] == 0) for x in te])
    m0 = logreg.fit(Xs_tr, y_tr, l2=10); a0 = acc(logreg.logit(m0, gaps(te, 0)), y_te)
    m1 = logreg.fit(X2_tr, y_tr, l2=10); a1 = acc(logreg.logit(m1, gaps(te, 1)), y_te)
    m2 = mlp.fit(X2_tr, y_tr, hidden=32, epochs=6); a2 = acc(mlp.logit(m2, gaps(te, 1)), y_te)
    print("held-out game %d  gap-frame accuracy: current shape LR %.3f | rich shape LR %.3f | rich shape MLP %.3f" % (g, a0, a1, a2), flush=True)

"""Shared machinery for training and tuning on the sample games.

A 'case' is one sample game under one degradation setting and seed. For each case we cache the
features and design matrices once; training and tuning then only re-run the cheap parts.
Scores come from evaluation/possession_metrics.py -- the official scorer's own code.
"""

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
import sample_games  # noqa: E402
import degrade  # noqa: E402
from solution import features, logreg, pipeline  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "data", "cache", "dev")
GAMES = (1, 2, 3)
CASES = [(g, s, seed) for g in GAMES for s, seed in (("light", 0), ("medium", 1), ("heavy", 2))]


def blocks_of(labels):
    return [(int(a), int(b)) for a, b in pm.valid_runs(labels)]


def simulate_tags(labels, blocks, seed, fps=25, p_tag=0.55, p_team=0.6, p_wrong=0.05,
                  fp_rate=0.05):
    """Smart-Tagging-like set-piece windows around true restarts (sample games have no tags).

    Statistics follow what DATA.md and 01_smart_tagging.ipynb say about the real file: windows of
    a few seconds that open ~3.5 s before the restart; only some restarts tagged (59 set pieces in
    BAR-ATH for roughly 90+ real restarts); team missing on many (every throw-in is N/A).
    """
    rng = np.random.RandomState(1000 + seed)
    tags = []
    for s, e in blocks:
        lab = labels[s:e]
        for i in np.flatnonzero((lab[1:] != 2) & (lab[:-1] == 2)) + 1:
            if rng.rand() > p_tag:
                continue
            r = s + int(i)
            a = r - int(rng.uniform(2.5, 4.5) * fps)
            b = r + int(rng.uniform(1.0, 3.0) * fps)
            team = int(lab[i]) if rng.rand() < p_team else None
            if team is not None and rng.rand() < p_wrong:
                team = 1 - team
            tags.append((a, b, team))
        n_fp = rng.binomial(len(tags) + 1, fp_rate)
        for _ in range(n_fp):
            r = rng.randint(s, e)
            tags.append((r - 80, r + 50, None))
    return tags


def simulate_phases(labels, blocks, seed, fps=25, p_tag=0.6, p_correct=0.85):
    """Smart-Tagging-like team phases: noisy sub-intervals of real possessions, sometimes wrong.

    Deliberately pessimistic about boundaries (jittered by up to 2 s either way, so a phase can
    spill over a change of possession) because the real tags are 'phases, not possessions'.
    """
    rng = np.random.RandomState(2000 + seed)
    out = []
    for a, b, k in pm.possession_spells(labels, [(s, e) for s, e in blocks]):
        if b - a < 3 * fps or rng.rand() > p_tag:
            continue
        s = a + int(rng.uniform(-2, 2) * fps)
        e = b + int(rng.uniform(-2, 2) * fps)
        side = int(k) if rng.rand() < p_correct else 1 - int(k)
        if e > s:
            out.append((s, e, side))
    return out


def case_data(game_n, setting, seed):
    """(F, Xd, Xs, labels, blocks, tags) for a case, cached on disk."""
    if not os.path.isdir(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    path = os.path.join(CACHE_DIR, "case_%d_%s_%d.pkl" % (game_n, setting, seed))
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    g = sample_games.load_game(game_n)
    ball, xy = degrade.degrade(g, setting, seed)
    blocks = blocks_of(g.labels)
    F = features.compute(ball, xy, g.team, g.pitch, g.fps, blocks)
    Xd, _ = features.dead_design(F, g.pitch, g.fps, blocks)
    Xs, _ = features.shape_design(F, g.fps, blocks)
    Xo = features.owner_design(F, g.fps, blocks)
    tags = simulate_tags(g.labels, blocks, seed)
    out = (F, Xd.astype(np.float32), Xs.astype(np.float32), g.labels, blocks, tags,
           Xo.astype(np.float32))
    with open(path, "wb") as handle:
        pickle.dump(out, handle, protocol=4)
    return out


def graded_mask(labels):
    return labels != pm.X


def train(cases, subsample=3):
    """Fit the dead (MLP), owner (logistic) and shape (logistic) models on the given cases."""
    Xd_all, yd_all, Xs_all, ys_all, Xo_all = [], [], [], [], []
    for c in cases:
        F, Xd, Xs, labels, blocks, _, Xo = case_data(*c)
        m = graded_mask(labels)
        idx = np.flatnonzero(m)[::subsample]
        Xd_all.append(Xd[idx])
        yd_all.append((labels[idx] == pm.D).astype(float))
        alive = np.flatnonzero(m & (labels < 2))[::subsample]
        Xs_all.append(Xs[alive])
        Xo_all.append(Xo[alive])
        ys_all.append((labels[alive] == pm.H).astype(float))
    from solution import mlp
    dead = mlp.fit(np.vstack(Xd_all), np.concatenate(yd_all), hidden=32, epochs=6, seed=0)
    ys = np.concatenate(ys_all)
    shape = logreg.fit(np.vstack(Xs_all).astype(np.float64), ys, l2=10.0)
    owner = logreg.fit(np.vstack(Xo_all).astype(np.float64), ys, l2=10.0)
    return {"dead": dead, "shape": shape, "owner": owner}


_LOGITS = {}


def attach_logits(case, models):
    """Features + the three logits; memoised per (case, model set) within a process."""
    key = (case, id(models))
    if key not in _LOGITS:
        F, Xd, Xs, labels, blocks, tags, Xo = case_data(*case)
        F = dict(F)
        F["dead_logit"] = logreg.logit(models["dead"], Xd.astype(np.float64))
        F["shape_logit"] = logreg.logit(models["shape"], Xs.astype(np.float64))
        F["owner_logit"] = logreg.logit(models["owner"], Xo.astype(np.float64))
        F["episode"] = features.control_episodes(F, blocks).astype(np.float64)
        _LOGITS[key] = (F, labels, blocks, tags)
    return _LOGITS[key]


_LOGO = {}
LOGO_PATH = os.path.join(CACHE_DIR, "logo_models.pkl")


def logo_models():
    """Leave-one-game-out models: the model applied to game g never saw game g. Cached on disk."""
    if not _LOGO:
        if os.path.isfile(LOGO_PATH):
            with open(LOGO_PATH, "rb") as handle:
                _LOGO.update(pickle.load(handle))
        else:
            for g in GAMES:
                _LOGO[g] = train([c for c in CASES if c[0] != g])
            with open(LOGO_PATH, "wb") as handle:
                pickle.dump(dict(_LOGO), handle)
    return _LOGO


def score(labels, pred):
    r = pm.evaluate(labels, pred)
    return r


def run_case(case, p, models=None, use_tags=True):
    models = models or logo_models()[case[0]]
    F, labels, blocks, tags = attach_logits(case, models)
    T = len(labels)
    prior = None
    if use_tags:
        from solution import tagging
        prior = pipeline.tag_prior_from_setpieces(tags, F, p, T, 25)
        if p.get("w_phase", 0.0):
            prior = prior + tagging.phase_prior(simulate_phases(labels, blocks, case[2]), T,
                                                p["w_phase"])
    pred = pipeline.predict(F, blocks, p, prior)
    return labels, pred

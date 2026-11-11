"""Tracking (+ optional Smart Tagging) -> per-frame H/A/D. The whole method, in one place.

    features.compute        per-frame physics: who is near the ball, is play alive, who attacks
    logreg (dead model)     P(dead ball) from multi-scale visibility / location / tempo cues
    logreg (shape model)    P(home attacking) from collective movement only -- for ball gaps
    emissions               control evidence + the two logits (+ Smart Tagging restarts)
    decode.viterbi          one global segmentation with an L0 cost per possession change

Every constant used here lives in params.json, and every one of them was chosen on Metrica's
public sample games (other matches), never on BAR-ATH -- see dev/tune.py.
"""

import json
import os

import numpy as np

from . import decode, features, logreg, seqnet

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "params.json")
MODELS_PATH = os.path.join(HERE, "models.json")


def load_params(path=PARAMS_PATH):
    with open(path) as handle:
        return json.load(handle)


def prepare(ball, xy, team, pitch, fps, blocks, models):
    """Everything that does not depend on decoder params: features and the two logits."""
    F = features.compute(ball, xy, team, pitch, fps, blocks)
    Xd, _ = features.dead_design(F, pitch, fps, blocks)
    Xs, _ = features.shape_design(F, fps, blocks)
    Xo = features.owner_design(F, fps, blocks)
    F["dead_logit"] = logreg.logit(models["dead"], Xd)
    F["shape_logit"] = logreg.logit(models["shape"], Xs)
    F["owner_logit"] = logreg.logit(models["owner"], Xo)
    F["episode"] = features.control_episodes(F, blocks).astype(np.float64)
    Xh = features.hazard_design(F, fps, blocks)
    if "hazard" in models:
        F["hazard_logit"] = logreg.logit(models["hazard"], Xh)
    net = seqnet.load()
    if net is not None:
        F["seq_lp"] = seqnet.log_probs(net, seqnet.design(Xd, Xo, Xh), blocks)
    return F


def control(F, p):
    """Per-team control evidence in [0, 1]: close to the ball AND moving with it."""
    out = []
    for k in ("H", "A"):
        d = F["d" + k]
        r = F["r" + k]
        c = np.exp(-0.5 * (d / p["sigma"]) ** 2)
        rel = np.where(np.isfinite(r), np.exp(-np.maximum(0.0, r - p["v0"]) / p["vs"]), 1.0)
        c = np.where(np.isfinite(c), c * rel, 0.0)
        out.append(c)
    return out


def emissions(F, p, tag_prior=None):
    cH, cA = control(F, p)
    # Only the team that is clearly closer gets credit, so a duel is not evidence for either.
    z = p["w_control"] * (cH - cA)
    z = z + p["w_shape"] * np.clip(np.nan_to_num(F["shape_logit"]), -4, 4)
    z = z + p.get("w_owner", 0.0) * np.clip(np.nan_to_num(F["owner_logit"]), -8, 8)
    # A clear control episode (98% precise on the sample games) is close to decisive.
    z = z + p.get("w_episode", 0.0) * F["episode"]
    T = len(z)
    E = np.zeros((T, 4))
    E[:, decode.H] = 0.5 * z
    E[:, decode.A] = -0.5 * z
    dead = p["w_dead"] * (np.clip(np.nan_to_num(F["dead_logit"]), -8, 8) + p["dead_bias"])
    # Being clearly on the ball is evidence that play is alive, too.
    alive_bonus = p["w_alive_ctrl"] * np.maximum(cH, cA)
    E[:, decode.DH] = dead - alive_bonus
    E[:, decode.DA] = dead - alive_bonus
    # The temporal network's per-frame log-probabilities (home, away, dead), if shipped.
    if p.get("w_seq", 0.0) and "seq_lp" in F:
        lp = np.nan_to_num(F["seq_lp"], nan=np.log(1.0 / 3))
        E[:, decode.H] += p["w_seq"] * lp[:, 0]
        E[:, decode.A] += p["w_seq"] * lp[:, 1]
        E[:, decode.DH] += p["w_seq"] * lp[:, 2]
        E[:, decode.DA] += p["w_seq"] * lp[:, 2]
    if tag_prior is not None:
        E += tag_prior
    return E


def switch_costs(F, p):
    """Per-frame cost of a possession changing hands, or None for the constant cost.

    cost_t = switch * exp(-w_hazard * clip(hazard_logit_t + hazard_bias)): cheap where the hazard
    model sees a contest or a touch by the other side, dearer where nothing is happening.
    """
    if not p.get("w_hazard") or "hazard_logit" not in F:
        return None
    z = np.clip(np.nan_to_num(F["hazard_logit"]) + p.get("hazard_bias", 0.0), -3.0, 6.0)
    return p["switch"] * np.exp(-p["w_hazard"] * z)


def decode_blocks(E, blocks, p, T, sw=None):
    """Viterbi per graded block; everything outside the blocks is X (code 3)."""
    C = decode.transition_matrix(p)
    labels = np.full(T, 3, dtype=np.int8)
    for s, e in blocks:
        path = decode.viterbi(E[s:e], C, switch_cost=None if sw is None else sw[s:e])
        labels[s:e] = decode.to_labels(path)
    return labels


def predict(F, blocks, p, tag_prior=None):
    T = len(F["vis"])
    E = emissions(F, p, tag_prior)
    return decode_blocks(E, blocks, p, T, switch_costs(F, p))


# -- Smart Tagging as a prior ------------------------------------------------------------------

def tag_prior_from_setpieces(set_pieces, F, p, T, fps):
    """Emission bonuses from tagged set pieces: dead before the restart, alive after it.

    A tag is a window of a few seconds that *contains* a restart (01_smart_tagging.ipynb). The
    restart instant is placed where the ball first accelerates out of standstill inside the window;
    if the ball is not seen, at the window's centre. Before it the ball is dead, after it the
    restarting team has it -- when the tag names a team (throw-ins never do).

    set_pieces: [(start_frame, end_frame, team_code_or_None)]
    """
    prior = np.zeros((T, 4))
    if not set_pieces or p.get("w_tag", 0) == 0:
        return prior
    bs = np.nan_to_num(F["ball_speed"], nan=0.0)
    vis = np.nan_to_num(F["vis"]).astype(bool)
    pre = int(p.get("tag_pre_s", 2.0) * fps)
    post = int(p.get("tag_post_s", 1.0) * fps)
    for s, e, team in set_pieces:
        s = max(0, s)
        e = min(T, e)
        if e <= s:
            continue
        seg = np.flatnonzero(vis[s:e] & (bs[s:e] > p.get("tag_kick_speed", 6.0)))
        r = s + int(seg[0]) if len(seg) else (s + e) // 2
        lo = max(0, r - pre)
        prior[lo:r, decode.DH] += p["w_tag"]
        prior[lo:r, decode.DA] += p["w_tag"]
        hi = min(T, r + post)
        prior[r:hi, decode.H] += p["w_tag"]
        prior[r:hi, decode.A] += p["w_tag"]
        if team is not None:
            other = decode.A if team == 0 else decode.H
            prior[r:hi, other] -= p["w_tag_team"]
    return prior


# -- Smart Tagging phase weight, set from the phases' measured accuracy on this match -----------

def phase_weight(agreement, p):
    """Phase weight from the label-free agreement of phases with clear control episodes.

    The map (p['phase_weight_map'], [[agreement, weight], ...], interpolated linearly, flat
    outside) was measured on the sample games by simulating phases of known accuracy
    (dev/exp_phase_adapt.py): the agreement statistic tracks the true accuracy closely, and the
    better the phases the more weight they earn. The map sits a little below the optimum and is
    capped, because Metrica's phases come from the same tracking and their errors may be
    correlated with ours. Falls back to the fixed p['w_phase'] when there is nothing to measure.
    """
    table = p.get("phase_weight_map")
    if not table or agreement is None:
        return p.get("w_phase", 0.0)
    xs = [row[0] for row in table]
    ys = [row[1] for row in table]
    return float(np.interp(agreement, xs, ys))

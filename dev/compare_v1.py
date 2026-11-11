"""Score the team's rule-based v1 (origin/main solution/03_possession_v1.py) on the same 18
degraded sample-game cases as this pipeline, with the same simulated Smart Tagging.

    python dev/compare_v1.py <dir containing possession_v1.py and common.py>
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz, sample_games, degrade
sys.path.insert(0, sys.argv[1])
import common as C
import possession_v1 as V1

def run_v1(case):
    g = sample_games.load_game(case[0])
    ball_n, xy_n = degrade.degrade(g, case[1], case[2])
    L, W = g.pitch
    C.PITCH_L, C.PITCH_W = L, W
    ball = ball_n.astype(np.float64) * [L, W]
    players = xy_n.astype(np.float64) * [L, W]
    labels = g.labels
    graded = labels != 3
    p = dict(V1.PARAMS, second_half_kickoff=0)          # no BAR-ATH pre-kickoff stretch here
    filled = V1.fill_short_gaps(ball, p["gap_fill"])
    dead, _ = V1.dead_ball_mask(ball, filled, graded, p)
    ev = V1.tracking_evidence(players, filled, g.team, p)
    # same simulated tags as our pipeline gets: set pieces with their (noisy) team, phases
    blocks = hz.blocks_of(labels)
    events = []
    for a, b, t in hz.simulate_tags(labels, blocks, case[2]):
        events.append({"code": "SET PIECES", "start": a, "end": b,
                       "tags": {"Team": {0: "Barcelona", 1: "Athletic", None: "N/A"}[t]}})
    for a, b, t in hz.simulate_phases(labels, blocks, case[2]):
        events.append({"code": "BUILD UP", "start": a, "end": b,
                       "tags": {"Team": {0: "Barcelona", 1: "Athletic"}[t]}})
    prior, _ = V1.tagging_prior(events, len(labels), p)
    total = np.clip(ev + prior, -1, 1)
    out = np.full(len(labels), 3, dtype=np.int8)
    out[graded] = 2
    for s, e in C.runs(graded & ~dead):
        out[s:e] = V1.viterbi(total[s:e], p["switch_cost"])
    return labels, out

rows = {}
for case in hz.CASES:
    labels, pred = run_v1(case)
    r = hz.score(labels, pred)
    rows.setdefault(case[1], []).append((r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    print("  game %d %-6s seed %2d  spellF1 %.4f  macroF1 %.4f  spells %d/%d" % (case + (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"])), flush=True)
allr = [x for v in rows.values() for x in v]
for k, v in sorted(rows.items()):
    print("  %-7s spellF1 %.4f  macroF1 %.4f" % (k, np.mean([x[0] for x in v]), np.mean([x[1] for x in v])))
print("  ALL     spellF1 %.4f  macroF1 %.4f" % (np.mean([x[0] for x in allr]), np.mean([x[1] for x in allr])))

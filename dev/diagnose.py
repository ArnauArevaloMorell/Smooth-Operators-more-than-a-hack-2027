"""Where do the errors come from? Boundary recall by type, and spell matching detail.

    python dev/diagnose.py [game setting seed]
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz
pm = hz.pm

def boundaries(codes, runs):
    return pm.transitions(codes, runs)

def report(case, p):
    labels, pred = hz.run_case(case, p)
    runs = pm.valid_runs(labels)
    gt_t, pr_t = boundaries(labels, runs), boundaries(pred, runs)
    kinds = {"team switch": lambda f, t: f < 2 and t < 2, "alive->dead": lambda f, t: f < 2 and t == 2,
             "restart": lambda f, t: f == 2 and t < 2}
    print("case", case)
    for name, fn in kinds.items():
        g = [x for x in gt_t if fn(x[1], x[2])]; q = [x for x in pr_t if fn(x[1], x[2])]
        for tol in (12, 50):
            pairs, missed, spur = pm.match_transitions(g, q, tol, typed="pair")
            print("  {:<12} tol {:>2}  gt {:>4} pred {:>4}  recall {:.2f} precision {:.2f}".format(
                name, tol, len(g), len(q), len(pairs) / max(1, len(g)), len(pairs) / max(1, len(q))))
    r = hz.score(labels, pred)
    for th, v in r["spell_f1"].items():
        print("  IoU {:.1f}  P {:.2f} R {:.2f} F1 {:.3f}".format(th, v["precision"], v["recall"], v["f1"]))
    # gt spell length vs matched at 0.5
    gs = pm.possession_spells(labels, runs); ps = pm.possession_spells(pred, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
    ln = np.array([b - a for a, b, _ in gs]) / 25.0
    for lo, hi in ((0, 2), (2, 5), (5, 10), (10, 20), (20, 999)):
        m = (ln >= lo) & (ln < hi)
        print("  gt spells {:>3}-{:<3}s  n {:>4}  matched@0.5 {:.2f}".format(lo, hi, m.sum(), flags[m].mean() if m.any() else 0))
    D = labels == 2; Pd = pred == 2; g = labels != 3
    print("  dead frames: recall {:.3f} precision {:.3f}".format((D & Pd & g).sum() / (D & g).sum(), (D & Pd & g).sum() / (Pd & g).sum()))

if __name__ == "__main__":
    p = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
    case = (int(sys.argv[1]), sys.argv[2], int(sys.argv[3])) if len(sys.argv) > 3 else (3, "medium", 1)
    report(case, p)

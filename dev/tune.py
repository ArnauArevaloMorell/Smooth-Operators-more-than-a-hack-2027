"""Tune solution/params.json on the sample games -- other matches only, never BAR-ATH.

Objective: mean spellF1 over every (game, degradation) case, with leave-one-game-out models, so
each game is scored by models that never saw it. Coordinate descent over multiplicative steps:
cheap, deterministic, and with ~15 parameters against ~3,400 possessions there is little room to
overfit -- the real risk is transfer to BAR-ATH, which is why the degradations span a range.

    python dev/tune.py --rounds 3 --out solution/params.json
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402

# Which parameters move, and their bounds. Everything else in params.json stays fixed.
SPACE = {
    "sigma": (0.5, 4.0), "v0": (0.0, 6.0), "vs": (0.5, 20.0),
    "w_control": (0.05, 10.0), "w_shape": (0.0, 2.0),
    "w_dead": (0.02, 5.0), "dead_bias": (-4.0, 4.0), "w_alive_ctrl": (0.0, 5.0),
    "switch": (0.5, 200.0), "die": (0.5, 300.0), "restart": (0.5, 300.0),
    "restart_same": (-50.0, 50.0),
    "w_tag": (0.0, 5.0), "w_tag_team": (0.0, 5.0), "tag_pre_s": (0.0, 5.0),
    "tag_post_s": (0.0, 5.0), "w_owner": (0.0, 10.0), "w_phase": (0.0, 3.0), "w_episode": (0.0, 20.0), "w_hazard": (0.0, 2.0), "hazard_bias": (-4.0, 4.0), "w_seq": (0.0, 20.0),
}
ADDITIVE = {"dead_bias", "restart_same", "v0", "tag_pre_s", "tag_post_s", "w_shape",
            "w_alive_ctrl", "w_tag", "w_tag_team", "w_owner", "w_phase", "w_episode", "w_hazard", "hazard_bias", "w_seq"}


def _one(args):
    case, p = args
    labels, pred = hz.run_case(case, p)
    r = hz.score(labels, pred)
    return r["spell_f1_mean"], r["macro_f1"]


def objective(pool, p):
    res = pool.map(_one, [(c, p) for c in hz.CASES])
    return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))


def candidates(name, value):
    lo, hi = SPACE[name]
    if name in ADDITIVE:
        span = (hi - lo) / 8.0
        vals = [value - span, value - span / 2, value + span / 2, value + span]
    else:
        vals = [value / 2.0, value / 1.4, value * 1.4, value * 2.0]
    return [float(np.clip(v, lo, hi)) for v in vals if np.clip(v, lo, hi) != value]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(hz.ROOT, "solution", "params.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--metrica-only", action="store_true",
                    help="tune on the Metrica cases only (human-made labels)")
    ap.add_argument("--settings", nargs="*", default=None,
                    help="tune on these degradation settings only (e.g. medium heavy)")
    a = ap.parse_args()
    if a.metrica_only:
        hz.CASES[:] = [c for c in hz.CASES if c[0] in hz.METRICA_GAMES]
    if a.settings:
        hz.CASES[:] = [c for c in hz.CASES if c[1] in a.settings]
        print("tuning on {} cases: {}".format(len(hz.CASES), sorted(set(c[1] for c in hz.CASES))))
    p = json.load(open(a.params))
    hz.logo_models()
    for c in hz.CASES:
        hz.case_data(*c)
    with mp.Pool(min(4, mp.cpu_count())) as pool:
        best, best_m = objective(pool, p)
        print("start spellF1 {:.4f} macroF1 {:.4f}".format(best, best_m), flush=True)
        names = a.only or list(SPACE)
        for rnd in range(a.rounds):
            improved = False
            for name in names:
                for v in candidates(name, p[name]):
                    q = dict(p)
                    q[name] = v
                    s, m = objective(pool, q)
                    if s > best + 1e-4:
                        best, best_m, p = s, m, q
                        improved = True
                        print("  r{} {:<13} -> {:<9.4g} spellF1 {:.4f} macroF1 {:.4f}".format(
                            rnd, name, v, s, m), flush=True)
                        if a.out:
                            json.dump(p, open(a.out, "w"), indent=1)
            if not improved:
                break
    print("best spellF1 {:.4f} macroF1 {:.4f}".format(best, best_m))
    print(json.dumps(p, indent=1))
    if a.out:
        json.dump(p, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

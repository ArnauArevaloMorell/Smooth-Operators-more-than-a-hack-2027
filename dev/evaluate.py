"""Score the pipeline on every sample-game case with leave-one-game-out models.

    python dev/evaluate.py                    # params.json
    python dev/evaluate.py --baseline         # the starter's nearest-player baseline, same data

Also prints the dead-ball model's frame quality, which is the component that decides where
possessions start and end.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
from solution import pipeline  # noqa: E402


def baseline_pred(case, window=50):
    F, Xd, Xs, labels, blocks, tags, _ = hz.case_data(*case)
    code = np.full(len(labels), -1)
    vis = np.nan_to_num(F["vis"]).astype(bool)
    near_h = np.nan_to_num(F["dH"], nan=1e9) <= np.nan_to_num(F["dA"], nan=1e9)
    code[vis & near_h] = 0
    code[vis & ~near_h] = 1
    sys.path.insert(0, os.path.join(hz.ROOT, "starter"))
    import baseline
    lab = baseline.smooth(code, window)
    out = np.array([{"H": 0, "A": 1, "D": 2}[c] for c in lab], dtype=np.int8)
    out[labels == 3] = 3
    return labels, out


def summarise(rows):
    keys = ("spell", "macro", "n_pred", "n_gt", "adf1")
    by = {}
    for r in rows:
        by.setdefault(r["setting"], []).append(r)
    for s, rs in sorted(by.items()):
        print("  {:<7} spellF1 {:.4f}  macroF1 {:.4f}  aliveDeadF1 {:.3f}  spells pred/gt {}/{}".format(
            s, np.mean([r["spell"] for r in rs]), np.mean([r["macro"] for r in rs]),
            np.mean([r["adf1"] for r in rs]), sum(r["n_pred"] for r in rs),
            sum(r["n_gt"] for r in rs)))
    print("  {:<7} spellF1 {:.4f}  macroF1 {:.4f}".format(
        "ALL", np.mean([r["spell"] for r in rows]), np.mean([r["macro"] for r in rows])))
    del keys


def evaluate(p, cases=None, use_tags=True, verbose=True, baseline=False):
    rows = []
    for case in (cases or hz.CASES):
        if baseline:
            labels, pred = baseline_pred(case)
        else:
            labels, pred = hz.run_case(case, p, use_tags=use_tags)
        r = hz.score(labels, pred)
        rows.append({"case": case, "setting": case[1], "spell": r["spell_f1_mean"],
                     "macro": r["macro_f1"], "n_pred": r["n_pred_spells"],
                     "n_gt": r["n_gt_spells"], "adf1": r["alive_dead_f1"]})
        if verbose:
            print("  game {} {:<6}  spellF1 {:.4f}  macroF1 {:.4f}  aliveDead {:.3f}  "
                  "spells {:>4}/{:<4} team_acc {:.3f}".format(
                      case[0], case[1], r["spell_f1_mean"], r["macro_f1"],
                      r["alive_dead_f1"], r["n_pred_spells"], r["n_gt_spells"],
                      r["team_acc_both_alive"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=pipeline.PARAMS_PATH)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--no-tags", action="store_true")
    a = ap.parse_args()
    p = json.load(open(a.params))
    rows = evaluate(p, use_tags=not a.no_tags, baseline=a.baseline)
    summarise(rows)

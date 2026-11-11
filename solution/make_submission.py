"""Build the BAR-ATH possession file from the provided data. The one command that reproduces it.

    python -m solution.make_submission --team <team_name>
        -> evaluation/results/<team_name>.csv, validated with the official checks

Inputs, all from the gated bundle (DATA.md), tactical source only:
    BAR-ATH_tactical_atd.xml / _atd.txt      tracking: players by team, and the ball
    BAR-ATH_tactical_smart_tagging.xml       set pieces (restarts) and phases (optional prior)

Not used: the videos, the homography, the broadcast recording (see TECHNICAL.md for why).

Graded frames are taken from games.half_video_blocks (published boundaries, DATA.md); every other
frame is written as X, which is never scored. Deterministic: same inputs, same file, byte for byte.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from solution import epts, logreg, pipeline, report, tagging  # noqa: E402

LETTERS = np.array(["H", "A", "D", "X"])

# The checklist printed at the end: (level, message). OK = fine, WARN = look at it but the file
# is usable, STOP = do not submit this file before understanding why.
CHECKS = []


def check(condition, message, fail_level="STOP"):
    CHECKS.append(("OK" if condition else fail_level, message))


def build(data_dir=None, use_tags=True, params_path=pipeline.PARAMS_PATH, verbose=True):
    t0 = time.time()
    g = games.GAMES[games.GAME]
    meta, raw = games.require(["atd_meta", "atd_raw"], data_dir=data_dir)
    cache = os.path.join(data_dir or games.DATA_DIR, "cache")
    tr = epts.load_cached(meta, raw, cache)
    n = games.source_frames("tactical")
    if verbose:
        print("ATD: {} records, {} slots ({} home / {} away), ball channels {}, slot channels {}"
              .format(len(tr), tr.xy.shape[1], int((tr.team == 0).sum()),
                      int((tr.team == 1).sum()), ["x", "y"] + tr.ball_channels,
                      ["x", "y"] + tr.player_channels))
        print("teams: H = {}, A = {}".format(*tr.team_names))

    # ATD frame_id counts from 1; submission row = frame_id - 1 (DATA.md). Place records by id
    # rather than by line number, so a missing record cannot shift everything after it.
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    xy = np.full((n, tr.xy.shape[1], 2), np.nan, dtype=np.float32)
    rows = tr.frame_id - 1
    ok = (rows >= 0) & (rows < n)
    ball[rows[ok]] = tr.ball[ok]
    xy[rows[ok]] = tr.xy[ok]
    if verbose:
        print("aligned {} of {} video frames; ball visible on {:.1%} of all frames".format(
            int(ok.sum()), n, np.isfinite(ball[:, 0]).mean()))

    blocks = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
    graded = np.zeros(n, dtype=bool)
    for a_, b_ in blocks:
        graded[a_:b_] = True
    vis_g = float(np.isfinite(ball[graded, 0]).mean())
    check(len(tr) == n and np.array_equal(tr.frame_id, np.arange(1, n + 1)),
          "tracking has one record per video frame, ids 1..{} ({} records)".format(n, len(tr)))
    check(min(int((tr.team == 0).sum()), int((tr.team == 1).sum())) >= 11,
          "track slots split between both teams ({} home / {} away)".format(
              int((tr.team == 0).sum()), int((tr.team == 1).sum())))
    check("barc" in (tr.team_names[0] or "").lower(),
          "home team (H) is Barcelona: file says H = {}, A = {}".format(*tr.team_names), "WARN")
    check(0.50 <= vis_g <= 0.68,
          "ball visible on {:.1%} of graded frames (DATA.md says 59.3%)".format(vis_g))
    models = logreg.load(pipeline.MODELS_PATH)
    p = pipeline.load_params(params_path)
    F = pipeline.prepare(ball, xy, tr.team, g["pitch"], g["fps"], blocks, models)

    prior = np.zeros((n, 4))
    if use_tags:
        try:
            st_path = games.require(["smart_tagging"], data_dir=data_dir)[0]
            set_pieces, phases = tagging.load(st_path, tr.team_names, g["fps"])
            prior += pipeline.tag_prior_from_setpieces(set_pieces, F, p, n, g["fps"])
            prior += tagging.phase_prior(phases, n, p.get("w_phase", 0.0))
            if verbose:
                print("smart tagging: {} set pieces ({} with a team), {} team phases".format(
                    len(set_pieces), sum(1 for s in set_pieces if s[2] is not None), len(phases)))
            check(40 <= len(set_pieces) <= 80 and len(phases) >= 200,
                  "Smart Tagging read: {} set pieces (DATA.md: 59), {} team phases".format(
                      len(set_pieces), len(phases)), "WARN")
        except IOError as error:
            print("smart tagging not found, continuing without it: {}".format(error))
            check(False, "Smart Tagging file not found -- the result is weaker without it", "WARN")

    codes = pipeline.predict(F, blocks, p, prior)
    if verbose:
        print("decoded in {:.0f}s".format(time.time() - t0))
    flagged = report.print_report(F, codes, blocks, g["fps"])
    check(not flagged, "health report: every value inside its sample-game range" if not flagged
          else "health report flags: " + "; ".join(flagged), "WARN")
    return codes


def public_window_check(path):
    """Score against the public 5 minutes -- a correctness check (alignment, label meaning),
    never a tuning signal: RULES.md, and 17 possessions cannot rank anything anyway."""
    gt_paths = [os.path.join(ROOT, "evaluation", "gt", f)
                for f in os.listdir(os.path.join(ROOT, "evaluation", "gt"))
                if f.endswith("_possession_gt_public.csv")]
    if not gt_paths:
        return None
    gt = pm.label_codes(pm.read_label_csv(gt_paths[0]))
    r = pm.evaluate(gt, pm.label_codes(pm.read_label_csv(path)))
    print("\npublic 5-minute window (correctness check only): spellF1 {:.4f}  macroF1 {:.4f}  "
          "possessions {} vs {} in the truth".format(r["spell_f1_mean"], r["macro_f1"],
                                                    r["n_pred_spells"], r["n_gt_spells"]))
    print("  baselines there: smooth50 spellF1 0.3256 / macroF1 0.6207. Far below that means "
          "something is misaligned or swapped -- stop and investigate.")
    return r["spell_f1_mean"]


def print_checklist(path):
    print("\n" + "=" * 78 + "\nCHECKLIST\n" + "=" * 78)
    for level, message in CHECKS:
        print("  [{:^4}] {}".format(level, message))
    levels = {level for level, _ in CHECKS}
    print("-" * 78)
    if "STOP" in levels:
        print("  VERDICT: DO NOT SUBMIT YET -- fix the [STOP] lines first. Paste this console")
        print("           output (text only, no data) to get help.")
    elif "WARN" in levels:
        print("  VERDICT: FILE IS VALID AND SUBMITTABLE: {}".format(os.path.relpath(path, ROOT)))
        print("           but read the [WARN] lines -- they say where the real data differs from")
        print("           what the method was tuned on. Paste this console output to discuss.")
    else:
        print("  VERDICT: READY TO SUBMIT: {}".format(os.path.relpath(path, ROOT)))
    print("=" * 78)


def write(codes, path):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[codes]))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", default="submission", help="file name, without .csv")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "evaluation", "results"))
    ap.add_argument("--data-dir", default=None, help="defaults to MTAH_DATA_DIR or ./data")
    ap.add_argument("--no-tags", action="store_true", help="ignore the Smart Tagging")
    ap.add_argument("--params", default=pipeline.PARAMS_PATH)
    a = ap.parse_args()

    codes = build(a.data_dir, use_tags=not a.no_tags, params_path=a.params)
    path = write(codes, os.path.join(a.out_dir, a.team + ".csv"))

    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, games.GAMES[games.GAME]["n_video_frames"])
    check(reason is None, "file passes the official validator" + ("" if reason is None
                                                                   else ": " + reason))
    if reason:
        print_checklist(path)
        sys.exit(1)
    runs = [(a_, b_) for _, a_, b_ in
            [(h, s, e + 1) for h, s, e in games.half_video_blocks(games.GAME)]]
    st = pm.possession_stats(pm.label_codes(table), runs)
    print("wrote {}  (valid)  {} possessions, mean {:.1f}s, home share {:.1%}".format(
        os.path.relpath(path, ROOT), st["n_spells"], st["mean_duration_s"], st["home_share"]))
    print("sha256 {}".format(hashlib.sha256(open(path, "rb").read()).hexdigest()))
    spell = public_window_check(path)
    if spell is not None:
        check(spell >= 0.30, "public 5 minutes: spellF1 {:.3f} (starter baseline: 0.326; near 0 "
              "means frames are misaligned or H/A swapped)".format(spell),
              "STOP" if spell < 0.15 else "WARN")
    print_checklist(path)


if __name__ == "__main__":
    main()

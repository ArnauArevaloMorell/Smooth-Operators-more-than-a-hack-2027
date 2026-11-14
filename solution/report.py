"""Health report for a real run: summary statistics next to their range in simulation.

Nothing here changes the submission. It exists because the method was developed on other matches
with simulated noise: if the BAR-ATH tracking differs from that simulation in a way that matters,
one of these numbers will sit outside its reference range, and that is where to look first.
Only aggregate statistics are printed -- no frame-level data.
"""

import numpy as np

from . import features

# (low, high) over the 32 degraded development cases (Metrica sample games and DFL matches,
# dev/harness.py), see dev/report_ranges.py.
# Home share is a property of the match, not of the pipeline, so it has no range.
REFERENCE = {
    "ball visible, graded frames": (0.446, 0.716),
    "tracked players per graded frame": (19.102, 21.159),
    "clear control episodes, share of graded frames": (0.062, 0.258),
    "median ball-to-nearest-player distance (m)": (1.135, 2.835),
    "median mean player speed (m/s)": (1.425, 1.784),
    "predicted dead-ball share": (0.317, 0.527),
    "possessions per minute": (2.816, 3.773),
    "mean possession length (s)": (7.603, 12.329),
    "decoded team agrees with ownership model": (0.893, 0.952),
}


def stats(F, codes, blocks, fps):
    g = np.zeros(len(codes), dtype=bool)
    for s, e in blocks:
        g[s:e] = True
    vis = np.nan_to_num(F["vis"]).astype(bool)
    ep = features.control_episodes(F, blocks)
    near = np.fmin(F["dH"], F["dA"])
    alive = g & (codes < 2)
    graded_min = g.sum() / float(fps) / 60.0
    n_spells = 0
    lengths = []
    for s, e in blocks:
        c = codes[s:e]
        b = np.r_[0, np.flatnonzero(c[1:] != c[:-1]) + 1, len(c)]
        for a, z in zip(b[:-1], b[1:]):
            if c[a] < 2:
                n_spells += 1
                lengths.append(z - a)
    agree = np.sign(np.nan_to_num(F["owner_logit"][alive])) == np.where(codes[alive] == 0, 1, -1)
    return {
        "ball visible, graded frames": vis[g].mean(),
        "tracked players per graded frame": np.nanmean(F["n_players"][g]),
        "clear control episodes, share of graded frames": (ep[g] != 0).mean(),
        "median ball-to-nearest-player distance (m)": np.nanmedian(near[g & vis]),
        "median mean player speed (m/s)": np.nanmedian(F["ms"][g]),
        "predicted dead-ball share": (codes[g] == 2).mean(),
        "possessions per minute": n_spells / graded_min,
        "mean possession length (s)": np.mean(lengths) / fps if lengths else 0.0,
        "home share of possession": (codes[alive] == 0).mean() if alive.any() else 0.0,
        "decoded team agrees with ownership model": agree.mean() if alive.any() else 0.0,
    }


def print_report(F, codes, blocks, fps):
    print("\nhealth report  (value | range on the degraded development matches)")
    flagged = []
    for name, value in stats(F, codes, blocks, fps).items():
        lo_hi = REFERENCE.get(name)
        flag = ""
        if lo_hi:
            lo, hi = lo_hi
            pad = 0.15 * (hi - lo) + 1e-9
            flag = "" if lo - pad <= value <= hi + pad else "   <-- outside reference range"
            if flag:
                flagged.append(name)
            print("  {:<48} {:>7.3f} | {:.3f} .. {:.3f}{}".format(name, value, lo, hi, flag))
        else:
            print("  {:<48} {:>7.3f}".format(name, value))
    return flagged

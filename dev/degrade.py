"""Make clean sample-game tracking look like BAR-ATH's automatic, uncorrected tracking.

What the BAR-ATH ATD is like, and how each property is reproduced (all seeded, so reproducible):

  * Ball gaps are LONG, not frame drops. Read off the organisers' public baseline_raw.csv (which is
    ATD-derived: it marks every frame where the tracking has no ball), the ball disappears in ~148
    gaps over the match, median ~11 s, and is missing on 41% of graded frames. We inject gaps into
    live play with lengths resampled from that empirical distribution (GAP_QUANTILES below),
    half of them starting on fast balls (long passes and shots are where trackers lose it).
  * The ball is sometimes visible while dead (carried back to the restart spot): for a fraction of
    dead stretches we interpolate a slow ball path from where it went out to where play restarts.
  * No player identities, ~20.5 of 22 players tracked on average: slots drop out in bursts.
  * Position noise on everything.

Degradation strength is a parameter, and tuning averages over several settings so that nothing
is tuned to one guess of what the real noise looks like.
"""

import numpy as np

# Percentiles 0,5,...,100 of BAR-ATH in-play-or-dead ball gaps (frames), from baseline_raw.csv
# (an ATD-derived file the organisers publish; it carries no ground truth). Resampled by
# inverse-CDF interpolation.
GAP_QUANTILES = None  # filled lazily from the public baseline file, see _gap_quantiles()


def _gap_quantiles():
    global GAP_QUANTILES
    if GAP_QUANTILES is None:
        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(os.path.dirname(here), "evaluation"))
        import possession_metrics as pm
        raw = pm.label_codes(pm.read_label_csv(
            os.path.join(os.path.dirname(here), "evaluation", "results", "baseline_raw.csv")))
        graded = np.r_[11185:90652, 110808:190198]
        vis = (raw[graded] != pm.D).astype(np.int8)
        edges = np.diff(np.r_[0, vis, 0])
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        gaps = starts[1:] - stops[:-1]
        GAP_QUANTILES = np.percentile(gaps, np.arange(0, 101, 5))
    return GAP_QUANTILES


SETTINGS = {
    # alive_missing: target share of live frames with no ball; dead_visible: share of dead
    # stretches where the ball is seen being carried to the restart.
    # ball_offset: a slowly varying bias (redrawn every second, interpolated), the kind of error a
    # single-camera ball tracker makes -- a lofted ball projected to the ground, a ball partly
    # hidden by feet. It does not average out over frames. Sized so the median ball-to-nearest-
    # player distance spans BAR-ATH's (1.92 m, from the run's health report).
    "light": dict(alive_missing=0.15, dead_visible=0.5, ball_noise=0.25, ball_offset=0.4,
                  player_drop=0.04),
    "medium": dict(alive_missing=0.25, dead_visible=0.4, ball_noise=0.35, ball_offset=0.8,
                   player_drop=0.07),
    "heavy": dict(alive_missing=0.35, dead_visible=0.3, ball_noise=0.5, ball_offset=1.2,
                  player_drop=0.10),
    # beyond BAR-ATH on every axis -- training-only, so the network has seen worse than the match
    "xheavy": dict(alive_missing=0.42, dead_visible=0.3, ball_noise=0.6, ball_offset=1.6,
                   player_drop=0.12),
}


def degrade(game, setting="medium", seed=0):
    """(ball, xy) degraded copies of a sample Game's tracking, normalised coordinates."""
    cfg = SETTINGS[setting] if isinstance(setting, str) else setting
    rng = np.random.RandomState(seed)
    L, W = game.pitch
    ball = game.ball.copy()
    xy = game.xy.copy()
    labels = game.labels
    T = len(ball)
    alive = labels < 2

    # 0. Sources that track the ball through every stoppage (DFL) do not look like BAR-ATH, where
    # the ball is often lost while dead: hide it for all of a stoppage (45%) or its first part
    # (25%, the ball out of shot until it is fetched), keep it visible otherwise.
    dead = labels == 2
    if dead.any() and np.isfinite(ball[dead, 0]).mean() > 0.5:
        edges0 = np.diff(np.r_[0, dead.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(edges0 == 1), np.flatnonzero(edges0 == -1)):
            u = rng.rand()
            if u < 0.45:
                ball[s:e] = np.nan
            elif u < 0.70:
                ball[s:s + int((e - s) * rng.uniform(0.3, 0.9))] = np.nan

    # 1. Dead stretches where the ball is visible: interpolate from last seen to next seen.
    edges = np.diff(np.r_[0, (labels == 2).astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.rand() > cfg["dead_visible"] or s == 0 or e >= T:
            continue
        before = np.flatnonzero(np.isfinite(ball[max(0, s - 50):s, 0]))
        after = np.flatnonzero(np.isfinite(ball[e:min(T, e + 50), 0]))
        if not len(before) or not len(after):
            continue
        p0 = ball[max(0, s - 50) + before[-1]]
        p1 = ball[e + after[0]]
        # Visible for the latter part of the stretch (ball retrieved, then placed).
        v0 = s + int((e - s) * rng.uniform(0.0, 0.6))
        u = np.linspace(0, 1, e - v0)[:, None]
        ball[v0:e] = p0 * (1 - u) + p1 * u

    # 2. Gaps in live play, BAR-ATH-shaped.
    q = _gap_quantiles()
    speed = np.zeros(T)
    fin = np.isfinite(ball[:, 0])
    d = np.r_[0, np.hypot(np.diff(ball[:, 0]) * L, np.diff(ball[:, 1]) * W) * 25]
    speed[fin] = np.nan_to_num(d[fin])
    fast = np.flatnonzero(alive & (speed > 15))
    alive_idx = np.flatnonzero(alive)
    target = cfg["alive_missing"] * len(alive_idx)
    missing = (~np.isfinite(ball[:, 0])) & alive
    guard = 0
    while missing.sum() < target and guard < 10000:
        guard += 1
        length = int(np.interp(rng.rand(), np.linspace(0, 1, len(q)), q))
        length = max(5, min(length, 1500))
        start = int(fast[rng.randint(len(fast))]) if (rng.rand() < 0.5 and len(fast)) \
            else int(alive_idx[rng.randint(len(alive_idx))])
        stop = min(T, start + length)
        ball[start:stop] = np.nan
        missing[start:stop] = alive[start:stop]

    # 3. Noise.
    n = rng.normal(0, cfg["ball_noise"], size=ball.shape)
    off = cfg.get("ball_offset", 0.0)
    if off:
        knots = rng.normal(0, off / np.sqrt(2), size=(T // 25 + 2, 2))
        u = (np.arange(T) % 25 / 25.0)[:, None]
        i = np.arange(T) // 25
        n = n + knots[i] * (1 - u) + knots[i + 1] * u
    ball[:, 0] += n[:, 0] / L
    ball[:, 1] += n[:, 1] / W
    # Player noise is a slow drift, not per-frame jitter: the real ATD is smoothed (median mean
    # player speed 1.47 m/s on BAR-ATH, the same as the clean sample games), and white jitter
    # would inflate every speed -- which is exactly the mistake the first version made.
    # Offsets of ~0.2 m drawn every 2 s and linearly interpolated: positions are off, speeds
    # barely change (~0.15 m/s), as on the real, smoothed tracking.
    step = 50
    knots = rng.normal(0, 0.2, size=((T // step) + 2,) + xy.shape[1:])
    u = (np.arange(T) % step / float(step))[:, None, None]
    i = np.arange(T) // step
    pn = knots[i] * (1 - u) + knots[i + 1] * u
    xy[..., 0] += (pn[..., 0] / L).astype(np.float32)
    xy[..., 1] += (pn[..., 1] / W).astype(np.float32)

    # 4. Player dropouts in bursts (mean 50 frames).
    S = xy.shape[1]
    rate = cfg["player_drop"] / 50.0
    for s in range(S):
        starts = np.flatnonzero(rng.rand(T) < rate)
        for st in starts:
            xy[st:st + rng.geometric(1 / 50.0), s] = np.nan
    return ball.astype(np.float32), xy.astype(np.float32)

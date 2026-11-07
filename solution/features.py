"""Per-frame features from team-level tracking. numpy only. Everything in metres and seconds.

Three families, matching the three questions the decoder asks:

  control   who is touching the ball right now?  Distance from the ball to the nearest player of
            each team, and how well that player's velocity matches the ball's (a dribble moves
            with the player; a pass flying past a defender does not).
  dead      is the ball in play?  Ball visibility, where it was last seen (outside the lines, in a
            corner, on the spot), how fast everyone is moving (players walk during stoppages),
            and how fast the ball is moving -- each at several time scales.
  shape     which team is attacking, without the ball?  Collective movement toward a goal and
            how stretched each team is. This is what carries the decision through the ATD's long
            ball gaps (median ~11 s).

Windows never cross a block boundary (a half), so halftime cannot leak into either half.
"""

import numpy as np

SPEED_CAP = 11.0     # m/s; faster than this is a tracking glitch / slot identity switch
BALL_CAP = 40.0      # m/s
BALL_OUTLIER_M = 4.0


def _ffill_index(valid):
    """For each i, index of the last True at or before i (or -1)."""
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx)


def _bfill_index(valid):
    n = len(valid)
    idx = np.where(valid, np.arange(n), n)
    return np.minimum.accumulate(idx[::-1])[::-1]


def window_mean(x, half, valid=None):
    """Centred mean over +/-half frames, ignoring NaN (or ~valid). NaN where nothing valid."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x) if valid is None else (valid & np.isfinite(x))
    v = np.where(ok, x, 0.0)
    cs = np.r_[0.0, np.cumsum(v)]
    cn = np.r_[0, np.cumsum(ok)]
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - half)
    hi = np.minimum(n, i + half + 1)
    cnt = cn[hi] - cn[lo]
    out = np.full(n, np.nan)
    m = cnt > 0
    out[m] = (cs[hi] - cs[lo])[m] / cnt[m]
    return out


def window_mean_dir(x, before, after):
    """Mean over [i-before, i+after], NaN-aware. For past-only / future-only windows."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.r_[0.0, np.cumsum(v)]
    cn = np.r_[0, np.cumsum(ok)]
    n = len(x)
    i = np.arange(n)
    lo = np.maximum(0, i - before)
    hi = np.minimum(n, i + after + 1)
    cnt = cn[hi] - cn[lo]
    out = np.full(n, np.nan)
    m = cnt > 0
    out[m] = (cs[hi] - cs[lo])[m] / cnt[m]
    return out


def _clean_ball(B):
    """Drop isolated ball detections that disagree with their neighbourhood (false positives)."""
    B = B.copy()
    T = len(B)
    k = 3
    pad = np.full((k, 2), np.nan)
    Bp = np.vstack([pad, B, pad])
    stack = np.stack([Bp[j:j + T] for j in range(2 * k + 1)], axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
    dev = np.hypot(B[:, 0] - med[:, 0], B[:, 1] - med[:, 1])
    bad = np.isfinite(dev) & (dev > BALL_OUTLIER_M)
    B[bad] = np.nan
    return B


def _smooth_track(B, half=2):
    """NaN-aware centred moving average over +/-half frames, only where the ball is seen.

    Per-frame detection noise of a few tens of centimetres becomes several m/s of velocity noise
    once differentiated over a few frames; averaging 5 frames first is what makes 'is the ball
    moving with this player' answerable at all.
    """
    out = np.full(B.shape, np.nan)
    vis = np.isfinite(B[:, 0])
    for d in range(B.shape[1]):
        out[:, d] = window_mean(B[:, d], half)
    out[~vis] = np.nan
    return out


def _velocity(P, fps, lag=2, cap=SPEED_CAP):
    """Central-difference velocity over +/-lag frames. P (T, ..., 2). NaN where undefined."""
    V = np.full(P.shape, np.nan, dtype=np.float64)
    V[lag:-lag] = (P[2 * lag:] - P[:-2 * lag]) * (fps / (2.0 * lag))
    sp = np.hypot(V[..., 0], V[..., 1])
    V[sp > cap] = np.nan
    return V


def attack_sign(P, team, blocks):
    """Per block, +1 if the home team attacks toward +x, else -1 (home centroid on the left)."""
    signs = []
    home = team == 0
    for s, e in blocks:
        hx = np.nanmean(P[s:e][:, home, 0])
        ax = np.nanmean(P[s:e][:, ~home, 0])
        signs.append(1.0 if hx < ax else -1.0)
    return signs


def compute(ball, xy, team, pitch, fps, blocks):
    """Feature dict over the full timeline; frames outside blocks are left NaN.

    ball (T,2), xy (T,S,2) normalised; team (S,) 0/1; pitch (L, W) metres; blocks [(start, stop)].
    """
    L, W = pitch
    T = len(ball)
    scale = np.array([L, W])
    B_all = ball.astype(np.float64) * scale
    P_all = xy.astype(np.float64) * scale
    team = np.asarray(team)
    home = team == 0
    signs = attack_sign(P_all, team, blocks)

    names = ["vis", "dH", "dA", "rH", "rA", "ball_speed", "bx", "by",
             "ms", "n_players", "shape_vx", "shape_sx", "shape_sy", "shape_cx", "shape_dl"]
    F = {k: np.full(T, np.nan) for k in names}
    F["attack_sign"] = np.zeros(T)

    for (s, e), sign in zip(blocks, signs):
        B = _smooth_track(_clean_ball(B_all[s:e]))
        P = P_all[s:e]
        vis = np.isfinite(B[:, 0])
        VB = _velocity(B, fps, lag=3, cap=BALL_CAP)
        VP = _velocity(P, fps)
        SP = np.hypot(VP[..., 0], VP[..., 1])

        # control
        dist = np.hypot(P[..., 0] - B[:, None, 0], P[..., 1] - B[:, None, 1])  # (n,S)
        for k, mask in (("H", home), ("A", ~home)):
            dk = np.where(np.isfinite(dist[:, mask]), dist[:, mask], np.inf)
            j = np.argmin(dk, axis=1)
            dmin = dk[np.arange(len(dk)), j]
            dmin[~np.isfinite(dmin)] = np.nan
            vnear = VP[:, mask][np.arange(len(dk)), j]
            rel = np.hypot(VB[:, 0] - vnear[:, 0], VB[:, 1] - vnear[:, 1])
            F["d" + k][s:e] = dmin
            F["r" + k][s:e] = rel
        F["vis"][s:e] = vis
        F["ball_speed"][s:e] = np.hypot(VB[:, 0], VB[:, 1])
        F["bx"][s:e] = B[:, 0]
        F["by"][s:e] = B[:, 1]

        # collective motion
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                F["ms"][s:e] = np.nanmean(SP, axis=1)
                F["n_players"][s:e] = np.isfinite(P[..., 0]).sum(axis=1)
                # shape, oriented so that + means "looks like home is attacking"
                vx_all = np.nanmean(VP[..., 0], axis=1) * sign
                F["shape_vx"][s:e] = vx_all
                sxH = np.nanstd(P[:, home, 0], axis=1)
                sxA = np.nanstd(P[:, ~home, 0], axis=1)
                syH = np.nanstd(P[:, home, 1], axis=1)
                syA = np.nanstd(P[:, ~home, 1], axis=1)
                F["shape_sx"][s:e] = sxH - sxA
                F["shape_sy"][s:e] = syH - syA
                cx = np.nanmean(P[..., 0], axis=1)
                F["shape_cx"][s:e] = (cx - L / 2.0) * sign / L
                # defensive line height: home's deepest outfield vs away's (attacking team's
                # last line is higher up the pitch)
                xH = P[:, home, 0] * sign
                xA = P[:, ~home, 0] * sign
                F["shape_dl"][s:e] = (np.nanmean(xH, axis=1) - np.nanmean(xA, axis=1))
        F["attack_sign"][s:e] = sign
    return F


# -- design matrices for the two learned models -----------------------------------------------

def _block_apply(fn, x, blocks, T):
    out = np.full(T, np.nan)
    for s, e in blocks:
        out[s:e] = fn(x[s:e])
    return out


def dead_design(F, pitch, fps, blocks):
    """Matrix for the dead-ball classifier. Column order is the model contract -- keep it."""
    L, W = pitch
    T = len(F["vis"])
    vis = np.nan_to_num(F["vis"]).astype(bool)
    cols, names = [], []

    def add(name, x):
        names.append(name)
        cols.append(x)

    for sec in (1, 4, 10):
        add("vis_w%d" % sec, _block_apply(lambda v: window_mean(v, sec * fps), F["vis"], blocks, T))
    add("vis_past2", _block_apply(lambda v: window_mean_dir(v, 2 * fps, 0), F["vis"], blocks, T))
    add("vis_fut2", _block_apply(lambda v: window_mean_dir(v, 0, 2 * fps), F["vis"], blocks, T))

    # last / next known ball position, and time since / until
    bx_last = np.full(T, np.nan)
    by_last = np.full(T, np.nan)
    since = np.full(T, np.nan)
    until = np.full(T, np.nan)
    bx_next = np.full(T, np.nan)
    by_next = np.full(T, np.nan)
    for s, e in blocks:
        v = vis[s:e]
        fi = _ffill_index(v)
        bi = _bfill_index(v)
        n = e - s
        ok = fi >= 0
        bx_last[s:e][ok] = F["bx"][s:e][fi[ok]]
        by_last[s:e][ok] = F["by"][s:e][fi[ok]]
        since[s:e] = np.where(ok, np.arange(n) - fi, 50 * fps) / float(fps)
        okb = bi < n
        bx_next[s:e][okb] = F["bx"][s:e][bi[okb]]
        by_next[s:e][okb] = F["by"][s:e][bi[okb]]
        until[s:e] = np.where(okb, bi - np.arange(n), 50 * fps) / float(fps)
    add("log_since", np.log1p(since))
    add("log_until", np.log1p(until))

    def out_dist(x, y):
        return np.maximum.reduce([np.zeros_like(x), -x, x - L, -y, y - W])

    def zone(x, y):
        dt = np.minimum(y, W - y)                   # to touchline
        dg = np.minimum(x, L - x)                   # to goal line
        corner = np.hypot(dg, dt)
        centre = np.hypot(x - L / 2.0, y - W / 2.0)
        six = (dg < 5.5) & (np.abs(y - W / 2.0) < 9.16)
        return dt, dg, corner, centre, six

    for tag, (x, y) in (("last", (bx_last, by_last)), ("next", (bx_next, by_next))):
        od = np.nan_to_num(out_dist(x, y))
        add("out_%s" % tag, np.minimum(od, 5.0))
        add("isout_%s" % tag, (od > 0.5).astype(float))
        dt, dg, corner, centre, six = zone(np.nan_to_num(x, nan=L / 2), np.nan_to_num(y, nan=W / 2))
        add("touch_%s" % tag, (dt < 2.0).astype(float))
        add("corner_%s" % tag, (corner < 3.0).astype(float))
        add("centre_%s" % tag, (centre < 2.0).astype(float))
        add("six_%s" % tag, six.astype(float))

    ms = F["ms"]
    for sec in (1, 4, 10):
        add("ms_w%d" % sec, _block_apply(lambda v: window_mean(v, sec * fps), ms, blocks, T))
    add("ms_past3", _block_apply(lambda v: window_mean_dir(v, 3 * fps, 0), ms, blocks, T))
    add("ms_fut3", _block_apply(lambda v: window_mean_dir(v, 0, 3 * fps), ms, blocks, T))
    # relative to the half's typical tempo, so a slower match does not look 'dead'
    rel = np.full(T, np.nan)
    for s, e in blocks:
        med = np.nanmedian(ms[s:e])
        rel[s:e] = window_mean(ms[s:e], 4 * fps) / med if med > 0 else np.nan
    add("ms_rel4", rel)

    bs = np.where(vis, F["ball_speed"], np.nan)
    for sec in (1, 3):
        w = _block_apply(lambda v: window_mean(v, sec * fps), bs, blocks, T)
        add("bs_w%d" % sec, np.nan_to_num(w, nan=0.0))
        add("bs_w%d_nan" % sec, (~np.isfinite(w)).astype(float))
    add("n_players", np.nan_to_num(_block_apply(lambda v: window_mean(v, fps), F["n_players"],
                                                blocks, T), nan=0.0) / 22.0)
    # someone is on the ball -> play is probably alive
    cH, cA = control_kernel(F, 1.5)
    on_ball = np.maximum(cH, cA)
    for before, after in ((25, 25), (50, 0), (0, 50), (125, 125)):
        add("onball_%d_%d" % (before, after),
            _block_apply(lambda v: window_mean_dir(v, before, after), on_ball, blocks, T))
    X = np.column_stack(cols)
    X = np.nan_to_num(X, nan=0.0)
    return X, names


OWNER_SIGMAS = (1.0, 2.0, 4.0)
CONTROL_ON = 0.55      # kernel value that counts as "on the ball" (~1.1 m at sigma 1)
CONTROL_OFF = 0.25     # ...while the other team is clearly not
CONTROL_MIN_FRAMES = 3


def control_episodes(F, blocks):
    """+1 / -1 / 0 per frame: home / away player clearly controlling the ball for >= 3 frames."""
    T = len(F["vis"])
    cH, cA = control_kernel(F, 1.0)
    raw = np.where((cH > CONTROL_ON) & (cA < CONTROL_OFF), 1,
                   np.where((cA > CONTROL_ON) & (cH < CONTROL_OFF), -1, 0)).astype(np.int8)
    out = np.zeros(T, dtype=np.int8)
    for s, e in blocks:
        r = raw[s:e]
        edges = np.flatnonzero(np.diff(r) != 0) + 1
        bounds = np.r_[0, edges, len(r)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if r[a] != 0 and b - a >= CONTROL_MIN_FRAMES:
                out[s + a:s + b] = r[a]
    return out


def controller_features(F, fps, blocks):
    """Team of the last and of the next control episode, and how long ago / until.

    This is the possession rule itself, measured: the ball belongs to whoever last controlled it,
    through passes in flight and loose balls, until the other side controls it.
    """
    T = len(F["vis"])
    ep = control_episodes(F, blocks)
    last = np.zeros(T)
    nxt = np.zeros(T)
    since = np.full(T, 60.0)
    until = np.full(T, 60.0)
    for s, e in blocks:
        v = ep[s:e] != 0
        n = e - s
        fi = _ffill_index(v)
        bi = _bfill_index(v)
        ok = fi >= 0
        last[s:e][ok] = ep[s:e][fi[ok]]
        since[s:e][ok] = (np.arange(n) - fi)[ok] / float(fps)
        okb = bi < n
        nxt[s:e][okb] = ep[s:e][bi[okb]]
        until[s:e][okb] = (bi - np.arange(n))[okb] / float(fps)
    since = np.minimum(since, 60.0)
    until = np.minimum(until, 60.0)
    return (last, nxt, last * np.exp(-since / 2.0), nxt * np.exp(-until / 2.0),
            last * np.exp(-since / 10.0), nxt * np.exp(-until / 10.0), ep.astype(float),
            last * nxt, np.log1p(since), np.log1p(until))


def control_kernel(F, sigma, v0=2.5, vs=8.0):
    """(cH, cA): closeness to the ball x velocity agreement, 0 where the ball is not seen."""
    out = []
    for k in ("H", "A"):
        d, r = F["d" + k], F["r" + k]
        c = np.exp(-0.5 * (d / sigma) ** 2)
        rel = np.where(np.isfinite(r), np.exp(-np.maximum(0.0, r - v0) / vs), 1.0)
        out.append(np.where(np.isfinite(c), c * rel, 0.0))
    return out


def owner_design(F, fps, blocks):
    """Matrix for the 'which team owns the ball' classifier, + means home.

    Instantaneous control at three radii, plus the same evidence averaged over past-only,
    future-only and centred windows: the asymmetry between the past and the future is what
    locates a change of possession to within a few frames.
    """
    T = len(F["vis"])
    cols = []
    vis = np.nan_to_num(F["vis"])
    for sigma in OWNER_SIGMAS:
        cH, cA = control_kernel(F, sigma)
        diff = cH - cA
        cols.append(diff)
        for before, after in ((12, 0), (0, 12), (50, 0), (0, 50), (25, 25), (125, 125)):
            num = _block_apply(lambda v: window_mean_dir(v, before, after), diff, blocks, T)
            cols.append(num)
    dH = np.nan_to_num(F["dH"], nan=30.0)
    dA = np.nan_to_num(F["dA"], nan=30.0)
    cols += [np.minimum(dH, 30) - np.minimum(dA, 30), vis,
             _block_apply(lambda v: window_mean(v, 50), vis, blocks, T)]
    cols += list(controller_features(F, fps, blocks))
    Xs, _ = shape_design(F, fps, blocks)
    X = np.nan_to_num(np.column_stack(cols), nan=0.0)
    return np.column_stack([X, Xs])


def shape_design(F, fps, blocks):
    """Matrix for the ball-free 'which team is attacking' classifier. + means home."""
    T = len(F["vis"])
    cols, names = [], []
    for key in ("shape_vx", "shape_sx", "shape_sy", "shape_cx", "shape_dl"):
        for sec in (1, 4):
            names.append("%s_w%d" % (key, sec))
            cols.append(_block_apply(lambda v: window_mean(v, sec * fps), F[key], blocks, T))
    X = np.nan_to_num(np.column_stack(cols), nan=0.0)
    return X, names

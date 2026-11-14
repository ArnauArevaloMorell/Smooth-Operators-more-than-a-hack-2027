"""Metrica's public sample games as a development set -- other matches, so tuning on them is allowed.

    https://github.com/metrica-sports/sample-data   (fetched by dev/fetch_sample_data.sh)

RULES.md: "Hand-labelled data from other games is fine, in any amount." These three anonymised
matches come with tracking (player ids, ball) and human event data. We turn the events into a
frame-level H/A/D ground truth that follows CHALLENGE.md's possession rules, then *degrade* the
clean tracking until it looks like the BAR-ATH automatic tracking (see degrade.py), and that is
what every hyperparameter in solution/params.json is tuned on. Nothing here ever reads BAR-ATH.

Ground-truth derivation from events (the possession rules, made mechanical):
  * Ball is DEAD from the frame it leaves the pitch (BALL OUT / SHOT ... OUT end frame), from a
    whistled foul (FAULT RECEIVED, CHALLENGE ...FAULT without ADVANTAGE, BALL LOST OFFSIDE /
    HAND BALL), and from a goal, until the next restart.
  * A RESTART is a SET PIECE event, or the first controlled event after a dead stretch (goal
    kicks appear as PASS/GOAL KICK in games 1-2).
  * While alive, the owning team changes only on a CONTROLLED action by the other team: PASS,
    CARRY, SHOT, RECOVERY, SET PIECE, or the start of a BALL LOST (that team had it). Challenges,
    aerial duels and deflections never change possession on their own -- CHALLENGE.md.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLE_DIR = os.environ.get("MTAH_SAMPLE_DIR") or os.path.join(ROOT, "data", "sample-data")

H, A, D, X = 0, 1, 2, 3
RESTARTS = ("SET PIECE", "PASS", "SHOT")
CONTROLLED = ("PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL LOST")


class Game(object):
    """Clean tracking + derived labels for one sample match, normalised 0-1 coordinates."""

    def __init__(self, name, ball, xy, team, labels, period, pitch=(105.0, 68.0), fps=25):
        self.name, self.ball, self.xy, self.team = name, ball, xy, team
        self.labels, self.period, self.pitch, self.fps = labels, period, pitch, fps


# -- loading ---------------------------------------------------------------------------------

def _load_csv_game(n):
    base = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(n))
    if not os.path.isdir(base):
        base = SAMPLE_DIR
    xs, teams, ball = [], [], None
    period = frames = None
    for side, code in (("Home", 0), ("Away", 1)):
        path = os.path.join(base, "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(n, side))
        df = pd.read_csv(path, skiprows=2)
        cols = list(df.columns)
        player_cols = [c for c in cols if c.startswith("Player")]
        for c in player_cols:
            i = cols.index(c)
            xs.append(df.iloc[:, [i, i + 1]].values.astype(np.float32))
            teams.append(code)
        if ball is None:
            i = cols.index("Ball")
            ball = df.iloc[:, [i, i + 1]].values.astype(np.float32)
            period = df["Period"].values
            frames = df["Frame"].values
    xy = np.stack(xs, axis=1)
    ev = pd.read_csv(os.path.join(base, "Sample_Game_{}_RawEventsData.csv".format(n)))
    events = []
    for _, r in ev.iterrows():
        events.append({
            "team": 0 if r["Team"] == "Home" else 1,
            "type": str(r["Type"]).upper(),
            "sub": str(r["Subtype"]).upper() if isinstance(r["Subtype"], str) else "",
            "start": int(r["Start Frame"]) - 1,
            "end": int(r["End Frame"]) - 1 if r["End Frame"] > 0 else int(r["Start Frame"]) - 1,
            "period": int(r["Period"]),
        })
    assert np.all(frames == np.arange(1, len(frames) + 1))
    return ball, xy, np.array(teams, dtype=np.int8), period, events


def _load_epts_game():
    import sys
    sys.path.insert(0, ROOT)
    from solution.epts import load_epts
    base = os.path.join(SAMPLE_DIR, "Sample_Game_3")
    if not os.path.isdir(base):
        base = SAMPLE_DIR
    tr = load_epts(os.path.join(base, "Sample_Game_3_metadata.xml"),
                   os.path.join(base, "Sample_Game_3_tracking.txt"))
    raw = json.load(open(os.path.join(base, "Sample_Game_3_events.json")))
    team_ids = [t for t in ("FIFATMA", "FIFATMB")]
    events = []
    for e in raw["data"]:
        subs = e.get("subtypes")
        if isinstance(subs, list):
            sub = " ".join(s["name"] for s in subs)
        elif isinstance(subs, dict):
            sub = subs["name"]
        else:
            sub = ""
        start = e["start"]["frame"]
        end = e["end"]["frame"] if e["end"] and e["end"]["frame"] else start
        events.append({
            "team": team_ids.index(e["team"]["id"]),
            "type": e["type"]["name"].upper(), "sub": sub.upper(),
            "start": int(start) - 1, "end": int(end) - 1, "period": int(e["period"]),
        })
    period = np.zeros(len(tr), dtype=np.int64)
    # Periods from the events themselves: from the first to the last event of each.
    for p in (1, 2):
        s = [e["start"] for e in events if e["period"] == p]
        if s:
            period[min(s):max(e["end"] for e in events if e["period"] == p) + 1] = p
    return tr.ball, tr.xy, tr.team, period, events


# -- ground truth from events ----------------------------------------------------------------

def _kills(e):
    """Frame at which this event makes the ball dead, or None."""
    t, s = e["type"], e["sub"]
    if t == "BALL OUT":
        return e["end"]
    if t == "SHOT" and ("OUT" in s or "GOAL" in s):
        return e["end"]
    if t == "FAULT RECEIVED":
        return e["start"]
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return e["start"]
    if t == "BALL LOST" and ("OFFSIDE" in s or "HAND BALL" in s):
        return e["end"]
    if t == "CARD":
        return None
    return None


def labels_from_events(events, n_frames, period):
    """Frame-level H/A/D/X following the rules in the module docstring.

    Graded frames are each period's span from its first event (the kickoff) to its last event's
    end; everything else is X, mirroring the BAR-ATH ground truth's halftime and pre-match holes.
    """
    labels = np.full(n_frames, X, dtype=np.int8)
    for p in sorted(set(e["period"] for e in events)):
        evs = sorted([e for e in events if e["period"] == p], key=lambda e: (e["start"], e["end"]))
        lo = evs[0]["start"]
        hi = max(e["end"] for e in evs)
        # Points in time order: ("own", team, type) for controlled actions, ("kill",) for
        # whatever makes the ball dead. Then one walk with an explicit alive/dead state, so that
        # what players do with a dead ball (carrying it to the restart spot, which Sample Game 3
        # logs as CARRY) can never restart play: only a set piece or a kick does.
        points = []
        for order, e in enumerate(evs):
            if e["type"] in CONTROLLED:
                points.append((e["start"], 1, order, "own", e["team"], e["type"]))
            k = _kills(e)
            if k is not None:
                points.append((k, 0, order, "kill", None, None))
        points.sort(key=lambda x: (x[0], x[1], x[2]))
        seq = np.full(hi + 1 - lo, D, dtype=np.int8)
        state, last_f = D, lo
        for f, _, _, kind, team, etype in points:
            if kind == "kill":
                new = D
            elif state == D:
                if etype not in RESTARTS:
                    continue
                new = team
            else:
                new = team
            if new == state:
                continue
            f = min(max(f, lo), hi + 1)
            seq[last_f - lo:f - lo] = state
            state, last_f = new, f
        seq[last_f - lo:] = state
        labels[lo:hi + 1] = seq
    return labels


def load_game(n):
    """Game object for sample game 1, 2 or 3, or DFL match n - 4 (dev/dfl_games.py, games 4+)."""
    if n >= 4:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import dfl_games
        return dfl_games.load(dfl_games.match_ids()[n - 4])
    if n in (1, 2):
        ball, xy, team, period, events = _load_csv_game(n)
    else:
        ball, xy, team, period, events = _load_epts_game()
    labels = labels_from_events(events, len(ball), period)
    return Game("sample{}".format(n), ball.astype(np.float32), xy.astype(np.float32), team,
                labels, period)


def summary(labels, fps=25):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    runs = pm.valid_runs(labels)
    st = pm.possession_stats(labels, runs)
    graded = int(np.sum(labels != X))
    n_d = len([s for s in pm.segments(labels, runs) if s[2] == D])
    return ("{} graded min, {} possessions ({:.2f}/min, mean {:.1f}s), {} dead stretches, "
            "D share {:.1%}".format(graded // (60 * fps), st["n_spells"],
                                   st["n_spells"] / (graded / 60.0 / fps),
                                   st["mean_duration_s"], n_d,
                                   np.mean(labels[labels != X] == D)))


if __name__ == "__main__":
    for n in (1, 2, 3):
        g = load_game(n)
        print(g.name, g.xy.shape, summary(g.labels))

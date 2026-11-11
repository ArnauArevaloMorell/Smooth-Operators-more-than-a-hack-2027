"""The DFL / IDSSE open dataset as extra development matches (other games: allowed by RULES.md).

    Bassek, M. et al. (2025). An integrated dataset of spatiotemporal and event data in elite
    soccer. Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y
    Files: figshare doi:10.6084/m9.figshare.28196177, CC BY 4.0. dev/fetch_dfl_data.sh downloads
    the match-information and position files into data/dfl/ (gitignored).

Seven Bundesliga / 2. Bundesliga matches of full-pitch optical tracking at 25 Hz -- like Metrica's
-- and, crucially, the provider's own per-frame flags on the ball: BallStatus (1 alive, 0 dead)
and BallPossession (1 home, 2 guest). That is the same kind of answer the BAR-ATH ground truth is
built from ("a frame is D where the corrected tracking marks the ball dead, and otherwise takes
the team the tracking names as owning it", CHALLENGE.md), so these labels need no reconstruction
from events. They are used as-is.

Output: a sample_games.Game, with the two halves laid end to end and a 2,000-frame X gap between
them, so every tool that handles BAR-ATH's halftime handles this the same way.
"""

import glob
import os
import re
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DFL_DIR = os.environ.get("MTAH_DFL_DIR") or os.path.join(ROOT, "data", "dfl")
GAP = 2000

_FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)"[^>]*TeamId="([^"]+)"[^>]*PersonId="([^"]+)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]* X="(-?[\d.]+)" Y="(-?[\d.]+)"')
_BALLFLAGS = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def match_ids():
    """Match ids available on disk: raw positions + match information, or an already-built cache
    (so the 2.6 GB of raw XML can be deleted once parsed)."""
    out = set()
    for path in glob.glob(os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_*.xml")):
        mid = path.rsplit("_", 1)[-1][:-4]
        if glob.glob(os.path.join(DFL_DIR, "DFL_02_01_matchinformation_*_%s.xml" % mid)):
            out.add(mid)
    for path in glob.glob(os.path.join(DFL_DIR, "cache", "DFL-MAT-*.npz")):
        out.add(os.path.basename(path)[:-4])
    return sorted(out)


def _teams(mid):
    path = glob.glob(os.path.join(DFL_DIR, "DFL_02_01_matchinformation_*_%s.xml" % mid))[0]
    root = ET.parse(path).getroot()
    home = guest = None
    pitch = (105.0, 68.0)
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "Team":
            if el.get("Role") == "home":
                home = el.get("TeamId")
            elif el.get("Role") == "guest":
                guest = el.get("TeamId")
        if tag == "Environment" and el.get("PitchX"):
            pitch = (float(el.get("PitchX")), float(el.get("PitchY")))
    return home, guest, pitch


def parse(mid):
    """{half: {person: (N, X, Y)}}, ball flags, teams and pitch for one match. Streams the XML."""
    home, guest, pitch = _teams(mid)
    path = glob.glob(os.path.join(DFL_DIR, "DFL_04_03_positions_raw_observed_*_%s.xml" % mid))[0]
    sets = {}
    flags = {}
    team_of = {}
    cur = None
    buf_n, buf_x, buf_y, buf_p, buf_s = [], [], [], [], []

    def flush():
        if cur is None:
            return
        half, team, person = cur
        sets.setdefault(half, {})[person] = (np.array(buf_n, dtype=np.int64),
                                             np.array(buf_x, dtype=np.float32),
                                             np.array(buf_y, dtype=np.float32))
        if team == "BALL":
            flags[half] = (np.array(buf_n, dtype=np.int64), np.array(buf_p, dtype=np.int8),
                           np.array(buf_s, dtype=np.int8))

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = _FRAME.match(line)
                if m:
                    buf_n.append(int(m.group(1)))
                    buf_x.append(float(m.group(2)))
                    buf_y.append(float(m.group(3)))
                    if cur is not None and cur[1] == "BALL":
                        f = _BALLFLAGS.search(line)
                        buf_p.append(int(f.group(1)) if f else 0)
                        buf_s.append(int(f.group(2)) if f else 0)
            elif line.startswith("<FrameSet"):
                flush()
                m = _FRAMESET.match(line)
                cur = (m.group(1), m.group(2), m.group(3))
                team_of[m.group(3)] = m.group(2)
                buf_n, buf_x, buf_y, buf_p, buf_s = [], [], [], [], []
    flush()
    return sets, flags, team_of, home, guest, pitch


def build(mid):
    """(ball, xy, team, labels, pitch) on one timeline, normalised 0-1, labels H/A/D/X codes."""
    sets, flags, team_of, home, guest, pitch = parse(mid)
    L, W = pitch
    players = [p for p, t in team_of.items() if t in (home, guest)]
    team = np.array([0 if team_of[p] == home else 1 for p in players], dtype=np.int8)
    halves = [h for h in ("firstHalf", "secondHalf", "firstHalfExtra", "secondHalfExtra")
              if h in flags]
    spans = []
    total = 0
    for h in halves:
        n0, n1 = int(flags[h][0].min()), int(flags[h][0].max())
        spans.append((h, n0, n1, total))
        total += (n1 - n0 + 1) + GAP
    T = total - GAP
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    xy = np.full((T, len(players), 2), np.nan, dtype=np.float32)
    labels = np.full(T, 3, dtype=np.int8)
    for h, n0, n1, off in spans:
        bn, poss, status = flags[h]
        idx = off + (bn - n0)
        ball_person = [p for p, t in team_of.items() if t == "BALL"][0]
        _, bx, by = sets[h][ball_person]
        ball[idx, 0] = (bx + L / 2.0) / L
        ball[idx, 1] = (by + W / 2.0) / W
        lab = np.where(status == 1, np.where(poss == 1, 0, np.where(poss == 2, 1, 2)), 2)
        labels[idx] = lab.astype(np.int8)
        for j, p in enumerate(players):
            if p not in sets[h]:
                continue
            pn, px, py = sets[h][p]
            ok = (pn >= n0) & (pn <= n1)
            xy[off + (pn[ok] - n0), j, 0] = (px[ok] + L / 2.0) / L
            xy[off + (pn[ok] - n0), j, 1] = (py[ok] + W / 2.0) / W
    return ball, xy, team, labels, pitch


def load(mid):
    """Cached build(), as a sample_games.Game."""
    import sys
    sys.path.insert(0, HERE)
    import sample_games
    cache = os.path.join(DFL_DIR, "cache", mid + ".npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        ball, xy, team, labels, pitch = z["ball"], z["xy"], z["team"], z["labels"], \
            tuple(z["pitch"])
    else:
        ball, xy, team, labels, pitch = build(mid)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, ball=ball, xy=xy, team=team, labels=labels, pitch=np.array(pitch))
    return sample_games.Game("dfl_" + mid, ball, xy, team, labels, None, pitch=pitch, fps=25)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    import sample_games
    for mid in match_ids():
        g = load(mid)
        print(g.name, g.xy.shape, sample_games.summary(g.labels))

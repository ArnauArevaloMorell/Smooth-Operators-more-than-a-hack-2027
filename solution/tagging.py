"""Smart Tagging (SportsCode XML) -> set-piece windows and team-phase intervals, in video frames.

Standard library only. Time base per DATA.md: seconds from the first tactical video frame, so
frame = round(seconds * fps). Teams are mapped to sides by name against the ATD metadata's own
team names, with the spellings DATA.md documents ('Barcelona', 'Atletic Club') as a fallback.
'N/A' (every throw-in) maps to None and is handled as "restart, team unknown".
"""

import xml.etree.ElementTree as ET

import numpy as np

# Phase codes whose Team tag names the side IN possession, and those naming the side WITHOUT it.
IN_POSSESSION = ("build up", "progression", "creating chances", "attacking transition",
                 "ball in final third", "ball in the box", "players in the box")
OUT_OF_POSSESSION = ("defending in attacking third", "defending in middle third",
                     "defending in defensive third", "defensive transition")


def _side(name, team_names):
    if not name:
        return None
    n = name.strip().lower()
    if n in ("n/a", "na", ""):
        return None
    for side, full in enumerate(team_names):
        f = (full or "").strip().lower()
        if f and (f in n or n in f or f[:5] == n[:5]):
            return side
    if "barcelona" in n:
        return 0
    if "athletic" in n or "atletic" in n:
        return 1
    return None


def load(path, team_names, fps=25):
    """(set_pieces, phases): [(start, end, side|None)] and [(start, end, side_in_possession)]."""
    set_pieces, phases = [], []
    for inst in ET.parse(path).getroot().iter("instance"):
        code = (inst.findtext("code") or "").strip().lower()
        start = int(round(float(inst.findtext("start")) * fps))
        end = int(round(float(inst.findtext("end")) * fps))
        tags = {}
        for label in inst.findall("label"):
            tags[(label.findtext("group") or "").strip().lower()] = label.findtext("text")
        side = _side(tags.get("team"), team_names)
        if code == "set pieces":
            set_pieces.append((start, end, side))
        elif side is not None and code in IN_POSSESSION:
            phases.append((start, end, side))
        elif side is not None and code in OUT_OF_POSSESSION:
            phases.append((start, end, 1 - side))
    return set_pieces, phases


def phase_prior(phases, T, weight):
    """Small per-frame bonus to the side a tagged phase says is in possession.

    Phases are not possessions (DATA.md), so this is a gentle nudge for the ball-less stretches,
    never strong enough on its own to create or remove a possession change the tracking contradicts.
    """
    prior = np.zeros((T, 4))
    if weight == 0:
        return prior
    for s, e, side in phases:
        s, e = max(0, s), min(T, e)
        if e > s:
            prior[s:e, side] += weight / 2.0
            prior[s:e, 1 - side] -= weight / 2.0
    return prior


def phase_agreement(phases, episodes, T):
    """Label-free estimate of how often the phases' team is right on THIS match.

    episodes: +1 / -1 / 0 per frame, frames where a team clearly controls the ball (98-99% precise
    on the development games). Returns (agreement, n_frames): the share of episode frames inside
    phases whose side matches the phase's side. No ground truth involved.
    """
    implied = np.zeros(T)
    for s, e, side in phases:
        s, e = max(0, s), min(T, e)
        implied[s:e] += 1.0 if side == 0 else -1.0
    m = (episodes != 0) & (implied != 0)
    if not m.any():
        return None, 0
    return float(np.mean(np.sign(implied[m]) == episodes[m])), int(m.sum())

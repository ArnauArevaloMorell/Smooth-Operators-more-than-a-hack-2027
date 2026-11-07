"""FIFA/EPTS tracking reader, numpy only, driven by the metadata's own DataFormatSpecification.

Why not the starter's line-by-line parser: it is correct but slow (a Python loop over 46 slots x
199k frames), and it assumes the order of <Player> elements is the order of the slots in the raw
file. The EPTS metadata states the slot layout explicitly -- PlayerChannelRef -> PlayerChannel ->
Player -> teamId -- so we read it from there and only fall back to <Player> order if the spec is
absent. The same reader loads Metrica's public Sample Game 3, which is how it is tested.

Output is a Tracking object with everything in normalised 0-1 pitch units (as the file has it):

    frame_id   (T,)        int, as in the file (ATD counts from 1)
    ball       (T, 2)      x, y, NaN where the ball is not tracked
    ball_extra (T, k)      any further ball channels (z or speed, by name in ball_channels)
    xy         (T, S, 2)   per track slot x, y, NaN where the slot is empty
    extra      (T, S, k)   further per-slot channels (e.g. speed), by name in player_channels
    team       (S,)        0 = home (first <Team> in the file), 1 = away
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np


class Tracking(object):
    def __init__(self, frame_id, ball, ball_extra, ball_channels, xy, extra, player_channels,
                 team, team_names, fps):
        self.frame_id = frame_id
        self.ball = ball
        self.ball_extra = ball_extra
        self.ball_channels = ball_channels
        self.xy = xy
        self.extra = extra
        self.player_channels = player_channels
        self.team = team
        self.team_names = team_names
        self.fps = fps

    def __len__(self):
        return len(self.frame_id)


def _strip(tag):
    return tag.split("}", 1)[-1]


def _iter(root, name):
    return [el for el in root.iter() if _strip(el.tag) == name]


def parse_metadata(meta_path):
    """Teams, players, channels and the data format spec(s) from an EPTS metadata XML."""
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in _iter(root, "Team")]
    team_names = []
    for t in _iter(root, "Team"):
        name = [c for c in t if _strip(c.tag) == "Name"]
        team_names.append(name[0].text if name else t.get("id"))
    if len(teams) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(teams)))
    player_team = {p.get("id"): teams.index(p.get("teamId")) for p in _iter(root, "Player")}
    player_order = [p.get("id") for p in _iter(root, "Player")]

    # playerChannelId -> (playerId, channelId)
    channel_of = {}
    for pc in _iter(root, "PlayerChannel"):
        channel_of[pc.get("id")] = (pc.get("playerId"), pc.get("channelId"))

    fps = 25
    for el in _iter(root, "FrameRate"):
        try:
            fps = int(float(el.text))
        except (TypeError, ValueError):
            pass

    specs = []
    for spec in _iter(root, "DataFormatSpecification"):
        # Top level: StringRegister(frame) , SplitRegister(players ';') , ball registers
        children = list(spec)
        slots, ball_channels = [], []
        for child in children:
            tag = _strip(child.tag)
            if tag == "SplitRegister":
                inner = list(child)
                if inner and _strip(inner[0].tag) == "SplitRegister":
                    for group in inner:
                        refs = [r.get("playerChannelId") for r in group
                                if _strip(r.tag) == "PlayerChannelRef"]
                        slots.append(refs)
                else:
                    ball_channels.extend(r.get("channelId") for r in inner
                                         if _strip(r.tag) == "BallChannelRef")
            elif tag == "BallChannelRef":
                ball_channels.append(child.get("channelId"))
        specs.append({
            "start": int(spec.get("startFrame")),
            "end": int(spec.get("endFrame")),
            "slots": slots,
            "ball_channels": ball_channels,
        })
    return {
        "teams": teams, "team_names": team_names, "player_team": player_team,
        "player_order": player_order, "channel_of": channel_of, "specs": specs, "fps": fps,
    }


def _parse_block(lines, n_fields):
    """Parse many identically-shaped lines at once. Returns (n_lines, n_fields) float array."""
    text = ",".join(lines)
    text = text.replace(":", ",").replace(";", ",").replace("\n", "").replace("\r", "")
    flat = np.fromstring(text, sep=",")
    if flat.size != len(lines) * n_fields:
        # Rare: a malformed line. Fall back to a safe per-line parse.
        out = np.full((len(lines), n_fields), np.nan)
        for i, line in enumerate(lines):
            vals = re.split(r"[:;,]", line.strip())
            if len(vals) == n_fields:
                out[i] = [float(v) if v not in ("", "NaN", "nan") else np.nan for v in vals]
        return out
    return flat.reshape(len(lines), n_fields)


def load_epts(meta_path, raw_path):
    """Read an EPTS metadata + raw pair into a Tracking (see module docstring)."""
    meta = parse_metadata(meta_path)
    with open(raw_path) as handle:
        lines = [line for line in handle if line.strip()]

    # Field layout from the first line, confirmed against the spec where one exists.
    first = lines[0].rstrip("\n").split(":")
    if len(first) < 3:
        raise ValueError("unexpected raw line layout: {!r}".format(lines[0][:120]))
    first_slots = first[1].split(";")
    n_slots_line = len(first_slots)
    per_slot = len(first_slots[0].split(","))
    n_ball = len(first[2].split(","))

    specs = meta["specs"]
    try:
        # The spec must describe this file: every channel resolvable, slot count matching.
        for spec in specs:
            for refs in spec["slots"]:
                meta["channel_of"][refs[0]]
        if specs and len(specs) == 1 and len(specs[0]["slots"]) != n_slots_line:
            raise KeyError("slot count")
    except KeyError:
        specs = []
    if specs:
        # Global slot list = every distinct playerId across specs, in first-seen order. With a
        # single spec (BAR-ATH) this is just the file's slot order.
        slot_ids = []
        for spec in specs:
            for refs in spec["slots"]:
                pid = meta["channel_of"].get(refs[0], (None, None))[0]
                if pid not in slot_ids:
                    slot_ids.append(pid)
        player_channels = [meta["channel_of"][r][1] for r in specs[0]["slots"][0]]
        ball_channels = specs[0]["ball_channels"] or ["x", "y"][:n_ball]
    else:
        slot_ids = meta["player_order"][:n_slots_line]
        player_channels = ["x", "y", "speed"][:per_slot]
        ball_channels = ["x", "y", "z"][:n_ball]

    S = len(slot_ids)
    T = len(lines)
    n_extra = per_slot - 2
    frame_id = np.zeros(T, dtype=np.int64)
    xy = np.full((T, S, 2), np.nan, dtype=np.float32)
    extra = np.full((T, S, max(n_extra, 0)), np.nan, dtype=np.float32)
    ball = np.full((T, 2), np.nan, dtype=np.float32)
    ball_extra = np.full((T, max(n_ball - 2, 0)), np.nan, dtype=np.float32)

    # Assign line ranges to specs by frame id. Lines are in frame order.
    ids = np.array([int(l[:l.index(":")]) for l in lines], dtype=np.int64)
    spec_ranges = specs if specs else [{"start": ids[0], "end": ids[-1], "slots": None}]
    for spec in spec_ranges:
        sel = np.flatnonzero((ids >= spec["start"]) & (ids <= spec["end"]))
        if not len(sel):
            continue
        block_lines = [lines[i] for i in sel]
        parts = block_lines[0].rstrip("\n").split(":")
        n_sl = len(parts[1].split(";"))
        n_fields = 1 + n_sl * per_slot + n_ball
        arr = _parse_block(block_lines, n_fields)
        frame_id[sel] = arr[:, 0].astype(np.int64)
        pl = arr[:, 1:1 + n_sl * per_slot].reshape(len(sel), n_sl, per_slot)
        if spec.get("slots"):
            cols = [slot_ids.index(meta["channel_of"][refs[0]][0]) for refs in spec["slots"]]
        else:
            cols = list(range(n_sl))
        xy[sel[:, None], np.array(cols)[None, :], :] = pl[:, :, :2]
        if n_extra > 0:
            extra[sel[:, None], np.array(cols)[None, :], :] = pl[:, :, 2:]
        b = arr[:, 1 + n_sl * per_slot:]
        ball[sel] = b[:, :2]
        if n_ball > 2:
            ball_extra[sel] = b[:, 2:]

    team = np.array([meta["player_team"].get(pid, 0) for pid in slot_ids], dtype=np.int8)
    return Tracking(frame_id, ball, ball_extra, ball_channels[2:], xy, extra,
                    player_channels[2:], team, meta["team_names"], meta["fps"])


def load_cached(meta_path, raw_path, cache_dir):
    """load_epts with an .npz cache next to the data (the data folder is gitignored)."""
    if cache_dir and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    stamp = "{}_{}".format(os.path.getsize(raw_path), int(os.path.getmtime(raw_path)))
    cache = os.path.join(cache_dir, os.path.basename(raw_path) + "." + stamp + ".npz") \
        if cache_dir else None
    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=False)
        return Tracking(z["frame_id"], z["ball"], z["ball_extra"], list(z["ball_channels"]),
                        z["xy"], z["extra"], list(z["player_channels"]), z["team"],
                        list(z["team_names"]), int(z["fps"]))
    tr = load_epts(meta_path, raw_path)
    if cache:
        np.savez(cache, frame_id=tr.frame_id, ball=tr.ball, ball_extra=tr.ball_extra,
                 ball_channels=np.array(tr.ball_channels, dtype="U16"), xy=tr.xy,
                 extra=tr.extra, player_channels=np.array(tr.player_channels, dtype="U16"),
                 team=tr.team, team_names=np.array(tr.team_names, dtype="U64"), fps=tr.fps)
    return tr

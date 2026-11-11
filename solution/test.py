"""Self-tests that need no match data.  python -m solution.test

1. Loader agreement: write a synthetic file in the BAR-ATH ATD layout (46 track slots of
   'x,y,speed', ball 'x,y,z', frame ids from 1), once WITH a DataFormatSpecification and once
   without, and check that solution.epts reads exactly what the organisers' own parser in
   starter/baseline.py reads -- same team per slot, same nearest-team call on every frame.
2. Decoder sanity: a clean synthetic possession sequence is recovered exactly.
3. Submission writer: a file built by make_submission.write passes the official validator.
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import baseline  # noqa: E402
import possession_metrics as pm  # noqa: E402
from solution import decode, epts, make_submission  # noqa: E402


def _write_synthetic(folder, with_spec, T=600, S=46, seed=0):
    rng = np.random.RandomState(seed)
    teams = ["T_HOME", "T_AWAY"]
    slot_team = [teams[i % 2] for i in range(S)]
    xml = ['<?xml version="1.0" encoding="utf-8"?>', "<main><Metadata>",
           "<GlobalConfig><FrameRate>25</FrameRate></GlobalConfig>", "<Teams>"]
    xml += ['<Team id="{0}"><Name>{0}</Name></Team>'.format(t) for t in teams]
    xml += ["</Teams><Players>"]
    xml += ['<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name><ShirtNumber>{0}</ShirtNumber>'
            '</Player>'.format(i + 1, slot_team[i]) for i in range(S)]
    xml += ["</Players>", "<PlayerChannels>"]
    for i in range(S):
        for c in ("x", "y", "speed"):
            xml.append('<PlayerChannel id="player{0}_{1}" playerId="P{0}" channelId="{1}"/>'
                       .format(i + 1, c))
    xml += ["</PlayerChannels></Metadata>"]
    if with_spec:
        xml.append('<DataFormatSpecifications><DataFormatSpecification startFrame="1" '
                   'endFrame="{}" separator=":"><StringRegister name="frameCount"/>'
                   '<SplitRegister separator=";">'.format(T))
        for i in range(S):
            xml.append('<SplitRegister separator=",">' + "".join(
                '<PlayerChannelRef playerChannelId="player{}_{}"/>'.format(i + 1, c)
                for c in ("x", "y", "speed")) + "</SplitRegister>")
        xml.append('</SplitRegister><SplitRegister separator=","><BallChannelRef channelId="x"/>'
                   '<BallChannelRef channelId="y"/><BallChannelRef channelId="z"/>'
                   '</SplitRegister></DataFormatSpecification></DataFormatSpecifications>')
    xml.append("</main>")
    meta = os.path.join(folder, "m.xml")
    raw = os.path.join(folder, "r.txt")
    open(meta, "w").write("\n".join(xml))
    with open(raw, "w") as h:
        for f in range(1, T + 1):
            groups = []
            for i in range(S):
                if rng.rand() < 0.5:
                    groups.append("NaN,NaN,NaN")
                else:
                    groups.append("{:.5f},{:.5f},{:.2f}".format(rng.rand(), rng.rand(),
                                                                rng.rand() * 7))
            b = "NaN,NaN,NaN" if rng.rand() < 0.4 else "{:.5f},{:.5f},{:.2f}".format(
                rng.rand(), rng.rand(), rng.rand())
            h.write("{}:{}:{}\n".format(f, ";".join(groups), b))
    return meta, raw


def test_loader():
    pitch = (105.3, 68.0)
    for with_spec in (True, False):
        folder = tempfile.mkdtemp()
        meta, raw = _write_synthetic(folder, with_spec)
        tr = epts.load_epts(meta, raw)
        ref_team = baseline.team_of_each_track(meta)
        assert np.array_equal(tr.team, ref_team), "team per slot differs"
        T = len(tr)
        ref = baseline.nearest_player_labels(raw, ref_team, T, pitch)
        mine = np.full(T, -1)
        B = tr.ball.astype(np.float64) * pitch
        P = tr.xy.astype(np.float64) * pitch
        d = np.hypot(P[..., 0] - B[:, None, 0], P[..., 1] - B[:, None, 1])
        d = np.where(np.isfinite(d), d, np.inf)
        j = np.argmin(d, axis=1)
        vis = np.isfinite(B[:, 0]) & np.isfinite(d.min(axis=1))
        mine[vis] = tr.team[j[vis]]
        assert np.array_equal(mine, ref), "nearest-team disagrees on {} frames".format(
            int((mine != ref).sum()))
        assert tr.extra.shape[2] == 1 and np.array_equal(tr.frame_id, np.arange(1, T + 1))
    print("ok  loader agrees with starter/baseline.py on BAR-ATH-layout files (with/without spec)")


def test_decoder():
    T = 1000
    truth = np.r_[np.full(300, 0), np.full(100, 2), np.full(400, 1), np.full(200, 0)]
    E = np.zeros((T, 4))
    E[truth == 0, 0] = 1.0
    E[truth == 1, 1] = 1.0
    E[truth == 2, 2:] = 1.0
    E += np.random.RandomState(0).normal(0, 0.8, E.shape)
    C = decode.transition_matrix({"switch": 20, "die": 20, "restart": 20, "restart_same": 0})
    got = decode.to_labels(decode.viterbi(E, C))
    acc = (got == truth).mean()
    assert acc > 0.97, acc
    print("ok  decoder recovers a noisy synthetic sequence ({:.1%} frames)".format(acc))


def test_models_match_features():
    """The shipped models must expect exactly the columns the feature code produces."""
    from solution import features, logreg, pipeline
    T = 600
    rng = np.random.RandomState(0)
    ball = rng.rand(T, 2).astype(np.float32)
    xy = rng.rand(T, 22, 2).astype(np.float32)
    team = np.array([0] * 11 + [1] * 11, dtype=np.int8)
    blocks = [(0, T)]
    F = features.compute(ball, xy, team, (105.3, 68.0), 25, blocks)
    widths = {"dead": features.dead_design(F, (105.3, 68.0), 25, blocks)[0].shape[1],
              "shape": features.shape_design(F, 25, blocks)[0].shape[1],
              "owner": features.owner_design(F, 25, blocks).shape[1],
              "hazard": features.hazard_design(F, 25, blocks).shape[1]}
    models = logreg.load(pipeline.MODELS_PATH)
    for name, width in widths.items():
        assert len(models[name]["mean"]) == width, (name, len(models[name]["mean"]), width)
    from solution import seqnet
    nets = seqnet.load()
    if nets is not None:
        seq_width = widths["dead"] + widths["owner"] + widths["hazard"]
        for net in nets:
            assert len(net["mean"]) == seq_width, ("seqnet", len(net["mean"]), seq_width)
        widths["sequence net x%d" % len(nets)] = seq_width
    print("ok  shipped models match the feature code ({})".format(
        ", ".join("{} {}".format(k, v) for k, v in widths.items())))


def test_writer():
    n = 199319
    codes = np.full(n, 3, dtype=np.int8)
    codes[11185:90652] = 0
    path = os.path.join(tempfile.mkdtemp(), "t.csv")
    make_submission.write(codes, path)
    assert pm.validate_submission(pm.read_label_csv(path), n) is None
    print("ok  writer output passes the official validator")


if __name__ == "__main__":
    test_loader()
    test_decoder()
    test_models_match_features()
    test_writer()

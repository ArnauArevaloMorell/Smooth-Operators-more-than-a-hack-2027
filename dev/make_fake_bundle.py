"""Write Sample Game 3 in the exact BAR-ATH bundle layout, to smoke-test make_submission.

    python dev/make_fake_bundle.py <out_dir>
    MTAH_DATA_DIR=<out_dir> python -m solution.make_submission --team smoke

46 track slots of 'x,y,speed', ball 'x,y,z', file names from games.asset_paths(), and a
SportsCode Smart Tagging XML built from simulated tags. The output is NOT a BAR-ATH submission --
only a check that the whole command runs on files shaped like the real ones.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz, sample_games, degrade
sys.path.insert(0, hz.ROOT)
import games

out = sys.argv[1]
paths = games.asset_paths(data_dir=out)
os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)
g = sample_games.load_game(3)
ball, xy = degrade.degrade(g, "medium", 1)
S = 46
T = len(ball)
# pack 35 sample slots into 46 (extra slots empty), keep team order
team = np.r_[g.team, np.zeros(S - len(g.team), dtype=np.int8)]
team[len(g.team):] = np.arange(S - len(g.team)) % 2
XY = np.full((T, S, 2), np.nan, dtype=np.float32); XY[:, :len(g.team)] = xy
xml = ['<?xml version="1.0" encoding="utf-8"?>', "<main><Metadata><GlobalConfig><FrameRate>25</FrameRate></GlobalConfig><Teams>",
       '<Team id="FCB"><Name>FC Barcelona</Name></Team><Team id="ATH"><Name>Athletic Club</Name></Team></Teams><Players>']
xml += ['<Player id="T{0}" teamId="{1}"><Name>Track {0}</Name><ShirtNumber>{0}</ShirtNumber></Player>'.format(i + 1, ("FCB", "ATH")[team[i]]) for i in range(S)]
xml += ["</Players><PlayerChannels>"]
xml += ['<PlayerChannel id="track{0}_{1}" playerId="T{0}" channelId="{1}"/>'.format(i + 1, c) for i in range(S) for c in ("x", "y", "speed")]
xml += ['</PlayerChannels></Metadata><DataFormatSpecifications><DataFormatSpecification startFrame="1" endFrame="{}" separator=":"><StringRegister name="frameCount"/><SplitRegister separator=";">'.format(T)]
xml += ['<SplitRegister separator=",">' + "".join('<PlayerChannelRef playerChannelId="track{}_{}"/>'.format(i + 1, c) for c in ("x", "y", "speed")) + "</SplitRegister>" for i in range(S)]
xml += ['</SplitRegister><SplitRegister separator=","><BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="z"/></SplitRegister></DataFormatSpecification></DataFormatSpecifications></main>']
open(paths["atd_meta"], "w").write("\n".join(xml))
with open(paths["atd_raw"], "w") as h:
    for f in range(T):
        grp = ";".join("NaN,NaN,NaN" if not np.isfinite(XY[f, i, 0]) else "%.5f,%.5f,%.2f" % (XY[f, i, 0], XY[f, i, 1], 0.0) for i in range(S))
        b = "NaN,NaN,NaN" if not np.isfinite(ball[f, 0]) else "%.5f,%.5f,0.00" % (ball[f, 0], ball[f, 1])
        h.write("%d:%s:%s\n" % (f + 1, grp, b))
tags = hz.simulate_tags(g.labels, hz.blocks_of(g.labels), 1)
inst = []
for i, (a, b, t) in enumerate(tags):
    team_txt = {0: "Barcelona", 1: "Atletic Club", None: "N/A"}[t]
    inst.append("<instance><ID>%d</ID><start>%.2f</start><end>%.2f</end><code>SET PIECES</code><label><group>Team</group><text>%s</text></label><label><group>Half</group><text>1st Half</text></label></instance>" % (i, a / 25., b / 25., team_txt))
inst.append("<instance><ID>9999</ID><start>500.0</start><end>520.0</end><code>BUILD UP</code><label><group>Team</group><text>Barcelona</text></label></instance>")
open(paths["smart_tagging"], "w").write("<file><ALL_INSTANCES>" + "".join(inst) + "</ALL_INSTANCES></file>")
print("fake bundle in", out)

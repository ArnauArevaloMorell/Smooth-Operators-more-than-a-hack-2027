"""Summarise a decoder scan: mean held-out spellF1 per setting, and a leave-one-game-out choice."""
import pickle, sys
import numpy as np
keys, grid, res = pickle.load(open(sys.argv[1], "rb"))
cases = sorted({c for c, _ in res}, key=str)
settings = sorted({v for _, v in res})
games = sorted({c[0] for c in cases})
mean = {v: np.mean([res[(c, v)][0] for c in cases]) for v in settings}
heavy = {v: np.mean([res[(c, v)][0] for c in cases if c[1] == "heavy"]) for v in settings}
ratio = {v: np.sum([res[(c, v)][1] for c in cases]) / np.sum([res[(c, v)][2] for c in cases]) for v in settings}
print(keys)
for v in sorted(settings, key=lambda v: -mean[v])[:12]:
    print(v, "ALL %.4f heavy %.4f  pred/true spells %.3f" % (mean[v], heavy[v], ratio[v]))
cur = tuple(1.0 if k.endswith("_mult") else None for k in keys)
logo = []
for g in games:
    tr = [c for c in cases if c[0] != g]; te = [c for c in cases if c[0] == g]
    best = max(settings, key=lambda v: np.mean([res[(c, v)][0] for c in tr]))
    logo.append((g, best, np.mean([res[(c, best)][0] for c in te])))
print("leave-one-game-out:", [(g, b, round(s, 4)) for g, b, s in logo], "mean %.4f" % np.mean([s for *_, s in logo]))

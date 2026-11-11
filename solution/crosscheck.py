"""Label-free cross-check of the evidence sources on a run.

How often does each source of team evidence agree with the others, on frames the decoder calls
alive? Printed next to the range on the degraded development matches. A source whose agreement
with the rest drops on the real match is the one that does not transfer -- found without labels.
Only aggregate statistics are printed.
"""

import numpy as np

from . import features

# filled from dev/crosscheck_ranges.py
REFERENCE = {
    "ball control (clear) vs decoded, ball seen": (0.953, 0.976),
    "ball control (clear) vs phase tags": (0.751, 0.88),
    "ball control (clear) vs sequence net": (0.937, 0.954),
    "ownership model vs ball control (clear)": (0.949, 0.97),
    "ownership model vs decoded, ball gap": (0.936, 0.969),
    "ownership model vs decoded, ball seen": (0.968, 0.979),
    "ownership model vs phase tags": (0.768, 0.867),
    "ownership model vs sequence net": (0.896, 0.938),
    "ownership model vs shape model": (0.93, 0.961),
    "phase tags vs decoded, ball gap": (0.754, 0.936),
    "phase tags vs decoded, ball seen": (0.785, 0.927),
    "sequence net vs decoded, ball gap": (0.848, 0.916),
    "sequence net vs decoded, ball seen": (0.935, 0.954),
    "sequence net vs phase tags": (0.77, 0.869),
    "shape model vs ball control (clear)": (0.896, 0.917),
    "shape model vs decoded, ball gap": (0.941, 0.972),
    "shape model vs decoded, ball seen": (0.92, 0.952),
    "shape model vs phase tags": (0.777, 0.864),
    "shape model vs sequence net": (0.884, 0.91),
}


def signs(F, phases=None):
    """Per-frame team sign (+1 home, -1 away, 0 no opinion) of each evidence source."""
    out = {}
    out["ownership model"] = np.sign(np.nan_to_num(F["owner_logit"]))
    out["shape model"] = np.sign(np.nan_to_num(F["shape_logit"]))
    cH, cA = features.control_kernel(F, 1.5)
    d = cH - cA
    out["ball control (clear)"] = np.where(np.abs(d) > 0.3, np.sign(d), 0.0)
    if "seq_lp" in F:
        lp = np.nan_to_num(F["seq_lp"])
        out["sequence net"] = np.sign(lp[:, 0] - lp[:, 1])
    if phases is not None:
        T = len(d)
        implied = np.zeros(T)
        for s, e, side in phases:
            implied[max(0, s):min(T, e)] += 1.0 if side == 0 else -1.0
        out["phase tags"] = np.sign(implied)
    return out


def stats(F, codes, blocks, phases=None):
    g = np.zeros(len(codes), dtype=bool)
    for s, e in blocks:
        g[s:e] = True
    alive = g & (codes < 2)
    vis = np.nan_to_num(F["vis"]).astype(bool)
    decoded = np.where(codes == 0, 1.0, -1.0)
    S = signs(F, phases)
    res = {}
    for name, sg in S.items():
        for where, m in (("ball seen", alive & vis), ("ball gap", alive & ~vis)):
            mm = m & (sg != 0)
            if mm.sum() > 100:
                res["%s vs decoded, %s" % (name, where)] = float(np.mean(sg[mm] == decoded[mm]))
    names = list(S)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = S[names[i]], S[names[j]]
            mm = alive & (a != 0) & (b != 0)
            if mm.sum() > 100:
                res["%s vs %s" % (names[i], names[j])] = float(np.mean(a[mm] == b[mm]))
    return res


def print_crosscheck(F, codes, blocks, phases=None):
    print("\nevidence cross-check  (agreement | range on the degraded development matches)")
    flagged = []
    for name, value in stats(F, codes, blocks, phases).items():
        lo_hi = REFERENCE.get(name)
        if lo_hi:
            lo, hi = lo_hi
            pad = 0.15 * (hi - lo) + 1e-9
            flag = "" if lo - pad <= value <= hi + pad else "   <-- outside"
            if flag:
                flagged.append(name)
            print("  {:<56} {:>6.3f} | {:.3f} .. {:.3f}{}".format(name, value, lo, hi, flag))
        else:
            print("  {:<56} {:>6.3f}".format(name, value))
    return flagged

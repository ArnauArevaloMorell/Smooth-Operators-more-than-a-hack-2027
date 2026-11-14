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
    "ball control (clear) vs decoded, ball seen": (0.901, 0.968),
    "ball control (clear) vs phase tags": (0.732, 0.888),
    "ball control (clear) vs sequence net": (0.89, 0.966),
    "ownership model vs ball control (clear)": (0.941, 0.972),
    "ownership model vs decoded, ball gap": (0.847, 0.943),
    "ownership model vs decoded, ball seen": (0.909, 0.95),
    "ownership model vs phase tags": (0.725, 0.872),
    "ownership model vs sequence net": (0.885, 0.941),
    "ownership model vs shape model": (0.908, 0.963),
    "phase tags vs decoded, ball gap": (0.67, 0.906),
    "phase tags vs decoded, ball seen": (0.734, 0.877),
    "sequence net vs decoded, ball gap": (0.973, 0.993),
    "sequence net vs decoded, ball seen": (0.974, 0.991),
    "sequence net vs phase tags": (0.72, 0.878),
    "shape model vs ball control (clear)": (0.838, 0.942),
    "shape model vs decoded, ball gap": (0.845, 0.944),
    "shape model vs decoded, ball seen": (0.856, 0.928),
    "shape model vs phase tags": (0.717, 0.872),
    "shape model vs sequence net": (0.85, 0.925),
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

"""Viterbi segmentation into possessions. Four hidden states, three output labels.

    H   home in possession, ball alive
    A   away in possession, ball alive
    DH  dead ball, last possession was home      -> written as D
    DA  dead ball, last possession was away      -> written as D

Splitting D in two lets the decoder learn the football fact that a restart usually goes to the
*other* team (throw-ins, goal kicks after an attack, free kicks after a foul by the side in
possession are all against the last owner -- corners are the exception), without any rule that
names a frame.

The decoder maximises  sum_t emission[t, state_t]  -  sum of transition costs.  A constant
switch cost is an L0 penalty on the number of possessions: the possession count comes out of the
evidence, and the timing of each change sits where the evidence turns, which is what spellF1's
IoU matching rewards.
"""

import numpy as np

H, A, DH, DA = 0, 1, 2, 3
NEG = -1e18


def transition_matrix(p):
    """Cost matrix C[from, to] from params (all >= 0; inf = forbidden)."""
    inf = np.inf
    C = np.zeros((4, 4))
    C[H, A] = C[A, H] = p["switch"]
    C[H, DH] = C[A, DA] = p["die"]
    C[H, DA] = C[A, DH] = inf
    C[DH, H] = C[DA, A] = p["restart"] + p["restart_same"]
    C[DH, A] = C[DA, H] = p["restart"]
    C[DH, DA] = C[DA, DH] = inf
    return C


def viterbi(E, C, init=None):
    """Best state path for emissions E (T, 4) and costs C (4, 4)."""
    T = E.shape[0]
    score = -C.T  # score[to, from]
    back = np.zeros((T, 4), dtype=np.int8)
    v = E[0].copy() if init is None else E[0] + init
    # Plain Python over 4 states is faster than numpy for tiny per-step arrays.
    sc = [[float(score[j, i]) for i in range(4)] for j in range(4)]
    vv = [float(x) for x in v]
    El = E.tolist()
    bk = []
    for t in range(1, T):
        e = El[t]
        nv = [0.0] * 4
        b = [0] * 4
        for j in range(4):
            row = sc[j]
            best, bi = vv[0] + row[0], 0
            for i in (1, 2, 3):
                c = vv[i] + row[i]
                if c > best:
                    best, bi = c, i
            nv[j] = best + e[j]
            b[j] = bi
        bk.append(b)
        vv = nv
    if bk:
        back[1:] = np.array(bk, dtype=np.int8)
    path = np.zeros(T, dtype=np.int8)
    path[-1] = int(np.argmax(vv))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def to_labels(path):
    """States -> label codes 0=H, 1=A, 2=D (the scorer's codes)."""
    out = path.copy()
    out[out >= 2] = 2
    return out

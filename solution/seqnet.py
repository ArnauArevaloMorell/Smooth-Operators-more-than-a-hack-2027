"""Temporal convolutional network over the per-frame features -- numpy inference only.

Trained with PyTorch on the development matches (dev/train_seqnet.py, Metrica sample games +
DFL/IDSSE open data, never BAR-ATH) and shipped as plain weights in solution/seqnet.npz, so the
submission needs nothing beyond numpy and is reproducible.

Architecture (identical in dev/train_seqnet.py):
    input   the feature designs of solution/features.py (dead-ball, ownership, hazard), sampled
            every STRIDE frames (5 per second), standardised
    stem    1x1 convolution to C channels
    blocks  residual: x + W2 . relu(dilated 3-tap convolution(x)), dilations 1, 2, 4, ..., 64
            or ..., 128 (stored per network) -- each step sees about +/-25 s or +/-51 s of the
            match around it; the shipped networks use 128 (+/-51 s)
    head    1x1 convolution to 3 logits: home, away, dead

Output: per-frame log-probabilities (T, 3) at the video frame rate, interpolated between steps.
Halves are processed separately, so nothing crosses halftime.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "seqnet.npz")
STRIDE = 5


def design(Xd, Xo, Xh):
    """The network's input: the three feature designs side by side."""
    return np.hstack([Xd, Xo, Xh]).astype(np.float32)


def load(path=WEIGHTS_PATH):
    """The shipped network(s): a list of weight dicts (an ensemble), or None if not shipped.

    The file stores each member under a prefix 'n<i>_'; a file without prefixes is one network.
    """
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=False)
    members = sorted({k.split("_", 1)[0] for k in z.files if k.startswith("n") and
                      k.split("_", 1)[0][1:].isdigit()})
    if not members:
        return [{k: z[k] for k in z.files}]
    return [{k[len(m) + 1:]: z[k] for k in z.files if k.startswith(m + "_")} for m in members]


def save(nets, path=WEIGHTS_PATH):
    arrays = {}
    for i, net in enumerate(nets):
        for k, v in net.items():
            arrays["n%d_%s" % (i, k)] = v
    np.savez(path, **arrays)


def _conv(x, w, b, dilation=1):
    """x (C_in, T); w (C_out, C_in, K); zero padding keeps length T (as torch padding=d*(K-1)/2)."""
    k = w.shape[2]
    if k == 1:
        return w[:, :, 0] @ x + b[:, None]
    pad = dilation * (k - 1) // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    T = x.shape[1]
    out = b[:, None] + sum(w[:, :, j] @ xp[:, j * dilation:j * dilation + T] for j in range(k))
    return out


def forward(net, X, prenormalised=False):
    """Logits (T_steps, 3) for one block of strided features X (T_steps, D)."""
    if not prenormalised:
        X = (X - net["mean"]) / net["std"]
    x = X.T.astype(np.float64)
    x = _conv(x, net["stem_w"], net["stem_b"])
    n_blocks = int(net["n_blocks"])
    for i in range(n_blocks):
        d = int(net["dilations"][i])
        h = np.maximum(_conv(x, net["b%d_w1" % i], net["b%d_b1" % i], d), 0.0)
        x = x + _conv(h, net["b%d_w2" % i], net["b%d_b2" % i])
    return _conv(x, net["head_w"], net["head_b"]).T


def log_probs(nets, X_full, blocks, X_swapped=None):
    """Per-frame (T, 3) log-probabilities over the full timeline; NaN outside blocks.

    nets: one weight dict, or a list of them -- an ensemble, whose probabilities are averaged.
    X_swapped: the same match's features computed with the two teams' roles swapped. If given,
    each network also reads that view and its home/away outputs are swapped back before averaging
    (test-time team swap: any residual bias toward one side cancels out).
    """
    if isinstance(nets, dict):
        nets = [nets]
    T = len(X_full)
    out = np.full((T, 3), np.nan)
    # Per-match normalisation (networks trained with it): standardise by THIS match's own
    # statistics over its graded, strided frames -- label-free adaptation to the tracker.
    steps = [np.arange(s + STRIDE // 2, e, STRIDE) for s, e in blocks]
    allX = np.vstack([X_full[i] for i in steps if len(i)]) if steps else X_full[:0]
    m_mean = allX.mean(axis=0) if len(allX) else 0.0
    m_std = allX.std(axis=0) + 1e-6 if len(allX) else 1.0
    for (s, e), idx in zip(blocks, steps):
        if len(idx) < 2:
            continue
        prob = 0.0
        n_views = 0
        views = [(X_full, False)] + ([(X_swapped, True)] if X_swapped is not None else [])
        for net in nets:
            for X_view, swapped in views:
                if int(net.get("permatch", 0)):
                    z = forward(net, (X_view[idx] - m_mean) / m_std, prenormalised=True)
                else:
                    z = forward(net, X_view[idx])
                z = z - z.max(axis=1, keepdims=True)
                p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
                prob = prob + (p[:, [1, 0, 2]] if swapped else p)
                n_views += 1
        lp = np.log(prob / n_views)
        frames = np.arange(s, e)
        for k in range(3):
            out[s:e, k] = np.interp(frames, idx, lp[:, k])
    return out

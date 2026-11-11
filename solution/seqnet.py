"""Temporal convolutional network over the per-frame features -- numpy inference only.

Trained with PyTorch on the development matches (dev/train_seqnet.py, Metrica sample games +
DFL/IDSSE open data, never BAR-ATH) and shipped as plain weights in solution/seqnet.npz, so the
submission needs nothing beyond numpy and is reproducible.

Architecture (identical in dev/train_seqnet.py):
    input   the feature designs of solution/features.py (dead-ball, ownership, hazard), sampled
            every STRIDE frames (5 per second), standardised
    stem    1x1 convolution to C channels
    blocks  residual: x + W2 . relu(dilated 3-tap convolution(x)), dilations 1, 2, 4, ..., 64
            -- each step sees about +/-25 s of the match around it
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
    if not os.path.isfile(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


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


def forward(net, X):
    """Logits (T_steps, 3) for one block of strided features X (T_steps, D)."""
    x = ((X - net["mean"]) / net["std"]).T.astype(np.float64)
    x = _conv(x, net["stem_w"], net["stem_b"])
    n_blocks = int(net["n_blocks"])
    for i in range(n_blocks):
        d = int(net["dilations"][i])
        h = np.maximum(_conv(x, net["b%d_w1" % i], net["b%d_b1" % i], d), 0.0)
        x = x + _conv(h, net["b%d_w2" % i], net["b%d_b2" % i])
    return _conv(x, net["head_w"], net["head_b"]).T


def log_probs(net, X_full, blocks):
    """Per-frame (T, 3) log-probabilities over the full timeline; NaN outside blocks."""
    T = len(X_full)
    out = np.full((T, 3), np.nan)
    for s, e in blocks:
        idx = np.arange(s + STRIDE // 2, e, STRIDE)
        if len(idx) < 2:
            continue
        z = forward(net, X_full[idx])
        z = z - z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        frames = np.arange(s, e)
        for k in range(3):
            out[s:e, k] = np.interp(frames, idx, lp[:, k])
    return out

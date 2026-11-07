"""A one-hidden-layer binary classifier in numpy: fit in dev/, ship as JSON, infer anywhere.

Deterministic (fixed seed, fixed batch order), dependency-free, a few seconds to train on the
sample games. Output is a logit, so it plugs into the decoder exactly like the logistic models.
"""

import numpy as np


def fit(X, y, hidden=32, epochs=6, lr=3e-3, l2=1e-4, batch=1024, seed=0, weights=None):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    n, d = Z.shape
    rng = np.random.RandomState(seed)
    W1 = rng.normal(0, 1.0 / np.sqrt(d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0 / np.sqrt(hidden), hidden)
    b2 = np.array(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    sw = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    step = 0
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            x, t, w = Z[idx], y[idx], sw[idx]
            h = np.tanh(x @ W1 + b1)
            z = h @ W2 + b2
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - t) * w / len(idx)
            gW2 = h.T @ g + l2 * W2
            gb2 = np.array(g.sum())
            gh = np.outer(g, W2) * (1 - h * h)
            gW1 = x.T @ gh + l2 * W1
            gb1 = gh.sum(axis=0)
            step += 1
            for i, gr in enumerate((gW1, gb1, gW2, gb2)):
                m[i] = 0.9 * m[i] + 0.1 * gr
                v[i] = 0.999 * v[i] + 0.001 * gr * gr
                mh = m[i] / (1 - 0.9 ** step)
                vh = v[i] / (1 - 0.999 ** step)
                params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
            W1, b1, W2, b2 = params
    return {"kind": "mlp", "mean": mu.tolist(), "std": sd.tolist(), "W1": W1.tolist(),
            "b1": b1.tolist(), "W2": W2.tolist(), "b2": float(b2)}


def logit(model, X):
    Z = (np.asarray(X, dtype=np.float64) - np.array(model["mean"])) / np.array(model["std"])
    h = np.tanh(Z @ np.array(model["W1"]) + np.array(model["b1"]))
    return h @ np.array(model["W2"]) + model["b2"]

"""L2-regularised logistic regression in numpy, with optional pairwise-product expansion.

Small on purpose: the models are trained on other games (dev/train_models.py), and shipped as a
JSON file of coefficients so the submission pipeline needs nothing beyond numpy, and is exactly
reproducible.
"""

import json

import numpy as np


def _expand(X, quad):
    if not quad:
        return X
    n = X.shape[1]
    extra = [X[:, i] * X[:, j] for i in range(n) for j in range(i, n)]
    return np.column_stack([X] + extra)


def fit(X, y, l2=1.0, quad=False, iters=60, weights=None):
    """Newton/IRLS. Returns a model dict (mean, std, coef, intercept)."""
    Z = _expand(X, quad)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0) + 1e-9
    Z = (Z - mu) / sd
    n, d = Z.shape
    Zb = np.column_stack([Z, np.ones(n)])
    w = np.zeros(d + 1)
    sw = np.ones(n) if weights is None else weights
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0
    for _ in range(iters):
        z = Zb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = Zb.T @ (sw * (p - y)) + reg * w
        Hm = (Zb * (sw * p * (1 - p))[:, None]).T @ Zb + np.diag(reg + 1e-9)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return {"mean": mu.tolist(), "std": sd.tolist(), "coef": w[:-1].tolist(),
            "intercept": float(w[-1]), "quad": bool(quad)}


def logit(model, X):
    if model.get("kind") == "mlp":
        from . import mlp
        return mlp.logit(model, X)
    Z = _expand(X, model["quad"])
    Z = (Z - np.array(model["mean"])) / np.array(model["std"])
    return Z @ np.array(model["coef"]) + model["intercept"]


def save(models, path):
    with open(path, "w") as handle:
        json.dump(models, handle, indent=1, sort_keys=True)


def load(path):
    with open(path) as handle:
        return json.load(handle)

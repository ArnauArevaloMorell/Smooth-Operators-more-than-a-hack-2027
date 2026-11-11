"""Train the temporal convolutional network (solution/seqnet.py) on the development matches.

Needs PyTorch (training only -- the submission runs the exported weights with numpy):

    python -m venv .torchenv && .torchenv/bin/pip install torch numpy pandas
    .torchenv/bin/python dev/train_seqnet.py --logo      # held-out log-probs for every case
    .torchenv/bin/python dev/train_seqnet.py --final     # -> solution/seqnet.npz

Data: Metrica sample games 1-3 (event-derived labels) and the 7 DFL/IDSSE matches (provider
flags), degraded to look like BAR-ATH's tracking (dev/degrade.py). Never BAR-ATH.
Deterministic: fixed seeds, single-threaded-equivalent ops on CPU.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
from solution import seqnet  # noqa: E402

C = 64
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
CROP = 512            # steps = 102 s
BATCH = 32
STEPS = 3000
LR = 2e-3


def case_seq(case):
    """[(X_steps, y_steps)] per half, cached."""
    path = os.path.join(hz.CACHE_DIR, "seq_%d_%s_%d.npz" % case)
    if os.path.isfile(path):
        z = np.load(path)
        n = int(z["n"])
        return [(z["X%d" % i], z["y%d" % i]) for i in range(n)]
    F, Xd, Xs, labels, blocks, tags, Xo = hz.case_data(*case)
    Xh, _ = hz.hazard_data(case)
    X = seqnet.design(Xd, Xo, Xh)
    out = []
    for s, e in blocks:
        idx = np.arange(s + seqnet.STRIDE // 2, e, seqnet.STRIDE)
        out.append((X[idx], labels[idx].astype(np.int64)))
    np.savez(path, n=len(out), **{("X%d" % i): x for i, (x, _) in enumerate(out)},
             **{("y%d" % i): y for i, (_, y) in enumerate(out)})
    return out


class Net(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.stem = nn.Conv1d(d_in, C, 1)
        self.c1 = nn.ModuleList([nn.Conv1d(C, C, 3, dilation=d, padding=d) for d in DILATIONS])
        self.c2 = nn.ModuleList([nn.Conv1d(C, C, 1) for _ in DILATIONS])
        self.drop = nn.Dropout(0.1)
        self.head = nn.Conv1d(C, 3, 1)

    def forward(self, x):
        x = self.stem(x)
        for c1, c2 in zip(self.c1, self.c2):
            x = x + c2(self.drop(torch.relu(c1(x))))
        return self.head(x)


def train(cases, seed=0, steps=STEPS, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    seqs = [sq for c in cases for sq in case_seq(c)]
    allX = np.vstack([x for x, _ in seqs])
    mean = allX.mean(axis=0)
    std = allX.std(axis=0) + 1e-6
    seqs = [(((x - mean) / std).astype(np.float32), y) for x, y in seqs]
    lengths = np.array([len(y) for _, y in seqs], dtype=float)
    net = Net(allX.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(ignore_index=3)
    net.train()
    for step in range(steps):
        xb, yb = [], []
        for _ in range(BATCH):
            i = rng.choice(len(seqs), p=lengths / lengths.sum())
            x, y = seqs[i]
            a = rng.randint(0, max(1, len(y) - CROP))
            xs, ys = x[a:a + CROP], y[a:a + CROP]
            if len(ys) < CROP:
                xs = np.pad(xs, ((0, CROP - len(ys)), (0, 0)))
                ys = np.pad(ys, (0, CROP - len(ys)), constant_values=3)
            xb.append(xs.T)
            yb.append(ys)
        xb = torch.from_numpy(np.stack(xb))
        yb = torch.from_numpy(np.stack(yb))
        loss = lossf(net(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 500 == 0 or step == steps - 1):
            print("  step %d loss %.4f" % (step, loss.item()), flush=True)
    net.eval()
    return export(net, mean, std)


def export(net, mean, std):
    w = {"mean": mean.astype(np.float64), "std": std.astype(np.float64),
         "n_blocks": np.array(len(DILATIONS)), "dilations": np.array(DILATIONS),
         "stem_w": net.stem.weight.detach().double().numpy(),
         "stem_b": net.stem.bias.detach().double().numpy(),
         "head_w": net.head.weight.detach().double().numpy(),
         "head_b": net.head.bias.detach().double().numpy()}
    for i, (c1, c2) in enumerate(zip(net.c1, net.c2)):
        w["b%d_w1" % i] = c1.weight.detach().double().numpy()
        w["b%d_b1" % i] = c1.bias.detach().double().numpy()
        w["b%d_w2" % i] = c2.weight.detach().double().numpy()
        w["b%d_b2" % i] = c2.bias.detach().double().numpy()
    # numpy inference must reproduce torch
    x = np.random.RandomState(1).randn(300, len(mean)) * std + mean
    with torch.no_grad():
        t = net(torch.from_numpy(((x - mean) / std).T[None].astype(np.float32)))[0].numpy().T
    n = seqnet.forward(w, x)
    assert np.max(np.abs(t - n)) < 1e-3, np.max(np.abs(t - n))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logo", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=3, help="ensemble size for --final")
    ap.add_argument("--metrica-only", action="store_true",
                    help="LOGO on the Metrica folds, training on Metrica cases only (ablation)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    for c in hz.CASES:
        case_seq(c)
    if a.logo:
        folds = sorted(set(hz.fold_of(g) for g in hz.GAMES), key=str)
        if a.metrica_only:
            folds = [f for f in folds if f in hz.METRICA_GAMES]
        for f in folds:
            train_cases = [c for c in hz.CASES if hz.fold_of(c[0]) != f and
                           (not a.metrica_only or c[0] in hz.METRICA_GAMES)]
            print("fold", f, "training on", len(train_cases), "cases", flush=True)
            w = train(train_cases, steps=a.steps)
            np.savez(os.path.join(hz.CACHE_DIR, "seqnet_fold_%s.npz" % f), **w)
            for c in hz.CASES:
                if hz.fold_of(c[0]) != f:
                    continue
                F, Xd, Xs, labels, blocks, tags, Xo = hz.case_data(*c)
                Xh, _ = hz.hazard_data(c)
                lp = seqnet.log_probs(w, seqnet.design(Xd, Xo, Xh), blocks)
                prefix = "seqlpM" if a.metrica_only else "seqlp"
                np.save(os.path.join(hz.CACHE_DIR, "%s_%d_%s_%d.npy" % ((prefix,) + c)),
                        lp.astype(np.float32))
    if a.final:
        nets = []
        for seed in range(a.seeds):
            print("final network, seed", seed, flush=True)
            nets.append(train(hz.CASES, seed=seed, steps=a.steps))
            seqnet.save(nets)          # saved after each member, so a partial ensemble is usable
        print("wrote", seqnet.WEIGHTS_PATH, "with", len(nets), "networks")


if __name__ == "__main__":
    main()

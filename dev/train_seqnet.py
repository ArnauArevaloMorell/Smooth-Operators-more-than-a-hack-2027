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
STEPS = 1500
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


def seq_direct(game_n, setting, seed, flip=False):
    """Like case_seq, built straight from the game (no big case cache); flip=True swaps the two
    teams' roles and the H/A labels -- the same match seen from the other side."""
    path = os.path.join(hz.CACHE_DIR, "seqX_%d_%s_%d_%d.npz" % (game_n, setting, seed, int(flip)))
    if os.path.isfile(path):
        z = np.load(path)
        return [(z["X%d" % i], z["y%d" % i]) for i in range(int(z["n"]))]
    import degrade
    import sample_games
    from solution import features
    g = sample_games.load_game(game_n)
    ball, xy = degrade.degrade(g, setting, seed)
    team = (1 - g.team) if flip else g.team
    labels = g.labels.copy()
    if flip:
        labels = np.where(labels == 0, 1, np.where(labels == 1, 0, labels)).astype(labels.dtype)
    blocks = hz.blocks_of(g.labels)
    F = features.compute(ball, xy, team, g.pitch, g.fps, blocks)
    Xd, _ = features.dead_design(F, g.pitch, g.fps, blocks)
    Xo = features.owner_design(F, g.fps, blocks)
    Xh = features.hazard_design(F, g.fps, blocks)
    X = seqnet.design(Xd, Xo, Xh)
    out = []
    for s, e in blocks:
        idx = np.arange(s + seqnet.STRIDE // 2, e, seqnet.STRIDE)
        out.append((X[idx], labels[idx].astype(np.int64)))
    np.savez(path, n=len(out), **{("X%d" % i): x for i, (x, _) in enumerate(out)},
             **{("y%d" % i): y for i, (_, y) in enumerate(out)})
    return out


MORE = (("light", 40), ("medium", 41), ("heavy", 42))   # extra DFL draws for --more
MORE2_DFL = (("light", 43), ("medium", 44), ("heavy", 45))  # --more2: three more per DFL match
MORE2_METRICA = (("light", 50), ("medium", 51), ("heavy", 52))  # ... and three per Metrica match
MORE3_DFL = (("light", 46), ("medium", 47), ("heavy", 48))  # --more3: a third round on top of --more2
MORE3_METRICA = (("light", 53), ("medium", 54), ("heavy", 55))


def training_seqs(cases, flip=False, xheavy=False, more=False, more2=False, more3=False):
    """[(X, y, case_key)] for training: the cases, plus their flipped and extra-heavy versions."""
    out = [(x, y, c) for c in cases for x, y in case_seq(c)]
    games = sorted(set(c[0] for c in cases))
    if more or more2 or more3:
        extra = [(g, st, sd) for g in games if g in hz.DFL_GAMES for st, sd in MORE]
        if more2 or more3:
            extra += [(g, st, sd) for g in games if g in hz.DFL_GAMES for st, sd in MORE2_DFL]
            extra += [(g, st, sd) for g in games if g in hz.METRICA_GAMES for st, sd in MORE2_METRICA]
        if more3:
            extra += [(g, st, sd) for g in games if g in hz.DFL_GAMES for st, sd in MORE3_DFL]
            extra += [(g, st, sd) for g in games if g in hz.METRICA_GAMES for st, sd in MORE3_METRICA]
        out += [(x, y, c) for c in extra for x, y in seq_direct(*c)]
        if flip:
            out += [(x, y, c + ("flip",)) for c in extra for x, y in seq_direct(*c, flip=True)]
    if flip:
        out += [(x, y, c + ("flip",)) for c in cases for x, y in seq_direct(*c, flip=True)]
    if xheavy:
        extra = [(g, "xheavy", 30 + i) for g in games for i in range(2 if g in hz.METRICA_GAMES else 1)]
        out += [(x, y, c) for c in extra for x, y in seq_direct(*c)]
        if flip:
            out += [(x, y, c + ("flip",)) for c in extra for x, y in seq_direct(*c, flip=True)]
    return out


class Net(nn.Module):
    def __init__(self, d_in, dilations=DILATIONS):
        super().__init__()
        self.dilations = tuple(dilations)
        self.stem = nn.Conv1d(d_in, C, 1)
        self.c1 = nn.ModuleList([nn.Conv1d(C, C, 3, dilation=d, padding=d) for d in self.dilations])
        self.c2 = nn.ModuleList([nn.Conv1d(C, C, 1) for _ in self.dilations])
        self.drop = nn.Dropout(0.1)
        self.head = nn.Conv1d(C, 3, 1)

    def forward(self, x):
        x = self.stem(x)
        for c1, c2 in zip(self.c1, self.c2):
            x = x + c2(self.drop(torch.relu(c1(x))))
        return self.head(x)


def boundary_weights(y, radius=2, weight=2.0):
    """1 + weight on steps within +/-radius of a label change (possession change or stoppage)."""
    w = np.ones(len(y), dtype=np.float32)
    ch = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] != 3) & (y[:-1] != 3)) + 1
    for c in ch:
        w[max(0, c - radius):c + radius] += weight
    return w


def train(cases, seed=0, steps=STEPS, verbose=True, flip=False, xheavy=False, permatch=False,
          boundary=0.0, more=False, dilations=DILATIONS, more2=False, more3=False):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    triples = training_seqs(cases, flip=flip, xheavy=xheavy, more=more, more2=more2, more3=more3)
    # mean / std without stacking every sequence (memory: the data set is ~4 GB)
    n = sum(len(x) for x, _, _ in triples)
    mean = sum(x.sum(axis=0, dtype=np.float64) for x, _, _ in triples) / n
    var = sum(((x - mean) ** 2).sum(axis=0) for x, _, _ in triples) / n
    std = np.sqrt(var) + 1e-6
    d_in = triples[0][0].shape[1]
    if permatch:
        # each (match, degradation, side) standardised by its own statistics, as at inference
        stats = {}
        for x, _, key in triples:
            stats.setdefault(key, []).append(x)
        stats = {k: (np.vstack(v).mean(axis=0), np.vstack(v).std(axis=0) + 1e-6)
                 for k, v in stats.items()}
        seqs = [(((x - stats[k][0]) / stats[k][1]).astype(np.float32), y) for x, y, k in triples]
    else:
        seqs = []
        while triples:                      # normalise and drop the raw copy one at a time
            x, y, _ = triples.pop()
            seqs.append((((x - mean) / std).astype(np.float32), y))
        seqs.reverse()
    wts = [boundary_weights(y, weight=boundary) if boundary else np.ones(len(y), np.float32)
           for _, y in seqs]
    lengths = np.array([len(y) for _, y in seqs], dtype=float)
    net = Net(d_in, dilations)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(ignore_index=3, reduction="none")
    net.train()
    for step in range(steps):
        xb, yb, wb = [], [], []
        for _ in range(BATCH):
            i = rng.choice(len(seqs), p=lengths / lengths.sum())
            x, y = seqs[i]
            a = rng.randint(0, max(1, len(y) - CROP))
            xs, ys, ws = x[a:a + CROP], y[a:a + CROP], wts[i][a:a + CROP]
            if len(ys) < CROP:
                xs = np.pad(xs, ((0, CROP - len(ys)), (0, 0)))
                ys = np.pad(ys, (0, CROP - len(ys)), constant_values=3)
                ws = np.pad(ws, (0, CROP - len(ws)))
            xb.append(xs.T)
            yb.append(ys)
            wb.append(ws)
        xb = torch.from_numpy(np.stack(xb))
        yb = torch.from_numpy(np.stack(yb))
        wb = torch.from_numpy(np.stack(wb))
        per = lossf(net(xb), yb)
        valid = (yb != 3).float()
        loss = (per * wb * valid).sum() / (wb * valid).sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (step % 500 == 0 or step == steps - 1):
            print("  step %d loss %.4f" % (step, loss.item()), flush=True)
    net.eval()
    return export(net, mean, std, permatch)


def export(net, mean, std, permatch=False):
    w = {"mean": mean.astype(np.float64), "std": std.astype(np.float64),
         "n_blocks": np.array(len(net.dilations)), "dilations": np.array(net.dilations),
         "permatch": np.array(int(permatch)),
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
    n = seqnet.forward(w, (x - mean) / std, prenormalised=True)
    assert np.max(np.abs(t - n)) < 1e-3, np.max(np.abs(t - n))
    return w


def main():
    global C
    ap = argparse.ArgumentParser()
    ap.add_argument("--logo", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seeds", type=int, default=3, help="ensemble size for --final")
    ap.add_argument("--flip", action="store_true", help="add team-swapped copies of the training data")
    ap.add_argument("--xheavy", action="store_true", help="add extra-heavy-noise training cases")
    ap.add_argument("--permatch", action="store_true", help="per-match input normalisation")
    ap.add_argument("--seed", type=int, default=0, help="training seed for --logo (seed-noise checks)")
    ap.add_argument("--more", action="store_true", help="three extra noise draws per DFL match")
    ap.add_argument("--more2", action="store_true",
                    help="--more plus three further draws per DFL match and three per Metrica match")
    ap.add_argument("--out", default=None,
                    help="--final: write the ensemble here instead of solution/seqnet.npz")
    ap.add_argument("--more3", action="store_true",
                    help="--more2 plus a third round: three more draws per DFL and per Metrica match")
    ap.add_argument("--dilations", type=int, nargs="+", default=list(DILATIONS))
    ap.add_argument("--boundary", type=float, default=0.0,
                    help="extra loss weight near label changes (0 = plain cross-entropy)")
    ap.add_argument("--tag", default="", help="output prefix for --logo variants")
    ap.add_argument("--metrica-folds", action="store_true", help="--logo on the Metrica folds only")
    ap.add_argument("--dfl-folds", action="store_true", help="--logo on the DFL folds only")
    ap.add_argument("--channels", type=int, default=C, help="network width")
    ap.add_argument("--resume", action="store_true",
                    help="--final: keep the members already in solution/seqnet.npz, train the rest")
    ap.add_argument("--metrica-only", action="store_true",
                    help="LOGO on the Metrica folds, training on Metrica cases only (ablation)")
    a = ap.parse_args()
    C = a.channels
    torch.set_num_threads(4)
    for c in hz.CASES:
        case_seq(c)
    if a.logo:
        folds = sorted(set(hz.fold_of(g) for g in hz.GAMES), key=str)
        if a.metrica_only or a.metrica_folds:
            folds = [f for f in folds if f in hz.METRICA_GAMES]
        if a.dfl_folds:
            folds = [f for f in folds if f not in hz.METRICA_GAMES]
        for f in folds:
            train_cases = [c for c in hz.CASES if hz.fold_of(c[0]) != f and
                           (not a.metrica_only or c[0] in hz.METRICA_GAMES)]
            print("fold", f, "training on", len(train_cases), "cases", flush=True)
            w = train(train_cases, seed=a.seed, steps=a.steps, flip=a.flip, xheavy=a.xheavy,
                      permatch=a.permatch, boundary=a.boundary, more=a.more,
                      dilations=a.dilations, more2=a.more2, more3=a.more3)
            np.savez(os.path.join(hz.CACHE_DIR, "seqnet%s_fold_%s.npz" % (a.tag, f)), **w)
            for c in hz.CASES:
                if hz.fold_of(c[0]) != f:
                    continue
                F, Xd, Xs, labels, blocks, tags, Xo = hz.case_data(*c)
                Xh, _ = hz.hazard_data(c)
                lp = seqnet.log_probs(w, seqnet.design(Xd, Xo, Xh), blocks)
                prefix = "seqlpM" if a.metrica_only else ("seqlp" + a.tag)
                np.save(os.path.join(hz.CACHE_DIR, "%s_%d_%s_%d.npy" % ((prefix,) + c)),
                        lp.astype(np.float32))
    if a.final:
        # --resume continues a partly written ensemble (members are saved one by one)
        out = a.out or seqnet.WEIGHTS_PATH
        nets = (seqnet.load(out) or []) if a.resume else []
        for seed in range(len(nets), a.seeds):
            print("final network, seed", seed, flush=True)
            nets.append(train(hz.CASES, seed=seed, steps=a.steps, flip=a.flip, xheavy=a.xheavy,
                              permatch=a.permatch, boundary=a.boundary, more=a.more,
                              dilations=a.dilations, more2=a.more2, more3=a.more3))
            seqnet.save(nets, out)     # saved after each member, so a partial ensemble is usable
        print("wrote", out, "with", len(nets), "networks")


if __name__ == "__main__":
    main()

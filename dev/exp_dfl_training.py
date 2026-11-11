"""Does adding the DFL matches to training help? Metrica-only vs Metrica+DFL models, scored on
held-out matches only (leave-one-group-out), Metrica and DFL cases reported separately."""
import json, os, sys, pickle
import multiprocessing as mp
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import harness as hz

def models_for(kind):
    path = os.path.join(hz.CACHE_DIR, "exp_models_%s.pkl" % kind)
    if os.path.isfile(path):
        return pickle.load(open(path, "rb"))
    out = {}
    for f in sorted(set(hz.fold_of(g) for g in hz.GAMES), key=str):
        train = [c for c in hz.CASES if hz.fold_of(c[0]) != f and (kind == "both" or c[0] in hz.METRICA_GAMES)]
        out[f] = hz.train(train)
        print("trained", kind, f, len(train), "cases", flush=True)
    pickle.dump(out, open(path, "wb"))
    return out

def score(args):
    case, kind = args
    p = json.load(open(os.path.join(hz.ROOT, "solution", "params.json")))
    M = pickle.load(open(os.path.join(hz.CACHE_DIR, "exp_models_%s.pkl" % kind), "rb"))[hz.fold_of(case[0])]
    labels, pred = hz.run_case(case, p, models=M)
    r = hz.score(labels, pred)
    return case, kind, r["spell_f1_mean"], r["macro_f1"]

if __name__ == "__main__":
    models_for("metrica"); models_for("both")
    with mp.Pool(4) as pool:
        res = pool.map(score, [(c, k) for c in hz.CASES for k in ("metrica", "both")])
    for k in ("metrica", "both"):
        for name, sel in (("Metrica cases", lambda c: c[0] in hz.METRICA_GAMES), ("DFL cases", lambda c: c[0] in hz.DFL_GAMES)):
            for s in ("light", "medium", "heavy"):
                r = [x for x in res if x[1] == k and sel(x[0]) and x[0][1] == s]
                if r: print("%-8s %-14s %-6s spellF1 %.4f macroF1 %.4f (n=%d)" % (k, name, s, np.mean([x[2] for x in r]), np.mean([x[3] for x in r]), len(r)))
        r = [x for x in res if x[1] == k]
        print("%-8s ALL spellF1 %.4f macroF1 %.4f\n" % (k, np.mean([x[2] for x in r]), np.mean([x[3] for x in r])), flush=True)

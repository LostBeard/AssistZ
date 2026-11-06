"""
Fit the open-set probe on the bank's training folders and the lab's recordings, and read it on validation folders only. The test folders are not touched here. Usage: open_fit.py [lab weight, default 3] [out npz] [L2, default 1e-5]
"""
import collections, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_data import BANK, CLASSES, GUNLIKE, bank
LAB_WEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(BANK, "open_probe.npz")
rows, emb, y, sp, names, subs = bank()
lab = np.load(os.path.join(BANK, "open_lab.npz"))
lx, ly = lab["X"].astype(np.float64), np.array([CLASSES.index(c) for c in lab["y"]])
K = len(CLASSES)
tr = (sp == "train") & (y >= 0)
Xb = emb[tr].reshape(-1, 512).astype(np.float64)
yb = np.repeat(y[tr], emb.shape[1])
X = np.concatenate([Xb, lx])
Y = np.concatenate([yb, ly])
src = np.concatenate([np.ones(len(yb)), LAB_WEIGHT * np.ones(len(ly))])
print(f"training {len(Y)}: bank {len(yb)}, lab {len(ly)}; per class { {CLASSES[i]: int((Y == i).sum()) for i in range(K)} }", flush=True)
count = np.array([src[Y == i].sum() for i in range(K)])
w = src / (K * np.maximum(count[Y], 1))
L2 = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-5
from scipy.optimize import minimize
n, dim = X.shape
O = np.zeros((n, K))
O[np.arange(n), Y] = 1


def loss(v):
    A, B = v[:K * dim].reshape(K, dim), v[K * dim:]
    z = X @ A.T + B
    m = z.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(axis=1))
    l = float((w * (lse - z[np.arange(n), Y])).sum()) + 0.5 * L2 * float((A * A).sum())
    p = np.exp(z - m)
    p /= p.sum(axis=1, keepdims=True)
    r = (p - O) * w[:, None]
    return l, np.concatenate([(r.T @ X + L2 * A).ravel(), r.sum(axis=0)])


res = minimize(loss, np.zeros(K * dim + K), jac=True, method="L-BFGS-B", options=dict(maxiter=400, maxcor=20))
print(f"L-BFGS: {res.nit} steps, loss {res.fun:.4f}, {res.message}", flush=True)
A, B = res.x[:K * dim].reshape(K, dim), res.x[K * dim:]
np.savez(OUT, A=A, B=B, classes=np.array(CLASSES))


def probs(e):
    z = e @ A.T + B
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


own = [CLASSES.index("own_move"), CLASSES.index("own_shot")]
GUN, GAS, ATT, IV = CLASSES.index("gunfire"), CLASSES.index("gas_strike"), CLASSES.index("infected_attack"), CLASSES.index("infected_voice")
val = sp == "val"
print("held-out validation folders:", {g: len(set(subs[val & (names == g)].tolist())) for g in sorted(set(names[val].tolist()))})
print("guns in validation:", sorted(set(subs[val & (names == "gunshot")].tolist())))
for lv, level in enumerate((-6, -24)):
    p = probs(emb[:, lv])
    p[:, own] = -1
    top, conf = p.argmax(axis=1), p.max(axis=1)
    print(f"\n=== validation, {level} dB")
    for bar in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        said = lambda c: (top == c) & (conf >= bar)
        notgun = val & ~np.isin(names, list(GUNLIKE))
        notart = val & ~np.isin(names, ["artillery", "explosion"])
        notinf = val & ~np.isin(names, ["infected voice", "infected movement"])
        realg = val & (names == "gunshot")
        print(f" bar {bar}: false gunfire {100 * said(GUN)[notgun].mean():5.1f}%  false gas {100 * said(GAS)[notart].mean():5.1f}%  false infected attack {100 * said(ATT)[notinf].mean():5.1f}%  | real gunshots called gunfire {100 * said(GUN)[realg].mean():5.1f}% of {realg.sum()}")
    print(f" {'group':<20}{'n':>6}   top 3 names at bar 0 (%)")
    for g in sorted(set(names[val].tolist()), key=lambda g: -(names[val] == g).sum()):
        k = val & (names == g)
        c = collections.Counter(top[k].tolist()).most_common(3)
        print(f" {g:<20}{k.sum():>6}   " + ", ".join(f"{CLASSES[i]} {100 * v / k.sum():.0f}" for i, v in c))

"""
Whole-gun folds. Every gun outside the old test folders takes a turn as a gun the probe never trained on. False alarms are read on the validation kinds of sound, which never train. The old test folders stay out of all of this. Usage: open_cv.py <folds> <class weight power, 1 = balanced> [lab weight, default 3] [L2, default 1e-4]
"""
import hashlib, json, os, sys, time
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from open_data import BANK, CLASSES, GUNLIKE, bank
FOLDS, POWER = int(sys.argv[1]), float(sys.argv[2])
LAB = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
L2 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-4
K, DIM = len(CLASSES), 512
rows, emb, y, sp, names, subs = bank()
lab = np.load(os.path.join(BANK, "open_lab.npz"))
lx, ly = lab["X"].astype(np.float64), np.array([CLASSES.index(c) for c in lab["y"]])
GUNG = ["gunshot", "gunshot silenced", "gun tail", "gun handling", "gun unclear"]
isgun = np.isin(names, GUNG) & np.array([not rows[i]["path"].lower().startswith("tails/") for i in range(len(rows))])
guns = sorted(set(subs[isgun & (sp != "test")].tolist()), key=lambda g: hashlib.sha1(g.encode()).hexdigest())
fold_of = {g: i % FOLDS for i, g in enumerate(guns)}
print(f"{len(guns)} guns in {FOLDS} folds, power {POWER}, lab weight {LAB}, L2 {L2}", flush=True)
OWN = [CLASSES.index("own_move"), CLASSES.index("own_shot")]
GUN, GAS, ATT = CLASSES.index("gunfire"), CLASSES.index("gas_strike"), CLASSES.index("infected_attack")
BAR = np.full(K, 0.3)
for c in ("gas_strike", "heli_crash"):
    BAR[CLASSES.index(c)] = 0.5


def fit(mask):
    Xb = emb[mask].reshape(-1, DIM).astype(np.float64)
    X = np.concatenate([Xb, lx])
    Y = np.concatenate([np.repeat(y[mask], emb.shape[1]), ly])
    src = np.concatenate([np.ones(len(Xb)), LAB * np.ones(len(ly))])
    count = np.array([src[Y == i].sum() for i in range(K)])
    w = src * np.maximum(count[Y], 1) ** -POWER
    w /= w.sum()
    n = len(Y)
    O = np.zeros((n, K))
    O[np.arange(n), Y] = 1

    def loss(v):
        A, B = v[:K * DIM].reshape(K, DIM), v[K * DIM:]
        z = X @ A.T + B
        m = z.max(axis=1, keepdims=True)
        l = float((w * (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)) - z[np.arange(n), Y])).sum()) + 0.5 * L2 * float((A * A).sum())
        p = np.exp(z - m)
        p /= p.sum(axis=1, keepdims=True)
        r = (p - O) * w[:, None]
        return l, np.concatenate([(r.T @ X + L2 * A).ravel(), r.sum(axis=0)])

    res = minimize(loss, np.zeros(K * DIM + K), jac=True, method="L-BFGS-B", options=dict(maxiter=400, maxcor=20))
    return res.x[:K * DIM].reshape(K, DIM), res.x[K * DIM:], res.nit


def said(A, B, e):
    z = e @ A.T + B
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    p[:, OWN] = -1
    top = p.argmax(axis=1)
    conf = p[np.arange(len(p)), top]
    out = top.copy()
    out[conf < BAR[top]] = -1
    return out


# Every clip's word from the fold that never saw its gun; everything else from fold 0.
words = np.full((len(rows), 2), -9)
falses = []
t0 = time.time()
for f in range(FOLDS):
    held = isgun & np.isin(subs, [g for g in guns if fold_of[g] == f])
    train = (y >= 0) & (sp != "test") & ~held & ~((sp == "val") & ~isgun)
    A, B, nit = fit(train)
    for lv in range(2):
        c = said(A, B, emb[:, lv])
        words[held, lv] = c[held]
        v = (sp == "val") & ~isgun
        if f == 0:
            words[v, lv] = c[v]
    print(f" fold {f}: held {held.sum()} clips, {nit} steps, {time.time() - t0:.0f} s", flush=True)
out = {}
for lv, level in enumerate((-6, -24)):
    c = words[:, lv]
    v = (sp == "val") & ~isgun
    notgun = (v & ~np.isin(names, list(GUNLIKE))) | (isgun & (names == "gun handling") & (c != -9))
    notart = v & ~np.isin(names, ["artillery", "explosion"])
    notinf = v & ~np.isin(names, ["infected voice", "infected movement"])
    shots = isgun & (names == "gunshot") & (c != -9)
    print(f"\n=== {level} dB, bars .3, rare .5")
    print(f"false gunfire {100 * (c[notgun] == GUN).mean():.2f}% of {notgun.sum()}; of that gun handling {100 * (c[isgun & (names == 'gun handling') & (c != -9)] == GUN).mean():.2f}%, not guns {100 * (c[v & ~np.isin(names, list(GUNLIKE))] == GUN).mean():.2f}%")
    print(f"false gas {100 * (c[notart] == GAS).mean():.2f}%; false attack {100 * (c[notinf] == ATT).mean():.2f}%")
    print(f"unseen gunshots called gunfire {100 * (c[shots] == GUN).mean():.1f}% of {shots.sum()}, {len(set(subs[shots].tolist()))} guns")
    for g2 in ("gunshot silenced", "gun tail"):
        k = isgun & (names == g2) & (c != -9)
        print(f"  {g2} {100 * (c[k] == GUN).mean():.0f}% of {k.sum()}")
    per = sorted(((100 * (c[shots & (subs == g)] == GUN).mean(), g, int((shots & (subs == g)).sum())) for g in set(subs[shots].tolist())))
    print("  worst guns:", ", ".join(f"{g} {v:.0f}% of {n}" for v, g, n in per[:10]))
    print("  best guns:", ", ".join(f"{g} {v:.0f}%" for v, g, n in per[-5:]))
    wrong = {}
    for i in c[shots & (c != GUN)].tolist():
        wrong["no name" if i == -1 else CLASSES[i]] = wrong.get("no name" if i == -1 else CLASSES[i], 0) + 1
    print("  shots missed went to:", sorted(wrong.items(), key=lambda kv: -kv[1])[:6])
print(f"\nall folds {time.time() - t0:.0f} s")

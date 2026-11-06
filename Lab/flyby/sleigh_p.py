"""Would a confidence floor on world events separate true crashes from false ones? True crashes: world runs, each left out in turn, with their own fold's probe. False: the app's p on recordings with no crash. Read-only, and this has now looked at held-out data: any floor chosen from it needs fresh runs to test."""
import json, multiprocessing as mp, os, sys
import numpy as np
from world_classes import GATE, WORLD, STORM, ATTACK, SURVEY, CLASSES, U, gun_rows, world_rows, storm_rows, attack_rows, survey_rows, init, strike, triggers

if __name__ == "__main__":
    samples = sys.argv[1]
    with mp.Pool(4, initializer=init, initargs=(os.path.join(samples, "world"),)) as pool:
        located = pool.map(strike, [(n, when, dist) for n in WORLD for when, ev, dist in triggers(n) if ev == "gas"])
    strikes = {(n, t, d): b for n, t, d, b in located}
    world = {n: world_rows(n, strikes) for n in WORLD}
    base = [r for n in GATE for r in gun_rows(n, "_emb")] + [r for n in STORM for r in storm_rows(n)] + [r for n in ATTACK for r in attack_rows(n)] + [r for n in SURVEY for r in survey_rows(n)]

    def fit(d, steps=800, lr=0.5, l2=1.0):
        X = np.stack([r["x"] for r in d])
        y = np.array([CLASSES.index(r["y"]) for r in d])
        K = len(CLASSES)
        n, dim = X.shape
        count = np.bincount(y, minlength=K)
        w = 1.0 / (K * count[y])
        A, B = np.zeros((K, dim)), np.zeros(K)
        Y = np.eye(K)[y]
        for _ in range(steps):
            z = X @ A.T + B
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            r = (p - Y) * w[:, None]
            A -= lr * (r.T @ X + l2 / n * A)
            B -= lr * r.sum(axis=0)
        return A, B

    def soft(A, B, x):
        z = A @ x + B
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()

    true = {c: [] for c in ("gas_strike", "heli_crash", "sleigh_crash")}
    false_lo = {c: [] for c in true}
    for n in WORLD:
        A, B = fit(base + [r for m in WORLD if m != n for r in world[m]])
        for r in world[n]:
            p = soft(A, B, r["x"])
            c = CLASSES[int(np.argmax(p))]
            if c in true:
                (true if r["y"] == c else false_lo)[c].append(float(p.max()))
    print("Left-out world runs, probe's p when it names the class:")
    for c in true:
        t, f = np.array(true[c]), np.array(false_lo[c])
        print(f"  {c:<13} right {len(t):3d}: p10 {np.percentile(t, 10):.2f} median {np.median(t):.2f}; wrong {len(f):3d}: " + (f"median {np.median(f):.2f} max {f.max():.2f}" if len(f) else "none"))
    print("\nThe app's p on false world words (candidate 2E602B8D0E43, tag _r3):")
    W = ("gas strike", "heli crash", "sleigh crash")
    for name, lo in (("20261104_121745", 0), ("20261104_110232", 0), ("20261104_010112", 0), ("20261104_024720", 0)):
        ev = [json.loads(x) for x in open(f"{U}{name}/replay_r3_s0.jsonl", encoding="utf-8")]
        g = [e for e in ev if e["raw"] == "sleigh crash"]
        print(f"  {name[-6:]} sleigh crash {len(g):3d}: p {sorted(round(e['p'], 2) for e in g)[:24]}")
    t = np.array(true["sleigh_crash"])
    print("\nTrue sleigh events kept at a floor:", {f: f"{int((t >= f).sum())}/{len(t)}" for f in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)})
    fp = [e["p"] for name in ("20261104_121745", "20261104_110232", "20261104_010112", "20261104_024720") for e in map(json.loads, open(f"{U}{name}/replay_r3_s0.jsonl", encoding="utf-8")) if e["raw"] == "sleigh crash"]
    fp = np.array(fp)
    print("False sleigh events left at a floor:", {f: f"{int((fp >= f).sum())}/{len(fp)}" for f in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)})

"""
Can distance stop reading quiet guns as far? Two step probe (past 120 m, then near or far ridge), with the mode 33 guns in training, each mode 33 run held out; then fit on them all, the rifles on fresh runs never trained on. Compared with the shipped probe. Usage: gun_distance_cv.py <mode 33 run> [...]
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
from distance_model import load
ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/"
FAR = ["sniper_20261105_055044", "sniper_20261105_060740", "sniper_20261105_062337", "sniper_20261105_063920"]
NEW = [os.path.basename(p.replace("\\", "/").rstrip("/")) for p in sys.argv[1:]]
CUT = 120.0
shipped = load("D:/users/tj/Projects/AssistZ/AssistZ/AssistZ/wwwroot/models/distance-probe.json")


def rows(run, tag):
    reports, _ = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
    out = []
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near and 0 < r[3] <= 2000:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            if "embed" in e:
                out.append((np.array(e["embed"], dtype=np.float64), float(r[3]), r[2], e["label"]))
    return out


base = []
for path in sorted(glob.glob(ROOT + "sniper_*/shots_ai.csv")):
    run = os.path.dirname(path).replace("\\", "/")
    name = os.path.basename(run)
    if name in FAR or name in NEW:
        continue
    tag = next((t for t in ("_dist", "_ship", "_dir", "_rev", "_base", "_vote", "_emb", "_app") if os.path.exists(f"{run}/replay{t}_s0.jsonl")), None)
    if not tag:
        continue
    first = open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8").readline()
    if first and "embed" in json.loads(first):
        base += rows(run, tag)
for name in FAR:
    base += rows(ROOT + name, "_far")
new = {n: rows(ROOT + n, "_w") for n in NEW}
print(f"base {len(base)} shots; new guns {sum(len(v) for v in new.values())} shots over {len(new)} runs")


def ridge(X, d):
    y = np.log2(d)
    m, s = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - m) / s
    w = np.linalg.solve(Z.T @ Z + 100.0 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda q: 2 ** (((q - m) / s) @ w + y.mean())


def logistic(X, y):
    m, s = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - m) / s
    weight = np.where(y, 0.5 / max(y.sum(), 1), 0.5 / max((~y).sum(), 1))
    a, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(800):
        p = 1 / (1 + np.exp(-np.clip(Z @ a + b, -30, 30)))
        r = (p - y) * weight
        a -= 0.5 * (Z.T @ r + a / len(Z))
        b -= 0.5 * r.sum()
    return lambda q: ((q - m) / s) @ a + b


def probe(train):
    X = np.stack([t[0] for t in train])
    d = np.array([t[1] for t in train])
    far = d > CUT
    n, f, g = ridge(X[~far], d[~far]), ridge(X[far], d[far]), logistic(X, far)
    return lambda q: max(CUT, f(q)) if g(q) > 0 else n(q)


held = {}
for name in NEW:
    p = probe(base + [t for m, v in new.items() if m != name for t in v])
    for t in new[name]:
        held[id(t)] = float(p(t[0]))
STEPS = (50, 150, 300, 600)
print("\nMedian read by true range (shipped -> new guns in training, each run held out); within a factor of 2")
print(f"{'gun':<14}" + "".join(f"{s:>16}" for s in STEPS) + "   factor of 2: shipped -> new")
for w in ("CZ75", "FNX45", "MP5K", "FAL", "Mosin9130", "Winchester70"):
    g = [t for v in new.values() for t in v if t[2] == w]
    line = f"{w:<14}"
    for s in STEPS:
        k = [t for t in g if min(STEPS, key=lambda c: abs(c - t[1])) == s]
        line += f"{np.median([shipped(t[0]) for t in k]):>8.0f} ->{np.median([held[id(t)] for t in k]):>5.0f}" if k else f"{'-':>16}"
    a = np.array([shipped(t[0]) / t[1] for t in g]); b = np.array([held[id(t)] / t[1] for t in g])
    print(line + f"   {100 * ((a < 2) & (a > 0.5)).mean():.0f}% -> {100 * ((b < 2) & (b > 0.5)).mean():.0f}%")
print("\nPistol or MP5K at 50 m reading past 300 m: shipped " + str(sum(shipped(t[0]) >= 300 for v in new.values() for t in v if t[2] in ("CZ75", "FNX45", "MP5K") and t[1] < 100)) + ", new " + str(sum(held[id(t)] >= 300 for v in new.values() for t in v if t[2] in ("CZ75", "FNX45", "MP5K") and t[1] < 100)) + f" of {sum(1 for v in new.values() for t in v if t[2] in ('CZ75', 'FNX45', 'MP5K') and t[1] < 100)}")
print("\nDo the rifles hold, runs never trained on? Fit on everything above.")
p = probe(base + [t for v in new.values() for t in v])
for name, tag, label in (("sniper_20261105_074844", "_f1", "near clear"), ("sniper_20261105_080020", "_f1", "near rain"), ("sniper_20261105_151838", "_f2", "near rain 2"), ("sniper_20261105_071655", "_f1", "far clear"), ("sniper_20261105_073259", "_f1", "far rain"), ("sniper_20261105_150229", "_f2", "far clear 2")):
    g = rows(ROOT + name, tag)
    d = np.array([t[1] for t in g])
    a = np.array([shipped(t[0]) for t in g]); b = np.array([float(p(t[0])) for t in g])
    if label.startswith("near"):
        print(f"  {label:<12} n {len(g)}: median error {np.median(np.abs(a - d)):.1f} -> {np.median(np.abs(b - d)):.1f} m; within 25% {100 * (np.abs(a - d) <= 0.25 * d).mean():.1f}% -> {100 * (np.abs(b - d) <= 0.25 * d).mean():.1f}%; under 80 m reading 100+: {(a[d < 80] >= 100).sum()} -> {(b[d < 80] >= 100).sum()} of {(d < 80).sum()}")
    else:
        ra, rb = a / d, b / d
        print(f"  {label:<12} n {len(g)}: within a factor of 2 {100 * ((ra < 2) & (ra > 0.5)).mean():.1f}% -> {100 * ((rb < 2) & (rb > 0.5)).mean():.1f}%; correlation {np.corrcoef(np.log2(a), np.log2(d))[0, 1]:.2f} -> {np.corrcoef(np.log2(b), np.log2(d))[0, 1]:.2f}")

"""
Does training on world events cut the false alarms without costing gunfire? Leave one world run out: train on the 10 gate runs plus 2 world runs as "other"; test on the third, and on the recordings neither version trained on.
The test recordings are the same as the shipped probe's, so the two are compared on equal terms.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

U = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
GATE = ["20261103_160950", "20261103_164111", "20261103_190530", "20261103_200816", "20261103_231331", "20261103_232232", "20261103_233343", "20261104_001133", "20261104_002001", "20261104_002833"]
WORLD = ["20261104_053416", "20261104_055325", "20261104_061239"]
HOLD = ["20261103_153510", "20261104_010112", "20261104_024720", "20261104_021109", "20261104_021644"]


def rows(name, tag):
    reports, _ = calibrated_reports(U + name)
    out = []
    for line in open(f"{U}{name}/replay{tag}_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        near = [r for r in reports if -0.05 <= t - r[0] <= 0.15]
        if near:
            out.append((np.array(e["embed"]), 1.0, near[0][2], e["clap"], t, near[0][3]))
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            out.append((np.array(e["embed"]), 0.0, "", e["clap"], t, 0.0))
    return out


def fit(d):
    X = np.stack([r[0] for r in d])
    y = np.array([r[1] for r in d])
    w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    a, b = np.zeros(X.shape[1]), 0.0
    for _ in range(800):
        p = 1 / (1 + np.exp(-(X @ a + b)))
        r = w * (p - y)
        a -= 0.5 * (X.T @ r + 1.0 / len(y) * a)
        b -= 0.5 * r.sum()
    return lambda x: float(x @ a + b)


gate = [r for n in GATE for r in rows(n, "_emb")]
world = {n: rows(n, "_app") for n in WORLD}
hold = [r for n in HOLD for r in rows(n, "_app")]


def score(name, f, test_world):
    g = [r for r in hold if r[1] == 1]
    o = [r for r in hold if r[1] == 0]
    aug = [r for r in g if r[2] == "AugShort" and r[5] > 100]
    print(f"  {name:<34} gunfire {sum(f(r[0]) > 0 for r in g)}/{len(g)}, 150 m AUG {sum(f(r[0]) > 0 for r in aug)}/{len(aug)}; no shot {sum(f(r[0]) > 0 for r in o)}/{len(o)}; world events {sum(f(r[0]) > 0 for r in test_world)}/{len(test_world)}")


print(f"gate {len(gate)}, world {sum(len(v) for v in world.values())}, held out {len(hold)}")
shipped = fit(gate)
print("Shipped probe (trained on the gate runs only):")
for n in WORLD:
    score(n[-6:] + " tested", shipped, world[n])
print("Refit: gate runs + the other two world runs as not gunfire:")
for n in WORLD:
    f = fit(gate + [r for m in WORLD if m != n for r in world[m]])
    score(n[-6:] + " held out", f, world[n])
    m = [f(r[0]) for r in hold if r[1] == 1]
    print(f"     gunfire margin min {min(m):+.2f}, median {np.median(m):+.2f}; held-out world false alarms: " + ", ".join(f"{r[4]:.0f} s {r[3]}" for r in world[n] if f(r[0]) > 0)[:150])
print("Shipped probe's world false alarms (time, CLAP's word):")
for n in WORLD:
    print(f"  {n[-6:]}: " + ", ".join(f"{r[4]:.0f} s {r[3]}" for r in world[n] if shipped(r[0]) > 0))

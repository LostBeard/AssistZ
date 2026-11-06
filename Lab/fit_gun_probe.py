"""
Fits the gun probe AssistZ ships: which of ten guns, from CLAP's audio embedding, on every recording with a logged weapon. Class-balanced multinomial on the standardised embedding, as tested each run held out by Lab/weapon_cv.py. Only the weights ship. Re-verify on fresh runs: Lab/gun_gate.py.
Usage: python fit_gun_probe.py <out.json> <checkpoint>
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports
R = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261105_"
RUNS = {"055044": "_far", "060740": "_far", "062337": "_far", "063920": "_far", "071655": "_f1", "073259": "_f1", "150229": "_f2",
        "194158": "_w", "195820": "_w", "201808": "_w", "203409": "_w", "205010": "_w", "210617": "_w",
        "043053": "_dist", "044227": "_dist", "051052": "_ship", "052232": "_ship", "074844": "_f1", "080020": "_f1", "092756": "_ctl", "151838": "_f2"}
GUNS = ["AKM", "AugShort", "CZ75", "FAL", "FNX45", "M4A1", "MP5K", "Mosin9130", "SVD", "Winchester70"]
xs, ys = [], []
for name, tag in RUNS.items():
    reports, _ = calibrated_reports(R + name)
    ev = [json.loads(x) for x in open(f"{R}{name}/replay{tag}_s0.jsonl", encoding="utf-8")]
    for r in reports:
        if r[3] > 2000 or r[2] not in GUNS:
            continue
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            if "embed" in e:
                xs.append(e["embed"])
                ys.append(GUNS.index(r[2]))
X = np.array(xs, dtype=np.float64)
y = np.array(ys)
mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
Z = (X - mean) / std
K, n = len(GUNS), len(y)
count = np.bincount(y, minlength=K)
w = 1.0 / (K * count[y])
A, B = np.zeros((K, Z.shape[1])), np.zeros(K)
Y = np.eye(K)[y]
for _ in range(800):
    z = Z @ A.T + B
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    g = (p - Y) * w[:, None]
    A -= 0.5 * (g.T @ Z + A / n)
    B -= 0.5 * g.sum(axis=0)
probe = dict(Checkpoint=sys.argv[2], Classes=GUNS, Mean=[float(v) for v in mean], Std=[float(v) for v in std], Weights=[[float(v) for v in row] for row in A], Bias=[float(v) for v in B], Fit=dict(Steps=800, LearningRate=0.5, L2=1.0, ClassBalanced=True, ExamplesPerClass=[int(c) for c in count]))
with open(sys.argv[1], "w", encoding="ascii", newline="\n") as f:
    json.dump(probe, f, separators=(",", ":"))
got = np.argmax(Z @ A.T + B, axis=1)
print(f"{n} shots {dict(zip(GUNS, count.tolist()))}; training accuracy {(got == y).mean():.3f}; {os.path.getsize(sys.argv[1])} bytes")

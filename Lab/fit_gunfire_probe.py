"""
Fit the gunfire probe AssistZ ships, as JSON for SpawnDev.ILGPU.ML's ClapProbe.FromJson. Same algorithm as ClapProbe.FitLogistic (two classes: "other", "gunfire"; 800 steps, lr 0.5, L2 1 / N, class-balanced), on the embeddings
of the replays. Truth from the game's logs: events -50..+150 ms around a logged report are gunfire; events over 1.5 s from any report are other; the rest are left out.
A run with no shot log (lab mode 11's world events) has every event as other. <tag> may list several, comma separated: the first that exists for each run.
Usage: python fit_gunfire_probe.py <tag> <out.json> <checkpoint> <train run> [...]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

TAG, OUT, CHECKPOINT, RUNS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
X, y = [], []
for run in RUNS:
    reports, _ = calibrated_reports(run)
    path = next(p for p in (f"{run}/replay{t}_s0.jsonl" for t in TAG.split(",")) if os.path.exists(p))
    for line in open(path, encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        if any(-0.05 <= t - r[0] <= 0.15 for r in reports):
            X.append(e["embed"]); y.append(1.0)
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            X.append(e["embed"]); y.append(0.0)
X = np.array(X, dtype=np.float64)
y = np.array(y)
w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
a, b = np.zeros(X.shape[1]), 0.0
for _ in range(800):
    p = 1 / (1 + np.exp(-(X @ a + b)))
    r = w * (p - y)
    a -= 0.5 * (X.T @ r + 1.0 / len(y) * a)
    b -= 0.5 * r.sum()
probe = dict(Classes=["other", "gunfire"], Dim=int(X.shape[1]), Weights=[[float(np.float32(v)) for v in a]], Bias=[float(np.float32(b))], Checkpoint=CHECKPOINT,
             Fit=dict(Steps=800, LearningRate=0.5, L2=1.0, ClassBalanced=True, ExamplesPerClass=[int((y == 0).sum()), int(y.sum())]))
with open(OUT, "w", encoding="ascii", newline="\n") as f:
    json.dump(probe, f, separators=(",", ":"))
z = X @ a + b
print(f"{len(y)} examples ({int(y.sum())} gunfire); training accuracy gunfire {(z[y == 1] > 0).mean():.3f}, other {(z[y == 0] <= 0).mean():.3f}; wrote {OUT}, {os.path.getsize(OUT)} bytes")

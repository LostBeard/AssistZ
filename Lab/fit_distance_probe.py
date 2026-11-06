"""
Fits the distance probe AssistZ ships, in two steps, from CLAP's audio embedding on every labelled gunfire recording, in every weather: is the shot past 120 m (class-balanced logistic), then ridge 100 on the log2 distance for its side, near on the AI's shots up to 120 m, far on the lab's far ladders (mode 29). Only the weights ship. The method was tested in Lab/flyby/far_cv2.py; re-verify the shipped weights on fresh runs (Lab/far_gate.py, Lab/distance_shipped.py).
Usage: python fit_distance_probe.py <out.json> <checkpoint>
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/"
FAR = ["sniper_20261105_055044", "sniper_20261105_060740", "sniper_20261105_062337", "sniper_20261105_063920"]
# Mode 33, six more guns at 50 to 600 m: without them, quiet guns read far (a pistol at 50 m read 535).
NEWGUNS = ["sniper_20261105_194158", "sniper_20261105_195820", "sniper_20261105_201808", "sniper_20261105_203409", "sniper_20261105_205010", "sniper_20261105_210617"]
CUT = 120.0
xn, dn, xf, df = [], [], [], []


def rows(run, tag):
    reports, _ = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near and r[3] > 0:
            yield min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))["embed"], float(r[3])


for path in sorted(glob.glob(ROOT + "sniper_*/shots_ai.csv")):
    run = os.path.dirname(path).replace("\\", "/")
    if os.path.basename(run) in FAR or os.path.basename(run) in NEWGUNS:
        continue
    tag = next((t for t in ("_dist", "_ship", "_dir", "_rev", "_base", "_vote", "_emb", "_app") if os.path.exists(f"{run}/replay{t}_s0.jsonl")), None)
    if not tag:
        continue
    first = open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8").readline()
    if not first or "embed" not in json.loads(first):
        continue
    for x, d in rows(run, tag):
        if d > 2000:
            continue   # a shooter left far away, not a shot anyone hears
        (xf if d > CUT else xn).append(x)
        (df if d > CUT else dn).append(d)
for name, tag in [(n, "_far") for n in FAR] + [(n, "_w") for n in NEWGUNS]:
    for x, d in rows(ROOT + name, tag):
        (xf if d > CUT else xn).append(x)
        (df if d > CUT else dn).append(d)


def ridge(X, d):
    y = np.log2(d)
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    w = np.linalg.solve(Z.T @ Z + 100.0 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return dict(Mean=[float(v) for v in mean], Std=[float(v) for v in std], Weights=[float(v) for v in w], Intercept=float(y.mean()))


def logistic(X, far):
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    weight = np.where(far, 0.5 / far.sum(), 0.5 / (~far).sum())
    a, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(800):
        p = 1 / (1 + np.exp(-np.clip(Z @ a + b, -30, 30)))
        r = (p - far) * weight
        a -= 0.5 * (Z.T @ r + a / len(Z))
        b -= 0.5 * r.sum()
    return dict(Mean=[float(v) for v in mean], Std=[float(v) for v in std], Weights=[float(v) for v in a], Intercept=float(b))


Xn, Xf = np.array(xn, dtype=np.float64), np.array(xf, dtype=np.float64)
dn, df = np.array(dn), np.array(df)
probe = dict(Checkpoint=sys.argv[2], CutMeters=CUT, Near=ridge(Xn, dn), FarGate=logistic(np.concatenate([Xn, Xf]), np.concatenate([np.zeros(len(Xn), dtype=bool), np.ones(len(Xf), dtype=bool)])), Far=ridge(Xf, df),
             Fit=dict(NearExamples=len(dn), FarExamples=len(df), Ridge=100.0, Target="log2 metres", FarRangeMetres=[float(df.min()), float(df.max())]))
with open(sys.argv[1], "w", encoding="ascii", newline="\n") as f:
    json.dump(probe, f, separators=(",", ":"))
print(f"near {len(dn)} shots, far {len(df)} shots ({df.min():.0f}-{df.max():.0f} m); {os.path.getsize(sys.argv[1])} bytes")

"""
The distance test registered in docs/PLAN.md before mode 28 ran. Fit on every earlier recording, in every weather, ridge 100 on the log of distance; the mode 28 runs never trained on. Limits on their events: median error at most 10 m; at least 75% within 25%; correlation at least 0.7; for at least 2 of the 3 weapons the median estimate rises with each step of true distance.
Usage: python distance_gate.py <tag> <test run> [...]
"""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/"
TAG, TEST = sys.argv[1], [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]


def load(run, tags):
    tag = next((t for t in tags if os.path.exists(f"{run}/replay{t}_s0.jsonl")), None)
    if not tag:
        return []
    first = open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8").readline()
    if not first or "embed" not in json.loads(first):
        return []
    reports, shift = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
    out = []
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            out.append((r[2], float(r[3]), np.array(e["embed"], dtype=np.float64), e["label"], float(e["az"])))
    print(f"  {os.path.basename(run)}: {len(reports)} shots, {len(out)} with an event, clock {shift:.3f}")
    return out


test_names = {os.path.basename(p) for p in TEST}
train = []
print("training:")
for path in sorted(glob.glob(ROOT + "sniper_*/shots_ai.csv")):
    run = os.path.dirname(path).replace("\\", "/")
    if os.path.basename(run) not in test_names:
        train += load(run, ("_dir", "_rev", "_base", "_vote", "_emb", "_app"))
X = np.stack([t[2] for t in train])
y = np.log2([t[1] for t in train])
mx, sx = X.mean(axis=0), X.std(axis=0) + 1e-6
Z = (X - mx) / sx
w = np.linalg.solve(Z.T @ Z + 100.0 * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
est = lambda q: 2 ** (((q - mx) / sx) @ w + y.mean())
print(f"{len(train)} training events\ntest:")
results = []
rows = []
for run in TEST:
    rows += [(os.path.basename(run)[-6:],) + t for t in load(run, (TAG,))]
d = np.array([r[2] for r in rows])
p = est(np.stack([r[3] for r in rows]))
W = np.array([r[1] for r in rows])
err = np.abs(p - d)
base = np.abs(np.median(y) * 0 + 2 ** np.median(y) - d)
within = (err <= 0.25 * d).mean()
corr = np.corrcoef(np.log2(p), np.log2(d))[0, 1]
print(f"\n{len(rows)} test events; true distance p5-p95 {np.percentile(d, 5):.0f}-{np.percentile(d, 95):.0f} m")
print(f"{'PASS' if np.median(err) <= 10 else 'FAIL'} median error {np.median(err):.1f} m (limit 10); always the training median {np.median(base):.1f}")
print(f"{'PASS' if within >= 0.75 else 'FAIL'} within 25%: {100 * within:.1f}% (limit 75); always the training median {100 * (base <= 0.25 * d).mean():.1f}%")
print(f"{'PASS' if corr >= 0.7 else 'FAIL'} correlation {corr:.2f} (limit 0.7)")
steps = (30, 55, 80, 95)
rising = 0
print("Median estimate by true distance, per weapon (events):")
for wname in sorted(set(W)):
    meds = []
    txt = []
    for c in steps:
        m = (W == wname) & (np.abs(d - c) <= 9)
        if m.sum() >= 5:
            meds.append(float(np.median(p[m])))
            txt.append(f"{c}m: {meds[-1]:.0f} ({m.sum()})")
        else:
            meds.append(None)
            txt.append(f"{c}m: - ({m.sum()})")
    have = [v for v in meds if v is not None]
    up = len(have) == 4 and all(a < b for a, b in zip(have, have[1:]))
    rising += up
    print(f"  {wname:<9} {'  '.join(txt)}  {'rises' if up else 'does not'}")
print(f"{'PASS' if rising >= 2 else 'FAIL'} weapons whose estimate rises with each step: {rising} of 3 (at least 2)")
for run in sorted({r[0] for r in rows}):
    m = np.array([r[0] == run for r in rows])
    print(f"  run {run}: {m.sum()} events, error {np.median(err[m]):.1f} m, within 25% {100 * (err[m] <= 0.25 * d[m]).mean():.1f}%, correlation {np.corrcoef(np.log2(p[m]), np.log2(d[m]))[0, 1]:.2f}")
print("By true distance: " + "  ".join(f"{lo}-{hi}m: true {np.median(d[(d >= lo) & (d < hi)]):.0f} est {np.median(p[(d >= lo) & (d < hi)]):.0f} err {np.median(err[(d >= lo) & (d < hi)]):.0f} ({((d >= lo) & (d < hi)).sum()})" for lo, hi in ((0, 42), (42, 65), (65, 88), (88, 130)) if ((d >= lo) & (d < hi)).sum()))

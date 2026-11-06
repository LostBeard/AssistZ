"""Where does the 3.6 m come from? Each run left out, the estimate by true distance, by weapon, by run. Exploratory."""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/"
rows = []
for path in sorted(glob.glob(ROOT + "sniper_*/shots_ai.csv")):
    run = os.path.dirname(path).replace("\\", "/")
    tags = [t for t in ("_dir", "_rev", "_base", "_vote", "_emb", "_app") if os.path.exists(f"{run}/replay{t}_s0.jsonl")]
    if not tags:
        continue
    first = open(f"{run}/replay{tags[0]}_s0.jsonl", encoding="utf-8").readline()
    if not first or "embed" not in json.loads(first):
        continue
    reports, _ = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{tags[0]}_s0.jsonl", encoding="utf-8")]
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            rows.append(dict(run=os.path.basename(run)[-6:], weapon=r[2], dist=float(r[3]), x=np.array(e["embed"], dtype=np.float32)))
d = np.array([r["dist"] for r in rows])
y = np.log2(d)
runs = np.array([r["run"] for r in rows])
W = np.array([r["weapon"] for r in rows])
X = np.stack([r["x"] for r in rows]).astype(np.float64)
pred = np.full(len(y), np.nan)
for run in sorted(set(runs)):
    te = runs == run
    mx, sx = X[~te].mean(axis=0), X[~te].std(axis=0) + 1e-6
    Z = (X[~te] - mx) / sx
    w = np.linalg.solve(Z.T @ Z + 100.0 * np.eye(Z.shape[1]), Z.T @ (y[~te] - y[~te].mean()))
    pred[te] = ((X[te] - mx) / sx) @ w + y[~te].mean()
est = 2 ** pred
print("By true distance:")
for lo, hi in ((0, 32), (32, 38), (38, 44), (44, 48), (48, 52), (52, 58), (58, 62), (62, 70), (70, 80), (80, 95), (95, 120), (120, 200), (200, 1000)):
    m = (d >= lo) & (d < hi)
    if m.sum() > 2:
        print(f"  {lo:>4}-{hi:<4} m: {m.sum():5d} events, true median {np.median(d[m]):5.1f}, estimate median {np.median(est[m]):5.1f} (p10 {np.percentile(est[m], 10):5.1f}, p90 {np.percentile(est[m], 90):5.1f}), error {np.median(np.abs(est[m] - d[m])):5.1f}")
print("By weapon and true distance (median estimate):")
for w in sorted(set(W)):
    row = []
    for c in (25, 40, 50, 65, 75, 90, 150):
        m = (W == w) & (np.abs(d - c) < 8)
        if m.sum() > 5:
            row.append(f"{c}m:{np.median(est[m]):.0f}({m.sum()})")
    print(f"  {w:<9} {' '.join(row)}")
print("By run (left out): events, true median, estimate median, error, within-run correlation:")
for run in sorted(set(runs)):
    m = runs == run
    r = np.corrcoef(pred[m], y[m])[0, 1] if d[m].std() > 1 else float("nan")
    print(f"  {run}: {m.sum():4d}, {np.median(d[m]):6.1f}, {np.median(est[m]):6.1f}, {np.median(np.abs(est[m] - d[m])):5.1f}, r {r:.2f}")
g = (d > 30) & (d < 75)
print(f"Only 30-75 m: {g.sum()} events, error {np.median(np.abs(est[g] - d[g])):.1f} m (always the median {np.median(np.abs(np.median(d[g]) - d[g])):.1f}), correlation {np.corrcoef(pred[g], y[g])[0, 1]:.2f}")

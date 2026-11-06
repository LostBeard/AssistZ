"""Weapon names, a fairer look. The first number mixed weapons with distance and with which run they came from. Here AKM, AUG and M4A1 only, each run left out in turn; scored overall, by true distance, and per burst by majority, which is what a player would see. Also fire cadence alone. Exploratory, not gated."""
import csv, glob, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/"
W = ("AKM", "AugShort", "M4A1")
rows = []
for path in sorted(glob.glob(ROOT + "sniper_*/shots_ai.csv")):
    run = os.path.dirname(path).replace("\\", "/")
    name = os.path.basename(run)
    tags = [t for t in ("_dir", "_rev", "_base", "_vote", "_emb") if os.path.exists(f"{run}/replay{t}_s0.jsonl")]
    if not tags:
        continue
    first = open(f"{run}/replay{tags[0]}_s0.jsonl", encoding="utf-8").readline()
    if not first or "embed" not in json.loads(first):
        continue
    reports, _ = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{tags[0]}_s0.jsonl", encoding="utf-8")]
    times = sorted(r[0] for r in reports)
    burst_of = {}
    b = 0
    for i, t in enumerate(times):
        if i and t - times[i - 1] >= 1.0:
            b += 1
        burst_of[t] = b
    for r in reports:
        if r[2] not in W:
            continue
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            rows.append(dict(run=name, weapon=r[2], dist=r[3], burst=(name, burst_of[r[0]]), t=r[0], x=np.array(e["embed"])))
print(len(rows), "events from", len({r['run'] for r in rows}), "runs:", {w: sum(r["weapon"] == w for r in rows) for w in W})
print("runs holding each weapon:", {w: len({r['run'] for r in rows if r['weapon'] == w}) for w in W})


def fit(X, y, K=3, steps=500, lr=0.5, l2=1.0):
    n, dim = X.shape
    w = 1.0 / (K * np.maximum(np.bincount(y, minlength=K)[y], 1))
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


X = np.stack([r["x"] for r in rows])
y = np.array([W.index(r["weapon"]) for r in rows])
runs = np.array([r["run"] for r in rows])
pred = np.full(len(rows), -1)
for run in sorted(set(runs)):
    te = runs == run
    if len(set(y[~te].tolist())) < 3:
        continue
    A, B = fit(X[~te], y[~te])
    pred[te] = np.argmax(X[te] @ A.T + B, axis=1)
ok = pred >= 0
print(f"\nEach event alone: {(pred[ok] == y[ok]).sum()}/{ok.sum()} ({100 * (pred[ok] == y[ok]).mean():.0f}%); chance 33%")
for i, w in enumerate(W):
    m = ok & (y == i)
    print(f"  {w:<9} {(pred[m] == i).sum()}/{m.sum()} ({100 * (pred[m] == i).mean():.0f}%) said {[int((pred[m] == j).sum()) for j in range(3)]} of {W}")
print("By true distance:", {int(k): f"{100 * (pred[ok & (np.round(np.array([r['dist'] for r in rows]) / 25) * 25 == k)] == y[ok & (np.round(np.array([r['dist'] for r in rows]) / 25) * 25 == k)]).mean():.0f}% of {(ok & (np.round(np.array([r['dist'] for r in rows]) / 25) * 25 == k)).sum()}" for k in sorted(set(np.round(np.array([r['dist'] for r in rows]) / 25) * 25))})
by = {}
for r, p in zip(rows, pred):
    if p >= 0:
        by.setdefault(r["burst"], []).append((p, W.index(r["weapon"])))
right = sum(np.bincount([a for a, b in v], minlength=3).argmax() == v[0][1] for v in by.values())
print(f"Whole bursts by majority: {right}/{len(by)} ({100 * right / max(len(by), 1):.0f}%)")
for i, w in enumerate(W):
    g = [v for v in by.values() if v[0][1] == i]
    print(f"  {w:<9} {sum(np.bincount([a for a, b in v], minlength=3).argmax() == i for v in g)}/{len(g)}")
gaps = {w: [] for w in W}
for run in sorted(set(runs)):
    for w in W:
        t = sorted(r["t"] for r in rows if r["run"] == run and r["weapon"] == w)
        gaps[w] += [b - a for a, b in zip(t, t[1:]) if b - a < 1.0]
print("Event gaps within a burst, median s:", {w: round(float(np.median(g)), 3) if g else None for w, g in gaps.items()})

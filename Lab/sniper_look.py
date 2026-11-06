"""Can sound tell a sniper rifle from the others at range? Far ladders (mode 29): SVD against AKM, AUG, M4A1, each run held out, by range, class-balanced logistic on the standardised embedding. Single shots in all cases here. Usage: sniper_look.py"""
import json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
R = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261105_"
RUNS = {"055044": "_far", "060740": "_far", "062337": "_far", "063920": "_far", "071655": "_f1", "073259": "_f1", "150229": "_f2"}
STEPS = (150, 200, 300, 400, 600, 800)
rows = []
for name, tag in RUNS.items():
    reports, _ = calibrated_reports(R + name)
    ev = [json.loads(x) for x in open(f"{R}{name}/replay{tag}_s0.jsonl", encoding="utf-8")]
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            if "embed" in e:
                rows.append(dict(run=name, w=r[2], step=min(STEPS, key=lambda s: abs(s - r[3])), x=e["embed"], level=e["level"]))
print(f"{len(rows)} shots with an embedding; per weapon " + str({w: sum(r["w"] == w for r in rows) for w in sorted({r["w"] for r in rows})}))


def logistic(X, y):
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    weight = np.where(y, 0.5 / max(y.sum(), 1), 0.5 / max((~y).sum(), 1))
    a, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(800):
        p = 1 / (1 + np.exp(-np.clip(Z @ a + b, -30, 30)))
        r = (p - y) * weight
        a -= 0.5 * (Z.T @ r + a / len(Z))
        b -= 0.5 * r.sum()
    return lambda q: ((np.asarray(q, dtype=np.float64) - mean) / std) @ a + b


score = {}
for run in RUNS:
    tr = [r for r in rows if r["run"] != run]
    te = [r for r in rows if r["run"] == run]
    if not te:
        continue
    f = logistic(np.array([r["x"] for r in tr], dtype=np.float64), np.array([r["w"] == "SVD" for r in tr]))
    a = [r for r in te if r["w"] == "SVD"]
    b = [r for r in te if r["w"] != "SVD"]
    ok_a = sum(f(r["x"]) > 0 for r in a)
    ok_b = sum(f(r["x"]) <= 0 for r in b)
    print(f"  held out {run}: SVD called sniper {ok_a}/{len(a)}; others called not {ok_b}/{len(b)}")
    for r in te:
        score[id(r)] = f(r["x"]) > 0
print("By range, held out: SVD called sniper | others wrongly called sniper:")
for s in STEPS:
    a = [r for r in rows if r["step"] == s and r["w"] == "SVD"]
    b = [r for r in rows if r["step"] == s and r["w"] != "SVD"]
    print(f"  {s:>4} m: {sum(score[id(r)] for r in a)}/{len(a)} | {sum(score[id(r)] for r in b)}/{len(b)}")
print("By weapon, called sniper:", {w: f"{sum(score[id(r)] for r in rows if r['w'] == w)}/{sum(r['w'] == w for r in rows)}" for w in sorted({r['w'] for r in rows})})

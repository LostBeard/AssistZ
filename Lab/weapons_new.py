"""Does "sniper" generalise? Fit SVD against AKM, AUG, M4A1 on the far ladders, then call weapons never heard, from lab mode 33, by weapon and range. Named gunfire too. Usage: weapons_new.py <tag> <run> [...]"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
from sniper_look import logistic, rows as train  # runs its own first look when imported
TAG = sys.argv[1]
f = logistic(np.array([r["x"] for r in train], dtype=np.float64), np.array([r["w"] == "SVD" for r in train]))
STEPS = (50, 150, 300, 600)
out = []
for run in [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]:
    reports, shift = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    n = 0
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            n += 1
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            out.append(dict(w=r[2], step=min(STEPS, key=lambda s: abs(s - r[3])), d=r[3], sniper=bool(f(e["embed"]) > 0), label=e["label"], dist=e.get("dist"), level=e["level"]))
    print(f"\n{os.path.basename(run)}: {len(reports)} shots, clock {shift:+.3f} s, {n} with an event; per weapon logged {dict(collections.Counter(r[2] for r in reports))}")
print(f"\n{'weapon':<14}{'heard':>6}{'gunfire':>8}  called sniper by range (50, 150, 300, 600 m)        median level")
for w in sorted({r["w"] for r in out}):
    g = [r for r in out if r["w"] == w]
    by = " ".join(f"{sum(r['sniper'] for r in g if r['step'] == s)}/{sum(r['step'] == s for r in g)}" for s in STEPS)
    print(f"{w:<14}{len(g):>6}{sum(r['label'] in ('gunshot', 'automatic fire') for r in g):>8}  {sum(r['sniper'] for r in g)}/{len(g)}:  {by:<30} {np.median([r['level'] for r in g]):6.1f}")
print("\nNamed, by weapon:")
for w in sorted({r["w"] for r in out}):
    print(f"  {w:<14}{dict(collections.Counter(r['label'] for r in out if r['w'] == w).most_common(4))}")
print("\nDistance read, median by true range and weapon:")
for w in sorted({r["w"] for r in out}):
    g = [r for r in out if r["w"] == w and r["dist"]]
    print(f"  {w:<14}" + " ".join(f"{s}: {np.median([r['dist'] for r in g if r['step'] == s]):.0f}" for s in STEPS if any(r["step"] == s for r in g)))

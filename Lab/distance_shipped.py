"""
The shipped distance probe on a run it never trained on, from the app's own dist= (docs/PLAN.md). No refit: this scores the weights in wwwroot/models/distance-probe.json as the app ran them. Limits in each run: median error at most 10 m, at least 75% within 25%, correlation at least 0.7, at most 5% of shots under 80 m reading 100 m or more.
Usage: python distance_shipped.py <tag> <run> [...]
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

TAG = sys.argv[1]
for run in [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]:
    reports, shift = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    rows = []
    unmeasured = 0
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if not near:
            continue
        e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
        if "dist" in e:
            rows.append((r[2], float(r[3]), e["dist"]))
        else:
            unmeasured += 1
    W = np.array([x[0] for x in rows])
    d = np.array([x[1] for x in rows])
    p = np.array([x[2] for x in rows])
    err = np.abs(p - d)
    within = (err <= 0.25 * d).mean()
    corr = np.corrcoef(np.log2(p), np.log2(d))[0, 1]
    print(f"{os.path.basename(run)}: {len(reports)} shots, {len(rows)} with a distance, {unmeasured} heard but not named gunfire, clock {shift:.3f}; {len(ev)} events, {sum('dist' in e for e in ev)} with a distance")
    print(f"  {'PASS' if np.median(err) <= 10 else 'FAIL'} median error {np.median(err):.1f} m (limit 10)")
    print(f"  {'PASS' if within >= 0.75 else 'FAIL'} within 25%: {100 * within:.1f}% (limit 75)")
    print(f"  {'PASS' if corr >= 0.7 else 'FAIL'} correlation {corr:.2f} (limit 0.7)")
    close = d < 80
    wrongly = (p[close] >= 100).mean()
    print(f"  {'PASS' if wrongly <= 0.05 else 'FAIL'} shots under 80 m reading 100 m or more: {(p[close] >= 100).sum()}/{close.sum()} ({100 * wrongly:.1f}%, at most 5)")
    for name in sorted(set(W)):
        print(f"  {name:<9} " + "  ".join(f"{c}m: {np.median(p[(W == name) & (np.abs(d - c) <= 9)]):.0f} ({((W == name) & (np.abs(d - c) <= 9)).sum()})" for c in (30, 55, 80, 95) if ((W == name) & (np.abs(d - c) <= 9)).sum() >= 5))

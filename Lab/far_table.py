"""
What a far ladder run sounds like to the app, by weapon and range: shots with an event, named gunfire, what else they were called, the shipped probe's estimate, level. The probe is computed from the embedding on every event at a shot, gunfire or not.
Usage: python far_table.py <tag> <run> [...]
"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
from distance_model import load
est = load("D:/users/tj/Projects/AssistZ/AssistZ/AssistZ/wwwroot/models/distance-probe.json")
TAG = sys.argv[1]
rows = []
for run in [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]:
    reports, shift = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    n = 0
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0])) if near else None
        n += e is not None
        rows.append(dict(w=r[2], d=r[3], miss=r[4], ev=e))
    print(f"{os.path.basename(run)}: {len(reports)} shots, clock {shift:+.3f} s, {n} with an event, {len(ev)} events")
steps = (150, 200, 300, 400, 600, 800)
near_step = lambda d: min(steps, key=lambda s: abs(s - d))
print("\nweapon     range  shots event gunfire  probe med  level  called")
for wn in sorted({r["w"] for r in rows}):
    for s in steps:
        rr = [r for r in rows if r["w"] == wn and near_step(r["d"]) == s]
        if not rr:
            continue
        e = [r["ev"] for r in rr if r["ev"]]
        g = [x for x in e if x["raw"] in ("gunshot", "automatic fire")]
        p = [est(x["embed"]) for x in e if "embed" in x]
        c = collections.Counter(x["label"] for x in e)
        print(f"{wn:<10} {s:5d} {len(rr):6d} {len(e):5d} {len(g):7d} " + (f"{np.median(p):10.0f}" if p else "         -") + (f" {np.median([x['level'] for x in e]):6.1f}" if e else "      -") + "  " + str(dict(c.most_common(4))))
print("\nall weapons   range  shots event gunfire  probe p25/med/p75")
for s in steps:
    rr = [r for r in rows if near_step(r["d"]) == s]
    if rr:
        e = [r["ev"] for r in rr if r["ev"]]
        g = [x for x in e if x["raw"] in ("gunshot", "automatic fire")]
        p = [est(x["embed"]) for x in e if "embed" in x]
        print(f"              {s:5d} {len(rr):6d} {len(e):5d} {len(g):7d} " + (f"{np.percentile(p, 25):6.0f} {np.median(p):5.0f} {np.percentile(p, 75):5.0f}" if p else "     -"))
print("\nnear (2 m) against wide (25 m): event, gunfire")
for m in sorted({r["miss"] for r in rows}):
    rr = [r for r in rows if r["miss"] == m]
    e = [r["ev"] for r in rr if r["ev"]]
    print(f"  miss {m:.0f}: {len(rr)} shots, {len(e)} events, {sum(x['raw'] in ('gunshot', 'automatic fire') for x in e)} gunfire")

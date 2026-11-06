"""
The far gunfire test registered in docs/PLAN.md before its runs, from the app's own labels and dist=. In each run:
- at least 85% of shots with an event named gunfire, and at least 75% at any range with 10 or more;
- at least 85% of those read 100 m or more;
- at least 75% of those within a factor of 2;
- at most 2% of events with no shot within 1.5 s called gunfire.
Usage: python far_gate.py <tag> <run> [...]
"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

TAG = sys.argv[1]
GUN = ("gunshot", "automatic fire")
STEPS = (150, 200, 300, 400, 600, 800)
near_step = lambda d: min(STEPS, key=lambda s: abs(s - d))
every = True
for run in [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]:
    reports, shift = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    rows = []
    for r in reports:
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            rows.append((min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0])), r[3], near_step(r[3]), r[2]))
    none = [e for e in ev if all(abs(e["sample"] / 48000.0 - r[0]) > 1.5 for r in reports)]
    named = [x for x in rows if x[0]["label"] in GUN]
    measured = [x for x in named if "dist" in x[0]]
    print(f"{os.path.basename(run)}: {len(reports)} shots, clock {shift:+.3f} s, {len(rows)} with an event, {len(ev)} events")
    print("  range  shots heard gunfire  median read  called")
    worst = 1.0
    for s in STEPS:
        total = sum(near_step(r[3]) == s for r in reports)
        h = [x for x in rows if x[2] == s]
        g = [x for x in h if x[0]["label"] in GUN]
        m = [x[0]["dist"] for x in g if "dist" in x[0]]
        print(f"  {s:5d} {total:6d} {len(h):5d} {len(g):7d} " + (f"{np.median(m):12.0f}" if m else "           -") + "  " + str(dict(collections.Counter(x[0]["label"] for x in h).most_common(4))))
        if len(h) >= 10:
            worst = min(worst, len(g) / len(h))
    ratio = np.array([x[0]["dist"] / x[1] for x in measured]) if measured else np.array([0.0])
    checks = [
        (len(named) / max(len(rows), 1) >= 0.85, f"named gunfire {len(named)}/{len(rows)} ({100 * len(named) / max(len(rows), 1):.1f}%, need 85)"),
        (worst >= 0.75, f"worst range with 10+ heard {100 * worst:.0f}% (need 75)"),
        ((ratio >= 100 / np.array([x[1] for x in measured])).mean() >= 0.85 if measured else False, f"read 100 m or more {100 * (np.array([x[0]['dist'] for x in measured]) >= 100).mean():.1f}% (need 85)" if measured else "read 100 m: none"),
        (((ratio < 2) & (ratio > 0.5)).mean() >= 0.75, f"within a factor of 2 {100 * ((ratio < 2) & (ratio > 0.5)).mean():.1f}% (need 75)"),
        (sum(e["label"] in GUN for e in none) / max(len(none), 1) <= 0.02, f"no shot called gunfire {sum(e['label'] in GUN for e in none)}/{len(none)} (at most 2%)"),
    ]
    # The third check is on the reading itself.
    checks[2] = ((np.array([x[0]["dist"] for x in measured]) >= 100).mean() >= 0.85 if measured else False, checks[2][1])
    for ok, text in checks:
        print(f"  {'PASS' if ok else 'FAIL'} {text}")
        every = every and bool(ok)
print("ALL FAR CHECKS PASS" if every else "FAR CHECKS FAILED")

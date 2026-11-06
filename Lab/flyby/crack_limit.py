"""
Does a playing crack block the next one? soundObjectsLimit = 1, samples about 1 s long, shots 0.55 s apart. Uses the on/off of flyby_match2.json, and the shot log for spacing. Also rescoring without run 190530.
"""
import json, os, sys
import numpy as np
from scipy.stats import fisher_exact
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

samples, runs = sys.argv[1], sys.argv[2:]
rows = json.load(open(os.path.join(samples, "flyby_match2.json")))
for r in rows:
    r["on"] = r["crack"] >= 0.35 and 0.85 <= r["crack_pitch"] <= 1.18
fires = {}
for run in runs:
    reports, _ = calibrated_reports(run)
    fires[os.path.basename(run.rstrip("/"))[-15:]] = [(t - d / 330.0) for t, _, _, d, _ in reports]
for r in rows:
    f = fires[r["run"]]
    t = f[r["i"]]
    earlier = [x for x in f if x < t - 0.001]
    r["gap"] = t - max(earlier) if earlier else 99.0
    # Time since the last known ON crack in the same run, among the shots that qualified.
    ons = [f[o["i"]] for o in rows if o["run"] == r["run"] and o["on"] and f[o["i"]] < t - 0.001]
    r["gap_on"] = t - max(ons) if ons else 99.0
print("Chance of on by time since the previous SHOT (any):")
for lo, hi in ((0, 0.45), (0.45, 0.6), (0.6, 0.8), (0.8, 1.2), (1.2, 3), (3, 1e9)):
    for name, sel in (("under 8 m", lambda r: r["passed"] < 8), ("15 m or more", lambda r: r["passed"] >= 15)):
        g = [r for r in rows if lo <= r["gap"] < hi and sel(r)]
        if g:
            print(f"  gap {lo}-{hi if hi < 1e9 else 'inf'} s, {name:<13}: {sum(r['on'] for r in g):3d}/{len(g):<3d}")
print("Chance of on by time since the previous ON crack known in this run (qualifying shots only, so a lower bound on activity):")
for lo, hi in ((0, 0.8), (0.8, 1.2), (1.2, 2), (2, 5), (5, 1e9)):
    g = [r for r in rows if lo <= r["gap_on"] < hi]
    if g:
        print(f"  {lo}-{hi if hi < 1e9 else 'inf'} s: {sum(r['on'] for r in g):3d}/{len(g):<3d} = {100 * sum(r['on'] for r in g) / len(g):.0f}%")
keep = [r for r in rows if r["run"] != "20261103_190530"]
print(f"\nWithout run 190530 ({len(keep)} shots):")
for lo, hi in ((0, 8), (8, 15), (15, 25), (25, 1e9)):
    g = [r for r in keep if lo <= r["passed"] < hi]
    print(f"  pass {lo}-{hi if hi < 1e9 else 'inf'} m: {sum(r['on'] for r in g)}/{len(g)} = {100 * sum(r['on'] for r in g) / max(1, len(g)):.0f}%")
a = [r for r in keep if r["passed"] < 8]
b = [r for r in keep if r["passed"] >= 15]
p = fisher_exact([[sum(r["on"] for r in a), sum(not r["on"] for r in a)], [sum(r["on"] for r in b), sum(not r["on"] for r in b)]])[1]
print(f"  under 8 m {sum(r['on'] for r in a)}/{len(a)} vs 15 m or more {sum(r['on'] for r in b)}/{len(b)}, Fisher p = {p:.2e}")
g = [r for r in rows if r["run"] == "20261103_190530"]
if g: print(f"\nRun 190530: {len(g)} shots, passes {min(r['passed'] for r in g):.0f}-{max(r['passed'] for r in g):.0f} m, best crack correlation median {np.median([r['crack'] for r in g]):.3f}, max {max(r['crack'] for r in g):.3f}")

"""
The registered gun naming test, from the app's own gun= on fresh runs (docs/PLAN.md). Over all runs given:
- each gun named 10+ times: right at least 85% of the time when named; all named together at least 90%;
- family said: right at least 85%;
- pistol or MP5K under 100 m read 300 m or more: at most 10%.
Usage: python gun_gate.py <tag> <run> [...]
"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports
TAG = sys.argv[1]
NAME = {"AKM": "AKM", "AugShort": "AUG", "M4A1": "M4A1", "SVD": "SVD", "Winchester70": "Winchester", "CZ75": "CZ75", "FNX45": "FNX45"}
FAMILY = {"SVD": "sniper", "Mosin9130": "sniper", "Winchester70": "sniper", "AKM": "assault", "AugShort": "assault", "M4A1": "assault", "FAL": "assault", "MP5K": "smg", "CZ75": "pistol", "FNX45": "pistol"}
rows = []
for run in [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]:
    reports, shift = calibrated_reports(run)
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    n = 0
    for r in reports:
        if r[3] > 2000 or r[2] not in FAMILY:
            continue
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            n += 1
            rows.append(dict(w=r[2], d=r[3], gun=e.get("gun"), family=e.get("family"), dist=e.get("dist"), label=e["label"]))
    print(f"{os.path.basename(run)}: {len(reports)} shots, {n} with an event, clock {shift:+.3f} s")
named = [r for r in rows if r["gun"]]
fam = [r for r in rows if r["family"]]
print(f"{len(rows)} events at a shot: named a gun {len(named)}, a family at all {len(fam)}, neither {len(rows) - len(fam)}; not gunfire {sum(r['label'] not in ('gunshot', 'automatic fire') for r in rows)}")
every = True
print(f"{'said':<12}{'times':>6}{'right':>7}")
for said in sorted({r["gun"] for r in named}):
    g = [r for r in named if r["gun"] == said]
    right = sum(NAME.get(r["w"]) == said for r in g)
    ok = len(g) < 10 or right / len(g) >= 0.85
    every = every and ok
    print(f"{said:<12}{len(g):>6}{right:>7}  {100 * right / len(g):.0f}% {'PASS' if ok else 'FAIL'}; truly {dict(collections.Counter(r['w'] for r in g).most_common(3))}")
right = sum(NAME.get(r["w"]) == r["gun"] for r in named)
ok = right / max(len(named), 1) >= 0.90
every = every and ok
print(f"{'PASS' if ok else 'FAIL'} all named: {right}/{len(named)} ({100 * right / max(len(named), 1):.1f}%, need 90)")
right = sum(FAMILY[r["w"]] == r["family"] for r in fam)
ok = right / max(len(fam), 1) >= 0.85
every = every and ok
print(f"{'PASS' if ok else 'FAIL'} family: {right}/{len(fam)} ({100 * right / max(len(fam), 1):.1f}%, need 85)")
for f in ("sniper", "assault", "pistol", "smg"):
    g = [r for r in fam if r["family"] == f]
    print(f"   said {f}: {len(g)}, right {sum(FAMILY[r['w']] == f for r in g)}; of true {f} guns said so {sum(r['family'] == f for r in rows if FAMILY[r['w']] == f)}/{sum(FAMILY[r['w']] == f for r in rows)}")
quiet = [r for r in rows if r["w"] in ("CZ75", "FNX45", "MP5K") and r["d"] < 100 and r["dist"]]
if quiet:
    bad = sum(r["dist"] >= 300 for r in quiet)
    ok = bad / len(quiet) <= 0.10
    every = every and ok
    print(f"{'PASS' if ok else 'FAIL'} quiet guns under 100 m reading 300 m or more: {bad}/{len(quiet)} (at most 10%)")
print("ALL GUN CHECKS PASS" if every else "GUN CHECKS FAILED")

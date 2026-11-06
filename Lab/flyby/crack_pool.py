"""Crack on-rate by pass distance, discovery and holdout separately and pooled. The bin edges are exploration: only the pre-registered 8 m and 15 m comparison was tested."""
import json, os, sys
from scipy.stats import fisher_exact
s = sys.argv[1]
disc = [r for r in json.load(open(os.path.join(s, "flyby_match2_discovery.json"))) if r["run"] != "20261103_190530"]
hold = json.load(open(os.path.join(s, "flyby_match2.json")))
on = lambda r: r["crack"] >= 0.35 and 0.85 <= r["crack_pitch"] <= 1.18
print("pass m      discovery         holdout           pooled")
for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 21), (21, 24), (24, 30), (30, 1e9)):
    cells = []
    for g in (disc, hold, disc + hold):
        x = [r for r in g if lo <= r["passed"] < hi]
        cells.append(f"{sum(on(r) for r in x):3d}/{len(x):<3d} {100 * sum(on(r) for r in x) / max(1, len(x)):3.0f}%")
    print(f"{str(lo) + '-' + (str(hi) if hi < 1e9 else 'inf'):<8} {cells[0]:>14}  {cells[1]:>14}  {cells[2]:>14}")
allr = disc + hold
for edge in (8, 12, 15, 18):
    a = [r for r in allr if r["passed"] < edge]
    b = [r for r in allr if r["passed"] >= edge]
    p = fisher_exact([[sum(on(r) for r in a), sum(not on(r) for r in a)], [sum(on(r) for r in b), sum(not on(r) for r in b)]])[1]
    print(f"split at {edge:2d} m: under {sum(on(r) for r in a)}/{len(a)} = {100 * sum(on(r) for r in a) / len(a):.0f}%, over {sum(on(r) for r in b)}/{len(b)} = {100 * sum(on(r) for r in b) / len(b):.0f}%, p = {p:.1e}")
print("by run:", {run: f"{sum(on(r) for r in allr if r['run'] == run)}/{sum(r['run'] == run for r in allr)}" for run in sorted({r["run"] for r in allr})})

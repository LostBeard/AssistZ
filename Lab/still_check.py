"""The registered check on a live run with the character still: net heading at most 15 degrees, at most 5% of events call a side. Usage: still_check.py <run> <console>"""
import csv, re, sys
import numpy as np
run, console = sys.argv[1].replace("\\", "/"), sys.argv[2]
pose = list(csv.DictReader(open(run + "/pose.csv")))
yaw = np.array([float(p["yaw"]) for p in pose])
print(f"game log yaw {yaw.min():.2f} to {yaw.max():.2f}")
pat = re.compile(r"\[AssistZ\] event .* yaw=(-?[\d.]+) track=(\d+) side=(\w+) conf=([\d.]+) turned=(-?[\d.]+)")
ev = [(float(m[1]), m[3], float(m[5])) for m in map(pat.search, open(console, encoding="utf-8")) if m]
view = np.array([e[0] for e in ev])
sides = sum(e[1] != "Unknown" for e in ev)
net = abs(view[-1] - view[0])
print(f"{len(ev)} events; heading {view[0]:.1f} to {view[-1]:.1f}; every 30th: " + " ".join(f"{v:.0f}" for v in view[::30]))
print(f"{'PASS' if net <= 15 else 'FAIL'} net heading {net:.1f} deg (at most 15)")
print(f"{'PASS' if sides <= 0.05 * len(ev) else 'FAIL'} events calling a side {sides}/{len(ev)} ({100 * sides / max(len(ev), 1):.1f}%, at most 5)")

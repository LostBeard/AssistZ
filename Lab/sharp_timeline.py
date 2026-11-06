"""Does the picture sharpen after a teleport? Laplacian variance over the visible game area (menus masked), by seconds since arrival, per grid point, from the frames of the short test."""
import csv, datetime, glob, json, os, sys
import numpy as np
from PIL import Image

frames = sorted(glob.glob(r"L:\AssistZLabWork\frames\test2\*.jpg"))
run = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261105_014605"
pose = list(csv.DictReader(open(run + "/pose.csv")))
g = np.array([int(p["game_ms"]) / 1000 for p in pose])
u = np.array([sum(int(x) * k for x, k in zip(p["utc"].split(":"), (3600, 60, 1))) for p in pose], dtype=float)
off = np.median(u + 0.5 - g)
tp = g + off
pos = [(round(float(p["x"]) / 100), round(float(p["z"]) / 100)) for p in pose]
arrive = []
for i in range(1, len(pos)):
    if pos[i] != pos[i - 1] and abs(pos[i][0] - pos[i - 1][0]) + abs(pos[i][1] - pos[i - 1][1]) >= 5:
        arrive.append((tp[i], pose[i]["x"], pose[i]["z"]))
print(len(arrive), "arrivals:", [(round(a[0] % 86400), a[1], a[2]) for a in arrive][:6])
rows = []
for f in frames:
    s = os.path.basename(f)[:-4]
    t = int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + int(s[6:]) / 1000
    prev = [a for a in arrive if a[0] <= t]
    if not prev:
        continue
    a = np.asarray(Image.open(f).convert("L"), dtype=np.float32)[200:700, 560:1400]
    lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
    i = int(np.argmin(np.abs(tp - t)))
    rows.append((prev[-1][0], t - prev[-1][0], float(lap.var()), float(a.mean()), pose[i]["yaw"]))
by = {}
for a, dt, v, m, y in rows:
    by.setdefault(a, []).append((dt, v, y))
for a, r in list(by.items())[:9]:
    print(f"point at {round(a) % 86400}: " + " ".join(f"{dt:.0f}s:{v:.0f}" for dt, v, y in r[::3]))
print("yaw over time in one point:", [(round(dt, 1), y) for dt, v, y in list(by.values())[1][::3]])
for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13)):
    v = [x[2] for x in rows if lo <= x[1] < hi]
    if v:
        print(f"  {lo}-{hi} s after arrival: median sharpness {np.median(v):.0f} of {len(v)} frames")

"""
Labels the survey's frames from the pose logs: one frame per grid point and heading, focused, settled (at least 2.5 s after arrival, at least 0.6 s after the turn, the last in its dwell). Writes <frames dir>/manifest.csv and reports coverage and gaps.
Usage: python make_manifest.py <frames dir> <vision run> [...]
"""
import csv, glob, json, os, random, sys
import numpy as np

out_dir, runs = sys.argv[1], sys.argv[2:]
frames = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
stamp = lambda f: (lambda s: int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + int(s[6:9]) / 1000)(os.path.basename(f))
ftime = np.array([stamp(f) for f in frames])
focus = np.array([f.endswith("_f.jpg") for f in frames])
views = {}
expected = {}
no_frame = 0
offsets = []
for run in runs:
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    g = np.array([int(p["game_ms"]) / 1000 for p in pose])
    u = np.array([sum(int(x) * k for x, k in zip(p["utc"].split(":"), (3600, 60, 1))) for p in pose], dtype=float)
    off = float(np.median(u + 0.5 - g))
    offsets.append(round(off, 2))
    world = [e for e in csv.DictReader(open(run + "/world.csv")) if e["event"] == "view"]
    x = np.array([float(p["x"]) for p in pose])
    z = np.array([float(p["z"]) for p in pose])
    yaw = np.array([float(p["yaw"]) for p in pose])
    for e in world:
        end = int(e["game_ms"]) / 1000
        key = (round(float(e["x"])), round(float(e["z"])), int(float(e["distance"])))
        hd = key[2]
        expected[key] = float(e["y"])
        # The dwell ends at this log line; the turn was 1.5 s before it, or for heading 0 the arrival was 8 s before it.
        lo = end - (1.5 if hd else 8.0) + (0.6 if hd else 2.5) + off
        hi = end + off - 0.1
        m = (ftime >= lo) & (ftime <= hi)
        idx = np.where(m & focus)[0]
        if not len(idx):
            no_frame += 1
            continue
        i = int(idx[-1])
        k = int(np.argmin(np.abs(g + off - ftime[i])))
        if abs(x[k] - key[0]) > 3 or abs(z[k] - key[1]) > 3:
            continue
        true_yaw = float(yaw[k])
        err = abs((true_yaw - (hd if hd <= 180 else hd - 360) + 180) % 360 - 180)
        if err > 2:
            continue
        views[key] = (frames[i], float(x[k]), float(e["y"]), float(z[k]), true_yaw, os.path.basename(run))
path = os.path.join(out_dir, "manifest.csv")
with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "x", "y", "z", "yaw", "grid_x", "grid_z", "heading", "run"])
    for key, v in sorted(views.items()):
        w.writerow([os.path.basename(v[0]), v[1], v[2], v[3], v[4], key[0], key[1], key[2], v[5]])
points = {k[:2] for k in expected}
got = {k[:2] for k in views}
full = sum(all((p[0], p[1], h) in views for h in (0, 90, 180, 270)) for p in points)
print(f"{out_dir}: {len(frames)} frames, focused {int(focus.sum())}; clock offsets {offsets}")
print(f"  expected {len(expected)} views at {len(points)} points; labelled {len(views)}; points with all 4 headings {full}; no focused frame in the dwell {no_frame}")
missing = sorted(k for k in expected if k not in views)
cols = {}
for k in missing:
    cols.setdefault(k[0], 0)
    cols[k[0]] += 1
print("  missing views by column x:", cols)
xs = sorted({k[0] for k in points})
zs = sorted({k[1] for k in points})
print(f"  x {xs[0]}..{xs[-1]}, z {zs[0]}..{zs[-1]}, columns {xs}")
print("  map, X has all 4 headings, o some, . sea or none (north at top):")
for zv in reversed(range(zs[0], zs[-1] + 1, 1000)):
    print("   " + "".join(("X" if all((xv, zv, h) in views for h in (0, 90, 180, 270)) else "o") if (xv, zv) in got else ("?" if (xv, zv) in points else ".") for xv in range(xs[0], xs[-1] + 1, 1000)))
random.seed(5)
s = random.sample(sorted(views), 4)
print("  samples:")
for k in s:
    print("   ", k, views[k][0])

"""Is the live score's disagreement a clock? Wide search: for each shift of the passes against the log, the median bearing error of sightings and the share that land near the placed infected, and for still and moving phases apart. How the wall clock is read: pose.csv's own utc column, checked for resolution. Usage: live_shift.py <run> <console>"""
import csv, datetime, json, math, os, re, sys
import numpy as np
run, console = sys.argv[1].replace("\\", "/"), sys.argv[2]
pose = list(csv.DictReader(open(run + "/pose.csv")))
env = list(csv.DictReader(open(run + "/env.csv")))
print("pose utc samples:", [p["utc"] for p in pose[100:104]], "rows a second", round(len(pose) / ((int(pose[-1]["game_ms"]) - int(pose[0]["game_ms"])) / 1000), 1))
sec = lambda u: sum(float(v) * k for v, k in zip(u.split(":"), (3600, 60, 1)))
diff = np.array([sec(p["utc"]) - int(p["game_ms"]) / 1000 for p in pose])
print(f"utc minus game: median {np.median(diff):.2f}, p5 {np.percentile(diff, 5):.2f}, p95 {np.percentile(diff, 95):.2f}")
off = np.median(diff)
pg = np.array([int(p["game_ms"]) / 1000 for p in pose])
yaw = np.array([float(p["yaw"]) + float(p["aim_lr"]) for p in pose])
px, pz = np.array([float(p["x"]) for p in pose]), np.array([float(p["z"]) for p in pose])
eg = np.array([int(e["game_ms"]) / 1000 for e in env])
ex, ez = np.array([float(e["placed_x"]) for e in env]), np.array([float(e["placed_z"]) for e in env])
seen = []
for line in open(console, encoding="utf-8", errors="replace"):
    m = re.search(r"\[AssistZ\] sight utc=(\S+) frame=\S+ share=\S+ since=\d+ ms=([\d.]+) .*? seen=(\S*) ", line)
    if m and m[3]:
        for s in m[3].split(";"):
            b, me, c = (float(v) for v in s.split(":"))
            seen.append((sec(m[1]) - off, float(m[2]) / 1000, b, me, c))
print(len(seen), "sightings")
rate = np.abs(np.gradient(np.unwrap(np.radians(yaw)), pg)) * 180 / math.pi


def err(t, b):
    i = int(np.argmin(np.abs(pg - t)))
    j = int(np.argmin(np.abs(eg - t)))
    if ex[j] == 0:
        return None
    rel = (math.degrees(math.atan2(ex[j] - px[i], ez[j] - pz[i])) - yaw[i] + 180) % 360 - 180
    return abs(b - rel), rate[i], math.hypot(ex[j] - px[i], ez[j] - pz[i])


best = []
for shift in np.arange(-70, 70.01, 0.25):
    e = [err(t - ms + shift, b) for t, ms, b, me, c in seen]
    e = [x[0] for x in e if x]
    if len(e) > 10:
        best.append((np.median(e), shift, np.mean(np.array(e) <= 3), len(e)))
best.sort(key=lambda x: x[0])
print("best shifts, seconds added to the pass's start time:", [(round(float(s), 2), round(float(m), 2), round(float(f) * 100)) for m, s, f, n in best[:6]])
s0 = next(s for m, s, f, n in best if abs(s) < 0.13)
print("at 0 s: median", [round(float(m), 2) for m, s, f, n in best if abs(s) < 0.13], "share within 3 deg", [round(float(f) * 100) for m, s, f, n in best if abs(s) < 0.13])
shift = float(best[0][1])
rows = [err(t - ms + shift, b) + (me,) for t, ms, b, me, c in seen if err(t - ms + shift, b)]
a = np.array(rows)
print(f"at the best shift {shift:.2f} s: bearing median {np.median(a[:, 0]):.2f}, within 3 deg {100 * (a[:, 0] <= 3).mean():.0f}%; distance within 25% {100 * (np.abs(a[:, 3] - a[:, 2]) <= 0.25 * a[:, 2]).mean():.0f}%")
for name, k in (("view still (under 5 deg/s)", a[:, 1] < 5), ("view turning", a[:, 1] >= 5)):
    if k.sum():
        print(f"  {name}: {k.sum()} sightings, bearing median {np.median(a[k, 0]):.2f}, within 3 deg {100 * (a[k, 0] <= 3).mean():.0f}%")
print("   utc of a pass is logged when it ends; the frame was copied one pass earlier.")

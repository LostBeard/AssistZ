"""Fix the azimuth curve from both direction runs and register it before a fresh run. Symmetric, so it cannot favour a side from a few events. Writes the table into the handoff; logs the runs."""
import csv, json, math, os, subprocess, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
fold = lambda a: (180 - a if a > 90 else (-180 - a if a < -90 else a))
pts = []
for name in ("20261105_000715", "20261105_001851"):
    run = ROOT + name
    reports, _ = calibrated_reports(run)
    shots = list(csv.DictReader(open(run + "/shots_ai.csv")))
    server = list(csv.DictReader(open(run + "/aishots_server.csv")))
    sg = np.array([int(s["game_ms"]) for s in server])
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    pg = np.array([int(p["game_ms"]) for p in pose])
    ev = [json.loads(x) for x in open(f"{run}/replay_dir_s0.jsonl", encoding="utf-8")]
    for r, s in zip(reports, shots):
        g = int(s["game_ms"])
        k = int(np.argmin(np.abs(sg - g)))
        if abs(sg[k] - g) > 1500:
            continue
        p = pose[int(np.argmin(np.abs(pg - g)))]
        rel = (math.degrees(math.atan2(float(server[k]["mx"]) - float(p["x"]), float(server[k]["mz"]) - float(p["z"]))) - float(p["yaw"]) + 180) % 360 - 180
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if near:
            e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            pts.append((float(e["az"]), fold(rel)))
az = np.array([p[0] for p in pts])
tr = np.array([p[1] for p in pts])
print(len(pts), "points; az range", round(az.min()), round(az.max()))
xs, ys = [0.0], [0.0]
for lo in range(0, 90, 10):
    hi = lo + 10
    # Both sides together: az and truth mirrored.
    m = ((az >= lo) & (az < hi) & (lo > 0 or az > 0)) | ((az <= -lo) & (az > -hi) & (lo > 0 or az < 0))
    if m.sum() >= 8:
        xs.append(lo + 5.0)
        ys.append(float(np.median(np.sign(az[m]) * tr[m])))
ys = list(np.maximum.accumulate(ys))
xs.append(90.0)
ys.append(90.0)
ys = [round(min(y, 90.0), 1) for y in ys]
print("curve:", list(zip(xs, ys)))
json.dump({"az": xs, "true": ys}, open(sys.argv[1], "w"))
path = "D:/users/tj/Projects/AssistZ/AssistZ/docs/lab-handoff.md"
text = open(path, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in text else "\n"
text = text.replace("\r\n", "\n")
old = """Front or back is reported. It is not expected to work: the app has no cue for it.
"""
new = old + f"""
**Result: FAILED at the sides, close overall** (`sniper_20261105_000715`, `001851`; 1370 shots, 717 with an event; `Lab/direction_truth.py`). Left or right 627/633 (99.1%). Folded angle median 10.1 deg, p90 29.6, correlation 0.97, over a truth spread of 55 deg. Over the limit at true +-60 to +-120: 90 deg reads about 60. Systematic, so correctable. (`Lab/direction_next.py`) A curve fixed on one run, tested on the other: median 10.0 -> 6.7 and 10.1 -> 8.9, p90 29.5 -> 19.8 and 29.7 -> 21.0, worst bearing median 25 -> 15 and 28 -> 14.
Front or back: not in the stereo mix. Level differs at most 0.7 dB, above 4 kHz at most 1.1 dB, and an embedding probe trained on one run and tested on the other reads 54% and 55% balanced against a coin's 50%. So it needs another source. Heading from the video feed plus the player turning would do it: the angle's change tells front from back. To be tested with video.
Detection, a correction: I told TJ the detector misses most shots in storms. It does not. Bursts heard 83/83 across 9 runs in every weather. Shots without their own event come from pacing: at most one event each 0.17 s against shots about each 0.15 s. Distant single shots remain open: older SVD runs 15/20, 14/18, 21/42.

### Pre-registered: azimuth curve (written before its fresh run)
Curve fixed on both runs, mirrored, az to true: {list(zip(xs, ys))}. Fresh `-DirectionTest` run, never used to fit. Limits on the fresh run with the curve, against the same run raw: folded median at most 10 deg; p90 at most 25; worst bearing's median at most 18; left or right not below 95%. Pass: the HUD uses the curve (raw az stays in the logs).
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new)
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write(text.replace("\n", nl))
print("handoff registered")

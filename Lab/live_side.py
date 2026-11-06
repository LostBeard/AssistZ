"""Did the character turn, and are the live front/back calls right? Truth from the shooter's and listener's logged positions. Events and labels pair in order. Usage: live_side.py <run> <console> <rate> <offset>"""
import csv, math, re, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
run, console, K, OFF = sys.argv[1].replace("\\", "/"), sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
pose = list(csv.DictReader(open(run + "/pose.csv")))
yaw = np.array([float(p["yaw"]) for p in pose])
print(f"pose: {len(pose)} rows; yaw first {yaw[0]:.2f} last {yaw[-1]:.2f} min {yaw.min():.2f} max {yaw.max():.2f}; moved {math.hypot(float(pose[-1]['x']) - float(pose[0]['x']), float(pose[-1]['z']) - float(pose[0]['z'])):.2f} m")
evp = re.compile(r"\[AssistZ\] event az=(-?[\d.]+) .* yaw=(-?[\d.]+) track=(\d+) side=(\w+) conf=([\d.]+) turned=(-?[\d.]+)")
lap = re.compile(r"\[AssistZ\] label utc=\S+ end=(\d+) label=(.+?) raw=.* az=(-?[\d.]+) level")
events, labels = [], []
for line in open(console, encoding="utf-8"):
    m = evp.search(line)
    if m:
        events.append(dict(az=float(m[1]), view=float(m[2]), track=int(m[3]), side=m[4], conf=float(m[5]), turned=float(m[6])))
    m = lap.search(line)
    if m:
        labels.append(dict(end=int(m[1]), label=m[2], az=float(m[3])))
print(f"{len(events)} events, {len(labels)} labels; az in order agrees on {sum(abs(a['az'] - b['az']) < 0.11 for a, b in zip(events, labels))}")
reports, _ = calibrated_reports(run)
shots = list(csv.DictReader(open(run + "/shots_ai.csv")))
server = list(csv.DictReader(open(run + "/aishots_server.csv")))
sg = np.array([int(s["game_ms"]) for s in server])
pg = np.array([int(p["game_ms"]) for p in pose])
truth = []
for r, s in zip(reports, shots):
    g = int(s["game_ms"])
    k = int(np.argmin(np.abs(sg - g)))
    if abs(sg[k] - g) > 1500:
        continue
    p = pose[int(np.argmin(np.abs(pg - g)))]
    rel = (math.degrees(math.atan2(float(server[k]["mx"]) - float(p["x"]), float(server[k]["mz"]) - float(p["z"]))) - float(p["yaw"]) + 180) % 360 - 180
    truth.append((r[0], rel))
tt = np.array([t[0] for t in truth])
right = wrong = unclear = unmatched = 0
rows = []
for e, l in zip(events, labels):
    if e["side"] == "Unknown":
        continue
    t = l["end"] / 48000.0 * K + OFF
    i = int(np.argmin(np.abs(tt - t)))
    if abs(tt[i] - t) > 0.5:
        unmatched += 1
        continue
    rel = truth[i][1]
    want = "Front" if abs(rel) <= 75 else ("Back" if abs(rel) >= 105 else None)
    if want is None:
        unclear += 1
    elif want == e["side"]:
        right += 1
    else:
        wrong += 1
    rows.append((e["track"], e["side"], want, round(rel), e["az"], round(e["turned"], 1), round(e["conf"], 2), l["label"]))
print(f"front/back calls: right {right}, wrong {wrong}, near the side {unclear}, no shot near {unmatched}; of {sum(e['side'] != 'Unknown' for e in events)}")
for r in rows[:24]:
    print("  track {} said {} true {} rel {} az {} turned {} conf {} {}".format(*r))
v = np.array([e["view"] for e in events])
print("view yaw at every 30th event: " + " ".join(f"{x:.0f}" for x in v[::30]))

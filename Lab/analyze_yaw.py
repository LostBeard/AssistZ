"""
Checks AssistZ's video yaw tracker against the DayZ pose log for one lab run directory (lab_yaw_run):
console.txt ([AssistZ] yaw lines: utc, shift px, score), pose.csv (server: utc s, game_ms, yaw, aim_lr), turns.json
(mouse segments). Per turn segment: integrated tracker shift (px) vs the pose log's change of view direction
(yaw + aim_lr, degrees) -> pixels per degree, and how consistent that ratio is across slow/medium/fast/sweep turns.
Usage: python analyze_yaw.py <run_dir>
"""
import sys, os, re, csv, json, datetime
import numpy as np

run = sys.argv[1]
yaw = []
for line in open(os.path.join(run, "console.txt"), encoding="utf-8"):
    m = re.search(r"\] yaw utc=(\d\d):(\d\d):(\d\d\.\d+) ts=([\d.]+) shift=(-?[\d.]+) score=(-?[\d.]+)", line)
    if m:
        t = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
        yaw.append((t, float(m[5]), float(m[6])))
yaw = np.array(yaw)
pose = list(csv.DictReader(open(os.path.join(run, "pose.csv"))))

def tsec(iso):
    d = datetime.datetime.fromisoformat(iso)
    return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1e6

# Pose rows: utc has 1 s resolution; game_ms is exact. Map game_ms to utc with the offset that fits all rows.
g = np.array([int(r["game_ms"]) / 1000 for r in pose])
u = np.array([sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) for r in pose], dtype=float)
offset = np.median(u + 0.5 - g)          # utc floors the second: +0.5 centres it

def utc_iso_of_game(ms):
    t = ms / 1000 + offset
    h, rem = divmod(t, 3600); m, s = divmod(rem, 60)
    return f"2000-01-01T{int(h):02d}:{int(m):02d}:{s:09.6f}+00:00"

if os.path.exists(os.path.join(run, "turns.csv")):
    # Server-scripted turns (lab mode 2): start/end rows with game_ms, exact.
    turns, open_seg = [], None
    for r in csv.DictReader(open(os.path.join(run, "turns.csv"))):
        if r["event"] == "start":
            open_seg = {"label": f"seg {r['segment']} rate {float(r['aim_change']):+.3f}", "steps": 1,
                        "start_utc": utc_iso_of_game(int(r["game_ms"]))}
        elif r["event"] == "end" and open_seg:
            open_seg["end_utc"] = utc_iso_of_game(int(r["game_ms"]))
            turns.append(open_seg); open_seg = None
else:
    turns = json.load(open(os.path.join(run, "turns.json")))["segments"]
view = np.array([float(r["yaw"]) + float(r["aim_lr"]) for r in pose])
view = np.degrees(np.unwrap(np.radians(view)))
pt = g + offset

def view_at(t):
    return float(np.interp(t, pt, view))

print(f"{len(yaw)} tracker samples, {len(pose)} pose rows; median score {np.median(yaw[:, 2]):.3f}")
print(f"{'segment':14} {'tracker px':>10} {'view deg':>9} {'px/deg':>8}  min score")
ratios = []
for s in turns:
    if not s["steps"] or "end_utc" not in s:
        continue
    t0, t1 = tsec(s["start_utc"]) - 0.3, tsec(s["end_utc"]) + 0.8
    sel = (yaw[:, 0] >= t0) & (yaw[:, 0] <= t1)
    px = yaw[sel, 1].sum()
    deg = view_at(t1) - view_at(t0)
    r = px / deg if abs(deg) > 0.5 else float("nan")
    ratios.append(r)
    print(f"{s['label']:14} {px:10.1f} {deg:9.1f} {r:8.2f}  {yaw[sel, 2].min() if sel.any() else float('nan'):.3f}")
ratios = np.array([r for r in ratios if np.isfinite(r)])
if len(ratios):
    print(f"px/deg mean {ratios.mean():.2f}, spread {ratios.std() / abs(ratios.mean()) * 100:.1f}% over {len(ratios)} segments")
# Still periods: tracker should read ~0 px when no turn is happening.
still = np.ones(len(yaw), bool)
for s in turns:
    if s["steps"] and "end_utc" in s:
        still &= ~((yaw[:, 0] >= tsec(s["start_utc"]) - 0.3) & (yaw[:, 0] <= tsec(s["end_utc"]) + 0.8))
print(f"still frames: {still.sum()}, |shift| mean {np.abs(yaw[still, 1]).mean():.3f} px, max {np.abs(yaw[still, 1]).max():.2f} px")

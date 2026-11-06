"""
Front/back resolution vs ground truth for a lab mode-3 run (two world-fixed repeating beacons while the view turns).
Inputs in <run_dir>: console.txt (AssistZ lab console: "<iso utc> [AssistZ] event ... yaw= track= side= conf= turned="),
beacon.csv (server: game_ms, utc, bearing_rel = body-relative bearing of the source at detonation, aim_lr, ...),
pose.csv (server pose log, used to map game_ms to UTC).
For each detonation: the AssistZ event that follows it (sound travels 60 m in ~0.18 s), the true head-relative bearing
(bearing_rel - aim_lr) and the true side, against the tracker's side. Usage: python analyze_frontback.py <run_dir>
"""
import sys, os, re, csv, datetime, collections
import numpy as np

run = sys.argv[1]
events = []
for line in open(os.path.join(run, "console.txt"), encoding="utf-8"):
    m = re.match(r"(\S+) \[AssistZ\] event az=(-?[\d.]+) .*? track=(\d+) side=(\w+) conf=([\d.]+) turned=([\d.]+)", line)
    if m:
        t = datetime.datetime.fromisoformat(m[1])
        events.append(dict(t=t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6, az=float(m[2]),
                           track=int(m[3]), side=m[4], conf=float(m[5]), turned=float(m[6])))
pose = list(csv.DictReader(open(os.path.join(run, "pose.csv"))))
g = np.array([int(r["game_ms"]) / 1000 for r in pose])
u = np.array([sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) for r in pose], dtype=float)
offset = np.median(u + 0.5 - g)
beacons = list(csv.DictReader(open(os.path.join(run, "beacon.csv"))))

def norm(a):
    return (a + 180) % 360 - 180

rows = []
for b in beacons:
    t = int(b["game_ms"]) / 1000 + offset
    cands = [e for e in events if t + 0.05 <= e["t"] <= t + 1.3]
    if not cands:
        continue
    e = cands[0]
    head = norm(float(b["bearing_rel"]) - float(b["aim_lr"]))
    truth = "Front" if abs(head) < 90 else "Back"
    rows.append(dict(head=head, truth=truth, **e))

print(f"{len(beacons)} detonations, {len(events)} AssistZ events, {len(rows)} matched")
conf = collections.Counter((r["truth"], r["side"]) for r in rows)
print("truth \\ tracker   Front  Back  Unknown")
for truth in ("Front", "Back"):
    print(f"  {truth:6}          {conf[(truth, 'Front')]:5} {conf[(truth, 'Back')]:5} {conf[(truth, 'Unknown')]:8}")
decided = [r for r in rows if r["side"] != "Unknown"]
correct = [r for r in decided if r["side"] == r["truth"]]
if decided:
    print(f"decided {len(decided)}/{len(rows)}, correct {len(correct)}/{len(decided)} ({100 * len(correct) / len(decided):.0f}%)")
tracks = collections.defaultdict(list)
for r in rows:
    tracks[r["track"]].append(r)
for tid, rs in sorted(tracks.items()):
    first = next((i for i, r in enumerate(rs) if r["side"] != "Unknown"), None)
    print(f"  track {tid}: {len(rs)} events, true side {collections.Counter(r['truth'] for r in rs).most_common(1)[0][0]},"
          f" decided after {first + 1 if first is not None else '-'} events, final {rs[-1]['side']} (turned {rs[-1]['turned']:.0f} deg)")
# Bearing error: the per-event front/back label is ill-posed for a source near +-90 (its true side flips as the view
# sweeps), so score the full estimate instead. Tracker estimate: calibrated azimuth (DayZ pan scale 0.91) on its
# resolved side; baseline: stereo alone (always assume the front reading).
PAN = 0.91
def unfold(cal, side):
    if side == "Back":
        return 180 - cal if cal >= 0 else -180 - cal
    return cal
err_tracker, err_front = [], []
for r in rows:
    cal = max(-90.0, min(90.0, r["az"] / PAN))
    err_front.append(abs(norm(cal - r["head"])))
    if r["side"] != "Unknown":
        err_tracker.append(abs(norm(unfold(cal, r["side"]) - r["head"])))
if err_tracker:
    et, ef = np.array(err_tracker), np.array(err_front)
    print(f"bearing error, resolved events ({len(et)}): median {np.median(et):.1f} deg, p90 {np.percentile(et, 90):.1f}, "
          f"within 30 deg {np.mean(et <= 30) * 100:.0f}%")
    print(f"bearing error, stereo alone ({len(ef)}): median {np.median(ef):.1f} deg, p90 {np.percentile(ef, 90):.1f}, "
          f"within 30 deg {np.mean(ef <= 30) * 100:.0f}%")
for r in rows[:30]:
    print(f"    head {r['head']:7.1f} ({r['truth']:5}) az {r['az']:6.1f} track {r['track']:3} -> {r['side']:7} conf {r['conf']:.2f}")

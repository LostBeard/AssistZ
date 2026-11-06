"""
The live video pass against the game's log: a lab mode 34 run with the driver. Each sighting's bearing and distance, by wall clock, against the placed infected. Slots sighted within 5 s; sightings at rest; inventory flagged while the driver held it open. Usage: live_sight_score.py <run> <console>
"""
import csv, datetime, json, math, os, re, sys
import numpy as np
run, console = sys.argv[1].replace("\\", "/"), sys.argv[2]
# Seconds added to a pass's start time, from live_shift.py: the log's utc is whole seconds. -0.75 by a wide search on this run.
SHIFT = float(sys.argv[3]) if len(sys.argv) > 3 else -0.75
pose = list(csv.DictReader(open(run + "/pose.csv")))
env = list(csv.DictReader(open(run + "/env.csv")))
world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
sec = lambda u: sum(float(v) * k for v, k in zip(u.split(":"), (3600, 60, 1)))
off = np.median([sec(p["utc"]) - int(p["game_ms"]) / 1000 for p in pose])
pg = np.array([int(p["game_ms"]) / 1000 for p in pose])
eg = np.array([int(e["game_ms"]) / 1000 for e in env])
passes = []
for line in open(console, encoding="utf-8", errors="replace"):
    m = re.search(r"\[AssistZ\] sight utc=(\S+) frame=(\S+) share=(\S+) since=(\d+) ms=([\d.]+) gate=[\d.]+ inventory=(\d) boxes=(\d+) seen=(\S*) ", line)
    if m:
        seen = [tuple(float(v) for v in s.split(":")) for s in m[8].split(";") if s]
        passes.append(dict(game=sec(m[1]) - off - float(m[5]) / 1000 + SHIFT, inventory=int(m[6]), seen=seen, boxes=int(m[7])))
print(f"{len(passes)} passes, game {passes[0]['game']:.0f} to {passes[-1]['game']:.0f} s")
spans = [(int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000, a["event"], float(a["distance"])) for a, b in zip(world, world[1:])]


def truth(g):
    e = env[int(np.argmin(np.abs(eg - g)))]
    p = pose[int(np.argmin(np.abs(pg - g)))]
    if float(e.get("placed_x") or 0) == 0:
        return None
    dx, dz = float(e["placed_x"]) - float(p["x"]), float(e["placed_z"]) - float(p["z"])
    rel = (math.degrees(math.atan2(dx, dz)) - float(p["yaw"]) - float(p["aim_lr"]) + 180) % 360 - 180
    return math.hypot(dx, dz), rel, int(e.get("placed_los") or 0)


slots, rest_seen, away, errs, meters = [], 0, 0, [], []
rest_minutes = 0.0
for lo, hi, ev, dist in spans:
    ps = [p for p in passes if lo + 1 <= p["game"] < hi and not p["inventory"]]
    if ev == "rest":
        rest_minutes += sum(1 for _ in ps) * 0.0  # time counted below
        if ps:
            rest_minutes += (hi - lo - 1) / 60
        rest_seen += sum(bool(p["seen"]) for p in ps)
        continue
    start = first = None
    for p in ps:
        t = truth(p["game"])
        if not t:
            continue
        near, rel, los = t
        inview = 10 <= near < 30 and abs(rel) < 43.6 and los >= 1
        if inview and start is None:
            start = p["game"]
        hit = False
        for b, m_, c in p["seen"]:
            tol = max(3.0, math.degrees(math.atan(1.5 / max(near, 1))))
            if abs(b - rel) <= tol:
                hit = True
                errs.append(abs(b - rel))
                meters.append((near, m_))
            else:
                away += 1
        if inview and hit and first is None:
            first = p["game"]
    if start is not None:
        last = max((p["game"] for p in ps if (lambda t: t and 10 <= t[0] < 30 and abs(t[1]) < 43.6 and t[2] >= 1)(truth(p["game"]))), default=start)
        if last - start >= 5:
            slots.append((dist, last - start, None if first is None else first - start))
got = [s for s in slots if s[2] is not None and s[2] <= 5]
print(f"sighted within 5 s, slots in view at 10-30 m for 5 s or more: {len(got)}/{len(slots)}; {[(s[0], round(s[1]), None if s[2] is None else round(s[2], 1)) for s in slots if s not in got]}")
print(f"passes at rest with a sighting: {rest_seen}, in {rest_minutes:.1f} min; sightings away from the infected during slots: {away}")
if errs:
    m = np.array(meters)
    r = (np.abs(m[:, 1] - m[:, 0]) <= 0.25 * m[:, 0]).mean() * 100
    print(f"bearing off the log: median {np.median(errs):.2f} deg, p90 {np.percentile(errs, 90):.2f}, {len(errs)} sightings; distance within 25% {r:.0f}%, median error {np.median(np.abs(m[:, 1] - m[:, 0])):.1f} m")
drive = os.path.join(run, "see_drive.json")
if os.path.exists(drive):
    inv = []
    for p in json.load(open(drive))["phases"]:
        if p["name"] == "inventory":
            a = datetime.datetime.fromisoformat(p["start_utc"]); b = datetime.datetime.fromisoformat(p["end_utc"])
            inv.append((a.hour * 3600 + a.minute * 60 + a.second + a.microsecond / 1e6 - off, b.hour * 3600 + b.minute * 60 + b.second + b.microsecond / 1e6 - off))
    held = [p for p in passes if any(a + 1.5 <= p["game"] < b - 0.5 for a, b in inv)]
    near = [p for p in passes if any(a - 1.5 <= p["game"] < b + 2.5 for a, b in inv)]
    print(f"inventory: {sum(p['inventory'] for p in held)}/{len(held)} passes flagged while held open; flagged elsewhere {sum(p['inventory'] for p in passes if p not in near)}; {len(inv)} openings; sightings on held-open passes {sum(bool(p['seen']) for p in held)}")


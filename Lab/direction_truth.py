"""
Direction against the truth, all the way round (mode 25). Limits registered in the handoff before the runs: left or right right at least 95% where at least 15 deg off centre; folded angle median at most 15 deg, p90 at most 30, overall and per bearing (bearings with 5+ heard shots). Front or back reported.
Usage: python direction_truth.py <tag> <run> [...]
"""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

TAG, RUNS = sys.argv[1], sys.argv[2:]
fold = lambda a: (180 - a if a > 90 else (-180 - a if a < -90 else a))
rows = []
for run in RUNS:
    reports, shift = calibrated_reports(run)
    shots = list(csv.DictReader(open(run + "/shots_ai.csv")))
    server = list(csv.DictReader(open(run + "/aishots_server.csv")))
    sg = np.array([int(s["game_ms"]) for s in server])
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    pg = np.array([int(p["game_ms"]) for p in pose])
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    n0 = len(rows)
    for r, s in zip(reports, shots):
        g = int(s["game_ms"])
        k = int(np.argmin(np.abs(sg - g)))
        if abs(sg[k] - g) > 1500:
            continue
        p = pose[int(np.argmin(np.abs(pg - g)))]
        dx, dz = float(server[k]["mx"]) - float(p["x"]), float(server[k]["mz"]) - float(p["z"])
        rel = (math.degrees(math.atan2(dx, dz)) - float(p["yaw"]) + 180) % 360 - 180
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0])) if near else None
        rows.append(dict(run=os.path.basename(run)[-6:], rel=rel, dist=math.hypot(dx, dz), az=float(e["az"]) if e else None, level=float(e["level"]) if e else None, label=e["label"] if e else None))
    print(f"{os.path.basename(run)}: {len(rows) - n0} shots, clock {shift:.3f}, yaw {pose[0]['yaw']} to {pose[-1]['yaw']}, moved {math.hypot(float(pose[-1]['x']) - float(pose[0]['x']), float(pose[-1]['z']) - float(pose[0]['z'])):.1f} m")
heard = [r for r in rows if r["az"] is not None]
print(f"\nShots with an event: {len(heard)}/{len(rows)}; distance median {np.median([r['dist'] for r in rows]):.0f} m")
print(f"{'bearing':>7} {'shots':>5} {'heard':>5} {'|err| med':>9} {'p90':>5} {'mean az':>8} {'folded':>7}  named gunfire")
all_ok = True
for b in range(-180, 180, 30):
    g = [r for r in rows if abs((r["rel"] - b + 180) % 360 - 180) <= 15]
    h = [r for r in g if r["az"] is not None]
    if not h:
        print(f"{b:>7} {len(g):>5} {0:>5}")
        continue
    err = np.array([r["az"] - fold(r["rel"]) for r in h])
    med, p90 = np.median(np.abs(err)), np.percentile(np.abs(err), 90)
    ok = len(h) < 5 or (med <= 15 and p90 <= 30)
    all_ok = all_ok and ok
    gun = sum(r["label"] in ("gunshot", "automatic fire") for r in h)
    print(f"{b:>7} {len(g):>5} {len(h):>5} {med:>9.1f} {p90:>5.1f} {np.mean([r['az'] for r in h]):>8.1f} {fold(b):>7} {gun:>6}/{len(h)} {'' if ok else 'OVER'}")
err = np.array([r["az"] - fold(r["rel"]) for r in heard])
med, p90 = np.median(np.abs(err)), np.percentile(np.abs(err), 90)
side = [(np.sign(r["az"]), np.sign(fold(r["rel"]))) for r in heard if abs(fold(r["rel"])) >= 15]
lr = sum(a == b for a, b in side) / max(len(side), 1)
print(f"\nFolded angle: median |error| {med:.1f} deg (at most 15), p90 {p90:.1f} (at most 30), correlation {np.corrcoef([r['az'] for r in heard], [fold(r['rel']) for r in heard])[0, 1]:.2f}, spread of truth {np.std([fold(r['rel']) for r in heard]):.0f} deg")
print(f"Left or right: {sum(a == b for a, b in side)}/{len(side)} ({100 * lr:.1f}%, at least 95)")
front = [r for r in heard if abs(r["rel"]) <= 75]
back = [r for r in heard if abs(r["rel"]) >= 105]
print(f"Front or back: the app says nothing. Heard in front {len(front)} (level {np.mean([r['level'] for r in front]) if front else 0:.1f} dB), behind {len(back)} (level {np.mean([r['level'] for r in back]) if back else 0:.1f} dB).")
ok = lr >= 0.95 and med <= 15 and p90 <= 30 and all_ok
print("DIRECTION GATE " + ("PASS" if ok else "FAIL"))

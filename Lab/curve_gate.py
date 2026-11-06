"""The registered azimuth curve on a fresh run. Limits: folded median at most 10 deg; p90 at most 25; worst bearing's median at most 18; left or right not below 95%. Usage: python curve_gate.py <curve.json> <tag> <run>"""
import csv, json, math, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

c = json.load(open(sys.argv[1]))
xs, ys = np.array(c["az"]), np.array(c["true"])
curve = lambda a: float(np.sign(a) * np.interp(abs(a), xs, ys))
tag, run = sys.argv[2], sys.argv[3]
fold = lambda a: (180 - a if a > 90 else (-180 - a if a < -90 else a))
reports, shift = calibrated_reports(run)
shots = list(csv.DictReader(open(run + "/shots_ai.csv")))
server = list(csv.DictReader(open(run + "/aishots_server.csv")))
sg = np.array([int(s["game_ms"]) for s in server])
pose = list(csv.DictReader(open(run + "/pose.csv")))
pg = np.array([int(p["game_ms"]) for p in pose])
ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
rows = []
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
        rows.append((float(e["az"]), rel))
print(f"{len(reports)} shots, {len(rows)} with an event, clock {shift:.3f}")
result = {}
for name, f in (("raw", lambda a: a), ("curve", curve)):
    err = np.array([f(a) - fold(t) for a, t in rows])
    worst, table = 0.0, []
    for b in range(-90, 91, 30):
        g = [abs(f(a) - fold(t)) for a, t in rows if abs(fold(t) - b) <= 15]
        if len(g) >= 5:
            m = float(np.median(g))
            worst = max(worst, m)
            table.append(f"{b}:{m:.0f}({len(g)})")
    side = [(np.sign(f(a)), np.sign(fold(t))) for a, t in rows if abs(fold(t)) >= 15]
    lr = sum(x == y for x, y in side) / max(len(side), 1)
    result[name] = (float(np.median(np.abs(err))), float(np.percentile(np.abs(err), 90)), worst, lr)
    print(f"{name:<6} median {result[name][0]:5.1f}, p90 {result[name][1]:5.1f}, worst bearing median {worst:5.1f}, left or right {sum(x == y for x, y in side)}/{len(side)} ({100 * lr:.1f}%); {' '.join(table)}")
m, p, w, lr = result["curve"]
ok = m <= 10 and p <= 25 and w <= 18 and lr >= 0.95
print(("PASS" if ok else "FAIL") + f": curve median {m:.1f} (at most 10), p90 {p:.1f} (at most 25), worst bearing {w:.1f} (at most 18), left or right {100 * lr:.1f}% (at least 95)")

"""
Lab mode 6 (Expansion AI shooters): turn the CLIENT's shot log (aishots_client.csv: the listener's own game process,
where the shot sound is rendered, with a level barrel direction) into analyze_shots.py's shots.csv.
- Time: client game_ms moved onto the server clock (pose.csv's clock) by the median offset of the shots both
  processes logged (same shooter + weapon, in order). Measured, not assumed: on run sniper_20261103_153510 it was
  48.2 s, constant to +-12 ms over 45 shots.
- distance: muzzle to the listener's head (pose.csv position + 1.6 m) at that time.
- miss: the barrel line's closest approach to the head (the weapon's own memory points; the bullet's real path also
  carries the AI's aim error and dispersion, so this is the barrel line, not the bullet).
Usage: python aishots_to_shots.py <run dir> [shooter substring]   -> <run dir>/shots_ai.csv
"""
import csv, math, sys
import numpy as np

run = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else ""
client = list(csv.DictReader(open(run + "/aishots_client.csv")))
server = list(csv.DictReader(open(run + "/aishots_server.csv")))
pose = list(csv.DictReader(open(run + "/pose.csv")))

offsets = []
for key in {(r["shooter"], r["weapon"]) for r in client}:
    c = [int(r["game_ms"]) for r in client if (r["shooter"], r["weapon"]) == key]
    s = [int(r["game_ms"]) for r in server if (r["shooter"], r["weapon"]) == key]
    if len(c) == len(s):
        offsets += [b - a for a, b in zip(c, s)]
off = float(np.median(offsets))
print(f"client->server offset {off / 1000:.3f} s (spread {np.ptp(offsets):.0f} ms over {len(offsets)} shots)")

pg = np.array([int(r["game_ms"]) for r in pose], dtype=float)
P = np.array([[float(r["x"]), float(r["y"]) + 1.6, float(r["z"])] for r in pose])

out = open(run + "/shots_ai.csv", "w", newline="")
w = csv.writer(out)
w.writerow(["shot", "game_ms", "distance", "miss", "fired", "shooter", "weapon"])
n = 0
for r in client:
    if only and only not in r["shooter"]:
        continue
    g = int(r["game_ms"]) + off
    head = [float(np.interp(g, pg, P[:, i])) for i in range(3)]
    m = [float(r[k]) for k in ("mx", "my", "mz")]
    d = [float(r[k]) for k in ("dx", "dy", "dz")]
    v = [head[i] - m[i] for i in range(3)]
    t = sum(v[i] * d[i] for i in range(3))
    dist = math.dist(m, head)
    miss = math.sqrt(max(0.0, dist * dist - t * t)) if t > 0 else dist
    n += 1
    w.writerow([n, int(g), f"{dist:.1f}", f"{miss:.2f}", "fired", r["shooter"], r["weapon"]])
out.close()
print(f"{n} shots -> {run}/shots_ai.csv")

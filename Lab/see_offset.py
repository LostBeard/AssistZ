"""How far are real boxes from the log's prediction? Per slot inside 30 m: the best person box nearest the column (any distance), its offset in px and what that means in metres at that distance; where the player stood and faced; whether the player moved. If one slot is off by a steady amount the truth is wrong there, not the detector."""
import collections, csv, json, math
import numpy as np
PX = 8.8 * 1600 / 960
F = 840.0
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
W = "L:/AssistZLabWork/frames/see"
for n in range(4, 13):
    d = W + str(n)
    man = {r["frame"]: r for r in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
    run = open(d + "/run.txt").read().strip().replace("\\", "/")
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    by = collections.defaultdict(list)
    for o in json.load(open(d + "_yolo.json")):
        by[o["file"]].append(o)
    frames = sorted(by, key=lambda f: float(man[f]["game"]))
    pp = pose[len(pose) // 2]
    xs = np.array([float(p["x"]) for p in pose[-int(len(pose) * 0.9):]])
    ys = np.array([float(p["yaw"]) for p in pose[-int(len(pose) * 0.9):]])
    print(f"see{n}: player at {float(pp['x']):.0f},{float(pp['z']):.0f}; x range {xs.max() - xs.min():.2f} m; yaw range {ys.max() - ys.min():.2f}; pose columns {list(pose[0].keys())[:9]}")
    for a, b in zip(world, world[1:]):
        if a["event"] != "infected":
            continue
        lo, hi = int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000
        rows = []
        for f in frames:
            g = float(man[f]["game"])
            m = man[f]
            if not lo + 1 <= g < hi or m["placed_rel"] == "":
                continue
            near, rel = float(m["placed_near"]), float(m["placed_rel"])
            if near >= 30 or abs(rel) >= 50:
                continue
            lin, tan = 800 + rel * PX, 800 + F * math.tan(math.radians(rel))
            got = [q for o in by[f] if o["kind"] in BARS for q in o["found"] if q["label"] == "person" and q["conf"] >= BARS[o["kind"]]]
            if not got:
                rows.append((near, rel, None, None, None))
                continue
            q = min(got, key=lambda q: abs(q["x"] + q["w"] / 2 - lin))
            cx = q["x"] + q["w"] / 2
            rows.append((near, rel, cx - lin, cx - tan, q["h"]))
        if not rows:
            continue
        have = [r for r in rows if r[2] is not None]
        if have:
            off = np.array([r[2] for r in have])
            off2 = np.array([r[3] for r in have])
            near = np.median([r[0] for r in have])
            print(f"   placed {float(a['distance']):>3.0f}: {len(have)}/{len(rows)} frames boxed; near {near:.0f} m, rel {np.median([r[1] for r in have]):6.1f}; linear offset median {np.median(off):6.0f} px (spread {np.percentile(off, 90) - np.percentile(off, 10):4.0f}); tan offset {np.median(off2):6.0f} px; height says {1268 / np.median([r[4] for r in have]):.0f} m; tan offset in metres at that range {np.median(off2) / F * near:.1f}")
        else:
            print(f"   placed {float(a['distance']):>3.0f}: 0/{len(rows)} frames boxed; near {np.median([r[0] for r in rows]):.0f} m")

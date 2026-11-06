"""Rescore with the truth the log really has: world.csv gives each placed infected's x,z at the moment it was placed. env's infected_x,z is the nearest of ALL infected, including natural ones (see1 has 11-13 about). Per slot: placed bearing and distance from the player; nearest infected; share of frames with a person box within tolerance of the placed column. Also the same for the nearest one's column. Boxes elsewhere reported: bar .5 and .35."""
import collections, csv, json, math, os, sys
import numpy as np
PX = 8.8 * 1600 / 960
for run in ("see1", "see2"):
    d = f"L:/AssistZLabWork/frames/{run}"
    man = {r["frame"]: r for r in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
    rd = open(d + "/run.txt").read().strip().replace("\\", "/")
    pose = list(csv.DictReader(open(rd + "/pose.csv")))
    pg = np.array([int(p["game_ms"]) / 1000 for p in pose])
    world = [w for w in csv.DictReader(open(rd + "/world.csv")) if w["event"] in ("infected", "rest")]
    res = collections.defaultdict(list)
    for o in json.load(open(d + "_yolo.json")):
        res[o["file"]].append(o)
    frames = sorted(res, key=lambda f: float(man[f]["game"]))
    print(run)
    for a, b in zip(world, world[1:]):
        lo, hi = int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000
        fs = [f for f in frames if lo + 1 <= float(man[f]["game"]) < hi]
        if not fs or a["event"] != "infected":
            continue
        p = pose[int(np.argmin(np.abs(pg - lo)))]
        dist = math.hypot(float(a["x"]) - float(p["x"]), float(a["z"]) - float(p["z"]))
        rel = (math.degrees(math.atan2(float(a["x"]) - float(p["x"]), float(a["z"]) - float(p["z"]))) - float(p["yaw"]) + 180) % 360 - 180
        x = 800 + rel * PX
        near = np.array([float(man[f]["near"]) for f in fs])
        nrel = [float(man[f]["rel"]) for f in fs if man[f]["rel"]]
        line = f"  placed {float(a['distance']):>4.0f} m: from log {dist:5.1f} m at {rel:6.1f} deg (x {x:5.0f}); nearest of all {np.median(near):5.1f} m at {np.median(nrel) if nrel else 0:6.1f} deg; state {collections.Counter(man[f]['state'] for f in fs).most_common(2)}"
        for bar in (0.5, 0.35):
            at = other = 0
            where = []
            for f in fs:
                got = [k for o in res[f] for k in o["found"] if k["label"] == "person" and k["conf"] >= bar]
                near_col = [k for k in got if k["x"] - 40 <= x <= k["x"] + k["w"] + 40]
                at += bool(near_col)
                if got and not near_col:
                    other += 1
                    where += [round(k["x"] + k["w"] / 2) for k in got]
            line += f" | bar {bar}: at placed {at}/{len(fs)}, elsewhere {other}" + (f" x~{int(np.median(where))}" if where else "")
        print(line)

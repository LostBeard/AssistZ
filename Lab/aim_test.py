"""Does the camera look along yaw + aim? For strong person boxes while an infected is out, how far is each box from the column predicted by yaw alone, yaw + aim_lr, and yaw - aim_lr, by driver phase. The right truth should pull boxes on to the column in sweeps."""
import collections, csv, datetime, json, math, os
import numpy as np
F = 840.34
RUNS = (29, 37, 38, 31, 32, 39, 34, 40, 41)
res = collections.defaultdict(lambda: collections.defaultdict(list))
for n in RUNS:
    d = f"L:/AssistZLabWork/frames/see{n}"
    run = open(d + "/run.txt").read().strip().replace("\\", "/")
    man = {m["frame"]: m for m in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    pg = np.array([int(p["game_ms"]) / 1000 for p in pose])
    aim = np.array([float(p["aim_lr"]) for p in pose])
    yaw = np.array([float(p["yaw"]) for p in pose])
    sec = lambda u: sum(float(v) * k for v, k in zip(u.split(":"), (3600, 60, 1)))
    off = np.median([sec(p["utc"]) - int(p["game_ms"]) / 1000 for p in pose])
    ph = []
    for p in json.load(open(run + "/see_drive.json"))["phases"]:
        a = datetime.datetime.fromisoformat(p["start_utc"]); b = datetime.datetime.fromisoformat(p["end_utc"])
        ph.append((a.hour * 3600 + a.minute * 60 + a.second + a.microsecond / 1e6 - off, b.hour * 3600 + b.minute * 60 + b.second + b.microsecond / 1e6 - off, p["name"]))
    flagged = set(json.load(open(d + "/inventory.json")))
    best = collections.defaultdict(list)
    for o in json.load(open(d + "_yolo.json")):
        if o["file"] in flagged:
            continue
        for b in o["found"]:
            if b["label"] == "person" and b["conf"] >= 0.6:
                best[o["file"]].append(b)
    for f, bs in best.items():
        m = man[f]
        if m["placed_rel"] == "" or float(m["placed_near"]) >= 40:
            continue
        g = float(m["game"])
        i = int(np.argmin(np.abs(pg - g)))
        k = next((k for a, b, k in ph if a <= g < b), "other")
        for name, sign in (("yaw only", 0), ("yaw + aim", 1), ("yaw - aim", -1)):
            rel = ((float(m["placed_rel"]) - sign * aim[i] + 180) % 360) - 180
            if abs(rel) >= 70:
                res[k][name].append(9999)
                continue
            x = 800 + F * math.tan(math.radians(rel))
            res[k][name].append(min(abs(b["x"] + b["w"] / 2 - x) for b in bs))
print("median px from the predicted column to the nearest strong person box, share within 40 px:")
for k, v in res.items():
    print(f"{k:<12}" + "; ".join(f"{name}: {np.median(a):6.0f} px, {100 * np.mean(np.array(a) <= 40):3.0f}% (n {len(a)})" for name, a in v.items()))
print("aim_lr range by phase, see34:")

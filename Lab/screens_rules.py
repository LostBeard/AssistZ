"""Setup 4's sightings per driver phase on the screens run: which screens get past the whole rule, not just the detector? Frames the inventory rule flags say nothing. With an extra rule to try: a box whose track began in a pause menu. Usage: screens_rules.py <n>"""
import collections, csv, datetime, json, math, os, sys
import numpy as np
n = sys.argv[1]
d = f"L:/AssistZLabWork/frames/see{n}"
run = open(d + "/run.txt").read().strip().replace("\\", "/")
man = {m["frame"]: m for m in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
pose = list(csv.DictReader(open(run + "/pose.csv")))
sec = lambda u: sum(float(v) * k for v, k in zip(u.split(":"), (3600, 60, 1)))
off = np.median([sec(p["utc"]) - int(p["game_ms"]) / 1000 for p in pose])
ph = []
for p in json.load(open(run + "/see_drive.json"))["phases"]:
    a = datetime.datetime.fromisoformat(p["start_utc"]); b = datetime.datetime.fromisoformat(p["end_utc"])
    ph.append((a.hour * 3600 + a.minute * 60 + a.second + a.microsecond / 1e6 - off, b.hour * 3600 + b.minute * 60 + b.second + b.microsecond / 1e6 - off, p["name"]))
flag = set(json.load(open(d + "/inventory.json")))
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
by = collections.defaultdict(list)
for o in json.load(open(d + "_yolo.json")):
    by[o["file"]].append(o)
F = 840.34
tracks = []
out = collections.defaultdict(collections.Counter)
examples = collections.defaultdict(list)
frames = sorted(by, key=lambda f: float(man[f]["game"]))
for f in frames:
    if f in flag:
        continue
    m = man[f]
    g = float(m["game"])
    name = next((k for a, b, k in ph if a <= g < b), "between")
    out[name]["frames"] += 1
    sighted = False
    for o in by[f]:
        if o["kind"] not in BARS:
            continue
        for b in o["found"]:
            if b["label"] != "person" or b["conf"] < BARS[o["kind"]] or (o["kind"] != "full" and abs(b["y"] - 120) < 2):
                continue
            cx, foot = b["x"] + b["w"] / 2, b["y"] + b["h"]
            t = next((t for t in tracks if g - t["last"] <= 5 and abs(t["cx"] - cx) <= 60 and abs(t["by"] - foot) <= 60), None)
            if t is None:
                t = dict(x0=cx, y0=foot, moved=False, best=0.0, n=0)
                tracks.append(t)
            t.update(cx=cx, by=foot, last=g)
            t["moved"] = t["moved"] or math.hypot(cx - t["x0"], foot - t["y0"]) >= 10
            t["best"] = max(t["best"], b["conf"])
            if t["moved"] and t["best"] >= 0.6:
                x = None
                if m["placed_rel"] != "" and abs(float(m["placed_rel"])) < 80:
                    x = 800 + F * math.tan(math.radians(float(m["placed_rel"])))
                near = float(m["placed_near"]) if m["placed_near"] != "" else None
                at = x is not None and near is not None and b["x"] - max(25, 1260 / max(near, 1)) <= x <= b["x"] + b["w"] + max(25, 1260 / max(near, 1))
                out[name]["sightings at the infected" if at else "sightings elsewhere"] += 1
                if not at:
                    examples[name].append((round(g), round(b["conf"], 2), round(b["x"]), round(b["y"]), round(b["h"])))
for k, v in out.items():
    print(f"{k:<12} {dict(v)}; first elsewhere {examples[k][:4]}")
print(f"{len(flag)} frames left out by the inventory rule")

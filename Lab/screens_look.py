"""What does the detector make of each screen? Per driver phase: frames, frames with a person box at the setup's bars, the tallest, and what the inventory rule said; a picture of each screen with its boxes; crawl speed from the pose log, for the prone key. Usage: screens_look.py <n>"""
import collections, csv, datetime, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
S = "C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/"
n = sys.argv[1]
d = f"L:/AssistZLabWork/frames/see{n}"
run = open(d + "/run.txt").read().strip().replace("\\", "/")
man = {m["frame"]: m for m in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
pose = list(csv.DictReader(open(run + "/pose.csv")))
world = list(csv.DictReader(open(run + "/world.csv")))
print("world tail:", [(w["event"], int(w["game_ms"]) // 1000) for w in world if w["event"] in ("die", "hour")])
sec = lambda u: sum(float(v) * k for v, k in zip(u.split(":"), (3600, 60, 1)))
off = np.median([sec(p["utc"]) - int(p["game_ms"]) / 1000 for p in pose])
drive = json.load(open(run + "/see_drive.json"))
print("driver stopped:", drive.get("stopped"), "phases", len(drive["phases"]))
ph = []
for p in drive["phases"]:
    a = datetime.datetime.fromisoformat(p["start_utc"]); b = datetime.datetime.fromisoformat(p["end_utc"])
    ph.append((a.hour * 3600 + a.minute * 60 + a.second + a.microsecond / 1e6 - off, b.hour * 3600 + b.minute * 60 + b.second + b.microsecond / 1e6 - off, p["name"]))
die = next((int(w["game_ms"]) / 1000 for w in world if w["event"] == "die"), 1e9)
flag = set(json.load(open(d + "/inventory.json")))
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
by = collections.defaultdict(list)
for o in json.load(open(d + "_yolo.json")):
    by[o["file"]].append(o)
stats = collections.defaultdict(lambda: collections.Counter())
heights = collections.defaultdict(list)
examples = {}
for f, os_ in by.items():
    g = float(man[f]["game"])
    if g >= die + 3:
        name = "dead"
    else:
        name = next((k for a, b, k in ph if a + 1.0 <= g < b - 0.3), None)
        if name is None:
            name = "before_driver" if g < ph[0][0] else "between"
    boxes = [(o["kind"], b) for o in os_ if o["kind"] in BARS for b in o["found"] if b["label"] == "person" and b["conf"] >= BARS[o["kind"]] and not (o["kind"] != "full" and abs(b["y"] - 120) < 2)]
    near = float(man[f]["placed_near"]) if man[f]["placed_near"] != "" else None
    s = stats[name]
    s["frames"] += 1
    s["with a person box"] += bool(boxes)
    s["with one at 0.6 or more"] += any(b["conf"] >= 0.6 for _, b in boxes)
    s["inventory rule"] += f in flag
    s["infected within 40 m"] += near is not None and near < 40
    heights[name] += [b["h"] for _, b in boxes]
    if name not in examples or (boxes and not examples[name][1]):
        if name not in examples or boxes:
            examples[name] = (f, boxes)
for k, v in stats.items():
    h = heights[k]
    print(f"{k:<14} {dict(v)}" + (f"; box height median {np.median(h):.0f} max {max(h):.0f}" if h else ""))
names = ["before_driver", "pause_menu", "map", "inventory", "crawl", "dead"]
T = (400, 225)
sheet = Image.new("RGB", (T[0] * 3, (T[1] + 14) * 2), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
for i, k in enumerate(names):
    if k not in examples:
        continue
    f, boxes = examples[k]
    im = Image.open(os.path.join(d, f)).convert("RGB")
    draw = ImageDraw.Draw(im)
    for _, b in boxes:
        draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=(0, 255, 0), width=3)
    sheet.paste(im.resize(T), ((i % 3) * T[0], (i // 3) * (T[1] + 14)))
    dr.text(((i % 3) * T[0] + 2, (i // 3) * (T[1] + 14) + T[1]), f"{k}: {len(boxes)} boxes, best {max([b['conf'] for _, b in boxes], default=0):.2f}", fill=(255, 255, 0))
sheet.save(S + f"screens_{n}.png")
pt = np.array([sec(p["utc"]) - off for p in pose])
x, z = np.array([float(p["x"]) for p in pose]), np.array([float(p["z"]) for p in pose])
for k in ("crawl",):
    sp = []
    for a, b, name in ph:
        if name == k:
            m = (pt >= a) & (pt <= b)
            if m.sum() > 3:
                sp.append(np.hypot(np.diff(x[m]), np.diff(z[m])).sum() / max(pt[m][-1] - pt[m][0], 0.1))
    print(f"{k} speed m/s by phase: {[round(float(v), 2) for v in sp]}")

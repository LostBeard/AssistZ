"""Moving trial: which phase does each box fall in? Boxes at and away from the infected; how tall the false ones are; how far off the log's column real boxes are by phase, since a moving view leans harder on frame timing. A sheet of the false boxes. Usage: moving_false.py <n>"""
import collections, csv, datetime, json, math, os, random, sys
import numpy as np
from PIL import Image, ImageDraw
n = sys.argv[1]
S = "C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/"
F = 840.34
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
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
def phase(g):
    return next((p[2] for p in ph if p[0] <= g < p[1]), "other")
world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
spans = [(int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000, a["event"]) for a, b in zip(world, world[1:])]
counts = collections.defaultdict(collections.Counter)
offs = collections.defaultdict(list)
false = []
for o in json.load(open(d + "_yolo.json")):
    if o["kind"] not in BARS:
        continue
    m = man[o["file"]]
    g = float(m["game"])
    name = phase(g)
    span = next((s for s in spans if s[0] + 1 <= g < s[1]), None)
    counts[name]["inferences"] += 1
    for b in o["found"]:
        if b["label"] != "person" or b["conf"] < BARS[o["kind"]]:
            continue
        at = False
        if m["placed_rel"] != "" and abs(float(m["placed_rel"])) < 43.6:
            x = 800 + F * math.tan(math.radians(float(m["placed_rel"])))
            near = float(m["placed_near"])
            tol = max(25.0, 1260.0 / max(near, 1.0))
            at = b["x"] - tol <= x <= b["x"] + b["w"] + tol
            if at and 8 <= near < 40:
                offs[name].append(abs(b["x"] + b["w"] / 2 - x))
        if at:
            counts[name]["at infected"] += 1
        else:
            counts[name]["away"] += 1
            counts[name]["away, 500 px or taller"] += b["h"] >= 500
            counts[name]["away, conf 0.6+"] += b["conf"] >= 0.6
            false.append((d, o["file"], b, o["kind"], name, span[2] if span else None, m["placed_near"], m["placed_rel"]))
for name, c in counts.items():
    extra = f"; off the column median {np.median(offs[name]):.0f} px, p90 {np.percentile(offs[name], 90):.0f}, n {len(offs[name])}" if offs[name] else ""
    print(f"{name:<12} {dict(c)}{extra}")
random.seed(4)
T, N = 150, 8
small = [f for f in false if f[2]["h"] < 500]
tall = [f for f in false if f[2]["h"] >= 500]
rows = [("under 500 px", random.sample(small, min(16, len(small)))), ("500 px or taller", random.sample(tall, min(8, len(tall))))]
sheet = Image.new("RGB", (T * N, 16 + sum(((len(g) + N - 1) // N) * (T + 14) + 16 for _, g in rows)), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
y = 0
for name, items in rows:
    dr.text((2, y + 1), f"away from the infected, {name}", fill=(255, 255, 0))
    y += 16
    for i, (dd, f, b, kind, pn, sp, near, rel) in enumerate(items):
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        side = max(120, 1.6 * max(b["w"], b["h"]))
        box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        crop = Image.open(os.path.join(dd, f)).convert("RGB").crop(tuple(int(v) for v in box)).resize((T, T), Image.BILINEAR)
        s = T / side
        ImageDraw.Draw(crop).rectangle([(b["x"] - box[0]) * s, (b["y"] - box[1]) * s, (b["x"] + b["w"] - box[0]) * s, (b["y"] + b["h"] - box[1]) * s], outline=(0, 255, 0))
        px, py = (i % N) * T, y + (i // N) * (T + 14)
        sheet.paste(crop, (px, py))
        dr.text((px + 2, py + T), f"{pn[:6]} {b['conf']:.2f} {str(near)[:4]}m r{str(rel)[:5]}", fill=(200, 200, 200))
    y += ((len(items) + N - 1) // N) * (T + 14)
sheet.save(S + f"see{n}_false.png")
print("sheet", sheet.size)

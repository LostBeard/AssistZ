"""Setup 3, by eye and by count: the one false sighting during a slot; what line of sight did to slots; does placed_los match the picture (frames with 0 rays at 10-30 m, within the view, against frames with 3); how close the distance limit is at dusk."""
import collections, csv, json, math, os
import numpy as np
from PIL import Image, ImageDraw
S = "C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/"
F = 840.34
W = "L:/AssistZLabWork/frames/see"
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
RUNS = {18: "dusk", 19: "dusk", 20: "dusk", 21: "rain", 22: "rain", 23: "rain", 24: "noon", 25: "noon", 26: "noon"}
blocked, clear = [], []
for n, cond in RUNS.items():
    d = W + str(n)
    man = {m["frame"]: m for m in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
    run = open(d + "/run.txt").read().strip().replace("\\", "/")
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    by = collections.defaultdict(list)
    for o in json.load(open(d + "_yolo.json")):
        by[o["file"]].append(o)
    frames = sorted(by, key=lambda f: float(man[f]["game"]))
    line = []
    for a, b in zip(world, world[1:]):
        if a["event"] != "infected":
            continue
        lo, hi = int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000
        fs = [f for f in frames if lo + 1 <= float(man[f]["game"]) < hi and man[f]["placed_rel"] != ""]
        angle = [f for f in fs if 10 <= float(man[f]["placed_near"]) < 30 and abs(float(man[f]["placed_rel"])) < 43.6]
        seen = [f for f in angle if int(man[f]["placed_los"]) >= 1]
        if float(a["distance"]) <= 20:
            line.append(f"{a['distance']}m:{len(seen)}/{len(angle)}")
        for f in angle:
            (blocked if int(man[f]["placed_los"]) == 0 else clear if int(man[f]["placed_los"]) == 3 else []).append((d, f, man[f], n))
    print(f"see{n} {cond}: slots at 10-30 m, seconds with line of sight / within the angle: {' '.join(line)}")
print("frames at 10-30 m within the angle: 0 rays", len(blocked), "; 3 rays", len(clear))
blocked_hits = 0
for d, f, m, n in blocked:
    x = 800 + F * math.tan(math.radians(float(m["placed_rel"])))
    res = json.load(open(d + "_yolo.json")) if False else None
print("--- sheet: top rows, 0 rays; then 3 rays; then the false box in see25")
import random
random.seed(3)
T, N = 170, 7
rows = [("0 rays", random.sample(blocked, min(14, len(blocked)))), ("3 rays", random.sample(clear, min(7, len(clear))))]
sheet = Image.new("RGB", (T * N, (T + 14) * 4), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
y = 0
for name, items in rows:
    for i, (d, f, m, n) in enumerate(items):
        x = 800 + F * math.tan(math.radians(float(m["placed_rel"])))
        crop = Image.open(os.path.join(d, f)).convert("RGB").crop((int(x - 100), 370, int(x + 100), 570)).resize((T, T), Image.BICUBIC)
        px, py = (i % N) * T, y + (i // N) * (T + 14)
        ImageDraw.Draw(crop).line((T / 2, T - 12, T / 2, T), fill=(255, 255, 0))
        sheet.paste(crop, (px, py))
        dr.text((px + 2, py + T), f"s{n} {name} {float(m['placed_near']):.0f}m", fill=(220, 220, 220))
    y += ((len(items) + N - 1) // N) * (T + 14)
# The false sighting: see25, game 629, around x 552, y 463, h 69.
d = W + "25"
man = {m["frame"]: m for m in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
near = sorted(man, key=lambda f: abs(float(man[f]["game"]) - 629))[:1][0]
fs = sorted(man, key=lambda f: float(man[f]["game"]))
i0 = fs.index(near)
for i, k in enumerate(range(-6, 8, 2)):
    f = fs[i0 + k * 2]
    crop = Image.open(os.path.join(d, f)).convert("RGB").crop((552 - 60, 400, 552 + 140, 600)).resize((T, T), Image.BICUBIC)
    sheet.paste(crop, (i * T, y))
    dr.text((i * T + 2, y + T), f"s25 {float(man[f]['game']):.0f}s placed {man[f]['placed_near'][:4]}m rel {man[f]['placed_rel']}", fill=(220, 220, 220))
sheet.save(S + "setup3_look.png")
print("saved", sheet.size)

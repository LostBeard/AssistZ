"""Contact sheet of YOLOv8's person boxes, for checking by eye: boxes at the infected's column, boxes away from it, and rest frames. Stays local. Usage: yolo_sheet.py <out png> <frames dir> <results json> [...]"""
import collections, csv, json, os, random, sys
from PIL import Image, ImageDraw
PX = 8.8 * 1600 / 960
out = sys.argv[1]
args = sys.argv[2:]
at, away, rest = [], [], []
for frames_dir, res in zip(args[0::2], args[1::2]):
    man = {r["frame"]: r for r in csv.DictReader(open(os.path.join(frames_dir, "manifest.csv"), encoding="utf-8"))}
    run = open(os.path.join(frames_dir, "run.txt")).read().strip().replace("\\", "/")
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    spans = [(int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000, a["event"]) for a, b in zip(world, world[1:])]
    for o in json.load(open(res)):
        m = man[o["file"]]
        g = float(m["game"])
        span = next((s for s in spans if s[0] + 1 <= g < s[1]), None)
        if not span:
            continue
        for d in o["found"]:
            if d["label"] != "person" or d["conf"] < 0.35:
                continue
            item = (os.path.join(frames_dir, o["file"]), d, m["near"], o["kind"])
            if span[2] == "rest":
                rest.append(item)
            elif m["rel"] and d["x"] - 25 <= 800 + float(m["rel"]) * PX <= d["x"] + d["w"] + 25:
                at.append(item)
            else:
                away.append(item)
print("at column", len(at), "away from it", len(away), "rest", len(rest))
random.seed(7)
groups = [("at", random.sample(at, min(24, len(at)))), ("away", random.sample(away, min(24, len(away)))), ("rest", random.sample(rest, min(24, len(rest))))]
T = 150
sheet = Image.new("RGB", (8 * T, sum(((len(g) + 7) // 8) * (T + 14) + 16 for _, g in groups)), (20, 20, 20))
draw = ImageDraw.Draw(sheet)
y = 0
for name, g in groups:
    draw.text((2, y + 2), name, fill=(255, 255, 0))
    y += 16
    for i, (path, d, near, kind) in enumerate(g):
        im = Image.open(path).convert("RGB")
        cx, cy = d["x"] + d["w"] / 2, d["y"] + d["h"] / 2
        side = max(120, 1.6 * max(d["w"], d["h"]))
        box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        crop = im.crop(tuple(int(v) for v in box)).resize((T, T), Image.BILINEAR)
        s = T / side
        ImageDraw.Draw(crop).rectangle([(d["x"] - box[0]) * s, (d["y"] - box[1]) * s, (d["x"] + d["w"] - box[0]) * s, (d["y"] + d["h"] - box[1]) * s], outline=(0, 255, 0))
        px, py = (i % 8) * T, y + (i // 8) * (T + 14)
        sheet.paste(crop, (px, py))
        draw.text((px + 2, py + T), f"{d['conf']:.2f} {float(near):.0f}m {kind}", fill=(200, 200, 200))
    y += ((len(g) + 7) // 8) * (T + 14)
sheet.save(out)
print("saved", out, sheet.size)

"""Full resolution crops where each infected should be, one per slot, with the same crop from a frame with it gone, to judge by eye what can be seen. Usage: see_crops.py <frames dir> <sheet.png>"""
import csv, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageChops
frames_dir, sheet = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(os.path.join(frames_dir, "manifest.csv"), encoding="utf-8")))
run = open(os.path.join(frames_dir, "run.txt")).read().strip().replace("\\", "/")
world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
g = np.array([float(r["game"]) for r in rows])
PX = 8.8 * 1600 / 960
slots = []
for a, b in zip(world, world[1:]):
    if a["event"] != "infected":
        continue
    lo, hi = int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000
    i = np.where((g >= lo + 4) & (g < lo + 12))[0]
    j = np.where((g >= hi + 6) & (g < hi + 16))[0]
    if len(i) and len(j):
        slots.append((float(a["distance"]), rows[i[len(i) // 2]], rows[j[len(j) // 2]]))
slots = slots[:12]
C = 300
CH = 200
out = Image.new("RGB", ((C * 2 + 8) * 2, (CH + 4) * ((len(slots) + 1) // 2)), (40, 40, 40))
for k, (dist, on, off) in enumerate(slots):
    rel = float(on["rel"]) if on["rel"] else 0
    im = Image.open(os.path.join(frames_dir, on["frame"])).convert("RGB")
    im2 = Image.open(os.path.join(frames_dir, off["frame"])).convert("RGB")
    W, H = im.size
    x = int(W / 2 + rel * PX * W / 1600)
    y = int(H * 0.5)
    box = (max(0, min(W - C, x - C // 2)), max(0, min(H - CH, y - CH // 2)), 0, 0)
    box = (box[0], box[1], box[0] + C, box[1] + CH)
    a, b = im.crop(box), im2.crop(box)
    d = ImageDraw.Draw(a)
    d.text((4, 4), f"{dist:.0f} m nearest {float(on['near']):.0f} state {on['state']} rel {rel:.0f}", fill=(255, 255, 0))
    out.paste(a, ((k % 2) * (C * 2 + 8), (k // 2) * (CH + 4)))
    out.paste(b, ((k % 2) * (C * 2 + 8) + C + 2, (k // 2) * (CH + 4)))
out.save(sheet)
print(len(slots), "slots", [s[0] for s in slots])

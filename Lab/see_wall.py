"""Past 30 m nothing fires at noon. Is the infected visible at its logged column? Native-size crops, magnified, by distance band, with the column marked; plus one whole frame. Usage: see_wall.py <frames dir> <out png>"""
import csv, os, sys
from PIL import Image, ImageDraw
PX = 8.8 * 1600 / 960
d, out = sys.argv[1], sys.argv[2]
man = list(csv.DictReader(open(d + "/manifest.csv", encoding="utf-8")))
bands = ((20, 30), (30, 40), (40, 50), (50, 70), (70, 100))
T, N = 200, 6
sheet = Image.new("RGB", (T * N, (T + 14) * len(bands)), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
for r, (lo, hi) in enumerate(bands):
    g = [m for m in man if m["placed_near"] and lo <= float(m["placed_near"]) < hi and abs(float(m["placed_rel"])) < 50]
    pick = g[:: max(1, len(g) // N)][:N]
    for i, m in enumerate(pick):
        x = 800 + float(m["placed_rel"]) * PX
        im = Image.open(os.path.join(d, m["frame"])).convert("RGB")
        side = 100
        box = (int(x - side / 2), 300, int(x + side / 2), 300 + side)
        y = 0
        crop = im.crop((int(x - 70), 420, int(x + 70), 560)).resize((T, T), Image.BICUBIC)
        ImageDraw.Draw(crop).line((T / 2, T - 14, T / 2, T), fill=(255, 255, 0))
        px, py = i * T, r * (T + 14)
        sheet.paste(crop, (px, py))
        dr.text((px + 2, py + T), f"{float(m['placed_near']):.0f} m rel {float(m['placed_rel']):.0f} {m['frame'][:6]}", fill=(220, 220, 220))
sheet.save(out)
print(out, sheet.size)


"""What are they, by eye? Person boxes in the setup under test, by group: at rest; away from the infected during a slot; at the column past 50 m; and frames missed at 10-20 m in rain. Also counts of false boxes by place per run, and how many touch the top of their tile. Stays local. Usage: see_look.py <out png>"""
import collections, csv, json, os, random, sys
from PIL import Image, ImageDraw
PX = 8.8 * 1600 / 960
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
W = "L:/AssistZLabWork/frames/see"
COND = {4: "dusk", 5: "dusk", 6: "dusk", 7: "rain", 8: "rain", 9: "rain", 10: "noon", 11: "noon", 12: "noon", 13: "night"}
groups = collections.defaultdict(list)
counts = collections.defaultdict(lambda: collections.Counter())
places = collections.defaultdict(collections.Counter)
edge = collections.Counter()
for n, cond in COND.items():
    d = W + str(n)
    man = {r["frame"]: r for r in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
    run = open(d + "/run.txt").read().strip().replace("\\", "/")
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    spans = [(int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000, a["event"]) for a, b in zip(world, world[1:])]
    by = collections.defaultdict(list)
    for o in json.load(open(d + "_yolo.json")):
        by[o["file"]].append(o)
    for f, os_ in by.items():
        m = man[f]
        g = float(m["game"])
        span = next((s for s in spans if s[0] + 1 <= g < s[1]), None)
        if not span:
            continue
        got = [(o["kind"], b) for o in os_ if o["kind"] in BARS for b in o["found"] if b["label"] == "person" and b["conf"] >= BARS[o["kind"]]]
        rel = float(m["placed_rel"]) if m["placed_rel"] != "" else None
        near = float(m["placed_near"]) if m["placed_near"] != "" else None
        x = 800 + rel * PX if rel is not None else None
        hit_any = False
        for kind, b in got:
            at = x is not None and b["x"] - 25 <= x <= b["x"] + b["w"] + 25
            hit_any = hit_any or at
            item = (d, f, b, kind, near, n)
            if span[2] == "rest":
                groups[f"rest {cond}"].append(item)
                counts[cond]["rest"] += 1
            elif at and near is not None and near >= 50:
                groups[f"at column past 50 m {cond}"].append(item)
            elif not at:
                groups[f"away in slot {cond}"].append(item)
                counts[cond]["away in slot"] += 1
            else:
                counts[cond]["at column"] += 1
            if not at:
                places[n][(round((b["x"] + b["w"] / 2) / 80) * 80, round((b["y"] + b["h"] / 2) / 80) * 80)] += 1
                edge[(cond, "touches tile top" if kind != "full" and abs(b["y"] - 120) < 2 else "not")] += 1
        if span[2] == "infected" and near is not None and 10 <= near < 20 and abs(rel) < 50 and not hit_any and cond == "rain":
            groups["missed 10-20 m rain"].append((d, f, dict(x=x - 40, y=380, w=80, h=160, conf=0), "miss", near, n))
for c, v in counts.items():
    print(c, dict(v))
print("false boxes touching a tile's top:", dict(edge))
for n, v in places.items():
    print(f"see{n} ({COND[n]}) places of false boxes:", v.most_common(5))
random.seed(5)
names = ["rest dusk", "away in slot dusk", "rest night", "at column past 50 m noon", "at column past 50 m rain", "away in slot noon", "away in slot rain", "missed 10-20 m rain"]
T, N = 150, 8
rows = [(k, random.sample(groups[k], min(N, len(groups[k])))) for k in names if groups[k]]
sheet = Image.new("RGB", (T * N, len(rows) * (T + 30)), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
y = 0
for name, items in rows:
    dr.text((2, y + 1), f"{name}: {len(groups[name])}", fill=(255, 255, 0))
    y += 16
    for i, (d, f, b, kind, near, n) in enumerate(items):
        im = Image.open(os.path.join(d, f)).convert("RGB")
        cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
        side = max(120, 1.6 * max(b["w"], b["h"]))
        box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        crop = im.crop(tuple(int(v) for v in box)).resize((T, T), Image.BILINEAR)
        s = T / side
        ImageDraw.Draw(crop).rectangle([(b["x"] - box[0]) * s, (b["y"] - box[1]) * s, (b["x"] + b["w"] - box[0]) * s, (b["y"] + b["h"] - box[1]) * s], outline=(0, 255, 0))
        sheet.paste(crop, (i * T, y))
        dr.text((i * T + 2, y + T), f"s{n} {b['conf']:.2f} {near if near is None else round(near)}m {kind}", fill=(200, 200, 200))
    y += T + 14
sheet.save(sys.argv[1])
print(sys.argv[1], sheet.size)

"""The moving test's open checks. 1: the inventory rule against the driver's phases, frames held open flagged; flagged frames outside. 2: is any sighting on an open inventory frame the rule missed? 3: the false sightings during slots: which phase, how far from the logged infected, and a sheet of all of them. Usage: moving_verdict.py"""
import collections, csv, datetime, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
S = "C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/"
RUNS = {29: "dusk", 37: "dusk", 38: "dusk", 31: "rain", 32: "rain", 39: "rain", 34: "noon", 40: "noon", 41: "noon"}
F = 840.34
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
tot_open = tot_flag = out_flag = 0
items = []
phase_counts = collections.Counter()
for n, cond in RUNS.items():
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
    inv = [(a, b) for a, b, k in ph if k == "inventory"]
    flagged = set(json.load(open(d + "/inventory.json")))
    held = [f for f, m in man.items() if any(a + 1.5 <= float(m["game"]) < b - 0.5 for a, b in inv)]
    near = [f for f, m in man.items() if any(a - 1.5 <= float(m["game"]) < b + 2.5 for a, b in inv)]
    tot_open += len(held)
    tot_flag += sum(f in flagged for f in held)
    out = [f for f in flagged if f not in near]
    out_flag += len(out)
    phase = lambda g: next((k for a, b, k in ph if a <= g < b), "other")
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    spans = [(int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000, a["event"]) for a, b in zip(world, world[1:])]
    by = collections.defaultdict(list)
    for o in json.load(open(d + "_yolo.json")):
        by[o["file"]].append(o)
    frames = [f for f in sorted(by, key=lambda f: float(man[f]["game"])) if f not in flagged]
    # plan4's tracks, as in see_score.
    tracks = []
    for f in frames:
        m = man[f]
        g = float(m["game"])
        span = next((s for s in spans if s[0] + 1 <= g < s[1]), None)
        for o in by[f]:
            if o["kind"] not in BARS:
                continue
            for b in o["found"]:
                if b["label"] != "person" or b["conf"] < BARS[o["kind"]] or (o["kind"] != "full" and abs(b["y"] - 120) < 2):
                    continue
                cx, by_ = b["x"] + b["w"] / 2, b["y"] + b["h"]
                t = next((t for t in tracks if g - t["last"] <= 5 and abs(t["cx"] - cx) <= 60 and abs(t["by"] - by_) <= 60), None)
                if t is None:
                    t = dict(x0=cx, y0=by_, moved=False, best=0.0, said=False)
                    tracks.append(t)
                t.update(cx=cx, by=by_, last=g)
                t["moved"] = t["moved"] or math.hypot(cx - t["x0"], by_ - t["y0"]) >= 10
                t["best"] = max(t["best"], b["conf"])
                if not (t["moved"] and t["best"] >= 0.6) or not span or span[2] != "infected" or t["said"]:
                    continue
                near_m = float(m["placed_near"]) if m["placed_near"] != "" else None
                rel = float(m["placed_rel"]) if m["placed_rel"] != "" else None
                if rel is None:
                    continue
                x = 800 + F * math.tan(math.radians(rel)) if abs(rel) < 89 else 1e9
                tol = max(25.0, 1260.0 / max(near_m, 1.0))
                at = b["x"] - tol <= x <= b["x"] + b["w"] + tol
                if not at:
                    t["said"] = True
                    k = phase(g)
                    phase_counts[k] += 1
                    items.append((d, f, b, k, n, near_m, rel, cx - x if abs(x) < 1e8 else None))
    print(f"see{n} {cond}: held open {len(held)}, flagged {sum(f in flagged for f in held)}; flagged away from any opening {len(out)}")
print(f"LIMIT inventory rule: {tot_flag}/{tot_open} held-open frames flagged ({100 * tot_flag / max(1, tot_open):.1f}%, limit 90); flagged elsewhere {out_flag}")
print("false sighting tracks during slots, by phase:", dict(phase_counts), "total", len(items))
T, N = 165, 7
sheet = Image.new("RGB", (T * N, (T + 14) * ((len(items) + N - 1) // N)), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
for i, (d, f, b, k, n, near_m, rel, off_x) in enumerate(items):
    cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    side = max(160, 1.6 * max(b["w"], b["h"]))
    box = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    crop = Image.open(os.path.join(d, f)).convert("RGB").crop(tuple(int(v) for v in box)).resize((T, T), Image.BILINEAR)
    s = T / side
    ImageDraw.Draw(crop).rectangle([(b["x"] - box[0]) * s, (b["y"] - box[1]) * s, (b["x"] + b["w"] - box[0]) * s, (b["y"] + b["h"] - box[1]) * s], outline=(0, 255, 0))
    px, py = (i % N) * T, (i // N) * (T + 14)
    sheet.paste(crop, (px, py))
    dr.text((px + 2, py + T), f"s{n} {k[:6]} {b['conf']:.2f} {near_m or 0:.0f}m off{(off_x or 0):.0f}", fill=(220, 220, 220))
sheet.save(S + "moving_false_all.png")
print("sheet", sheet.size)

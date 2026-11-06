"""Why did fresh dusk fall? Per run and slot: frames boxed, where the player stood, how bright. Native crops around the infected's column at 10-30 m, by run, with the box if any; false boxes in see16. Output: a sheet. Compare with see4 to see6."""
import collections, csv, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
S = "C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/"
F = 840.34
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
T, N = 170, 7
RUNS = (4, 5, 6, 14, 15, 16)
sheet = Image.new("RGB", (T * N, (T + 14) * len(RUNS)), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
for r, n in enumerate(RUNS):
    d = f"L:/AssistZLabWork/frames/see{n}"
    man = {m["frame"]: m for m in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
    run = open(d + "/run.txt").read().strip().replace("\\", "/")
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    env = list(csv.DictReader(open(run + "/env.csv")))
    world = list(csv.DictReader(open(run + "/world.csv")))
    by = collections.defaultdict(list)
    for o in json.load(open(d + "_yolo.json")):
        by[o["file"]].append(o)
    frames = sorted(by, key=lambda f: float(man[f]["game"]))
    near = [f for f in frames if man[f]["placed_near"] != "" and 10 <= float(man[f]["placed_near"]) < 30 and abs(float(man[f]["placed_rel"])) < 43]
    def got(f):
        x = 800 + F * math.tan(math.radians(float(man[f]["placed_rel"])))
        return [b for o in by[f] if o["kind"] in BARS for b in o["found"] if b["label"] == "person" and b["conf"] >= BARS[o["kind"]] and b["x"] - 25 <= x <= b["x"] + b["w"] + 25]
    hit = [f for f in near if got(f)]
    bright = np.mean([np.asarray(Image.open(os.path.join(d, f)).convert("L").crop((0, 200, 1600, 700))).mean() for f in frames[::60]])
    full = sum(1 for f in near if any(b in [q for o in by[f] if o["kind"] == "full" for q in o["found"]] for b in got(f)))
    print(f"see{n}: player {float(pose[len(pose) // 2]['x']):.0f},{float(pose[len(pose) // 2]['z']):.0f}; pitch {float(pose[len(pose) // 2]['pitch']):.1f}; brightness {bright:.1f}; overcast {float(env[len(env) // 2]['overcast']):.2f} rain {max(float(e['rain']) for e in env):.2f} fog {max(float(e['fog']) for e in env):.2f}; boxed 10-30 m {len(hit)}/{len(near)}; first infected type lines {[w['distance'] for w in world if w['event'] == 'hour']}")
    pick = near[:: max(1, len(near) // N)][:N]
    for i, f in enumerate(pick):
        m = man[f]
        x = 800 + F * math.tan(math.radians(float(m["placed_rel"])))
        h = 1268 / max(float(m["placed_near"]), 1)
        cy = 470
        side = 200
        box = (int(x - side / 2), cy - side // 2, int(x + side / 2), cy + side // 2)
        crop = Image.open(os.path.join(d, f)).convert("RGB").crop(box).resize((T, T), Image.BICUBIC)
        q = T / side
        for b in got(f):
            ImageDraw.Draw(crop).rectangle([(b["x"] - box[0]) * q, (b["y"] - box[1]) * q, (b["x"] + b["w"] - box[0]) * q, (b["y"] + b["h"] - box[1]) * q], outline=(0, 255, 0))
        sheet.paste(crop, (i * T, r * (T + 14)))
        dr.text((i * T + 2, r * (T + 14) + T), f"s{n} {float(m['placed_near']):.0f}m rel{float(m['placed_rel']):.0f}" + (" HIT" if got(f) else ""), fill=(220, 220, 220))
sheet.save(S + "dusk_look.png")
print("sheet", sheet.size)

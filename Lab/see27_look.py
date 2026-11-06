"""see27: when did the view move and the inventory open? Picture change frame to frame, the slots on the log's clock, a sheet to look at. Also the body pitch flag across all runs, to see whether it is noise."""
import csv, os
import numpy as np
from PIL import Image, ImageDraw
S = "C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/"
d = "L:/AssistZLabWork/frames/see27"
man = list(csv.DictReader(open(d + "/manifest.csv", encoding="utf-8")))
man.sort(key=lambda m: float(m["game"]))
run = open(d + "/run.txt").read().strip().replace("\\", "/")
world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
print("slots:", [(w["event"][:3], int(w["game_ms"]) // 1000, w["distance"]) for w in world])
def small(m):
    return np.asarray(Image.open(os.path.join(d, m["frame"])).convert("L").resize((80, 45)), dtype=np.float32)
ref = None
changes = []
step = man[::4]
for m in step:
    a = small(m)
    if ref is not None:
        changes.append((float(m["game"]), float(np.abs(a - ref).mean()), float(a.mean())))
    ref = a
big = [(round(g), round(c, 1), round(b, 1)) for g, c, b in changes if c > 6]
print("big picture changes (game s, change, brightness):", big[:30], "...", len(big))
pick = [g for g, c, b in changes if c > 6][:6]
wanted = sorted(set([660, 676, 680, 700, 760, 800, 860, 900] + [round(g) for g in pick]))[:16]
T = (300, 169)
sheet = Image.new("RGB", (T[0] * 4, (T[1] + 14) * ((len(wanted) + 3) // 4)), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
for i, g in enumerate(wanted):
    m = min(man, key=lambda m: abs(float(m["game"]) - g))
    im = Image.open(os.path.join(d, m["frame"])).convert("RGB").resize(T)
    sheet.paste(im, ((i % 4) * T[0], (i // 4) * (T[1] + 14)))
    dr.text(((i % 4) * T[0] + 2, (i // 4) * (T[1] + 14) + T[1]), f"{float(m['game']):.0f}s near {m['placed_near'][:5]} los {m['placed_los']} rel {m['placed_rel']}", fill=(220, 220, 220))
sheet.save(S + "see27_look.png")
for n in list(range(4, 13)) + list(range(14, 17)) + list(range(18, 28)):
    r = open(f"L:/AssistZLabWork/frames/see{n}/run.txt").read().strip().replace("\\", "/")
    p = list(csv.DictReader(open(r + "/pose.csv")))[20:]
    out = []
    for c in ("yaw", "pitch", "roll", "aim_lr", "aim_ud"):
        v = np.array([float(x[c]) for x in p])
        out.append(f"{c} {np.ptp(((v - v[0] + 180) % 360) - 180):.2f}")
    print(f"see{n}: " + ", ".join(out))

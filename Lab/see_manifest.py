"""
Frames against the game's log: for each frame, the infected's distance, state and where it should be on screen. Writes manifest.csv beside the frames and a contact sheet by eye. Horizontal field of view is measured against nothing yet, so the column is only a guide.
Usage: see_manifest.py <frames dir> <sheet.png>
"""
import csv, datetime, glob, json, math, os, re, sys
import numpy as np
from PIL import Image, ImageDraw
frames_dir, sheet = sys.argv[1], sys.argv[2]
run = open(os.path.join(frames_dir, "run.txt")).read().strip().replace("\\", "/")
files = sorted(f for f in glob.glob(os.path.join(frames_dir, "*.jpg")))
print(run, len(files), "frames; first", os.path.basename(files[0]), "last", os.path.basename(files[-1]))
grab = open(run + "/grab.txt", encoding="utf-8", errors="replace").read()
print("grab.txt head:", grab[:300].replace("\n", " | "))
pose = list(csv.DictReader(open(run + "/pose.csv")))
print("pose yaw", pose[0]["yaw"], pose[-1]["yaw"], "at", pose[len(pose) // 2]["x"], pose[len(pose) // 2]["z"])
env = list(csv.DictReader(open(run + "/env.csv")))
world = list(csv.DictReader(open(run + "/world.csv")))
print("world:", [(w["game_ms"], w["event"], w["distance"]) for w in world[:6]])
# utc of a frame: from its name, or from the file's time.
def stamp(path):
    m = re.search(r"(\d{8})_(\d{6})(?:_(\d+))?", os.path.basename(path))
    if m:
        t = datetime.datetime.strptime(m[1] + m[2], "%Y%m%d%H%M%S").replace(tzinfo=datetime.timezone.utc).timestamp()
        return t + (int(m[3]) / 1000 if m[3] else 0)
    return os.path.getmtime(path)


times = np.array([stamp(f) for f in files])
day0 = datetime.datetime.fromtimestamp(times[0], datetime.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
game = times - day0 - off
print(f"frames in game time {game[0]:.0f} to {game[-1]:.0f}; env {int(env[0]['game_ms']) / 1000:.0f} to {int(env[-1]['game_ms']) / 1000:.0f}")
et = np.array([int(e["game_ms"]) / 1000 for e in env])
pg = np.array([int(p["game_ms"]) / 1000 for p in pose])
rows = []
for f, g in zip(files, game):
    e = env[int(np.argmin(np.abs(et - g)))]
    p = pose[int(np.argmin(np.abs(pg - g)))]
    near = float(e["infected_nearest"])
    rel = None
    if near >= 0 and float(e["infected_x"]) != 0:
        rel = (math.degrees(math.atan2(float(e["infected_x"]) - float(p["x"]), float(e["infected_z"]) - float(p["z"]))) - float(p["yaw"]) + 180) % 360 - 180
    # The placed infected itself, on runs that log it: the nearest of all can be a natural one (see1 had 11-13 about).
    placed_near, placed_rel, placed_state = "", "", ""
    placed_los = e.get("placed_los", "")
    if float(e.get("placed_x") or 0) != 0:
        dx, dz = float(e["placed_x"]) - float(p["x"]), float(e["placed_z"]) - float(p["z"])
        placed_near = round(math.hypot(dx, dz), 2)
        placed_rel = round((math.degrees(math.atan2(dx, dz)) - float(p["yaw"]) + 180) % 360 - 180, 1)
        placed_state = int(e["placed_state"])
    rows.append(dict(frame=os.path.basename(f), game=round(float(g), 2), near=near, state=int(e["infected_state"]), worst=int(e["infected_worst"]), rel=round(rel, 1) if rel is not None else "", count=int(e["infected"]), placed_near=placed_near, placed_rel=placed_rel, placed_state=placed_state, placed_los=placed_los))
with open(os.path.join(frames_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("states:", {k: sum(r["state"] == k for r in rows) for k in sorted({r["state"] for r in rows})})
print("nearest in bands:", {b: sum(lo <= r["near"] < hi for r in rows) for b, (lo, hi) in dict(a0_15=(0, 15), b15_30=(15, 30), c30_50=(30, 50), d50_70=(50, 70), e70_100=(70, 100), f100_150=(100, 150)).items()}, "none", sum(r["near"] < 0 for r in rows))
print("in view (|rel| under 35), calm:", {b: sum(lo <= r["near"] < hi and r["rel"] != "" and abs(r["rel"]) < 35 and r["state"] == 0 for r in rows) for b, (lo, hi) in dict(a0_15=(0, 15), b15_30=(15, 30), c30_50=(30, 50), d50_70=(50, 70), e70_100=(70, 100), f100_150=(100, 150)).items()})
# Contact sheet: one frame per distance band with an infected in view, and one with none.
pick = []
for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 70), (70, 100), (100, 150)):
    c = [r for r in rows if lo <= r["near"] < hi and r["rel"] != "" and abs(r["rel"]) < 30]
    if c:
        pick.append(c[len(c) // 2])
none = [r for r in rows if r["near"] < 0 or r["near"] > 200]
if none:
    pick.append(none[len(none) // 2])
W, H = 620, 349
out = Image.new("RGB", (W * 2 + 6, (H + 4) * ((len(pick) + 1) // 2)), (40, 40, 40))
for i, r in enumerate(pick):
    im = Image.open(os.path.join(frames_dir, r["frame"])).convert("RGB").resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    d.text((6, 6), f"near {r['near']:.0f} m state {r['state']} rel {r['rel']} count {r['count']}", fill=(255, 255, 0))
    if r["rel"] != "":
        x = W / 2 + r["rel"] / 35 * W / 2     # a guide only: assumes about 70 degrees across
        d.line((x, H - 30, x, H - 6), fill=(255, 60, 60), width=2)
    out.paste(im, ((i % 2) * (W + 6), (i // 2) * (H + 4)))
out.save(sheet)
print("sheet", sheet, len(pick))

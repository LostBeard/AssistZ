"""Did the driver do what it should? Per phase: how far and fast the player went, how far the view turned. And a sheet of frames through one cycle, including the inventory. Usage: moving_check.py <n>"""
import csv, datetime, json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw
n = sys.argv[1]
S = "C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/"
d = f"L:/AssistZLabWork/frames/see{n}"
run = open(d + "/run.txt").read().strip().replace("\\", "/")
pose = list(csv.DictReader(open(run + "/pose.csv")))
drive = json.load(open(run + "/see_drive.json"))
print("stopped:", drive.get("stopped"), "phases", len(drive["phases"]))
sec = lambda u: sum(float(v) * k for v, k in zip(u.split(":"), (3600, 60, 1)))
pt = np.array([sec(p["utc"]) for p in pose])
x, z, y = (np.array([float(p[c]) for p in pose]) for c in ("x", "z", "yaw"))
g = np.array([int(p["game_ms"]) / 1000 for p in pose])
off = np.median(pt - g)
by = {}
for ph in drive["phases"]:
    a = datetime.datetime.fromisoformat(ph["start_utc"]); b = datetime.datetime.fromisoformat(ph["end_utc"])
    s0 = a.hour * 3600 + a.minute * 60 + a.second + a.microsecond / 1e6
    s1 = b.hour * 3600 + b.minute * 60 + b.second + b.microsecond / 1e6
    k = (pt >= s0) & (pt <= s1)
    if k.sum() < 3:
        continue
    dist = np.hypot(np.diff(x[k]), np.diff(z[k])).sum()
    t = pt[k][-1] - pt[k][0]
    yaw = ((y[k] - y[k][0] + 180) % 360) - 180
    by.setdefault(ph["name"], []).append((dist / max(t, 0.1), np.ptp(yaw), np.hypot(x[k][-1] - x[k][0], z[k][-1] - z[k][0])))
for name, v in by.items():
    a = np.array(v)
    print(f"{name:<12} x{len(v):<3} path speed median {np.median(a[:, 0]):5.2f} m/s (p10 {np.percentile(a[:, 0], 10):.2f}); yaw swing median {np.median(a[:, 1]):6.1f} deg; start to end {np.median(a[:, 2]):5.1f} m")
h = np.array([float(p["y"]) for p in pose])
print(f"whole run: area {np.ptp(x):.0f} x {np.ptp(z):.0f} m; yaw from {y.min():.0f} to {y.max():.0f}; under 0.5 m height (water) {(h < 0.5).mean() * 100:.0f}% of the time")
man = list(csv.DictReader(open(d + "/manifest.csv", encoding="utf-8")))
man.sort(key=lambda m: float(m["game"]))
first = drive["phases"][:14]
T = (300, 169)
pick = []
for ph in first[:7] + [next(p for p in drive["phases"] if p["name"] == "inventory")]:
    a = datetime.datetime.fromisoformat(ph["start_utc"]); b = datetime.datetime.fromisoformat(ph["end_utc"])
    mid = (a.hour * 3600 + a.minute * 60 + a.second + a.microsecond / 1e6 + b.hour * 3600 + b.minute * 60 + b.second + b.microsecond / 1e6) / 2
    pick.append((ph["name"], mid - off))
sheet = Image.new("RGB", (T[0] * 4, (T[1] + 14) * 2), (20, 20, 20))
dr = ImageDraw.Draw(sheet)
for i, (name, game) in enumerate(pick):
    m = min(man, key=lambda r: abs(float(r["game"]) - game))
    sheet.paste(Image.open(os.path.join(d, m["frame"])).convert("RGB").resize(T), ((i % 4) * T[0], (i // 4) * (T[1] + 14)))
    dr.text(((i % 4) * T[0] + 2, (i // 4) * (T[1] + 14) + T[1]), f"{name} {game:.0f}s near {m['placed_near'][:5]} rel {m['placed_rel']}", fill=(220, 220, 220))
sheet.save(S + f"see{n}_moving.png")
print("sheet", sheet.size)

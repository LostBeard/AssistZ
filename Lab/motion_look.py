"""
First look at video item 2: which way is the player moving? From a pair of frames 0.5 s apart: walking forward, the picture expands from the centre. The ground slides down and out, the left side left and the right side right. A turn shifts it all one way. So: block-match bands left, right and low in the picture, and compare with the truth from the pose log, speed along the view's heading. Not a test: this is to see if the signal is there. Usage: motion_look.py <run numbers>
"""
import csv, math, os, sys
import numpy as np
from PIL import Image
W, H = 400, 225
BANDS = {"left": (10, 110, 110, 180), "right": (290, 390, 110, 180), "low": (130, 270, 120, 190)}
SEARCH = 24


def grey(path):
    return np.asarray(Image.open(path).convert("L").resize((W, H), Image.BILINEAR), dtype=np.float32)


def shift(a, b, box):
    """Best (dx, dy) moving box in a to b, and the match's quality."""
    x0, x1, y0, y1 = box
    t = a[y0:y1, x0:x1]
    t = t - t.mean()
    best, at = -1e9, (0, 0)
    for dy in range(-SEARCH, SEARCH + 1, 2):
        for dx in range(-SEARCH, SEARCH + 1, 2):
            yy0, xx0 = y0 + dy, x0 + dx
            if yy0 < 0 or xx0 < 0 or yy0 + t.shape[0] > H or xx0 + t.shape[1] > W:
                continue
            c = b[yy0:yy0 + t.shape[0], xx0:xx0 + t.shape[1]]
            c = c - c.mean()
            s = float((t * c).sum()) / (float(np.sqrt((t * t).sum() * (c * c).sum())) + 1e-6)
            if s > best:
                best, at = s, (dx, dy)
    return at[0], at[1], best


rows = []
for n in sys.argv[1:]:
    d = f"L:/AssistZLabWork/frames/see{n}"
    run = open(d + "/run.txt").read().strip().replace("\\", "/")
    man = sorted(csv.DictReader(open(d + "/manifest.csv", encoding="utf-8")), key=lambda m: float(m["game"]))
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    pg = np.array([int(p["game_ms"]) / 1000 for p in pose])
    px, pz = np.array([float(p["x"]) for p in pose]), np.array([float(p["z"]) for p in pose])
    look = np.array([float(p["yaw"]) + float(p["aim_lr"]) for p in pose])
    flag = set()
    p_inv = d + "/inventory.json"
    if os.path.exists(p_inv):
        import json
        flag = set(json.load(open(p_inv)))
    pairs = [(a, b) for a, b in zip(man[::1], man[1::1]) if a["frame"] not in flag and b["frame"] not in flag][::3]
    for a, b in pairs[:360]:
        g0, g1 = float(a["game"]), float(b["game"])
        if g1 - g0 > 0.8:
            continue
        i, j = int(np.argmin(np.abs(pg - g0))), int(np.argmin(np.abs(pg - g1)))
        if i == j:
            continue
        dt = pg[j] - pg[i]
        h = math.radians(look[i])
        dx, dz = px[j] - px[i], pz[j] - pz[i]
        along = (dx * math.sin(h) + dz * math.cos(h)) / dt
        speed = math.hypot(dx, dz) / dt
        turn = ((look[j] - look[i] + 180) % 360 - 180) / dt
        A, B = grey(os.path.join(d, a["frame"])), grey(os.path.join(d, b["frame"]))
        L, R, G = shift(A, B, BANDS["left"]), shift(A, B, BANDS["right"]), shift(A, B, BANDS["low"])
        rows.append((int(n), along, speed, turn, L[0], R[0], G[0], G[1], L[2], R[2], G[2]))
r = np.array(rows, dtype=np.float32)
np.save("C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/motion.npy", r)
along, speed, turn, lx, rx, gx, gy = r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6], r[:, 7]
spread = rx - lx
print(f"{len(r)} pairs; still (under 0.3 m/s) {(speed < 0.3).sum()}, forward (along > 0.8) {(along > 0.8).sum()}, back (along < -0.8) {(along < -0.8).sum()}; truth along: {np.percentile(along, 5):.1f} to {np.percentile(along, 95):.1f} m/s")
for name, k in (("still", speed < 0.3), ("forward", along > 0.8), ("back", along < -0.8)):
    if k.sum():
        print(f"  {name:<8} n {k.sum():>4}: ground moves y median {np.median(gy[k]):5.1f} (p10 {np.percentile(gy[k], 10):5.1f}, p90 {np.percentile(gy[k], 90):5.1f}); left-right spread median {np.median(spread[k]):5.1f}")
flat = np.abs(turn) < 5
print("not turning only:")
for name, k in (("still", (speed < 0.3) & flat), ("forward", (along > 0.8) & flat), ("back", (along < -0.8) & flat)):
    if k.sum():
        print(f"  {name:<8} n {k.sum():>4}: ground y median {np.median(gy[k]):5.1f}; share y >= 2 {100 * (gy[k] >= 2).mean():.0f}%, y <= -2 {100 * (gy[k] <= -2).mean():.0f}%")
print("correlation of speed along the view with ground y:", round(float(np.corrcoef(along, gy)[0, 1]), 3), "; with spread:", round(float(np.corrcoef(along, spread)[0, 1]), 3))

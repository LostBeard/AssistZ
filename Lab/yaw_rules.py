"""
Rules for trusting a measured view turn, offline, on both kinds of data. Real turns: lab runs with the mouse or server turning the view, frame shift and score in console.txt, true yaw in pose.csv. False turns: live runs where the character stood still. A rule must keep the real and drop the false.
Usage: yaw_rules.py <console of a still run> [...]
"""
import csv, glob, os, re, sys
import numpy as np
pat = re.compile(r"\[AssistZ\] yaw utc=\S+ ts=([\d.]+) shift=(-?[\d.]+) score=(-?[\d.]+)")
PX = 8.8   # DayZ, per degree at 960 px (GameProfile)


def load(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            rows.append((float(m[1]), float(m[2]), float(m[3])))
    return np.array(rows)


RULES = {
    "shipped: score over 0.5": lambda s, q: q > 0.5,
    "and not at the search edge": lambda s, q: (q > 0.5) & (np.abs(s) < 240),
    "and score over 0.7": lambda s, q: (q > 0.7) & (np.abs(s) < 240),
    "and score over 0.8": lambda s, q: (q > 0.8) & (np.abs(s) < 240),
    "and score over 0.9": lambda s, q: (q > 0.9) & (np.abs(s) < 240),
}
print("=== Real turns: path length of measured turn against the logged yaw, degrees")
for run in sorted(glob.glob("D:/users/tj/Projects/AssistZ/Lab/runs/yaw_*")):
    con = run + "/console.txt"
    if not os.path.exists(con):
        continue
    a = load(con)
    if len(a) < 50:
        print(os.path.basename(run), "no yaw lines", len(a))
        continue
    s, q = a[:, 1], a[:, 2]
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    yaw = np.unwrap(np.radians([float(p["yaw"]) for p in pose]))
    lr = np.array([float(p["aim_lr"]) for p in pose])
    true_path = np.abs(np.diff(np.degrees(yaw))).sum()
    move = np.abs(s) > 1
    print(f"{os.path.basename(run)}: {len(a)} frames; true yaw path {true_path:.0f} deg; frames moving {move.sum()}; |shift| p50 {np.median(np.abs(s[move])) if move.any() else 0:.0f} p99 {np.percentile(np.abs(s[move]), 99) if move.any() else 0:.0f} max {np.abs(s).max():.0f}; at the edge (240+): {(np.abs(s) >= 240).sum()}; score while moving p1 {np.percentile(q[move], 1) if move.any() else 0:.2f} p10 {np.percentile(q[move], 10) if move.any() else 0:.2f} p50 {np.median(q[move]) if move.any() else 0:.2f}")
    for name, rule in RULES.items():
        k = rule(s, q)
        print(f"    {name:<28} path {np.abs(s[k]).sum() / PX:7.0f} deg, kept {100 * np.abs(s[k & move]).sum() / max(np.abs(s[move]).sum(), 1):5.1f}% of moving px, net {s[k].sum() / PX:7.0f}")
print("\n=== False turns: the character never turned; net and path should be near 0")
for con in sys.argv[1:]:
    a = load(con)
    s, q = a[:, 1], a[:, 2]
    print(f"{con[-40:]}: {len(a)} frames")
    for name, rule in RULES.items():
        k = rule(s, q)
        print(f"    {name:<28} path {np.abs(s[k]).sum() / PX:7.1f} deg, net {s[k].sum() / PX:7.1f} deg; over 12 deg in any 10 s: {sum(abs(s[k & (a[:, 0] >= t) & (a[:, 0] < t + 10)].sum()) / PX > 12 for t in np.arange(a[0, 0], a[-1, 0], 10))}")

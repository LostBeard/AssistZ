"""The isolated-jump rule: a shift of at least N px with no movement on either side is not a turn. Also at the search edge. Same data as yaw_rules.py. Usage: yaw_rules2.py <console of a still run> [...]"""
import csv, glob, os, re, sys
import numpy as np
pat = re.compile(r"\[AssistZ\] yaw utc=\S+ ts=([\d.]+) shift=(-?[\d.]+) score=(-?[\d.]+)")
PX = 8.8


def load(path):
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            rows.append((float(m[1]), float(m[2]), float(m[3])))
    return np.array(rows)


def isolated(s, n, rest=2.0):
    prev = np.abs(np.r_[0, s[:-1]]) < rest
    nxt = np.abs(np.r_[s[1:], 0]) < rest
    return (np.abs(s) >= n) & prev & nxt


RULES = {
    "shipped: score over 0.5": lambda s, q: q > 0.5,
    "edge 240": lambda s, q: (q > 0.5) & (np.abs(s) < 240),
    "edge + isolated 20": lambda s, q: (q > 0.5) & (np.abs(s) < 240) & ~isolated(s, 20),
    "edge + isolated 40": lambda s, q: (q > 0.5) & (np.abs(s) < 240) & ~isolated(s, 40),
    "edge + isolated 80": lambda s, q: (q > 0.5) & (np.abs(s) < 240) & ~isolated(s, 80),
}
print("=== Real turns")
for run in ("yaw_20261103_035356", "yaw_20261103_040343", "yaw_20261103_041604"):
    a = load(f"D:/users/tj/Projects/AssistZ/Lab/runs/{run}/console.txt")
    s, q = a[:, 1], a[:, 2]
    move = np.abs(s) > 1
    print(f"{run}: {move.sum()} moving frames; isolated at 20 px: {isolated(s, 20).sum()}, of them values {np.round(s[isolated(s, 20)][:8]).tolist()}; scores {np.round(q[isolated(s, 20)][:8], 2).tolist()}")
    for name, rule in RULES.items():
        k = rule(s, q)
        print(f"    {name:<24} path {np.abs(s[k]).sum() / PX:6.0f} deg, kept {100 * np.abs(s[k & move]).sum() / max(np.abs(s[move]).sum(), 1):5.1f}%, net {s[k].sum() / PX:6.0f}")
print("=== False turns: never turned")
for con in sys.argv[1:]:
    a = load(con)
    s, q = a[:, 1], a[:, 2]
    for name, rule in RULES.items():
        k = rule(s, q)
        windows = [abs(s[k & (a[:, 0] >= t) & (a[:, 0] < t + 10)].sum()) / PX for t in np.arange(a[0, 0], a[-1, 0], 5)]
        print(f"    {name:<24} path {np.abs(s[k]).sum() / PX:6.1f} deg, net {s[k].sum() / PX:6.1f}; 10 s windows with 12+ deg: {sum(w >= 12 for w in windows)}; largest {max(windows):.1f}")
    k = RULES["edge + isolated 40"](s, q)
    left = np.where(k & (np.abs(s) >= 5))[0]
    print("    left over at 5 px or more: " + " ".join(f"{s[i]:.0f}/{q[i]:.2f}" for i in left[:24]))

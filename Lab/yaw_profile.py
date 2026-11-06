"""What the picture did at each big false jump: where along the width it changed, and whether overall brightness moved. Usage: yaw_profile.py <console>"""
import re, sys
import numpy as np
rows = []
for line in open(sys.argv[1], encoding="utf-8"):
    m = re.search(r"yawprofile ts=([\d.]+) shift=(-?[\d.]+) score=(-?[\d.]+) before=([\d,]+) after=([\d,]+)", line)
    if m:
        rows.append((float(m[1]), float(m[2]), float(m[3]), np.array(m[4].split(","), dtype=float), np.array(m[5].split(","), dtype=float)))
print(f"{len(rows)} big jumps")
if not rows:
    sys.exit()
t0 = rows[0][0]
print("  t(s)   shift score | mean before after | change: p50 p90 max | bins changed >5 | where (0-47) | std before after")
for ts, s, q, a, b in rows[:40]:
    d = np.abs(b - a)
    big = np.where(d > 5)[0]
    print(f"{ts - t0:7.1f} {s:7.1f} {q:5.2f} | {a.mean():6.1f} {b.mean():6.1f} | {np.median(d):5.1f} {np.percentile(d, 90):5.1f} {d.max():5.1f} | {len(big):3d} | {big.min() if len(big) else -1:>2}-{big.max() if len(big) else -1:<2} | {a.std():5.1f} {b.std():5.1f}")
d = np.array([np.abs(r[4] - r[3]) for r in rows])
print("share of jumps where each part changed by more than 5 (left to right, 8 parts):", " ".join(f"{100 * (d[:, i:i + 6] > 5).any(axis=1).mean():.0f}%" for i in range(0, 48, 6)))
print("overall brightness change, median:", round(float(np.median([abs(r[4].mean() - r[3].mean()) for r in rows])), 1))
print("profile std, before: median", round(float(np.median([r[3].std() for r in rows])), 1), "(a flat profile makes correlation easy to fool)")
print("example before:", " ".join(f"{v:.0f}" for v in rows[0][3]))
print("example after :", " ".join(f"{v:.0f}" for v in rows[0][4]))

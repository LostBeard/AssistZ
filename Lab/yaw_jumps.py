"""Every measured turn in a live run with the character standing still: what do the jumps look like, frame by frame, and what is happening then? Usage: yaw_jumps.py <run> <console>"""
import csv, re, sys
import numpy as np
run, console = sys.argv[1].replace("\\", "/"), sys.argv[2]
pat = re.compile(r"\[AssistZ\] yaw utc=(\d+):(\d+):([\d.]+) ts=([\d.]+) shift=(-?[\d.]+) score=(-?[\d.]+)")
rows = []
for line in open(console, encoding="utf-8"):
    m = pat.search(line)
    if m:
        rows.append((int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]), float(m[4]), float(m[5]), float(m[6])))
a = np.array(rows)
print(f"{len(a)} frames over {a[-1, 1] - a[0, 1]:.0f} s ({len(a) / (a[-1, 1] - a[0, 1]):.1f}/s)")
s, q = a[:, 2], a[:, 3]
print(f"shift: |shift| median {np.median(np.abs(s)):.3f} px, p99 {np.percentile(np.abs(s), 99):.2f}, max {np.abs(s).max():.1f}; score median {np.median(q):.4f}, under 0.5: {(q < 0.5).sum()}")
print(f"sum of shifts with score over 0.5: {s[q > 0.5].sum():.1f} px")
big = np.where(np.abs(s) >= 5)[0]
print(f"frames with |shift| 5 px or more: {len(big)}; sum {s[big][q[big] > 0.5].sum():.1f} px")
print("values: " + " ".join(f"{x:.1f}" for x in s[big][:40]))
print("score at those: " + " ".join(f"{x:.2f}" for x in q[big][:40]))
small = np.abs(s) < 5
print(f"the rest sums to {s[small & (q > 0.5)].sum():.1f} px")
pad = 3
last = -99
shown = 0
for i in big:
    if i - last < 10:
        continue
    last = i
    lo, hi = max(i - pad, 0), min(i + pad + 1, len(s))
    print(f"  ts {a[i, 1]:7.2f} utc+{a[i, 0] - a[0, 0]:6.1f}: " + " ".join(f"{s[k]:.1f}/{q[k]:.2f}" for k in range(lo, hi)))
    shown += 1
    if shown >= 14:
        break
gaps = np.diff(a[:, 1])
print(f"frame gaps: median {np.median(gaps) * 1000:.0f} ms, p99 {np.percentile(gaps, 99) * 1000:.0f}, max {gaps.max() * 1000:.0f}; over 500 ms: {(gaps > 0.5).sum()}")
big_gap_near = sum(np.any(np.abs(a[i, 1] - a[1:, 1][gaps > 0.3]) < 0.1) for i in big)
print(f"jumps within 100 ms of a frame gap over 300 ms: {big_gap_near}/{len(big)}")

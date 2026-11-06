"""Lost audio from the app's console, and how long a name takes, by wall clock against the event's own console line in order. Usage: gap_score.py <console> <app log>"""
import datetime, re, sys
import numpy as np
gaps, first = [], None
ev, lab = [], []
for line in open(sys.argv[1], encoding="utf-8"):
    ts = datetime.datetime.fromisoformat(line.split(" ")[0]).timestamp()
    first = first or ts
    m = re.search(r"\[AssistZ\] audiogap utc=\S+ ms=([\d.]+) total=([\d.]+) count=(\d+)", line)
    if m:
        gaps.append((ts - first, float(m[1]), float(m[2])))
    m = re.search(r"\[AssistZ\] event az=(-?[\d.]+)", line)
    if m:
        ev.append((ts, m[1]))
    m = re.search(r"\[AssistZ\] label .* az=(-?[\d.]+) level", line)
    if m:
        lab.append((ts, m[1]))
# Pair by azimuth text: by order alone, one dropped name shifted every later pair and read as an 11 s delay.
used = [False] * len(ev)
delay = []
for (t, az) in lab:
    i = next((k for k, e in enumerate(ev) if not used[k] and e[1] == az and e[0] <= t), None)
    if i is not None:
        used[i] = True
        delay.append(t - ev[i][0])
delay = np.array(delay)
n = len(delay)
total = gaps[-1][2] if gaps else 0.0
print(f"  audio lost {total:.3f} s in {len(gaps)} gaps over {(ev[-1][0] - ev[0][0]) if ev else 0:.0f} s of events; biggest {max((g[1] for g in gaps), default=0):.0f} ms; in the first 60 s {sum(g[1] for g in gaps if g[0] < 60) / 1000:.3f} s")
print("  gaps (s into run: ms): " + " ".join(f"{g[0]:.0f}:{g[1]:.0f}" for g in gaps[:24]))
if n:
    print(f"  events {len(ev)}, named {len(lab)}; event line to name: median {np.median(delay):.2f} s, p95 {np.percentile(delay, 95):.2f}, max {delay.max():.2f}")
log = open(sys.argv[2], encoding="utf-8", errors="replace").read()
for key in ("Audio lost", "Skipped"):
    m = re.search(key + r"\s+(\S+)", log)
    print(f"  on screen {key}: {m[1] if m else '?'}")

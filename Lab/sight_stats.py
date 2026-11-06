"""How the live video pass did, from the app's console: time per pass, time between passes, whether capture replayed, what it saw. Usage: sight_stats.py <console>"""
import collections, re, sys
import numpy as np
ms, since, gate, boxes, inv, seen, frames, caps, fails = [], [], [], [], 0, 0, collections.Counter(), collections.Counter(), []
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    m = re.search(r"\[AssistZ\] sight utc=\S+ frame=(\S+) share=\S+ since=(\d+) ms=([\d.]+) gate=([\d.]+) inventory=(\d) boxes=(\d+) seen=(\S*) capture=(\S*)", line)
    if m:
        frames[m[1]] += 1
        since.append(int(m[2]))
        ms.append(float(m[3]))
        gate.append(float(m[4]))
        inv += int(m[5])
        boxes.append(int(m[6]))
        seen += bool(m[7])
        caps[m[8]] += 1
    elif "sight failed" in line or "sight loaded" in line:
        fails.append(line.strip()[:200])
print(fails[:4])
if len(ms) < 8:
    print(len(ms), "passes")
    sys.exit()
a, b = np.array(ms[3:]), np.array(since[3:])
print(f"{len(ms)} passes; frame sizes {dict(frames)}; capture {dict(caps)}")
print(f"LIMIT speed: {1000 / np.median(b):.2f} passes a second by median gap; {len(ms)} passes; gap median {np.median(b):.0f} p90 {np.percentile(b, 90):.0f} max {b.max()} ms (limit 2 a second)")
print(f"pass time median {np.median(a):.0f} p90 {np.percentile(a, 90):.0f} max {a.max():.0f} ms; inventory check median {np.median(gate):.1f} ms; inventory frames {inv}; passes with a person box {sum(x > 0 for x in boxes)}; with a sighting {seen}")

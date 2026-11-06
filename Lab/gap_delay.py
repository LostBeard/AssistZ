"""Event to name, pairing each name with the earliest unused event with the same azimuth text, so a dropped name does not shift every later pair. Usage: gap_delay.py <console> [...]"""
import datetime, re, sys
import numpy as np
for path in sys.argv[1:]:
    ev, lab = [], []
    for line in open(path, encoding="utf-8"):
        ts = datetime.datetime.fromisoformat(line.split(" ")[0]).timestamp()
        m = re.search(r"\[AssistZ\] event az=(-?[\d.]+)", line)
        if m:
            ev.append([ts, m[1], False])
        m = re.search(r"\[AssistZ\] label .* az=(-?[\d.]+) level", line)
        if m:
            lab.append((ts, m[1]))
    delays, unmatched = [], 0
    for t, az in lab:
        hit = next((e for e in ev if not e[2] and e[1] == az and e[0] <= t), None)
        if hit is None:
            unmatched += 1
            continue
        hit[2] = True
        delays.append(t - hit[0])
    d = np.array(delays)
    never = [i for i, e in enumerate(ev) if not e[2]]
    print(f"{path[-18:]}: {len(ev)} events, {len(lab)} names, {unmatched} names unmatched; never named: {len(never)} (event numbers {never[:6]}); delay median {np.median(d):.2f} s, p95 {np.percentile(d, 95):.2f}, max {d.max():.2f}; first 20 median {np.median(d[:20]):.2f}, last 20 {np.median(d[-20:]):.2f}")

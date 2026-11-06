"""
Timing of an M0 lab console (run_m0.ps1 -Query lab=1 -ConsoleOut <file>): the warm-up, the first analysis batches, whether the
steady state keeps up with the stream, and the event spacing.
Usage: python analyze_m0.py <console file> [...]
"""
import re, statistics, sys

for path in sys.argv[1:]:
    ev, batches, first, warm = [], [], None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        ts, _, text = line.partition(" ")
        try:
            ts = float(ts)
        except ValueError:
            continue
        first = first if first is not None else ts
        if "[AssistZ] event az=" in text:
            ev.append(ts)
        m = re.search(r"\[AssistZ\] warmup ms=([\d.]+)", text)
        if m:
            warm = float(m[1])
        m = re.search(r"abatch ms=([\d.]+) n=(\d+)", text)
        if m:
            batches.append((ts, float(m[1]), int(m[2])))
    gaps = [b - a for a, b in zip(ev, ev[1:])]
    ms = sorted(b[1] for b in batches[3:])
    span = batches[-1][0] - batches[3][0]
    total = sum(b[2] for b in batches[4:])
    head = ", ".join(f"{b[1]:.1f} ms n={b[2]}" for b in batches[:3])
    print(f"{path}: warmup {warm} ms; first batches {head}")
    print(f"  later batches median {statistics.median(ms):.1f} ms, p95 {ms[int(len(ms) * .95)]:.1f}, max {ms[-1]:.1f}; {total / span:.2f} windows/s (nominal 46.88)")
    print(f"  {len(ev)} events, gap median {statistics.median(gaps):.3f} s, min {min(gaps):.3f}, max {max(gaps):.3f}")

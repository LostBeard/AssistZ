import sys, os, re, glob, statistics as st
out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(out, "console_*_spectral*.txt"))):
    requested = int(re.search(r"spectral(\d)", f)[1])
    for line in open(f, encoding="utf-8"):
        m = re.search(r"abatch ms=([\d.]+) n=(\d+) spectral=(\d)", line)
        if m:
            # Key by what the PAGE reports it ran, and refuse a run that did not do what was requested.
            if int(m[3]) != requested:
                sys.exit(f"{os.path.basename(f)}: requested spectral={requested} but the page ran spectral={m[3]} - A/B invalid")
            rows.setdefault((int(m[3]), int(m[2])), []).append(float(m[1]))
print("spectral  frames  batches  median ms  p95 ms  mean ms")
for (spectral, n), v in sorted(rows.items()):
    v = sorted(v)
    p95 = v[min(len(v) - 1, int(0.95 * len(v)))]
    print(f"{spectral:8d}  {n:6d}  {len(v):7d}  {st.median(v):9.3f}  {p95:6.3f}  {st.mean(v):7.3f}")

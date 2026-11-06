"""Is the live sample clock drifting smoothly (a rate) or jumping (lost audio)? Then the live score again, with the rate fitted. Usage: live_drift.py <run> <console>"""
import datetime, json, os, re, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
run, console = sys.argv[1].replace("\\", "/"), sys.argv[2]
GUN = ("gunshot", "automatic fire")
pat = re.compile(r"\[AssistZ\] label utc=(\d+):(\d+):([\d.]+) end=(\d+) label=(.+?) raw=(.+?) clap=(.+?) p=([\d.]+) ms=([\d.]+) az=(-?[\d.]+) level=(-?[\d.]+)(?: dist=([\d.]+))?")
live = []
for line in open(console, encoding="utf-8"):
    m = pat.search(line)
    if m:
        live.append(dict(wall=int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]), end=int(m[4]) / 48000.0, label=m[5], dist=float(m[12]) if m[12] else None, ms=float(m[9])))
e = np.array([x["end"] for x in live])
w = np.array([x["wall"] for x in live])
A = np.polyfit(e, w, 1)
res = w - np.polyval(A, e)
print(f"wall seconds per live sample-second: {A[0]:.5f} ({100 * (A[0] - 1):+.2f}%); live span {e.max() - e.min():.0f} s, wall span {w.max() - w.min():.0f} s; residual p5 {np.percentile(res, 5):+.2f} p95 {np.percentile(res, 95):+.2f}")
# Lower envelope by 20 s bins: the fastest label in each bin follows the clock without the queue's delay.
bins = np.arange(e.min(), e.max(), 30)
low = [(b, np.min((w - e)[(e >= b) & (e < b + 30)])) for b in bins if ((e >= b) & (e < b + 30)).any()]
print("wall minus live clock, least in each 30 s: " + " ".join(f"{v - low[0][1]:+.2f}" for b, v in low))
reports, _ = calibrated_reports(run)
rt = np.array([r[0] for r in reports])
gun = np.array([x["label"] in GUN for x in live])
best = (-1, 0, 0)
for k in np.arange(0.990, 1.0151, 0.0005):
    t0 = e[gun] * k
    coarse = np.arange(20, 60, 0.02)
    for off in coarse:
        t = t0 + off
        h = int(np.sum(np.any((t[:, None] - rt[None, :] >= -0.05) & (t[:, None] - rt[None, :] <= 0.15), axis=1)))
        if h > best[0]:
            best = (h, k, off)
print(f"best rate {best[1]:.4f} offset {best[2]:.2f}: {best[0]} gunfire events on a shot, of {gun.sum()}")
k, off = best[1], best[2]
t = e * k + off
pairs, withev, named = [], 0, 0
times = t
for r in reports:
    if not t.min() - 1 <= r[0] <= t.max() + 1:
        continue
    idx = np.where((times - r[0] >= -0.05) & (times - r[0] <= 0.15))[0]
    if len(idx):
        withev += 1
        x = live[idx[np.argmin(np.abs(times[idx] - r[0]))]]
        if x["label"] in GUN:
            named += 1
            if x["dist"] and r[3] < 2000:
                pairs.append((r[3], x["dist"], r[2]))
d = np.array([p[0] for p in pairs])
p = np.array([p[1] for p in pairs])
err = np.abs(p - d)
print(f"LIVE, rate fitted: {withev} shots with an event, {named} named gunfire; distance on {len(pairs)}: median error {np.median(err):.1f} m, within 25% {100 * (err <= 0.25 * d).mean():.1f}%, correlation {np.corrcoef(np.log2(p), np.log2(d))[0, 1]:.2f}")
for wn in sorted({q[2] for q in pairs}):
    ww = np.array([q[2] == wn for q in pairs])
    print(f"  {wn:<9} " + "  ".join(f"{c}m: {np.median(p[ww & (np.abs(d - c) <= 9)]):.0f} ({(ww & (np.abs(d - c) <= 9)).sum()})" for c in (30, 55, 80, 95) if (ww & (np.abs(d - c) <= 9)).sum() >= 3))
rep = [json.loads(x) for x in open(f"{run}/replay_live_s0.jsonl", encoding="utf-8")]
pairs = []
for r in reports:
    near = [x for x in rep if -0.05 <= x["sample"] / 48000.0 - r[0] <= 0.15]
    if near:
        x = min(near, key=lambda q: abs(q["sample"] / 48000.0 - r[0]))
        if "dist" in x and r[3] < 2000:
            pairs.append((r[3], x["dist"], r[2]))
d = np.array([q[0] for q in pairs]); p = np.array([q[1] for q in pairs]); err = np.abs(p - d)
print(f"REPLAY, same recording: distance on {len(pairs)}: median error {np.median(err):.1f} m, within 25% {100 * (err <= 0.25 * d).mean():.1f}%, correlation {np.corrcoef(np.log2(p), np.log2(d))[0, 1]:.2f}")
for wn in sorted({q[2] for q in pairs}):
    ww = np.array([q[2] == wn for q in pairs])
    print(f"  {wn:<9} " + "  ".join(f"{c}m: {np.median(p[ww & (np.abs(d - c) <= 9)]):.0f} ({(ww & (np.abs(d - c) <= 9)).sum()})" for c in (30, 55, 80, 95) if (ww & (np.abs(d - c) <= 9)).sum() >= 3))
lag = w - e * k - off
meta = json.load(open(run + "/audio.json"))
st = datetime.datetime.fromisoformat(meta["start_utc"])
start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
lag -= start_s
print(f"sound ending to name, wall clock vs recording: median {np.median(lag):.2f} s, p5 {np.percentile(lag, 5):.2f}, p95 {np.percentile(lag, 95):.2f}; first 20 {np.median(lag[:20]):.2f}, last 20 {np.median(lag[-20:]):.2f}; CLAP median {np.median([x['ms'] for x in live]):.0f} ms")

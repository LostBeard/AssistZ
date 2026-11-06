"""
A live run of the app against the same run's loopback recording and shot log. Live events have the capture's own sample clock, so the single shift that matches the most named shots is found first. Then: shots heard and named, distance error, how long naming takes after the sound, and the same recording replayed through the app (tag) for comparison.
Usage: python live_score.py <run> <console file> <replay tag>
"""
import datetime, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

run, console, tag = sys.argv[1].replace("\\", "/").rstrip("/"), sys.argv[2], sys.argv[3]
GUN = ("gunshot", "automatic fire")
reports, shift = calibrated_reports(run)
rt = np.array([r[0] for r in reports])
meta = json.load(open(run + "/audio.json"))
start = datetime.datetime.fromisoformat(meta["start_utc"])
start_s = start.hour * 3600 + start.minute * 60 + start.second + start.microsecond / 1e6
pat = re.compile(r"\[AssistZ\] label utc=(\d+):(\d+):([\d.]+) end=(\d+) label=(.+?) raw=(.+?) clap=(.+?) p=([\d.]+) ms=([\d.]+) az=(-?[\d.]+) level=(-?[\d.]+)(?: dist=([\d.]+))?")
live = []
events = 0
for line in open(console, encoding="utf-8"):
    if "[AssistZ] event" in line:
        events += 1
    m = pat.search(line)
    if m:
        live.append(dict(wall=int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]), end=int(m[4]), label=m[5], raw=m[6], clap=m[7], ms=float(m[9]), az=float(m[10]), level=float(m[11]), dist=float(m[12]) if m[12] else None))
print(f"live: {events} events, {len(live)} labelled; recording: {len(reports)} shots, clock {shift:+.3f} s")
if not live:
    sys.exit("no labels: the app heard nothing")
ends = np.array([e["end"] / 48000.0 for e in live])
walls = np.array([e["wall"] for e in live])
gun = np.array([e["label"] in GUN for e in live])
rough = float(np.median(walls - ends)) - start_s
best = (-1, 0.0)
for off in np.arange(rough - 8, rough + 8, 0.01):
    t = ends[gun] + off
    hits = sum(np.any((t - r >= -0.05) & (t - r <= 0.15)) for r in rt)
    if hits > best[0]:
        best = (hits, float(off))
off = best[1]
print(f"live clock: {off:+.2f} s against the recording (rough {rough:+.2f}); shots matched {best[0]}")


def score(rows, name):
    """rows: dict(t, label, dist)."""
    times = np.array([x["t"] for x in rows])
    pairs = []
    with_event = named = 0
    last = max(times)
    in_span = [r for r in reports if min(times) - 1 <= r[0] <= last + 1]
    for r in in_span:
        k = np.where((times - r[0] >= -0.05) & (times - r[0] <= 0.15))[0]
        if len(k):
            with_event += 1
            e = rows[k[np.argmin(np.abs(times[k] - r[0]))]]
            if e["label"] in GUN:
                named += 1
                if e["dist"] and r[3] < 2000:
                    pairs.append((r[3], e["dist"]))
    pairs = np.array(pairs)
    none = [x for x in rows if np.all(np.abs(x["t"] - rt) > 1.5)]
    print(f"{name}: {len(rows)} events; {len(in_span)} shots in span, {with_event} with an event, {named} named gunfire; no shot within 1.5 s {len(none)}, called gunfire {sum(x['label'] in GUN for x in none)}")
    if len(pairs):
        err = np.abs(pairs[:, 1] - pairs[:, 0])
        print(f"  distance on {len(pairs)}: median error {np.median(err):.1f} m, within 25% {100 * (err <= 0.25 * pairs[:, 0]).mean():.1f}%, correlation {np.corrcoef(np.log2(pairs[:, 1]), np.log2(pairs[:, 0]))[0, 1]:.2f}")
    return min(times), last


lo, hi = score([dict(t=e["end"] / 48000.0 + off, label=e["label"], dist=e["dist"]) for e in live], "LIVE")
rep = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
score([dict(t=e["sample"] / 48000.0, label=e["label"], dist=e.get("dist")) for e in rep if lo <= e["sample"] / 48000.0 <= hi], "REPLAY, same span")
lag = walls - (start_s + ends + off)
ms = np.array([e["ms"] for e in live])
print(f"sound ending to name on screen: median {np.median(lag):.2f} s, p95 {np.percentile(lag, 95):.2f}, max {lag.max():.2f}; of it CLAP {np.median(ms):.0f} ms median, {ms.max():.0f} max")
d = np.diff(lag)
print(f"lag at start {np.median(lag[:20]):.2f} s, at end {np.median(lag[-20:]):.2f} s (growth means the app is falling behind); span {hi - lo:.0f} s")

"""
CLAP's words against forced conditions (lab mode 12). For each phase in world.csv: length, level, how many detector events, and which words. The detector only passes transients to CLAP, so steady rain may give no events at all: that is reported, not hidden.
Usage: python weather_words.py <run> <tag>
"""
import csv, datetime, json, sys
import numpy as np

RUN, TAG = sys.argv[1], sys.argv[2]
m = json.load(open(RUN + "/audio.json"))
dur = m["frames"] / 48000
st = datetime.datetime.fromisoformat(m["start_utc"])
s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
pose = list(csv.DictReader(open(RUN + "/pose.csv")))
off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
clock = lambda ms: ms / 1000 + off - s
world = [(clock(int(e["game_ms"])), e["event"], float(e["distance"])) for e in csv.DictReader(open(RUN + "/world.csv"))]
env = list(csv.DictReader(open(RUN + "/env.csv")))
env_t = np.array([clock(int(e["game_ms"])) for e in env])
a = np.load(RUN + "/audio.npy").astype(np.float64)
mono = a.mean(axis=1)
ev = [json.loads(x) for x in open(f"{RUN}/replay{TAG}_s0.jsonl", encoding="utf-8")]
phases = [("before", 0.0)] + [(f"{e}{'' if d == 0 else f' {d:.0f}m'}", t) for t, e, d in world]
print(f"{len(ev)} events in {dur:.0f} s")
print(f"{'phase':<14} {'from s':>6} {'for s':>5} {'dB':>5} {'high dB':>7} {'rain':>4} {'near inf':>8} {'events':>6}  words")
for i, (name, t0) in enumerate(phases):
    t1 = phases[i + 1][1] if i + 1 < len(phases) else dur
    t0c, t1c = max(0.0, t0), min(dur, t1)
    if t1c <= t0c:
        print(f"{name:<14} outside the recording")
        continue
    # 15 s in, once rain has had time to change.
    seg = mono[int((t0c + min(15, (t1c - t0c) / 3)) * 48000): int(t1c * 48000)]
    db = 20 * np.log10(np.sqrt((seg ** 2).mean()) + 1e-9)
    spec = np.abs(np.fft.rfft(seg[: min(len(seg), 480000)])) ** 2
    f = np.fft.rfftfreq(min(len(seg), 480000), 1 / 48000)
    high = 10 * np.log10(spec[f >= 3000].sum() / spec.sum() + 1e-12)
    k = (env_t >= t0c) & (env_t < t1c)
    rain = max([float(e["rain"]) for e, on in zip(env, k) if on] or [0])
    near = [float(e["infected_nearest"]) for e, on in zip(env, k) if on and float(e["infected_nearest"]) >= 0]
    g = [e for e in ev if t0c <= e["sample"] / 48000 < t1c]
    words = {}
    for e in g:
        words[e["label"]] = words.get(e["label"], 0) + 1
    print(f"{name:<14} {t0c:>6.0f} {t1c - t0c:>5.0f} {db:>5.0f} {high:>7.1f} {rain:>4.1f} {min(near) if near else -1:>8.0f} {len(g):>6}  {dict(sorted(words.items(), key=lambda x: -x[1])[:7])}")
print("\nThe loudest events in the storm, and every 'thunder' and 'zombie groaning' word:")
storm = next(t for t, e, d in world if e == "storm")
after = next(t for t, e, d in world if e == "clear_after")
top = sorted([e for e in ev if storm <= e["sample"] / 48000 < after], key=lambda e: -e["level"])[:10]
print("  " + ", ".join(f"{e['sample'] / 48000:.0f}s {e['label']} ({e['clap'][:8]}) {e['level']:.0f}dB" for e in top))
for w in ("thunder", "zombie groaning", "rain"):
    hits = [e for e in ev if e["clap"] == w or e["label"] == w]
    print(f"  {w}: {len(hits)}: " + ", ".join(f"{e['sample'] / 48000:.0f}s p{e['p']:.2f}" for e in hits)[:200])
print("\nProbe words:", {w: sum(e["label"] == w for e in ev) for w in ("gunshot", "automatic fire", "gas strike", "heli crash", "sleigh crash")})

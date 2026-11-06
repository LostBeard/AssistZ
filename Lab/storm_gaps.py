"""
Does what the storm adds to the audio follow the game's minimum time between strikes? Modes 13 to 15 set it by phase; world.csv logs it in the distance column.
A burst is at least 3 events, each under 0.5 s after the one before: the sharp start of a strike. Its quieter follow-up (usually 2 "rain" events about 9-11 s later) is not a burst. A burst belongs to the phase it starts in.
The first 20 s of a phase are left out of the gaps (the previous setting may still hold) and shown only in the count marked "all". env.csv's nearest infected is reported, since an attacking infected makes bursts of its own.
Usage: python storm_gaps.py <run> <tag>
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
clock = lambda g: int(g) / 1000 + off - s
world = [(clock(e["game_ms"]), e["event"], float(e["distance"])) for e in csv.DictReader(open(RUN + "/world.csv"))]
env = [(clock(e["game_ms"]), float(e["infected_nearest"])) for e in csv.DictReader(open(RUN + "/env.csv"))]
ev = sorted((json.loads(x) for x in open(f"{RUN}/replay{TAG}_s0.jsonl", encoding="utf-8")), key=lambda e: e["sample"])
runs = []
for e in ev:
    t = e["sample"] / 48000
    if runs and t - runs[-1][-1]["sample"] / 48000 < 0.5:
        runs[-1].append(e)
    else:
        runs.append([e])
bursts = [r for r in runs if len(r) >= 3]
print(f"{len(ev)} events, {len(bursts)} bursts, {dur:.0f} s")
print(f"{'phase':<12} {'setting':>7} {'from s':>6} {'for s':>5} {'all':>4} {'bursts':>6} {'min gap':>7} {'med gap':>7} {'infected':>8}  gaps")
for i, (t0, name, setting) in enumerate(world):
    t1 = min(world[i + 1][0] if i + 1 < len(world) else dur, dur)
    every = [r for r in bursts if t0 <= r[0]["sample"] / 48000 < t1]
    c = [r for r in every if r[0]["sample"] / 48000 >= t0 + 20]
    gaps = np.diff([r[0]["sample"] / 48000 for r in c])
    lo = f"{gaps.min():7.1f}" if len(gaps) else f"{'-':>7}"
    med = f"{np.median(gaps):7.1f}" if len(gaps) else f"{'-':>7}"
    near = [d for t, d in env if t0 <= t < t1 and d >= 0]
    print(f"{name:<12} {setting:>7.0f} {t0:>6.0f} {t1 - t0:>5.0f} {len(every):>4} {len(c):>6} {lo} {med} {min(near) if near else -1:>8.1f}  {[round(float(g), 1) for g in gaps][:20]}")

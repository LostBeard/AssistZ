"""Is 'infected attack' hearing agroed, or just close? Same distance bands, calm against agroed. And can a calm infected be heard at all: events per minute by what was out, by world.csv's phases and env.csv. Usage: infected_confound.py <tag> <run> [...]"""
import collections, csv, datetime, json, os, sys
import numpy as np
TAG = sys.argv[1]
rows, spans = [], []
for run in [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]:
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    env = list(csv.DictReader(open(run + "/env.csv")))
    et = np.array([int(e["game_ms"]) / 1000 + off - s for e in env])
    near = np.array([float(e["infected_nearest"]) for e in env])
    state = np.array([int(e["infected_state"]) for e in env])
    near[near < 0] = 1e9
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    name = os.path.basename(run)[-6:]
    for e in ev:
        t = e["sample"] / 48000.0
        i = int(np.argmin(np.abs(et - t)))
        rows.append(dict(run=name, t=t, state=int(state[i]), near=float(near[i]), label=e["label"], clap=e["clap"], level=e["level"]))
    world = [(int(w["game_ms"]) / 1000 + off - s, w["event"], float(w["distance"])) for w in csv.DictReader(open(run + "/world.csv"))]
    world = [w for w in world if w[1] in ("infected", "rest")]
    for a, b in zip(world, world[1:] + [(world[-1][0] + 60, "end", 0)]):
        spans.append(dict(run=name, kind=a[1], dist=a[2], lo=a[0], hi=b[0]))
AGRO = (1, 2, 3, 4)
print("Named 'infected attack', by true state and distance band (calm | agitated: disturbed, alerted, chase, fight):")
for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 30), (30, 60)):
    c = [r for r in rows if lo <= r["near"] < hi and r["state"] == 0]
    a = [r for r in rows if lo <= r["near"] < hi and r["state"] in AGRO]
    print(f"  {lo:>2}-{hi:<2} m: calm {sum(r['label'] == 'infected attack' for r in c)}/{len(c)} | agitated {sum(r['label'] == 'infected attack' for r in a)}/{len(a)};  calm level {np.median([r['level'] for r in c]) if c else 0:.1f} dB, agitated {np.median([r['level'] for r in a]) if a else 0:.1f} dB")
print("By state alone:")
for k, n in ((1, "disturbed"), (2, "alerted"), (3, "chase"), (4, "fight")):
    g = [r for r in rows if r["state"] == k and r["near"] <= 60]
    print(f"  {n:<10} {sum(r['label'] == 'infected attack' for r in g)}/{len(g)}, median distance {np.median([r['near'] for r in g]) if g else 0:.1f} m")
print("\nEvents per minute by what was out (only time when the nearest infected stayed calm counts for calm rates):")
for kind, d in (("rest", 0), ("infected", 4), ("infected", 10), ("infected", 20), ("infected", 40)):
    sp = [x for x in spans if x["kind"] == kind and x["dist"] == d]
    minutes = sum(x["hi"] - x["lo"] for x in sp) / 60
    in_sp = [r for r in rows if any(x["run"] == r["run"] and x["lo"] <= r["t"] < x["hi"] for x in sp)]
    calm = [r for r in in_sp if r["state"] == 0 or r["near"] > 60]
    print(f"  {kind:<9} {d:>2.0f} m: {minutes:5.1f} min, {len(in_sp) / max(minutes, 0.01):5.1f} events/min; of them with the infected calm {len(calm)}; words {dict(collections.Counter(r['label'] for r in calm).most_common(5))}")

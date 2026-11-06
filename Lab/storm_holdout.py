"""
The thunder class's gate on storm runs it never trained on, from the app's own replay output. Storm phase events: share named thunder, share named a world event, gunfire. Other phases: what else got called thunder. Registered gates: at least 80% thunder, at most 5% gas strike, heli crash or sleigh crash, 0 gunfire.
Usage: python storm_holdout.py <tag> <run> [...]
"""
import csv, datetime, json, sys
import numpy as np

TAG, RUNS = sys.argv[1], sys.argv[2:]
WORLD = ("gas strike", "heli crash", "sleigh crash")
GUN = ("gunshot", "automatic fire")
total = {"n": 0, "thunder": 0, "world": 0, "gun": 0}
for run in RUNS:
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    world = [(int(e["game_ms"]) / 1000 + off - s, e["event"], float(e["distance"])) for e in csv.DictReader(open(run + "/world.csv"))]
    env = [(int(e["game_ms"]) / 1000 + off - s, float(e["infected_nearest"])) for e in csv.DictReader(open(run + "/env.csv"))]
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    print(f"{run[-15:]}: {len(ev)} events")
    for i, (t0, name, setting) in enumerate(world):
        t1 = world[i + 1][0] if i + 1 < len(world) else 1e9
        g = [e for e in ev if t0 <= e["sample"] / 48000 < t1]
        words = {}
        for e in g:
            words[e["label"]] = words.get(e["label"], 0) + 1
        near = [d for t, d in env if t0 <= t < t1 and d >= 0]
        print(f"  {name:<8} {setting:>3.0f}: {len(g):3d} events, nearest infected {min(near) if near else -1:5.1f} m: {dict(sorted(words.items(), key=lambda x: -x[1])[:7])}")
        if name == "storm":
            total["n"] += len(g)
            total["thunder"] += words.get("thunder", 0)
            total["world"] += sum(words.get(w, 0) for w in WORLD)
            total["gun"] += sum(words.get(w, 0) for w in GUN)
n = max(total["n"], 1)
print(f"STORM PHASES: {total['n']} events; thunder {total['thunder']} ({100 * total['thunder'] / n:.1f}%, need 80), gas/heli/sleigh {total['world']} ({100 * total['world'] / n:.1f}%, at most 5), gunfire {total['gun']} (need 0)")

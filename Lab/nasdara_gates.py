"""
The shipped probe on Nasdara, a map it has never heard. Limits registered at 0c95df1; a measurement, not a ship gate. Every word in every phase is reported.
Usage: python nasdara_gates.py <tag> <run> [...]
"""
import csv, datetime, json, sys
import numpy as np

TAG, RUNS = sys.argv[1], sys.argv[2:]
GUN = ("gunshot", "automatic fire")
FALSE = ("gas strike", "heli crash", "thunder", "infected attack")


def words(g):
    w = {}
    for e in g:
        w[e["label"]] = w.get(e["label"], 0) + 1
    return dict(sorted(w.items(), key=lambda x: -x[1])[:9])


def rate(g, names):
    return sum(e["label"] in names for e in g)


attacks, storms, others = [], [], []
for run in RUNS:
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    world = [(int(e["game_ms"]) / 1000 + off - s, e["event"]) for e in csv.DictReader(open(run + "/world.csv"))]
    env = [(int(e["game_ms"]) / 1000 + off - s, float(e["infected_nearest"])) for e in csv.DictReader(open(run + "/env.csv"))]
    et, near = np.array([x[0] for x in env]), np.array([x[1] for x in env])
    ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    by = {}
    for e in ev:
        t = e["sample"] / 48000.0
        d = near[np.argmin(np.abs(et - t))]
        phase, since = "before", 0
        for i, (when, name) in enumerate(world):
            end = world[i + 1][0] if i + 1 < len(world) else float("inf")
            if when <= t < end:
                phase, since = name, t - when
        by.setdefault(phase, []).append(e)
        if phase == "attack":
            if since >= 3 and 0 <= d < 3:
                attacks.append(e)
        elif phase == "storm":
            storms.append(e)
        elif (phase == "rest" and since < 3) or (phase == "clear_after" and since < 20):
            continue
        else:
            others.append(e)
    print(f"{run[-15:]}: {len(ev)} events")
    for k, g in by.items():
        print(f"  {k:<12} {len(g):4d} {words(g)}")
a, st_, o = max(len(attacks), 1), max(len(storms), 1), max(len(others), 1)
print(f"{'PASS' if rate(attacks, ('infected attack',)) / a >= 0.7 and rate(attacks, ('thunder',)) / a <= 0.05 and rate(attacks, ('gas strike', 'heli crash')) / a <= 0.05 and rate(attacks, GUN) == 0 else 'FAIL'} attacks: {len(attacks)} scored; infected attack {rate(attacks, ('infected attack',))} ({100 * rate(attacks, ('infected attack',)) / a:.1f}%, need 70); thunder {rate(attacks, ('thunder',))}; gas or heli {rate(attacks, ('gas strike', 'heli crash'))}; gunfire {rate(attacks, GUN)}; {words(attacks)}")
print(f"{'PASS' if rate(storms, ('thunder',)) / st_ >= 0.8 and rate(storms, ('gas strike', 'heli crash')) / st_ <= 0.05 and rate(storms, ('infected attack',)) / st_ <= 0.05 and rate(storms, GUN) == 0 else 'FAIL'} storms: {len(storms)}; thunder {rate(storms, ('thunder',))} ({100 * rate(storms, ('thunder',)) / st_:.1f}%, need 80); gas or heli {rate(storms, ('gas strike', 'heli crash'))}; infected attack {rate(storms, ('infected attack',))}; gunfire {rate(storms, GUN)}; {words(storms)}")
print(f"{'PASS' if rate(others, FALSE) / o <= 0.05 and rate(others, GUN) == 0 else 'FAIL'} other phases: {len(others)}; event words {rate(others, FALSE)} ({100 * rate(others, FALSE) / o:.1f}%, at most 5); gunfire {rate(others, GUN)}; {words(others)}")

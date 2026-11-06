"""Can a calm infected be heard? Mode 31, using the worst state of any infected in range. Event rates and words: truly calm time with infected out, against rest time with none. And the agroed word by worst state. Usage: calm_check.py <tag> <run> [...]"""
import collections, csv, datetime, json, os, sys
import numpy as np
TAG = sys.argv[1]
tot = collections.defaultdict(lambda: dict(seconds=0.0, events=0, words=collections.Counter(), level=[]))
for run in [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]:
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    env = list(csv.DictReader(open(run + "/env.csv")))
    et = np.array([int(e["game_ms"]) / 1000 + off - s for e in env])
    near = np.array([float(e["infected_nearest"]) for e in env])
    near[near < 0] = 1e9
    worst = np.array([int(e["infected_worst"]) for e in env])
    world = [(int(w["game_ms"]) / 1000 + off - s, w["event"]) for w in csv.DictReader(open(run + "/world.csv"))]
    world = [w for w in world if w[1] in ("infected", "rest")]
    out = np.zeros(len(et), dtype=bool)
    for (a, ev), (b, _) in zip(world, world[1:] + [(world[-1][0] + 60, "")]):
        if ev == "infected":
            out |= (et >= a + 3) & (et < b)
    rest = np.zeros(len(et), dtype=bool)
    for (a, ev), (b, _) in zip(world, world[1:] + [(world[-1][0] + 20, "")]):
        if ev == "rest":
            rest |= (et >= a + 3) & (et < b)

    def bucket(i):
        if out[i] and worst[i] == 0:
            return "calm, in range"
        if out[i] and worst[i] > 0:
            return "agitated"
        if rest[i] and near[i] > 100:
            return "rest, none within 100 m"
        return None

    name = os.path.basename(run)[-6:]
    dt = np.median(np.diff(et))
    per = collections.Counter()
    for i in range(len(et)):
        b = bucket(i)
        if b:
            tot[(name, b)]["seconds"] += dt
            tot[("all", b)]["seconds"] += dt
    for line in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        i = int(np.argmin(np.abs(et - e["sample"] / 48000.0)))
        b = bucket(i)
        if b:
            for k in ((name, b), ("all", b)):
                tot[k]["events"] += 1
                tot[k]["words"][e["label"]] += 1
                tot[k]["level"].append(e["level"])
for k in sorted(tot, key=lambda x: (x[0] == "all", x[0], x[1])):
    v = tot[k]
    mins = v["seconds"] / 60
    w = v["words"]
    print(f"{k[0]:<7} {k[1]:<25} {mins:6.1f} min {v['events']:5d} events {v['events'] / max(mins, 0.01):6.1f}/min, level {np.median(v['level']) if v['level'] else 0:6.1f} dB; agroed word {w['infected attack']} ({100 * w['infected attack'] / max(v['events'], 1):.0f}%), groaning {w['zombie groaning']}; {dict(w.most_common(4))}")

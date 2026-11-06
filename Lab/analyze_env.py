"""
CLAP's words against the game's logged conditions, and the embedding probe on recordings it never trained on. env.csv: once a second, weather state and living infected within 200 m.
The script API has no lightning event, so thunder can only be compared with conditions.
Usage: python analyze_env.py <tag> <new run> [...] -- <train run> [...]
"""
import collections, csv, datetime, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

TAG = sys.argv[1]
split = sys.argv.index("--")
NEW, TRAIN = sys.argv[2:split], sys.argv[split + 1:]
GUN = ("gunshot", "automatic fire")


def train_rows(run):
    reports, _ = calibrated_reports(run)
    out = []
    for line in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        if any(-0.05 <= t - r[0] <= 0.15 for r in reports):
            out.append((np.array(e["embed"]), 1.0))
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            out.append((np.array(e["embed"]), 0.0))
    return out


def fit(rows, l2=1.0, steps=800, lr=0.5):
    X = np.stack([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    a, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ a + b)))
        a -= lr * (X.T @ (w * (p - y)) + l2 / len(y) * a)
        b -= lr * np.sum(w * (p - y))
    return lambda x: float(x @ a + b)


f = fit([r for run in TRAIN for r in train_rows(run)])
words = collections.Counter()
for run in NEW:
    name = os.path.basename(run.rstrip("/"))
    meta = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    offset = np.median([sum(int(v) * k for v, k in zip(r["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(r["game_ms"]) / 1000 for r in pose])
    env = list(csv.DictReader(open(run + "/env.csv")))
    env_t = np.array([int(r["game_ms"]) / 1000 + offset - start for r in env])
    reports, shift = calibrated_reports(run)
    ai = [(int(x["game_ms"]) / 1000 + offset - start + float(x["dist"]) / 330.0, float(x["speed"]), float(x["dist"])) for x in csv.DictReader(open(run + "/aipose.csv")) if x["ai"] != "decoy"]
    events = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    print(f"\n{name}: {meta['frames'] / 48000:.0f} s, {len(events)} events, {len(reports)} shots; env rows {len(env)}")
    print("  overcast {:.2f}-{:.2f}, rain {:.2f}-{:.2f}, wind {:.1f}-{:.1f}; infected within 200 m {}-{}, nearest ever {:.0f} m; vehicles {}".format(
        min(float(r["overcast"]) for r in env), max(float(r["overcast"]) for r in env), min(float(r["rain"]) for r in env), max(float(r["rain"]) for r in env),
        min(float(r["wind"]) for r in env), max(float(r["wind"]) for r in env), min(int(r["infected"]) for r in env), max(int(r["infected"]) for r in env),
        min([float(r["infected_nearest"]) for r in env if float(r["infected_nearest"]) >= 0] or [-1]), max(int(r["vehicles"]) for r in env)))
    table = collections.defaultdict(list)
    g, gc, o, oc, walk, walkc = 0, 0, 0, 0, 0, 0
    pg, po = 0, 0
    for e in events:
        t = e["sample"] / 48000.0
        k = int(np.argmin(np.abs(env_t - t)))
        ev = env[k]
        near = [r for r in reports if -0.05 <= t - r[0] <= 0.15]
        clear = all(abs(t - r[0]) > 1.5 for r in reports)
        m = f(np.array(e["embed"]))
        if near:
            g += 1
            gc += m > 0
            pg += e["label"] in GUN
        elif clear:
            o += 1
            oc += m > 0
            po += e["label"] in GUN
            b = min(ai, key=lambda x: abs(x[0] - t)) if ai else None
            walking = b is not None and abs(b[0] - t) < 1.0 and b[1] > 0.5 and b[2] < 35
            if walking:
                walk += 1
                walkc += m > 0
            else:
                table[e["clap"]].append((t, float(ev["rain"]), int(ev["infected"]), float(ev["infected_nearest"]), e["p"], m, e["level"]))
            words[e["clap"]] += 1
    print(f"  probe: gunfire called gunfire {gc}/{g} (prompts {pg}/{g}); no shot called gunfire {oc}/{o} (prompts {po}/{o}); AI walking called gunfire {walkc}/{walk}")
    print("  with no shot near and the AI not walking near, CLAP's words: word, n, p median, level median, rain max, infected nearest (min over those events; -1 = none within 200 m):")
    for w, rows in sorted(table.items(), key=lambda x: -len(x[1])):
        near = [r[3] for r in rows if r[3] >= 0]
        print(f"    {w:<18} {len(rows):3d}  p {np.median([r[4] for r in rows]):.2f}  level {np.median([r[6] for r in rows]):6.1f} dB  rain {max(r[1] for r in rows):.2f}  infected nearest {min(near) if near else -1:6.0f} m")
print("\nAll new recordings, every word on events with no shot near:", dict(words.most_common()))

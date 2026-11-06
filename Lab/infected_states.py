"""
First look: what do infected sound like by mind state? Each event gets the state and distance of the nearest infected at that moment, from env.csv, on the recording's clock (from audio.json and pose.csv: this mode has no shots to calibrate on, so good to a second or so). Then: can a held-out run's states be told apart from the embedding?
Usage: python infected_states.py <tag> <run> [...]
"""
import collections, csv, datetime, json, os, sys
import numpy as np
NAMES = {0: "calm", 1: "disturbed", 2: "alerted", 3: "chase", 4: "fight", 5: "cower", -1: "none"}
TAG = sys.argv[1]
rows = []
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
    print(f"{os.path.basename(run)}: env {len(env)} samples; state share of time: " + " ".join(f"{NAMES[k]} {100 * (state == k).mean():.0f}%" for k in sorted(set(state.tolist()))))
    for line in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        i = int(np.argmin(np.abs(et - t)))
        rows.append(dict(run=os.path.basename(run)[-6:], t=t, state=int(state[i]), near=float(near[i]), label=e["label"], clap=e["clap"], level=e["level"], x=e.get("embed")))
print(f"\n{len(rows)} events. By state of the nearest infected (within 60 m, else 'far or none'):")
group = lambda r: NAMES[r["state"]] if 0 <= r["near"] <= 60 else "far or none"
for g in ("calm", "disturbed", "alerted", "chase", "fight", "cower", "far or none"):
    rr = [r for r in rows if group(r) == g]
    if rr:
        print(f"  {g:<12} {len(rr):5d} events, median level {np.median([r['level'] for r in rr]):6.1f} dB, median distance {np.median([r['near'] for r in rr]):5.1f} m; named {dict(collections.Counter(r['label'] for r in rr).most_common(5))}; CLAP {dict(collections.Counter(r['clap'] for r in rr).most_common(3))}")
if len({r["run"] for r in rows}) < 2:
    sys.exit(0)
print("\nCan the embedding tell states apart? Each run held out; a class-balanced logistic on the standardised embedding.")


def logistic(X, y):
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    Z = (X - mean) / std
    weight = np.where(y, 0.5 / max(y.sum(), 1), 0.5 / max((~y).sum(), 1))
    a, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(800):
        p = 1 / (1 + np.exp(-np.clip(Z @ a + b, -30, 30)))
        r = (p - y) * weight
        a -= 0.5 * (Z.T @ r + a / len(Z))
        b -= 0.5 * r.sum()
    return lambda q: ((np.asarray(q, dtype=np.float64) - mean) / std) @ a + b


def test(name, positive, negative):
    pos = [r for r in rows if r["x"] and positive(r)]
    neg = [r for r in rows if r["x"] and negative(r)]
    tp = fn = tn = fp = 0
    per = []
    for run in sorted({r["run"] for r in rows}):
        train_p, train_n = [r for r in pos if r["run"] != run], [r for r in neg if r["run"] != run]
        test_p, test_n = [r for r in pos if r["run"] == run], [r for r in neg if r["run"] == run]
        if not train_p or not train_n or not (test_p or test_n):
            continue
        f = logistic(np.array([r["x"] for r in train_p + train_n], dtype=np.float64), np.array([True] * len(train_p) + [False] * len(train_n)))
        a = sum(f(r["x"]) > 0 for r in test_p); b = sum(f(r["x"]) <= 0 for r in test_n)
        tp += a; fn += len(test_p) - a; tn += b; fp += len(test_n) - b
        per.append(f"{run}: {a}/{len(test_p)} {b}/{len(test_n)}")
    print(f"  {name}: {len(pos)} against {len(neg)}; right {tp}/{tp + fn} and {tn}/{tn + fp}; {' | '.join(per)}")


test("calm vs agroed (chase, fight)", lambda r: r["near"] <= 60 and r["state"] == 0, lambda r: r["near"] <= 60 and r["state"] in (3, 4))
test("chase vs fight", lambda r: r["near"] <= 60 and r["state"] == 3, lambda r: r["near"] <= 60 and r["state"] == 4)
test("any infected near vs far or none", lambda r: 0 <= r["near"] <= 30, lambda r: r["near"] > 100 or r["near"] < 0)

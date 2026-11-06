"""Shipped five-class probe against round 7's candidate, same clips, weights only with no burst vote, on every condition never trained on. For TJ's decision; not a gate."""
import csv, datetime, json, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

R = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261104_"
S5 = json.load(open(sys.argv[1]))
S7 = json.load(open(sys.argv[2]))


def pred(p):
    A, B, C = np.array(p["Weights"], dtype=np.float64), np.array(p["Bias"], dtype=np.float64), p["Classes"]
    return lambda e: C[int(np.argmax(A @ np.array(e["embed"], dtype=np.float64) + B))]


f5, f7 = pred(S5), pred(S7)


def load(key):
    return [json.loads(x) for x in open(f"{R}{key}/replay_r7_s0.jsonl", encoding="utf-8")]


def phases(key):
    run = R + key
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    return [(int(e["game_ms"]) / 1000 + off - s, e["event"]) for e in csv.DictReader(open(run + "/world.csv"))]


def phase_of(world, t):
    now = "before"
    for when, name in world:
        if when <= t:
            now = name
    return now


def row(title, g, want=None):
    n = max(len(g), 1)
    def line(f):
        c = {}
        for e in g:
            k = f(e)
            c[k] = c.get(k, 0) + 1
        gun = c.get("gunfire", 0)
        world = sum(v for k, v in c.items() if k in ("gas_strike", "heli_crash", "sleigh_crash"))
        wanted = c.get(want, 0) if want else 0
        return f"{(str(wanted) + '/' + str(len(g)) + ' ' + want) if want else str(len(g)) + ' events'}; gunfire {gun}; gas, heli or sleigh {world}"
    print(f"{title:<38} shipped: {line(f5):<58} candidate: {line(f7)}")


for key, title in (("215004", "Real gunfire, rain storm"), ("215827", "Real gunfire, dry storm"), ("220700", "Real gunfire, rain, no storm")):
    reports, _ = calibrated_reports(R + key)
    ev = load(key)
    row(title, [e for e in ev if any(-0.05 <= e["sample"] / 48000 - r[0] <= 0.15 for r in reports)], "gunfire")
    row(title.replace("Real gunfire", "No shot near"), [e for e in ev if all(abs(e["sample"] / 48000 - r[0]) > 1.5 for r in reports)])
for key, title in (("221544", "Dry storm, Chernarus"), ("223129", "Dry storm, Nasdara"), ("224711", "Storm, Nasdara rain 0.3"), ("110232", "Rain storm, Chernarus")):
    w = phases(key)
    row(title, [e for e in load(key) if phase_of(w, e["sample"] / 48000) == "storm"], "thunder")
for key, title in (("224711", "Attacks, Nasdara"), ("144529", "Attacks, rain"), ("145757", "Attacks, rain"), ("152249", "Attacks, clear"), ("121745", "Attacks, clear")):
    w = phases(key)
    row(title, [e for e in load(key) if phase_of(w, e["sample"] / 48000) == "attack"], "infected_attack")
for key, title in (("121745", "Crow site, rest phases"), ("133615", "Survey, sites never heard"), ("160723", "Survey, sites never heard"), ("162444", "Survey, sites never heard")):
    w = phases(key)
    row(title, [e for e in load(key) if phase_of(w, e["sample"] / 48000) in ("rest", "clear", "clear_after", "site")])
for key, title in (("153510", "Five recordings, no shot"),):
    pass
GEMS = ("20261103_153510", "20261104_010112", "20261104_024720", "20261104_021109", "20261104_021644")
near, far = [], []
for name in GEMS:
    run = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_" + name
    reports, _ = calibrated_reports(run)
    ev = [json.loads(x) for x in open(run + "/replay_r7_s0.jsonl", encoding="utf-8")]
    near += [e for e in ev if any(-0.05 <= e["sample"] / 48000 - r[0] <= 0.15 for r in reports)]
    far += [e for e in ev if all(abs(e["sample"] / 48000 - r[0]) > 1.5 for r in reports)]
row("Five recordings, near a shot", near, "gunfire")
row("Five recordings, no shot", far)

"""
Can a probe name world events? Five classes: other, gunfire, gas_strike, heli_crash, sleigh_crash; the multinomial form of ClapProbe.FitLogistic (800 steps, lr 0.5, L2 1 / N, class-balanced). Labels from the game: a crash's events fall within 22 s of its trigger
in world.csv (the sound runs about 17 s, plus travel); a gas strike's within -2..+12 s of its Artillery_Close match, searched 150-240 s after its trigger, because a strike sounds minutes later. Each world run is held out in turn. The gunfire recordings
are held out as before. Samples are read locally from the player's install, never committed.
Usage: python world_classes.py <samples dir>
"""
import csv, datetime, glob, json, multiprocessing as mp, os, sys, time
import numpy as np
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from world_events import SOS, RATE, build
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

U = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
GATE = ["20261103_160950", "20261103_164111", "20261103_190530", "20261103_200816", "20261103_231331", "20261103_232232", "20261103_233343", "20261104_001133", "20261104_002001", "20261104_002833"]
WORLD = ["20261104_053416", "20261104_055325", "20261104_061239"]
HOLD = ["20261103_153510", "20261104_010112", "20261104_024720", "20261104_021109", "20261104_021644"]
STORM = ["20261104_081350", "20261104_084227", "20261104_174018", "20261104_175558", "20261104_181149", "20261104_195113", "20261104_200652"]   # thunder training, pre-registered; from 174018 on, dry storms (mode 24). 085810 (mode 14), 110232 (mode 15), 221544 and 223129 (mode 24) are never trained on.
ATTACK = ["20261104_120527", "20261104_130955", "20261104_170612", "20261104_171851", "20261104_182730", "20261104_202233"]   # modes 16 (clear) and 18 (rain) on Chernarus, 17 and 21 on Nasdara, infected attacks plus a storm, training. Everything in phase_gates7.py is never trained on.
GUNSTORM = ["20261104_193426", "20261104_194250", "20261104_214137"]   # labelled gunfire under weather: rain storm, dry storm, rain with no storm. 215004, 215827, 220700 are never trained on.
FAR = ["20261105_055044", "20261105_060740", "20261105_062337", "20261105_063920"]   # mode 29, the lab's own shooter at 150-800 m: clear, clear, heavy rain, storm. Tag _far.
SURVEY = ["20261104_132217"]   # mode 19, 12 sites, training. Mode 20 (133615) is never trained on. Natural strikes come from survey_strikes.py.
# No sleigh_crash class: StaticSantaCrash is active only in the Christmas missions (0 in dayzOffline), and 49 examples fired falsely. Real crash rows are dropped from training, not called other.
NEWGUNS = ["20261105_194158", "20261105_195820", "20261105_201808", "20261105_203409", "20261105_205010", "20261105_210617"]   # mode 33, six more guns, tag _w; 205010 heavy rain.
OWN = ["20261105_134028", "20261105_135100", "20261105_135931", "20261105_140809", "20261105_141634"]   # mode 32, the character alone, driven: clear, clear, clear, heavy rain, storm. Tag _own. Rows from own_cv.own_rows.
CLASSES = ["other", "gunfire", "gas_strike", "heli_crash", "thunder", "infected_attack", "own_move", "own_shot"]
_T = {}


def init(out):
    _T["t"] = [x for x in build(out) if x[0] == "artillery_close"]


def triggers(name):
    r = U + name
    if not os.path.exists(r + "/world.csv"):
        return []   # no world events and no weather: a clear lab run
    m = json.load(open(r + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(r + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    return [(int(e["game_ms"]) / 1000 + off - s, e["event"], float(e["distance"])) for e in csv.DictReader(open(r + "/world.csv"))]


def strike(job):
    name, t, dist = job
    a = np.load(U + name + "/audio.npy")
    lo, hi = int((t + 150) * 48000), min(len(a), int((t + 240) * 48000))
    seg = a[lo:hi].astype(np.float64)
    best = (0.0, 0.0, "")
    for c in range(seg.shape[1]):
        s = resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32)
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for group, base, pitch, y, rev in _T["t"]:
            if len(y) >= len(s):
                continue
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / e
            ncc[e < 0.0005] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > best[0]:
                best = (float(ncc[k]), lo / 48000 + k / RATE, base)
    return name, t, dist, best


def world_rows(name, strikes):
    trig = triggers(name)
    out = []
    for line in open(f"{U}{name}/replay_app_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        label, event = "other", ""
        for when, ev, dist in trig:
            if ev in ("heli", "sleigh") and when <= t <= when + 22:
                label = ev + "_crash"
                event = f"{ev}{dist:.0f}"
        for (run, tt, dist), (ncc, at, base) in strikes.items():
            if run == name and ncc >= 0.4 and at - 2 <= t <= at + 12:
                label = "gas_strike"
                event = f"gas{dist:.0f}"
        if label not in CLASSES:
            continue
        out.append(dict(x=np.array(e["embed"]), y=label, clap=e["clap"], t=t, event=event, run=name))
    return out


def storm_rows(name):
    """Pre-registered: every event in a storm phase is thunder; the 20 s after a storm is switched off are left out; everything else is other."""
    trig = triggers(name)
    out = []
    spans = []
    for i, (when, ev, dist) in enumerate(trig):
        if ev == "storm":
            spans.append((when, trig[i + 1][0] if i + 1 < len(trig) else float("inf")))
    for line in open(f"{U}{name}/replay_vote_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        label = "other"
        if any(a <= t < b for a, b in spans):
            label = "thunder"
        elif any(b <= t < b + 20 for a, b in spans):
            continue
        out.append(dict(x=np.array(e["embed"]), y=label, clap=e["clap"], t=t, event="thunder" if label == "thunder" else "", run=name))
    return out


def attack_rows(name):
    """Pre-registered at cb5cf01: infected_attack from 3 s after a spawn until its removal, while the nearest infected is under 3 m; the 3 s after a spawn or removal are left out; storm as in storm_rows, 20 s left out after; everything else other."""
    r = U + name
    trig = triggers(name)
    m = json.load(open(r + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(r + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    env = [(int(e["game_ms"]) / 1000 + off - s, float(e["infected_nearest"])) for e in csv.DictReader(open(r + "/env.csv"))]
    et = np.array([x[0] for x in env])
    near = np.array([x[1] for x in env])
    out = []
    for line in open(f"{r}/replay_vote_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        d = near[np.argmin(np.abs(et - t))]
        close = 0 <= d < 3
        label = "other"
        skip = False
        for i, (when, ev, dist) in enumerate(trig):
            end = trig[i + 1][0] if i + 1 < len(trig) else float("inf")
            if not when <= t < end:
                continue
            if ev == "attack":
                if t < when + 3:
                    skip = True
                elif close:
                    label = "infected_attack"
            elif ev == "rest":
                skip = t < when + 3
            elif ev == "storm":
                label = "thunder"
            elif ev == "clear_after":
                skip = t < when + 20
        if skip:
            continue
        out.append(dict(x=np.array(e["embed"]), y=label, clap=e["clap"], t=t, event=label if label != "other" else "", run=name))
    return out


def survey_rows(name):
    """Pre-registered at f8d04fa: other, leaving out 8 s after each move, anything with an infected under 3 m, and anything from -2 to +12 s around an Artillery_Close match of 0.4 or more."""
    r = U + name
    trig = [t for t in triggers(name) if t[1] == "site"]
    m = json.load(open(r + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(r + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    env = [(int(e["game_ms"]) / 1000 + off - s, float(e["infected_nearest"])) for e in csv.DictReader(open(r + "/env.csv"))]
    et = np.array([x[0] for x in env])
    near = np.array([x[1] for x in env])
    strikes = json.load(open(r + "/strikes.json"))
    out = []
    for line in open(f"{r}/replay_vote_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        if t < trig[0][0] or any(when <= t < when + 8 for when, ev, dist in trig):
            continue
        d = near[np.argmin(np.abs(et - t))]
        if 0 <= d < 3 or any(at - 2 <= t <= at + 12 for ncc, at in strikes):
            continue
        out.append(dict(x=np.array(e["embed"]), y="other", clap=e["clap"], t=t, event="", run=name))
    return out


def gunstorm_rows(name, tag="_vote"):
    """Pre-registered at round 7: near a logged shot (-50..+150 ms of its report) gunfire; over 1.5 s from any shot, thunder once a storm is on and other before that or with no storm; in between left out."""
    reports, _ = calibrated_reports(U + name)
    trig = triggers(name)
    storm = [when for when, ev, dist in trig if ev == "storm"]
    out = []
    for line in open(f"{U}{name}/replay{tag}_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        if any(-0.05 <= t - r[0] <= 0.15 for r in reports):
            label = "gunfire"
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            label = "thunder" if storm and t >= storm[0] + 5 else "other"
        else:
            continue
        out.append(dict(x=np.array(e["embed"]), y=label, clap=e["clap"], t=t, event="", run=name))
    return out


def gun_rows(name, tag):
    reports, _ = calibrated_reports(U + name)
    out = []
    for line in open(f"{U}{name}/replay{tag}_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        near = [r for r in reports if -0.05 <= t - r[0] <= 0.15]
        if near:
            out.append(dict(x=np.array(e["embed"]), y="gunfire", clap=e["clap"], t=t, event=near[0][2] + (" far" if near[0][3] > 100 else ""), run=name))
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            out.append(dict(x=np.array(e["embed"]), y="other", clap=e["clap"], t=t, event="", run=name))
    return out


def fit(d, steps=800, lr=0.5, l2=1.0):
    X = np.stack([r["x"] for r in d])
    y = np.array([CLASSES.index(r["y"]) for r in d])
    K = len(CLASSES)
    n, dim = X.shape
    count = np.bincount(y, minlength=K)
    w = 1.0 / (K * count[y])
    A, B = np.zeros((K, dim)), np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(steps):
        z = X @ A.T + B
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        r = (p - Y) * w[:, None]
        A -= lr * (r.T @ X + l2 / n * A)
        B -= lr * r.sum(axis=0)
    return lambda x: CLASSES[int(np.argmax(A @ x + B))]


if __name__ == "__main__":
    samples = sys.argv[1]
    out = os.path.join(samples, "world")
    t0 = time.time()
    jobs = [(n, when, dist) for n in WORLD for when, ev, dist in triggers(n) if ev == "gas"]
    with mp.Pool(4, initializer=init, initargs=(out,)) as pool:
        found = pool.map(strike, jobs)
    strikes = {(n, t, d): b for n, t, d, b in found}
    print(f"Gas strikes located in {time.time() - t0:.0f} s (Artillery_Close, 150-240 s after each trigger; 0.4 needed):")
    for (n, t, d), (ncc, at, base) in strikes.items():
        print(f"  {n[-6:]} gas {d:4.0f} m: {ncc:.3f} at {at:6.1f} s, {at - t:5.1f} s after the trigger, {base}")
    world = {n: world_rows(n, strikes) for n in WORLD}
    gate = [r for n in GATE for r in gun_rows(n, "_emb")] + [r for n in STORM for r in storm_rows(n)] + [r for n in ATTACK for r in attack_rows(n)] + [r for n in SURVEY for r in survey_rows(n)] + [r for n in GUNSTORM for r in gunstorm_rows(n)]
    hold = [r for n in HOLD for r in gun_rows(n, "_app")]
    print("\nExamples by class:", {c: sum(r["y"] == c for rows in world.values() for r in rows) for c in CLASSES}, "world; gate", {c: sum(r["y"] == c for r in gate) for c in CLASSES})
    conf = {a: {b: 0 for b in CLASSES} for a in CLASSES}
    per_event = {}
    for n in WORLD:
        f = fit(gate + [r for m in WORLD if m != n for r in world[m]])
        for r in world[n]:
            got = f(r["x"])
            conf[r["y"]][got] += 1
            if r["event"]:
                per_event.setdefault((n[-6:], r["event"]), []).append((r["y"], got))
        g = [r for r in hold if r["y"] == "gunfire"]
        o = [r for r in hold if r["y"] == "other"]
        aug = [r for r in g if r["event"] == "AugShort far"]
        print(f"held out {n[-6:]}: unseen recordings: gunfire {sum(f(r['x']) == 'gunfire' for r in g)}/{len(g)}, 150 m AUG {sum(f(r['x']) == 'gunfire' for r in aug)}/{len(aug)}, no shot called gunfire {sum(f(r['x']) == 'gunfire' for r in o)}/{len(o)}, no shot called a world event {sum(f(r['x']) not in ('other', 'gunfire') for r in o)}/{len(o)}")
    print("\nEvents in the held-out world run, truth (rows) vs probe (columns):")
    print(f"  {'':<14}" + " ".join(f"{c[:10]:>10}" for c in CLASSES))
    for a in CLASSES:
        print(f"  {a:<14}" + " ".join(f"{conf[a][b]:>10d}" for b in CLASSES))
    print("\nEach world event, held out: events named its class / events; (run, event)")
    summary = {}
    for (run, ev), v in sorted(per_event.items()):
        right = sum(a == b for a, b in v)
        summary.setdefault(ev.rstrip("0123456789"), []).append(right > 0)
        print(f"  {run} {ev:<12} {right:3d}/{len(v):<3d}")
    print("Events with at least one event named right:", {k: f"{sum(v)}/{len(v)}" for k, v in summary.items()})

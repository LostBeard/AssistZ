"""
Can the namer learn the player's own sounds? Two new classes from mode 32: own_move (events while the driver moves the character) and own_shot (events within 0.4 s of a shot the game logged). Each own run held out in turn. Then, fit on all own runs: how much do real sounds on runs never trained on get taken for own? Near and far gunfire, agroed infected, other characters' footsteps (mode 10).
Usage (from Lab/flyby): python own_cv.py <cache.pkl> <own run> [...]
"""
import collections, csv, datetime, glob, json, os, pickle, sys
import numpy as np
from world_classes import CLASSES as BASE, U, gunstorm_rows
from far_cv import FAR
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

CLASSES = BASE if "own_move" in BASE else BASE + ["own_move", "own_shot"]
GUN = "gunfire"


def fit(d):
    X = np.stack([r["x"] for r in d]).astype(np.float64)
    y = np.array([CLASSES.index(r["y"]) for r in d])
    K, n, dim = len(CLASSES), X.shape[0], X.shape[1]
    count = np.bincount(y, minlength=K)
    w = 1.0 / (K * count[y])
    A, B = np.zeros((K, dim)), np.zeros(K)
    Y = np.eye(K)[y]
    for _ in range(800):
        z = X @ A.T + B
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        r = (p - Y) * w[:, None]
        A -= 0.5 * (r.T @ X + 1.0 / n * A)
        B -= 0.5 * r.sum(axis=0)
    return A, B


word = lambda A, B, x: CLASSES[int(np.argmax(A @ np.asarray(x, dtype=np.float64) + B))]


def own_rows(run):
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    stamp = st.timestamp()
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    drive = json.load(open(run + "/own_drive.json"))
    ph = [(p["name"], datetime.datetime.fromisoformat(p["start_utc"]).timestamp() - stamp, datetime.datetime.fromisoformat(p["end_utc"]).timestamp() - stamp) for p in drive["phases"]]
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    env = list(csv.DictReader(open(run + "/env.csv")))
    et = np.array([int(e["game_ms"]) / 1000 + off - s for e in env])
    worst = np.array([int(e["infected_worst"]) for e in env])
    storm = os.path.exists(run + "/world.csv") and any(w["event"] == "storm" for w in csv.DictReader(open(run + "/world.csv")))
    shots = []
    if os.path.exists(run + "/aishots_server.csv"):
        shots = [int(r["game_ms"]) / 1000 + off - s for r in csv.DictReader(open(run + "/aishots_server.csv"))]
    shots = np.array(shots)
    # The shot log's clock against the sound: the shift putting most events just after a shot.
    ev = [json.loads(x) for x in open(run + "/replay_own_s0.jsonl", encoding="utf-8")]
    t = np.array([e["sample"] / 48000.0 for e in ev])
    if len(shots):
        fire = next(((a, b) for n, a, b in ph if n == "fire"), (0, 0))
        tf = t[(t >= fire[0] - 12) & (t < fire[1] + 12)]
        scored = [(sum(np.any((tf - (a + d) >= 0) & (tf - (a + d) <= 0.2)) for a in shots), d) for d in np.arange(-12, 12, 0.02)]
        hits, best = max(scored)
        print(f"  {os.path.basename(run)[-6:]}: {len(shots)} shots logged, game clock span {shots.min():.0f}-{shots.max():.0f} s, fire phase {fire[0]:.0f}-{fire[1]:.0f} s, clock shift {best:+.2f} s puts {hits} shots with an event")
        shots = shots + best
    out = []
    for e, x in zip(ev, t):
        name = next((n for n, a, b in ph if a <= x < b), None)
        i = int(np.argmin(np.abs(et - x)))
        if worst[i] > 0 or name is None:
            continue      # a real infected got involved, or outside the driver's plan
        if storm and name != "fire":
            continue      # in a storm, what is heard while moving is thunder
        if name == "fire":
            if len(shots) and np.any((x - shots >= -0.05) & (x - shots <= 0.4)):
                out.append(dict(x=np.array(e["embed"]), y="own_shot", run=os.path.basename(run), az=e["az"], word=e["label"]))
        elif not name.startswith("still"):
            out.append(dict(x=np.array(e["embed"]), y="own_move", run=os.path.basename(run), az=e["az"], word=e["label"]))
        else:
            out.append(dict(x=np.array(e["embed"]), y="other", run=os.path.basename(run), az=e["az"], word=e["label"]))
    return out


if __name__ == "__main__":
    cache = sys.argv[1]
    runs = [p.replace("\\", "/").rstrip("/") for p in sys.argv[2:]]
    base, _, _ = pickle.load(open(cache, "rb"))
    far = [r for n in FAR for r in gunstorm_rows(n, "_far")]
    own = {os.path.basename(r): own_rows(r) for r in runs}
    for name, rows in own.items():
        print(f"{name}: {dict(collections.Counter(r['y'] for r in rows))}; shipped words on moves {dict(collections.Counter(r['word'] for r in rows if r['y'] == 'own_move').most_common(4))}")
    print("\n=== Each own run held out: what its events are called by a namer that never heard this run")
    total = collections.defaultdict(collections.Counter)
    for name in own:
        A, B = fit(base + far + [r for n, rows in own.items() if n != name for r in rows])
        for r in own[name]:
            total[r["y"]][word(A, B, r["x"])] += 1
    for k in ("own_move", "own_shot", "other"):
        c = total[k]
        print(f"  truth {k:<9} {sum(c.values()):5d}: {dict(c.most_common(6))}")
    for k in ("own_move", "own_shot"):
        z = np.abs([r["az"] for rows in own.values() for r in rows if r["y"] == k])
        if len(z):
            print(f"  {k} |az|: under 5 {100 * (z < 5).mean():.0f}%, under 10 {100 * (z < 10).mean():.0f}%, under 15 {100 * (z < 15).mean():.0f}%, n {len(z)}")
    print("\n=== Fit on all own runs: real sounds on runs never trained on")
    A, B = fit(base + far + [r for rows in own.values() for r in rows])
    A0, B0 = fit(base + far)
    print(f"{'set':<36}{'events':>7}  {'called own':>11}  {'gunfire before':>15}  {'gunfire after':>14}")

    def show(name, xs, az=None):
        a = [word(A, B, x) for x in xs]
        b = [word(A0, B0, x) for x in xs]
        own_n = sum(w.startswith("own") for w in a)
        gate = ""
        if az is not None:
            gate = " ".join(f"|az|<{c}: {sum(w.startswith('own') and abs(z) < c for w, z in zip(a, az))}" for c in (5, 10, 15))
        print(f"{name:<36}{len(xs):>7}  {own_n:>6} {100 * own_n / max(len(xs), 1):>3.0f}%  {sum(w == GUN for w in b):>15}  {sum(w == GUN for w in a):>14}   after: {dict(collections.Counter(a).most_common(4))}  {gate}")

    def shots_of(run, tag):
        reports, _ = calibrated_reports(U + run)
        ev = [json.loads(x) for x in open(f"{U}{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
        out = []
        for r in reports:
            near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
            if near:
                out.append(min(near, key=lambda x: abs(x["sample"] / 48000.0 - r[0])))
        return out

    def show_shots(name, run, tag):
        rows = shots_of(run, tag)
        show(name, [e["embed"] for e in rows], [e["az"] for e in rows])

    show_shots("near gunfire, clear (074844)", "20261105_074844", "_f1")
    show_shots("near gunfire, rain (080020)", "20261105_080020", "_f1")
    show_shots("far gunfire, clear (071655)", "20261105_071655", "_f1")
    show_shots("far gunfire, rain (073259)", "20261105_073259", "_f1")
    for run in ("20261105_112656", "20261105_122339"):
        ev = [json.loads(x) for x in open(f"{U}{run}/replay_st_s0.jsonl", encoding="utf-8")]
        ag = [e for e in ev if e["label"] == "infected attack"]
        show(f"infected runs ({run[-6:]}), all named agroed", [e["embed"] for e in ag], [e["az"] for e in ag])
    # Other characters' footsteps: lab mode 10, an AI walking round a still listener.
    found = []
    for path in glob.glob(U + "*/server_script.txt"):
        run = os.path.dirname(path).replace("\\", "/")
        try:
            if "mode 10 walker" in open(path, encoding="utf-8", errors="replace").read():
                found.append(run)
        except OSError:
            pass
    print(f"  mode 10 runs found: {[os.path.basename(r)[-15:] for r in found]}")
    for run in found[:6]:
        tag = next((t for t in ("_r7", "_app", "_vote", "_emb", "_base") if os.path.exists(f"{run}/replay{t}_s0.jsonl")), None)
        if not tag:
            print("   ", os.path.basename(run), "no replay with embeddings")
            continue
        ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
        ev = [e for e in ev if "embed" in e]
        if ev:
            show(f"other's footsteps ({os.path.basename(run)[-6:]})", [e["embed"] for e in ev], [e["az"] for e in ev])

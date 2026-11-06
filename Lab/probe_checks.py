"""
Attempts to break the embedding probe (probe_eval.py). Usage: python probe_checks.py <tag> <test run> <train run> [...]
A. Shuffled training labels: must fall to chance, or the score means nothing.
B. A weapon never trained on.
C. The "other" events the prompts call gunfire: are they unlogged shots? If so, the truth labels are wrong.
D. Real footsteps: no shot near, the AI moving and close, not in melee.
E. How far inside the decision boundary run 4's AUG burst sits, against the training data.
"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports, envelope_db, HOP

TAG, TEST, TRAIN = sys.argv[1], sys.argv[2], sys.argv[3:]
GUN = ("gunshot", "automatic fire")
AUTO = ("AugShort", "AKM", "M4A1")
DT = HOP / 48000


def load(run):
    events = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    reports, shift = calibrated_reports(run)
    env = envelope_db(np.load(run + "/audio.npy").astype(np.float64))
    pose = []
    if os.path.exists(run + "/aipose.csv"):
        pose = [x for x in csv.DictReader(open(run + "/aipose.csv")) if x["ai"] != "decoy"]
    # Audio clock of the AI pose: the same offset the shot log has, game seconds -> recording seconds.
    meta = json.load(open(run + "/audio.json"))
    import datetime
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pp = list(csv.DictReader(open(run + "/pose.csv")))
    offset = np.median([sum(int(v) * k for v, k in zip(r["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(r["game_ms"]) / 1000 for r in pp])
    rows = []
    for e in events:
        t = e["sample"] / 48000.0
        near = [r for r in reports if -0.05 <= t - r[0] <= 0.15]
        gap = min((abs(t - r[0]) for r in reports), default=99)
        k = int(t / DT)
        ai = None
        for x in pose:
            tp = int(x["game_ms"]) / 1000 + offset - start_s + shift + float(x["dist"]) / 330.0
            if abs(tp - t) < 0.3:
                ai = x
                break
        row = dict(run=os.path.basename(run.rstrip("/\\")), t=t, x=np.array(e["embed"]), label=e["label"], clap=e["clap"], p=e["p"], weapon=near[0][2] if near else "", gap=gap,
                   level=float(env[k:k + 40].max()) if k < len(env) else -200, ai=ai)
        row["truth"] = "gunfire" if near else ("other" if gap > 1.5 else None)
        row["prompt"] = e["label"] in GUN
        rows.append(row)
    return rows


def fit(rows, y=None, l2=1.0, steps=800, lr=0.5):
    X = np.stack([r["x"] for r in rows])
    y = np.array([r["truth"] == "gunfire" for r in rows], dtype=float) if y is None else y
    w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    a, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ a + b)))
        a -= lr * (X.T @ (w * (p - y)) + l2 / len(y) * a)
        b -= lr * np.sum(w * (p - y))
    return lambda x: x @ a + b


def rate(rows, f, want=True):
    g = [r for r in rows]
    return f"{sum((f(r['x']) > 0) == want for r in g)}/{len(g)}"


data = {run: load(run) for run in TRAIN}
test = load(TEST)
everything = [r for rows in data.values() for r in rows if r["truth"]]
gun = [r for r in everything if r["truth"] == "gunfire"]
oth = [r for r in everything if r["truth"] == "other"]
print(f"training: {len(gun)} gunfire, {len(oth)} other")

print("\nA. Shuffled labels (5 shuffles), leave one run out: right on gunfire, right on other. Chance is about half each.")
rng = np.random.default_rng(7)
for s in range(5):
    gr = gn = orr = on = 0
    for held, rows in data.items():
        rest = [r for run, o in data.items() if run != held for r in o if r["truth"]]
        y = rng.permutation([float(r["truth"] == "gunfire") for r in rest])
        f = fit(rest, y)
        for r in rows:
            if r["truth"] == "gunfire":
                gn += 1
                gr += f(r["x"]) > 0
            elif r["truth"] == "other":
                on += 1
                orr += f(r["x"]) <= 0
    print(f"  shuffle {s}: gunfire {gr}/{gn}, other {orr}/{on}")

print("\nB. Weapons never trained on (gunfire called gunfire; other = all 'other' events in the held-out runs, called other)")
for name, excl in (("no AUG in training", ("AugShort",)), ("no AKM in training", ("AKM",)), ("no M4A1 in training", ("M4A1",)), ("no automatics at all (SVD only)", AUTO), ("no SVD or Scout (automatics only)", ("SVD", "Scout"))):
    rest = [r for r in everything if r["weapon"] not in excl]
    f = fit(rest)
    held = [r for r in everything if r["weapon"] in excl]
    burst = [r for r in test if r["weapon"] == "AugShort"]
    line = f"  {name:<36} held-out weapon {rate(held, f)}"
    if "AugShort" in excl:
        line += f"; run 4's AUG burst {rate(burst, f)}"
    print(line)

print("\nC. 'Other' events the prompts call gunfire (no logged report within 1.5 s):")
f_all = None
suspects = []
for held, rows in data.items():
    rest = [r for run, o in data.items() if run != held for r in o if r["truth"]]
    f = fit(rest)
    for r in rows:
        if r["truth"] == "other" and r["prompt"]:
            suspects.append((r, f(r["x"])))
print(f"  {len(suspects)} events; by run: " + str({k: sum(s[0]["run"] == k for s in suspects) for k in sorted({s[0]["run"] for s in suspects})}))
for r, m in suspects:
    print(f"    {r['run'][-15:]} {r['t']:7.2f} s  {r['clap']:<17} p {r['p']:.2f}  level {r['level']:6.1f} dB  nearest report {r['gap']:6.1f} s  probe margin {m:+.2f}  AI {r['ai']['state'] if r['ai'] else '-'}")

print("\nD. Real footsteps: no report within 1.5 s, AI moving > 0.5 m/s within 35 m, not in melee, held-out run")
steps = []
for held, rows in data.items():
    rest = [r for run, o in data.items() if run != held for r in o if r["truth"]]
    f = fit(rest)
    for r in rows:
        a = r["ai"]
        if r["truth"] == "other" and a and float(a["speed"]) > 0.5 and float(a["dist"]) < 35 and "Melee" not in a["state"]:
            steps.append((r, f(r["x"])))
print(f"  {len(steps)} events; probe calls gunfire {sum(m > 0 for _, m in steps)}; prompts call gunfire {sum(r['prompt'] for r, _ in steps)}; CLAP's word: " + str({w: sum(r['clap'] == w for r, _ in steps) for w in sorted({r['clap'] for r, _ in steps})}))
for r, m in steps[:14]:
    print(f"    {r['run'][-15:]} {r['t']:7.2f} s  {r['clap']:<17} p {r['p']:.2f} level {r['level']:6.1f} AI {float(r['ai']['speed']):.1f} m/s at {float(r['ai']['dist']):.0f} m {r['ai']['state']}  margin {m:+.2f}")

print("\nE. Decision margins (> 0 = gunfire), trained on everything except run 4:")
f = fit(everything)
for name, rows in (("training gunfire", gun), ("training other", oth), ("run 4 AUG burst", [r for r in test if r["weapon"] == "AugShort"]), ("run 4 CLAP 'footsteps' in the burst", [r for r in test if r["weapon"] == "AugShort" and r["clap"] == "footsteps"]), ("run 4 SVD and Scout", [r for r in test if r["weapon"] in ("SVD", "Scout")]), ("run 4 other", [r for r in test if r["truth"] == "other"])):
    m = np.array([f(r["x"]) for r in rows])
    print(f"  {name:<36} n={len(m):3d}  min {m.min():+6.2f}  median {np.median(m):+6.2f}  max {m.max():+6.2f}")

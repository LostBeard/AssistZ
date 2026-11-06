"""
Does the embedding probe call real footsteps gunfire? Lab mode 10 has no weapons at all, so every event is not gunfire by construction. The AI pose log
says whether the unarmed AI was walking near the listener when each event was heard. Trained only on the other recordings.
Usage: python probe_footsteps.py <tag> <footstep run> [...] -- <train run> [...]
"""
import csv, datetime, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

TAG = sys.argv[1]
split = sys.argv.index("--")
RUNS, TRAIN = sys.argv[2:split], sys.argv[split + 1:]
GUN = ("gunshot", "automatic fire")


def train_rows(run):
    reports, _ = calibrated_reports(run)
    rows = []
    for line in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        if any(-0.05 <= t - r[0] <= 0.15 for r in reports):
            rows.append((np.array(e["embed"]), 1.0))
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            rows.append((np.array(e["embed"]), 0.0))
    return rows


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
for run in RUNS:
    meta = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pp = list(csv.DictReader(open(run + "/pose.csv")))
    offset = np.median([sum(int(v) * k for v, k in zip(r["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(r["game_ms"]) / 1000 for r in pp])
    pose = list(csv.DictReader(open(run + "/aipose.csv")))
    speeds = [float(x["speed"]) for x in pose]
    dists = [float(x["dist"]) for x in pose]
    print(f"{os.path.basename(run.rstrip('/'))}: AI pose {len(pose)} rows, moving {sum(s > 0.5 for s in speeds)}, within 35 m {sum(d < 35 for d in dists)}, states {sorted({x['state'] for x in pose})[:6]}")
    print("  shots logged:", os.path.exists(run + "/aishots_server.csv"))
    events = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    walking, rest = [], []
    for e in events:
        t = e["sample"] / 48000.0
        best = min(pose, key=lambda x: abs(int(x["game_ms"]) / 1000 + offset - start_s + float(x["dist"]) / 330.0 - t))
        dt = abs(int(best["game_ms"]) / 1000 + offset - start_s + float(best["dist"]) / 330.0 - t)
        row = dict(t=t, clap=e["clap"], p=e["p"], m=f(np.array(e["embed"])), speed=float(best["speed"]), dist=float(best["dist"]), level=e["level"], prompt=e["label"] in GUN)
        (walking if dt < 1.0 and row["speed"] > 0.5 and row["dist"] < 35 else rest).append(row)
    for name, rows in (("AI walking within 35 m", walking), ("AI not walking near", rest)):
        print(f"  {name}: {len(rows)} events; probe calls gunfire {sum(r['m'] > 0 for r in rows)}, prompts call gunfire {sum(r['prompt'] for r in rows)}; CLAP's words {dict((w, sum(r['clap'] == w for r in rows)) for w in sorted({r['clap'] for r in rows}))}")
        if rows:
            print(f"     margin min {min(r['m'] for r in rows):+.2f} median {np.median([r['m'] for r in rows]):+.2f} max {max(r['m'] for r in rows):+.2f}; level dB median {np.median([r['level'] for r in rows]):.1f}")
    for r in sorted(walking, key=lambda r: -r["m"])[:12]:
        print(f"     {r['t']:7.2f} s  {r['clap']:<17} p {r['p']:.2f}  level {r['level']:6.1f}  AI {r['speed']:.1f} m/s at {r['dist']:.0f} m  margin {r['m']:+.2f}")

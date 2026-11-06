"""
Would a probe on CLAP's audio embeddings name DayZ's sounds better than the text prompts? Scored only on recordings it was not trained on.

Truth comes from the game's logs. An event ending -50..+150 ms around a logged report is gunfire of that weapon. An event with no report
within 1.5 s is "other"; it can still hold a far echo, so that label is the weak one. Events between the two are left out of training and
scoring. Footsteps have no ground truth of their own here, so "other" is the honest class, and the question is whether gunfire is told
from everything else, the AUG burst the prompts call footsteps above all.

Two rounds: leave one run out over the training recordings, and run 4 (sniper_20261103_153510) as a test only, because its AUG burst at
about 150 m is beyond the 90 m the lab's AI fires from.

Usage: python probe_eval.py <tag> <test run> <train run> [...]     reads <run>/replay<tag>_s0.jsonl
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab_clock import calibrated_reports

TAG, TEST, TRAIN = sys.argv[1], sys.argv[2], sys.argv[3:]
PROMPT_GUNFIRE = ("gunshot", "automatic fire")


def load(run):
    events = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
    reports, _ = calibrated_reports(run)
    rows = []
    for e in events:
        t = e["sample"] / 48000.0
        near = [r for r in reports if -0.05 <= t - r[0] <= 0.15]
        far = all(abs(t - r[0]) > 1.5 for r in reports)
        if near:
            truth, weapon = "gunfire", near[0][2]
        elif far:
            truth, weapon = "other", ""
        else:
            continue
        rows.append(dict(run=os.path.basename(run.rstrip("/\\")), t=t, truth=truth, weapon=weapon, x=np.array(e["embed"]), prompt=e["label"] in PROMPT_GUNFIRE, clap=e["clap"], distance=near[0][3] if near else 0.0))
    return rows


def fit_centroid(rows):
    centroids = {}
    for c in ("gunfire", "other"):
        m = np.mean([r["x"] for r in rows if r["truth"] == c], axis=0)
        centroids[c] = m / np.linalg.norm(m)
    return lambda x: (x @ centroids["gunfire"]) > (x @ centroids["other"])


def fit_logistic(rows, l2=1.0, steps=800, lr=0.5):
    """Class-balanced logistic regression on the embedding, full batch."""
    X = np.stack([r["x"] for r in rows])
    y = np.array([r["truth"] == "gunfire" for r in rows], dtype=float)
    w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    a, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ a + b)))
        a -= lr * (X.T @ (w * (p - y)) + l2 / len(y) * a)
        b -= lr * np.sum(w * (p - y))
    return lambda x: (x @ a + b) > 0


def report(name, rows, predict):
    def rate(sel, pred):
        g = [r for r in rows if sel(r)]
        return f"{sum(pred(r) for r in g)}/{len(g)}" if g else "-"
    methods = (("prompts", lambda r: r["prompt"]), ("probe", lambda r: bool(predict(r["x"]))))
    for method, pred in methods:
        weapons = " ".join(f"{w} {rate(lambda r, w=w: r['weapon'] == w, pred)}" for w in sorted({r["weapon"] for r in rows if r["weapon"]}))
        print(f"  {name:<24} {method:<8} gunfire named gunfire {rate(lambda r: r['truth'] == 'gunfire', pred):>9} ({weapons});  other named gunfire {rate(lambda r: r['truth'] == 'other', pred)}")


train = {run: load(run) for run in TRAIN}
for run, rows in train.items():
    print(f"{os.path.basename(run.rstrip('/'))}: {sum(r['truth'] == 'gunfire' for r in rows)} gunfire events, {sum(r['truth'] == 'other' for r in rows)} other")
for label, fit in (("nearest centroid", fit_centroid), ("logistic", fit_logistic)):
    print(f"\n=== {label}: leave one run out")
    pooled = []
    for held, rows in train.items():
        rest = [r for run, other in train.items() if run != held for r in other]
        predict = fit(rest)
        report(os.path.basename(held.rstrip("/"))[-15:], rows, predict)
        pooled += [dict(r, probe=bool(predict(r["x"]))) for r in rows]
    for method, key in (("prompts", "prompt"), ("probe", "probe")):
        g = [r for r in pooled if r["truth"] == "gunfire"]
        o = [r for r in pooled if r["truth"] == "other"]
        print(f"  {'ALL HELD OUT':<24} {method:<8} gunfire named gunfire {sum(r[key] for r in g)}/{len(g)};  other named gunfire {sum(r[key] for r in o)}/{len(o)}")
    print(f"=== {label}: trained on all of those, tested on run 4 (never trained on)")
    everything = [r for rows in train.values() for r in rows]
    predict = fit(everything)
    test = load(TEST)
    report("run 4 all events", test, predict)
    burst = [r for r in test if r["weapon"] == "AugShort"]
    print("  AUG burst, event by event (t, CLAP's word, probe says gunfire?):")
    print("   " + " ".join(f"{r['t']:.1f}{'' if r['clap'] != 'footsteps' else '(F)'}:{'Y' if predict(r['x']) else 'n'}" for r in burst))

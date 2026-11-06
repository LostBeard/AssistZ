"""Where did real gunfire go, and would a confidence bar on "own" stop it? Real sounds called own, by probability, against true own sounds. Survey site that went over. Usage: own_losses.py"""
import collections, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
R = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
OWN = ("own move", "own shot")
GUN = ("gunshot", "automatic fire")


def near(run, tag):
    reports, _ = calibrated_reports(R + run)
    ev = [json.loads(x) for x in open(f"{R}{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
    return [e for e in ev if any(-0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15 for r in reports)]


for run, name in (("20261104_215004", "rain storm"), ("20261104_215827", "dry storm"), ("20261104_220700", "rain no storm"), ("20261105_151838", "near rain fresh"), ("20261105_150229", "far clear fresh")):
    a = near(run, "_f2")
    c = collections.Counter(e["label"] for e in a)
    lost = [e for e in a if e["label"] in OWN]
    print(f"{name:<16} {len(a)} events near a shot: {dict(c.most_common(6))}")
    if lost:
        print(f"   called own: p {[round(e['p'], 2) for e in lost][:16]}  az {[round(e['az'], 1) for e in lost][:16]}")
print("\nReal sounds called own, p, over the near-shot sets above and older holdouts:")
real = []
for run in ("20261104_215004", "20261104_215827", "20261104_220700", "20261105_151838", "20261105_150229", "20261104_224711", "20261104_144529", "20261104_145757", "20261104_121745", "20261104_162444", "20261104_160723", "20261104_133615"):
    path = f"{R}{run}/replay_f2_s0.jsonl"
    if os.path.exists(path):
        real += [json.loads(x) for x in open(path, encoding="utf-8")]
false_own = np.array([e["p"] for e in real if e["label"] in OWN])
print(f"  {len(false_own)} of {len(real)}: p p10 {np.percentile(false_own, 10):.2f} p50 {np.median(false_own):.2f} p90 {np.percentile(false_own, 90):.2f}")
true_own = []
for run in ("20261105_143843", "20261105_144722"):
    true_own += [json.loads(x) for x in open(f"{R}{run}/replay_f2_s0.jsonl", encoding="utf-8")]
t = np.array([e["p"] for e in true_own if e["label"] in OWN])
print(f"  fresh own runs, named own: {len(t)}: p p10 {np.percentile(t, 10):.2f} p50 {np.median(t):.2f} p90 {np.percentile(t, 90):.2f}")
for bar in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    print(f"  bar {bar}: real still called own {(false_own >= bar).sum()}/{len(false_own)}; true own kept {(t >= bar).sum()}/{len(t)} ({100 * (t >= bar).mean():.0f}%)")
print("\nSurvey events near the site that went over:")
ev = [json.loads(x) for x in open(f"{R}20261104_162444/replay_f2_s0.jsonl", encoding="utf-8")]
print(" ", dict(collections.Counter(e["label"] for e in ev).most_common(10)))

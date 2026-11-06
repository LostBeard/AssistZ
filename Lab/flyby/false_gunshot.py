"""The false gunshots in dry thunder: where in the burst, how loud, how sure, and how do real shots compare? Read-only. These runs are now looked at, so anything chosen from this needs fresh runs to test."""
import json, sys
import numpy as np

R = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
GUN = ("gunshot", "automatic fire")
for name in ("20261104_175558", "20261104_181149"):
    ev = sorted((json.loads(x) for x in open(f"{R}{name}/replay_r5_s0.jsonl", encoding="utf-8")), key=lambda e: e["sample"])
    print(name)
    for i, e in enumerate(ev):
        if e["label"] in GUN:
            t = e["sample"] / 48000
            nb = [x for x in ev if abs(x["sample"] / 48000 - t) < 1.0]
            first = min(x["sample"] for x in nb) == e["sample"]
            print(f"  {t:6.1f} s {e['label']:<8} p {e['p']:.3f} level {e['level']:6.1f} clap {e['clap'][:14]:<14} az {e['az']:6.1f} first in burst {str(first):<5} neighbours {sorted(set(x['label'] for x in nb))} n={len(nb)}")
true_p, levels = [], []
for name in ("20261103_153510", "20261104_021109", "20261104_021644"):
    ev = [json.loads(x) for x in open(f"{R}{name}/replay_r5_s0.jsonl", encoding="utf-8")]
    for e in ev:
        if e["label"] in GUN:
            true_p.append(e["p"])
            levels.append(e["level"])
true_p = np.array(true_p)
levels = np.array(levels)
print(f"Real gunfire on the five: {len(true_p)} events; p min {true_p.min():.3f}, p5 {np.percentile(true_p, 5):.3f}, p25 {np.percentile(true_p, 25):.3f}, median {np.median(true_p):.3f}; level min {levels.min():.1f}, median {np.median(levels):.1f}, max {levels.max():.1f}")
print("Real gunfire kept at a floor:", {f: f"{int((true_p >= f).sum())}/{len(true_p)}" for f in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)})
for name in ("20261104_174018",):
    ev = [json.loads(x) for x in open(f"{R}{name}/replay_vote_s0.jsonl", encoding="utf-8")]
    print(f"Training run {name}: gunfire words under the shipped probe:", sum(e["label"] in GUN for e in ev))

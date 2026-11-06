"""
Is the probe a loudness detector? `make`: gain-shifted copies of two recordings under the work folder (never in the run folders). `score`: level alone as a
baseline on the same events, then the probe trained only on the unshifted recordings, tested on the shifted copies.
Usage: python gain_test.py make | python gain_test.py score <train run> [...]
"""
import json, os, shutil, sys
import numpy as np
from scipy.io import wavfile
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

BASE = "D:/users/tj/Projects/AssistZ/Lab/runs"
WORK = "L:/AssistZLabWork/gain"
COPIES = (("steps_plus30", "sniper_20261104_010112", 30.0), ("run4_minus20", "sniper_20261103_153510", -20.0), ("run4_plus12", "sniper_20261103_153510", 12.0))
GUN = ("gunshot", "automatic fire")

if sys.argv[1] == "make":
    for name, src, db in COPIES:
        out = f"{WORK}/{name}"
        os.makedirs(out, exist_ok=True)
        pcm = np.load(f"{BASE}/{src}/audio.npy").astype(np.float64) * 10 ** (db / 20)
        peak = float(np.abs(pcm).max())
        wavfile.write(out + "/audio.wav", 48000, (np.clip(pcm, -1, 1) * 32767).astype(np.int16))
        for f in ("audio.json", "pose.csv", "shots_ai.csv", "aipose.csv"):
            if os.path.exists(f"{BASE}/{src}/{f}"):
                shutil.copy(f"{BASE}/{src}/{f}", out)
        # lab_clock calibrates on audio.npy: the shot clock must stay the source recording's, so the shifted copy keeps a shifted npy too.
        np.save(out + "/audio.npy", np.clip(pcm, -1, 1).astype(np.float32))
        print(f"{name}: {db:+.0f} dB, peak {peak:.3f}, clipped samples {int((np.abs(pcm) > 1).sum())}")
    sys.exit(0)

TRAIN = sys.argv[2:]


def rows(run):
    reports, _ = calibrated_reports(run)
    out = []
    for line in open(f"{run}/replay_emb_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        t = e["sample"] / 48000.0
        near = [r for r in reports if -0.05 <= t - r[0] <= 0.15]
        if near:
            truth = 1.0
        elif all(abs(t - r[0]) > 1.5 for r in reports):
            truth = 0.0
        else:
            continue
        out.append(dict(x=np.array(e["embed"]), y=truth, level=e["level"], prompt=e["label"] in GUN, clap=e["clap"], weapon=near[0][2] if near else ""))
    return out


def fit(data, l2=1.0, steps=800, lr=0.5):
    X = np.stack([r["x"] for r in data])
    y = np.array([r["y"] for r in data])
    w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
    a, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ a + b)))
        a -= lr * (X.T @ (w * (p - y)) + l2 / len(y) * a)
        b -= lr * np.sum(w * (p - y))
    return lambda x: float(x @ a + b)


train = [r for run in TRAIN for r in rows(run)]
f = fit(train)
steps = rows(f"{BASE}/sniper_20261104_010112")
run4 = rows(f"{BASE}/sniper_20261103_153510")
levels = np.array([r["level"] for r in train])
truth = np.array([r["y"] for r in train])
best = max(((((levels > t) == (truth == 1))[truth == 1].mean() + ((levels > t) == (truth == 1))[truth == 0].mean()) / 2, t) for t in np.arange(-60, 0, 0.5))
T = best[1]
print(f"Level alone, best threshold on the training data: {T:.1f} dB, balanced accuracy {best[0]:.3f}")
print(f"  training gunfire level median {np.median(levels[truth == 1]):.1f} dB, other {np.median(levels[truth == 0]):.1f} dB")


def show(name, data):
    g = [r for r in data if r["y"] == 1]
    o = [r for r in data if r["y"] == 0]
    print(f"  {name:<28} events: gunfire {len(g):3d}, not {len(o):3d}.  gunfire called gunfire: probe {sum(f(r['x']) > 0 for r in g)}, level {sum(r['level'] > T for r in g)}, prompts {sum(r['prompt'] for r in g)};  not gunfire called gunfire: probe {sum(f(r['x']) > 0 for r in o)}, level {sum(r['level'] > T for r in o)}, prompts {sum(r['prompt'] for r in o)}")
    aug = [r for r in g if r["weapon"] == "AugShort"]
    if aug:
        print(f"  {'':<28} AUG burst: probe {sum(f(r['x']) > 0 for r in aug)}/{len(aug)}, level {sum(r['level'] > T for r in aug)}/{len(aug)}, prompts {sum(r['prompt'] for r in aug)}/{len(aug)};  level median {np.median([r['level'] for r in aug]):.1f} dB")
    if o:
        m = [f(r["x"]) for r in o]
        print(f"  {'':<28} not-gunfire margin max {max(m):+.2f}, median {np.median(m):+.2f};  level median {np.median([r['level'] for r in o]):.1f} dB")


print("\nUnshifted:")
show("real footsteps", steps)
show("run 4", run4)
print("\nShifted (probe trained on unshifted audio only):")
show("real footsteps +30 dB", rows(f"{WORK}/steps_plus30"))
show("run 4 -20 dB", rows(f"{WORK}/run4_minus20"))
show("run 4 +12 dB", rows(f"{WORK}/run4_plus12"))

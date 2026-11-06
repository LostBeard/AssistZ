"""Is the 470 s 'gas strike' thunder? Which wrong words fall in which storm cluster, and what is the spacing of the clusters? Read-only."""
import glob, json, os, sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from world_events import SOS, RATE

samples = sys.argv[1]
run = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261104_081350"
a = np.load(run + "/audio.npy").astype(np.float64)


def low(t0, t1):
    seg = a[int(t0 * 48000): int(t1 * 48000)]
    return [resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32) for c in range(seg.shape[1])]


def templates(paths, keep=3):
    out = []
    for p in paths:
        x, r = sf.read(p, always_2d=True)
        x = resample_poly(x.mean(axis=1), RATE, r)
        k = min(len(x), keep * RATE)
        s = int(np.argmax(np.convolve(x ** 2, np.ones(k), "valid")))
        x = x[s: s + k]
        for pitch in np.exp(np.arange(np.log(0.85), np.log(1.18), 0.01)):
            n = int(round(len(x) / pitch))
            y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
            nrm = np.linalg.norm(y)
            if nrm > 0:
                out.append((os.path.basename(p), float(pitch), y / nrm))
    return out


def best(sig, tmpl):
    top = (0.0, 0.0, "", 0.0)
    for s in sig:
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for name, pitch, y in tmpl:
            if len(y) >= len(s):
                continue
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / e
            ncc[e < 0.0005] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > top[0]:
                top = (float(ncc[k]), k / RATE, name, pitch)
    return top


th = templates(glob.glob(os.path.join(samples, "thunder", "*.ogg")))
ev = sorted((json.loads(x) for x in open(run + "/replay_vote_s0.jsonl", encoding="utf-8")), key=lambda e: e["sample"])
for t0, t1, note in ((465, 480, "the 'gas strike' at 470.5 s"), (480, 510, "clear_after, later"), (560, 600, "control, infected phase"), (120, 160, "control, clear phase")):
    m = best(low(t0, t1), th)
    print(f"thunder in {t0}-{t1} s, {note}: {m[0]:.3f} at {t0 + m[1]:.1f} s, {m[2][:24]}")
clusters = []
for e in ev:
    t = e["sample"] / 48000
    if not 315 <= t < 475:
        continue
    if clusters and t - clusters[-1][-1]["sample"] / 48000 < 6:
        clusters[-1].append(e)
    else:
        clusters.append([e])
print("\nStorm clusters: start, gap to the previous, loudest level, label by event, CLAP's word when the label differs")
prev = None
for c in clusters:
    t = c[0]["sample"] / 48000
    gap = "" if prev is None else f"{t - prev:5.1f}"
    prev = t
    print(f"  {t:6.1f} {gap:>5} {max(e['level'] for e in c):6.1f} dB  " + ", ".join(e["label"] for e in c) + "  | raw " + ", ".join(e.get("raw", "?") for e in c))
print("\nEvent keys:", sorted(ev[0].keys()))
print("Events with embedding:", sum("embedding" in e or "emb" in e for e in ev), "of", len(ev))

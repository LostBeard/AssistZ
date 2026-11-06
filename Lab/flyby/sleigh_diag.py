"""Mode 17's 'sleigh crash' words: when, how loud, what did the shipped probe say about the same clips, and do the game's crash samples match? Also the training run's rate. Read-only."""
import glob, json, os, sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from world_events import SOS, RATE

samples = sys.argv[1]
U = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
ship = json.load(open("D:/users/tj/Projects/AssistZ/AssistZ/AssistZ/wwwroot/models/sound-probe.json"))
A5, B5, C5 = np.array(ship["Weights"]), np.array(ship["Bias"]), ship["Classes"]
print("shipped probe classes:", C5)
paths = [p for p in glob.glob(os.path.join(samples, "world", "*.ogg")) if any(k in os.path.basename(p).lower() for k in ("sledge", "sleigh", "santa"))]
print(len(paths), "crash samples:", [os.path.basename(p) for p in paths][:6])
tmpl = []
for p in paths:
    x, r = sf.read(p, always_2d=True)
    y = resample_poly(x.mean(axis=1), RATE, r)
    k = min(len(y), 4 * RATE)
    s = int(np.argmax(np.convolve(y ** 2, np.ones(RATE) / RATE, "same")))
    y = y[max(0, s - RATE): max(0, s - RATE) + k]
    for pitch in (0.94, 0.97, 1.0, 1.03, 1.06):
        n = int(round(len(y) / pitch))
        z = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
        if np.linalg.norm(z) > 0:
            tmpl.append((os.path.basename(p)[:22], z / np.linalg.norm(z)))


def best(a, t0, t1):
    seg = np.array(a[int(max(0, t0) * 48000): int(t1 * 48000)], dtype=np.float64)
    top = (0.0, 0.0, "")
    for c in range(seg.shape[1]):
        s = resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32)
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for name, y in tmpl:
            if len(y) >= len(s):
                continue
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / e
            ncc[e < 0.0005] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > top[0]:
                top = (float(ncc[k]), max(0, t0) + k / RATE, name)
    return top


for name, tag in (("20261104_121745", "_at"), ("20261104_120527", "_vote")):
    run = U + name
    ev = sorted((json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")), key=lambda e: e["sample"])
    a = np.load(run + "/audio.npy", mmap_mode="r")
    print(f"\n{name}{tag}: {len(ev)} events")
    words = {}
    for e in ev:
        words[e["label"]] = words.get(e["label"], 0) + 1
    print("  labels:", dict(sorted(words.items(), key=lambda x: -x[1])[:8]))
    old = {}
    for e in ev:
        c = C5[int(np.argmax(A5 @ np.array(e["embed"], dtype=np.float64) + B5))]
        old[c] = old.get(c, 0) + 1
    print("  the shipped probe on the same clips:", dict(sorted(old.items(), key=lambda x: -x[1])))
    sl = [e for e in ev if e["label"] == "sleigh crash"]
    if not sl:
        continue
    groups = []
    for e in sl:
        t = e["sample"] / 48000
        if groups and t - groups[-1][-1]["sample"] / 48000 < 8:
            groups[-1].append(e)
        else:
            groups.append([e])
    print(f"  {len(sl)} sleigh crash events in {len(groups)} groups: start, n, loudest dB, az, CLAP word, crash sample match")
    for g in groups[:16]:
        t = g[0]["sample"] / 48000
        m = best(a, t - 3, t + 8)
        c = {}
        for e in g:
            c[e["clap"]] = c.get(e["clap"], 0) + 1
        print(f"    {t:6.1f} {len(g):3d} {max(e['level'] for e in g):6.1f} {np.median([e['az'] for e in g]):6.1f}  {max(c, key=c.get)[:16]:<16} {m[0]:.3f} {m[2]}")
    for t0, t1, note in ((5, 15, "control, early"), (250, 262, "control, attack phase")):
        m = best(a, t0, t1)
        print(f"  {note}: {m[0]:.3f} {m[2]}")

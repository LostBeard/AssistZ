"""Why did mode 13 miss the prediction? Every cluster in both storm runs, the shape of the game's thunder samples, and the sample match per cluster. Read-only."""
import csv, datetime, glob, json, os, sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from world_events import SOS, RATE

samples = sys.argv[1]
U = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
print("Thunder samples: seconds, and seconds of each envelope peak within 12 dB of the loudest (0.25 s RMS)")
tmpl = []
for p in sorted(glob.glob(os.path.join(samples, "thunder", "*.ogg"))):
    x, r = sf.read(p, always_2d=True)
    m = x.mean(axis=1)
    win = int(0.25 * r)
    env = np.sqrt(np.convolve(m ** 2, np.ones(win) / win, "same"))
    peaks = [i for i in range(win, len(env) - win, win // 2) if env[i] == env[max(0, i - r):i + r].max() and env[i] > env.max() * 0.25]
    print(f"  {os.path.basename(p):<26} {len(m) / r:5.1f} s  peaks {[round(i / r, 1) for i in peaks]}")
    y = resample_poly(m, RATE, r)
    k = min(len(y), 3 * RATE)
    s = int(np.argmax(np.convolve(y ** 2, np.ones(k), "valid")))
    y = y[s: s + k]
    for pitch in np.exp(np.arange(np.log(0.85), np.log(1.18), 0.02)):
        n = int(round(len(y) / pitch))
        z = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
        if np.linalg.norm(z) > 0:
            tmpl.append((os.path.basename(p), z / np.linalg.norm(z)))


def best(a, t0, t1):
    seg = a[int(max(0, t0) * 48000): int(t1 * 48000)].astype(np.float64)
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


for name in sys.argv[2:]:
    run = U + name
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s0 = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    world = [(int(e["game_ms"]) / 1000 + off - s0, e["event"], float(e["distance"])) for e in csv.DictReader(open(run + "/world.csv"))]
    a = np.load(run + "/audio.npy")
    ev = sorted((json.loads(x) for x in open(f"{run}/replay_vote_s0.jsonl", encoding="utf-8")), key=lambda e: e["sample"])
    clusters = []
    for e in ev:
        t = e["sample"] / 48000
        if clusters and t - clusters[-1][-1]["sample"] / 48000 < 6:
            clusters[-1].append(e)
        else:
            clusters.append([e])
    print(f"\n{name}: {len(ev)} events; phases " + ", ".join(f"{n} {d:.0f} at {t:.0f} s" for t, n, d in world))
    print("  start  gap   len   n   dB  thunder match  words")
    prev = None
    for c in clusters:
        t0, t1 = c[0]["sample"] / 48000, c[-1]["sample"] / 48000
        words = {}
        for e in c:
            words[e["label"]] = words.get(e["label"], 0) + 1
        top = sorted(words.items(), key=lambda x: -x[1])[:4]
        loud = max(e["level"] for e in c)
        mt = best(a, t0 - 2, t0 + 6) if loud > -32 else (0, 0, "")
        print(f"  {t0:6.1f} {'' if prev is None else f'{t0 - prev:5.1f}':>5} {t1 - t0:5.1f} {len(c):3d} {loud:5.0f}  {mt[0]:.3f} {mt[2][7:20]:<14} {dict(top)}")
        prev = t0

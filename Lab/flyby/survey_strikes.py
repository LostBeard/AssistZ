"""
Natural gas strikes in a survey run: the Artillery_Close match, searched around every burst (3+ events, each under 0.5 s after the last), so a real strike is not scored as a false word. Writes <run>/strikes.json, a list of [ncc, seconds]. Samples are read locally, never committed.
Usage: python survey_strikes.py <samples dir> <tag> <run name> [...]
"""
import json, multiprocessing as mp, os, sys
import numpy as np
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from world_events import SOS, RATE
from world_classes import U, init, _T


def search(job):
    name, t0, t1 = job
    a = np.load(U + name + "/audio.npy", mmap_mode="r")
    lo = int(max(0, t0) * 48000)
    seg = np.array(a[lo: int(t1 * 48000)], dtype=np.float64)
    best = (0.0, 0.0)
    for c in range(seg.shape[1]):
        s = resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32)
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for group, base, pitch, y, rev in _T["t"]:
            if len(y) >= len(s):
                continue
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / e
            ncc[e < 0.0005] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > best[0]:
                best = (float(ncc[k]), lo / 48000 + k / RATE)
    return name, t0, best


if __name__ == "__main__":
    samples, tag, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    jobs = []
    for n in names:
        ev = sorted((json.loads(x)["sample"] / 48000 for x in open(f"{U}{n}/replay{tag}_s0.jsonl", encoding="utf-8")))
        runs = []
        for t in ev:
            if runs and t - runs[-1][-1] < 0.5:
                runs[-1].append(t)
            else:
                runs.append([t])
        bursts = [r[0] for r in runs if len(r) >= 3]
        merged = []
        for t in bursts:
            if not merged or t - merged[-1] > 12:
                merged.append(t)
        print(n, len(ev), "events,", len(bursts), "bursts,", len(merged), "windows")
        jobs += [(n, t - 16, t + 14) for t in merged]
    with mp.Pool(4, initializer=init, initargs=(os.path.join(samples, "world"),)) as pool:
        found = pool.map(search, jobs)
    for n in names:
        hits = sorted({(round(b[0], 3), round(b[1], 2)) for m, t, b in found if m == n})
        strong = [list(h) for h in hits if h[0] >= 0.4]
        json.dump(strong, open(f"{U}{n}/strikes.json", "w"))
        print(n, "windows searched", len(hits), "best", hits[-1:] and max(h[0] for h in hits), "strikes at 0.4 or more:", strong)

"""
Of the unseen no-shot events a five-class probe calls a world event, how many are real? The natural gas strikes proven by sample match are labelled "other" only because the lab did not trigger them. Trained on the gate runs and all three world runs; tested on the five
unseen recordings. The world runs' own "other" events can also hold natural strikes, which would be wrong training labels, so those are searched as well and reported, not removed.
Usage: python world_fp.py <samples dir>
"""
import json, multiprocessing as mp, os, sys
import numpy as np
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from world_events import SOS, RATE
from world_classes import U, GATE, WORLD, HOLD, CLASSES, gun_rows, world_rows, fit, init, _T, strike, triggers

LEVEL_S = 1


def scan(job):
    name, t0, t1 = job
    a = np.load(U + name + "/audio.npy")
    lo, hi = int(t0 * 48000), min(len(a), int(t1 * 48000))
    seg = a[lo:hi].astype(np.float64)
    best = (0.0, 0.0, "")
    for c in range(seg.shape[1]):
        s = resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32)
        if len(s) < RATE:
            continue
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for group, base, pitch, y, rev in _T["t"]:
            if len(y) >= len(s):
                continue
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / e
            ncc[e < 0.0005] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > best[0]:
                best = (float(ncc[k]), lo / 48000 + k / RATE, base)
    return name, t0, best


if __name__ == "__main__":
    samples = sys.argv[1]
    out = os.path.join(samples, "world")
    with mp.Pool(4, initializer=init, initargs=(out,)) as pool:
        located = pool.map(strike, [(n, when, dist) for n in WORLD for when, ev, dist in triggers(n) if ev == "gas"])
        strikes = {(n, t, d): b for n, t, d, b in located}
        world = {n: world_rows(n, strikes) for n in WORLD}
        gate = [r for n in GATE for r in gun_rows(n, "_emb")]
        hold = [r for n in HOLD for r in gun_rows(n, "_app")]
        f = fit(gate + [r for rows in world.values() for r in rows])
        calls = [dict(r, got=f(r["x"])) for r in hold if r["y"] == "other"]
        flagged = [r for r in calls if r["got"] not in ("other", "gunfire")]
        # Group flagged events into bursts within 25 s, per recording, and search each for Artillery_Close.
        bursts = []
        for run in sorted({r["run"] for r in flagged}):
            times = sorted(r["t"] for r in flagged if r["run"] == run)
            start = times[0]
            prev = start
            for t in times[1:] + [1e9]:
                if t - prev > 25:
                    bursts.append((run, start, prev))
                    start = t
                prev = t
        found = pool.map(scan, [(run, max(0, a - 15), b + 15) for run, a, b in bursts])
    print(f"Unseen recordings: {len(calls)} no-shot events; {len(flagged)} called a world event; {len(bursts)} bursts")
    real = 0
    for (run, a, b), (_, _, m) in zip(bursts, found):
        g = [r for r in flagged if r["run"] == run and a <= r["t"] <= b]
        words = {}
        for r in g:
            words[r["got"]] = words.get(r["got"], 0) + 1
        is_gas = m[0] >= 0.4
        real += len(g) if is_gas else 0
        print(f"  {run[-6:]} {a:6.1f}-{b:6.1f} s: {len(g):2d} events {words}; Artillery_Close {m[0]:.3f} at {m[1]:6.1f} s -> {'REAL gas strike' if is_gas else 'no strike found'}; CLAP's words {sorted({r['clap'] for r in g})[:4]}")
    wrong = len(flagged) - real
    clean = len(calls) - real
    print(f"Called a world event, and a strike really is there: {real}. Not explained: {wrong} of {clean} no-shot events without a proven strike = {100 * wrong / max(1, clean):.1f}%")
    print("By probe word, for the unexplained:", {c: sum(r['got'] == c for (run, a, b), (_, _, m) in zip(bursts, found) if m[0] < 0.4 for r in flagged if r['run'] == run and a <= r['t'] <= b) for c in CLASSES[2:]})

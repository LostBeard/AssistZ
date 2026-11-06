"""
What is inside each rumble? The game's artillery and crash samples are searched in +-20 s around the 8 rumbles and in 5 control windows, low-passed at 4 kHz. Control within the method: the same samples time-reversed (same spectrum, cannot truly match).
Usage: python world_windows.py <samples dir>
"""
import json, multiprocessing as mp, os, sys, time
import numpy as np
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from world_events import SOS, RATE, GROUPS, build

RUNS = r"D:\users\tj\Projects\AssistZ\Lab\runs"
WINDOWS = [("160950", 170, "rumble, zone 9 s later"), ("164111", 168, "rumble, zone 4 s later"), ("232232", 172, "rumble, zone 8 s later"), ("234703", 40, "rumble, zone 9 s later"), ("021109", 46, "rumble, zone 9 s later"), ("024720", 56, "rumble, zone 5 s later"),
           ("142532", 52, "rumble, NO zone"), ("143035", 98, "rumble, NO zone"),
           ("233343", 92, "control: zone at 101, no rumble"), ("190530", 116, "control: zone at 125, no rumble"), ("001133", 55, "control: zone at 64, no rumble"), ("153510", 60, "control: nothing logged"), ("231331", 60, "control: nothing logged")]
FULL = {"160950": "20261103", "164111": "20261103", "232232": "20261103", "234703": "20261103", "021109": "20261104", "024720": "20261104", "142532": "20261103", "143035": "20261103", "233343": "20261103", "190530": "20261103", "001133": "20261104", "153510": "20261103", "231331": "20261103"}
_T = {}


def init(out):
    _T["t"] = build(out)


def scan(w):
    name, center, note = w
    run = rf"{RUNS}\sniper_{FULL[name]}_{name}"
    a = np.load(run + "/audio.npy")
    lo, hi = max(0, (center - 20) * 48000), (center + 20) * 48000
    seg = a[lo:hi].astype(np.float64)
    sig = [resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32) for c in range(seg.shape[1])]
    res = {}
    for group, base, pitch, y, rev in _T["t"]:
        for s in sig:
            if len(y) >= len(s):
                continue
            cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            for key, tpl in ((group, y), (group + "|rev", rev)):
                ncc = np.abs(fftconvolve(s, tpl[::-1], mode="valid")) / e
                ncc[e < 0.002] = 0
                k = int(np.argmax(ncc))
                if ncc[k] > res.get(key, (0,))[0]:
                    res[key] = (float(ncc[k]), lo / 48000 + k / RATE, base, pitch)
    return name, center, note, res


if __name__ == "__main__":
    out = os.path.join(sys.argv[1], "world")
    t0 = time.time()
    rows = []
    with mp.Pool(4, initializer=init, initargs=(out,)) as pool:
        for n, r in enumerate(pool.imap_unordered(scan, WINDOWS), 1):
            rows.append(r)
            print(f"  {n}/{len(WINDOWS)} at {time.time() - t0:.0f} s", flush=True)
    order = {w[0]: i for i, w in enumerate(WINDOWS)}
    rows.sort(key=lambda r: order[r[0]])
    json.dump(rows, open(os.path.join(sys.argv[1], "world_windows.json"), "w"))
    rev = [v[0] for r in rows for k, v in r[3].items() if k.endswith("|rev")]
    print(f"\nreversed-sample control over all windows: median {np.median(rev):.3f}, max {max(rev):.3f}")
    groups = list(GROUPS)
    print(f"{'recording':<8} {'at s':>5}  " + " ".join(f"{g[-12:]:>12}" for g in groups) + "   note")
    for name, center, note, res in rows:
        print(f"{name:<8} {center:>5}  " + " ".join(f"{res.get(g, (0,))[0]:>12.3f}" for g in groups) + f"   {note}")
    print("\nBest forward match in each window, and its reversed control for that group:")
    for name, center, note, res in rows:
        g = max(groups, key=lambda x: res.get(x, (0,))[0])
        v = res[g]
        print(f"  {name} {note:<32} {g:<18} {v[0]:.3f} at {v[1]:6.1f} s  {v[2]} x{v[3]:.3f}; reversed {res.get(g + '|rev', (0,))[0]:.3f}")

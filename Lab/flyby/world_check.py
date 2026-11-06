"""
Did the triggered world events reach the listener? For each logged event: level rise, the best sample match per sound group in the next 45 s, and the CLAP words and probe margin of the detector's events there.
Usage: python world_check.py <samples dir> <tag> <run> -- <train run> [...]
"""
import csv, datetime, json, multiprocessing as mp, os, sys
import numpy as np
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from world_events import SOS, RATE, GROUPS, build
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports

_T = {}


def init(out):
    _T["t"] = build(out)


def clock(run):
    meta = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    offset = np.median([sum(int(v) * k for v, k in zip(r["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(r["game_ms"]) / 1000 for r in pose])
    return lambda ms: ms / 1000 + offset - start


def scan(job):
    run, name, dist, t = job
    a = np.load(run + "/audio.npy", mmap_mode=None)
    lo, hi = max(0, int((t - 2) * 48000)), int((t + 45) * 48000)
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
                    res[key] = (float(ncc[k]), (lo / 48000 + k / RATE) - t, base, pitch)
    mono = seg.mean(axis=1)
    before = mono[: 2 * 48000]
    per = [20 * np.log10(np.sqrt((mono[i * 48000: (i + 1) * 48000] ** 2).mean()) + 1e-9) for i in range(0, len(mono) // 48000)]
    return name, dist, t, res, per


if __name__ == "__main__":
    samples, tag = sys.argv[1], sys.argv[2]
    run = sys.argv[3]
    train = sys.argv[5:]
    c = clock(run)
    events = list(csv.DictReader(open(run + "/world.csv")))
    jobs = [(run, e["event"], float(e["distance"]), c(int(e["game_ms"]))) for e in events]
    print("recording seconds:", [round(j[3]) for j in jobs], "of", json.load(open(run + "/audio.json"))["frames"] // 48000)
    out = os.path.join(samples, "world")
    with mp.Pool(4, initializer=init, initargs=(out,)) as pool:
        rows = pool.map(scan, jobs)
    groups = list(GROUPS)
    print(f"\n{'event':<7} {'m':>5} {'at s':>5} {'peak dB':>7} {'rise s':>6}  " + " ".join(f"{g[-11:]:>11}" for g in groups))
    for name, dist, t, res, per in rows:
        base = np.median(per[:2])
        peak = int(np.argmax(per))
        print(f"{name:<7} {dist:>5.0f} {t:>5.0f} {per[peak]:>7.1f} {peak - 2:>6d}  " + " ".join(f"{res.get(g, (0,))[0]:>11.3f}" for g in groups))
    print("\nBest match per event, with its time after the trigger, and the reversed control for that group:")
    for name, dist, t, res, per in rows:
        g = max(groups, key=lambda x: res.get(x, (0,))[0])
        v = res[g]
        print(f"  {name:<7} {dist:>5.0f} m: {g:<18} {v[0]:.3f} at +{v[1]:5.1f} s  {v[2]} x{v[3]:.3f}; reversed {res.get(g + '|rev', (0,))[0]:.3f}; level by 5 s from the trigger {[int(np.mean(per[i:i + 5])) for i in range(2, len(per) - 4, 5)]}")
    if os.path.exists(f"{run}/replay{tag}_s0.jsonl"):
        def fit_rows(r):
            reports, _ = calibrated_reports(r)
            o = []
            for line in open(f"{r}/replay{tag}_s0.jsonl", encoding="utf-8"):
                e = json.loads(line)
                s = e["sample"] / 48000.0
                if any(-0.05 <= s - x[0] <= 0.15 for x in reports):
                    o.append((np.array(e["embed"]), 1.0))
                elif all(abs(s - x[0]) > 1.5 for x in reports):
                    o.append((np.array(e["embed"]), 0.0))
            return o
        d = [x for r in train for x in fit_rows(r)]
        X = np.stack([x[0] for x in d])
        y = np.array([x[1] for x in d])
        w = np.where(y == 1, 0.5 / y.sum(), 0.5 / (1 - y).sum())
        a, b = np.zeros(X.shape[1]), 0.0
        for _ in range(800):
            p = 1 / (1 + np.exp(-(X @ a + b)))
            a -= 0.5 * (X.T @ (w * (p - y)) + 1.0 / len(y) * a)
            b -= 0.5 * np.sum(w * (p - y))
        ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
        print("\nDetector events within 45 s after each trigger: count, CLAP words, prompts gunfire, probe gunfire (trained without this run)")
        for name, dist, t, res, per in rows:
            g = [e for e in ev if t <= e["sample"] / 48000.0 <= t + 45]
            words = {}
            for e in g:
                words[e["clap"]] = words.get(e["clap"], 0) + 1
            print(f"  {name:<7} {dist:>5.0f} m: {len(g):3d} events  {dict(sorted(words.items(), key=lambda x: -x[1])[:5])}; prompts {sum(e['label'] in ('gunshot', 'automatic fire') for e in g)}; probe {sum(float(np.array(e['embed']) @ a + b) > 0 for e in g)}")
        print(f"  whole recording: {len(ev)} events; prompts gunfire {sum(e['label'] in ('gunshot', 'automatic fire') for e in ev)}; probe gunfire {sum(float(np.array(e['embed']) @ a + b) > 0 for e in ev)}")

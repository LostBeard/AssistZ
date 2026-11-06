"""
Uses the crack matches of flyby_match2.py (exact sample and pitch per shot). For every clear match: when the crack arrives after the fire, how loud it is, and what those follow: pass distance or
shooter distance. Then the crack is subtracted and the flyby samples are searched again in the remainder, at pitches near 1 (0.85-1.18, the crack's own config range is narrower), in case
the crack was masking them. Control: the same search before the fire.
Usage: python crack_locate.py <samples dir> <run> [...]
"""
import glob, json, multiprocessing as mp, os, sys
import numpy as np
import soundfile as sf
from scipy.stats import mannwhitneyu, spearmanr
from flyby_match2 import RATE, SOS, jobs, best
from scipy.signal import sosfiltfilt

MATCH = 0.35
PITCHES = np.exp(np.arange(np.log(0.85), np.log(1.18), 0.003))
_F = {}


def crack_template(samples, sample, pitch):
    x, rate = sf.read(os.path.join(samples, sample), always_2d=True)
    x = x.mean(axis=1).astype(np.float64)
    n = int(round(len(x) * RATE / rate / pitch))
    y = sosfiltfilt(SOS, np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x))
    k = int(np.argmax(np.convolve(y ** 2, np.ones(480), "same")))
    a = max(0, k - int(0.04 * RATE))
    return y[a: a + int(0.15 * RATE)], a


def init(samples):
    out = []
    for path in sorted(glob.glob(os.path.join(samples, "bullet_flyby_*.ogg"))):
        x, rate = sf.read(path, always_2d=True)
        x = x.mean(axis=1).astype(np.float64)
        for pitch in PITCHES:
            n = int(round(len(x) * RATE / rate / pitch))
            y = sosfiltfilt(SOS, np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x))
            k = int(np.argmax(np.convolve(y ** 2, np.ones(480), "same")))
            a = max(0, k - int(0.04 * RATE))
            y = y[a: a + int(0.15 * RATE)]
            nrm = np.linalg.norm(y)
            if nrm > 0:
                out.append((os.path.basename(path), float(pitch), (y / nrm).astype(np.float32)))
    _F["t"] = out


def search(job):
    channels, windows, meta = job
    res = dict(meta)
    for key, w in (("fly", "search"), ("ctrl", "control")):
        top = (0.0, "", 0.0, 0)
        for ch in channels:
            got = best(ch[windows[w][0]: windows[w][1]], _F["t"])
            if got[0] > top[0]:
                top = got
        res[key], res[key + "_sample"], res[key + "_pitch"] = top[:3]
    return res


if __name__ == "__main__":
    samples, runs = sys.argv[1], sys.argv[2:]
    lab = r"D:\users\tj\Projects\AssistZ\AssistZ\Lab"
    found = {(r["run"], r["i"]): r for r in json.load(open(os.path.join(samples, "flyby_match2.json")))}
    all_jobs = [j for run in runs for j in jobs(run, lab)]
    rows, residual_jobs = [], []
    for chans, wins, meta in all_jobs:
        r = found[(meta["run"], meta["i"])]
        chans = [c.astype(np.float64) for c in chans]
        y, a = crack_template(samples, r["crack_sample"], r["crack_pitch"])
        nrm2 = float((y ** 2).sum())
        k0, k1 = wins["crack"]
        # The channel where this template, at the found position, explains the most.
        pos = k0 + r["crack_k"]
        projections = []
        for c in chans:
            seg = c[pos: pos + len(y)]
            projections.append(float((seg * y[: len(seg)]).sum()) / nrm2 if len(seg) == len(y) else 0.0)
        matched = r["crack"] >= MATCH and 0.85 <= r["crack_pitch"] <= 1.18
        if matched:
            # Crack start = where sample position 0 would be: the template starts `a` samples into the file. Relative to the fire.
            start_ms = 1000 * (r["crack_k"] - a) / RATE
            rows.append(dict(run=meta["run"], i=meta["i"], passed=meta["passed"], d=meta["d"], arrival=meta["arrival_ms"], t_ms=start_ms, ncc=r["crack"], pitch=r["crack_pitch"],
                             gain_l=20 * np.log10(abs(projections[0]) + 1e-9), gain_r=20 * np.log10(abs(projections[1]) + 1e-9)))
            for n, c in enumerate(chans):
                seg_len = len(c[pos: pos + len(y)])
                c[pos: pos + seg_len] -= projections[n] * y[:seg_len]
        residual_jobs.append(([c.astype(np.float32) for c in chans], wins, meta))
    print(f"{len(rows)} of {len(all_jobs)} shots have a clear crack match (correlation >= {MATCH}, pitch 0.85-1.18)")
    pitch = np.array([r["pitch"] for r in rows])
    print(f"crack pitch: median {np.median(pitch):.3f}, 5-95% {np.percentile(pitch, 5):.3f}..{np.percentile(pitch, 95):.3f}")
    t = np.array([r["t_ms"] for r in rows])
    arr = np.array([r["arrival"] for r in rows])
    p = np.array([r["passed"] for r in rows])
    d = np.array([r["d"] for r in rows])
    gain = np.maximum([r["gain_l"] for r in rows], [r["gain_r"] for r in rows])
    print(f"\nWHEN (crack start after the fire): median {np.median(t):.1f} ms, spread 5-95% {np.percentile(t, 5):.1f}..{np.percentile(t, 95):.1f}")
    print(f"  crack start minus the bullet's arrival at the closest point: median {np.median(t - arr):+.1f} ms, spread {np.percentile(t - arr, 5):+.1f}..{np.percentile(t - arr, 95):+.1f}")
    print(f"  Spearman, crack time vs: arrival {spearmanr(t, arr).statistic:+.3f}; shooter distance {spearmanr(t, d).statistic:+.3f}; pass distance {spearmanr(t, p).statistic:+.3f}")
    print("  least squares ms = a + b * shooter m + c * pass m:", np.round(np.linalg.lstsq(np.c_[np.ones(len(t)), d, p], t, rcond=None)[0], 3).tolist(), "(1 / 785 m/s = 1.27 ms per m; 1 / 330 m/s = 3.03 ms per m)")
    print("\nHOW LOUD (crack gain in the recording, dB, louder channel, relative to the file's level)")
    print(f"  Spearman, gain vs pass distance {spearmanr(gain, p).statistic:+.3f}; vs shooter distance {spearmanr(gain, d).statistic:+.3f}")
    m = np.c_[np.ones(len(gain)), np.log10(np.maximum(p, 0.5)), np.log10(d)]
    sol = np.linalg.lstsq(m, gain, rcond=None)[0]
    print(f"  least squares dB = {sol[0]:.1f} {sol[1]:+.1f} log10(pass) {sol[2]:+.1f} log10(shooter); inverse-distance would be -20 per decade; residual sd {np.std(gain - m @ sol):.1f} dB")
    for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 25), (25, 1e9)):
        g = gain[(p >= lo) & (p < hi)]
        if len(g):
            print(f"    pass {lo}-{hi if hi < 1e9 else 'inf'} m: n={len(g):3d}, gain median {np.median(g):6.1f} dB")
    print(f"\nFlyby search after removing the crack, pitch {PITCHES[0]:.2f}-{PITCHES[-1]:.2f}, {len(residual_jobs)} shots:")
    res = []
    with mp.Pool(4, initializer=init, initargs=(samples,)) as pool:
        for r in pool.imap_unordered(search, residual_jobs):
            res.append(r)
    print(f"  {'pass m':<8} {'n':>3}  {'flyby at arrival':>18}  {'before fire':>18}  above control max")
    limit = max(r["ctrl"] for r in res)
    for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 35), (35, 1e9)):
        g = [r for r in res if lo <= r["passed"] < hi]
        if g:
            print(f"  {str(lo) + '-' + (str(hi) if hi < 1e9 else 'inf'):<8} {len(g):>3}  {np.median([r['fly'] for r in g]):>18.3f}  {np.median([r['ctrl'] for r in g]):>18.3f}  {sum(r['fly'] > limit for r in g)}")
    print(f"  control max {limit:.3f}. Above it:")
    for r in sorted([r for r in res if r["fly"] > limit], key=lambda x: -x["fly"])[:15]:
        print(f"    {r['run']} shot {r['i']:2d}: passed {r['passed']:4.1f} m, shooter {r['d']:.0f} m, flyby {r['fly']:.3f} {r['fly_sample'][7:14]} x{r['fly_pitch']:.3f}; control {r['ctrl']:.3f}")
    c = [r["fly"] for r in res if r["passed"] < 5]
    f = [r["fly"] for r in res if r["passed"] >= 15]
    print(f"  under 5 m vs 15 m or more: {np.median(c):.3f} vs {np.median(f):.3f}, p = {mannwhitneyu(c, f, alternative='greater').pvalue:.3f}")

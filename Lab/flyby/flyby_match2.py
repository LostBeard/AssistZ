"""
Second search for the game's bullet_flyby samples in the recordings: pitch steps of 0.3% (high-band correlation dies at a pitch error near 1%), and short windows so most SVD shots
qualify. Search window: arrival -60..+220 ms. Control: the same length before the fire (-300..-20 ms), and a shot only counts when no other report falls in -350..+350 ms around its fire.
The crack is searched too, in the window from the fire to arrival +220 ms. Samples are read locally from the player's install and are never committed.
Usage: python flyby_match2.py <samples dir> <run> [...]
"""
import csv, glob, json, multiprocessing as mp, os, sys, time
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfiltfilt
from scipy.stats import mannwhitneyu, spearmanr

RATE = 48000
V = 785.0
SOS = butter(6, 6000, btype="highpass", fs=RATE, output="sos")
PITCHES = np.exp(np.arange(np.log(0.6), np.log(1.6), 0.003))
_T = {}


def build(samples, pattern, keep_ms=150):
    out = []
    for path in sorted(glob.glob(os.path.join(samples, pattern))):
        x, rate = sf.read(path, always_2d=True)
        x = x.mean(axis=1).astype(np.float64)
        for pitch in PITCHES:
            n = int(round(len(x) * RATE / rate / pitch))
            y = sosfiltfilt(SOS, np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x))
            k = int(np.argmax(np.convolve(y ** 2, np.ones(480), "same")))
            a = max(0, k - int(0.04 * RATE))
            y = y[a: a + int(keep_ms / 1000 * RATE)]
            nrm = np.linalg.norm(y)
            if nrm > 0:
                out.append((os.path.basename(path), float(pitch), (y / nrm).astype(np.float32)))
    return out


def init(samples):
    _T["fly"] = build(samples, "bullet_flyby_*.ogg")
    _T["crack"] = build(samples, "sonicCrack_Close_[0-9]*.ogg")


def best(signal, tmpl):
    top = (0.0, "", 0.0, 0)
    cum = np.concatenate(([0.0], np.cumsum(signal.astype(np.float64) ** 2)))
    for name, pitch, y in tmpl:
        if len(y) >= len(signal):
            continue
        c = fftconvolve(signal, y[::-1], mode="valid")
        ncc = np.abs(c) / (np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-9)
        k = int(np.argmax(ncc))
        if ncc[k] > top[0]:
            top = (float(ncc[k]), name, pitch, k)
    return top


def work(job):
    channels, windows, meta = job
    res = dict(meta)
    for key, tmpl, w in (("fly", "fly", "search"), ("ctrl", "fly", "control"), ("crack", "crack", "crack")):
        top = (0.0, "", 0.0, 0)
        for ch in channels:
            got = best(ch[windows[w][0]: windows[w][1]], _T[tmpl])
            if got[0] > top[0]:
                top = got
        res[key], res[key + "_sample"], res[key + "_pitch"], res[key + "_k"] = top
    res["fly_ms"] = 1000 * res["fly_k"] / RATE - 60
    return res


def jobs(run, lab):
    sys.path.insert(0, lab)
    from lab_clock import calibrated_reports, envelope_db, HOP
    DT = HOP / RATE
    pcm = np.load(run + "/audio.npy").astype(np.float64)
    env = envelope_db(pcm)
    reports, shift = calibrated_reports(run)
    client = list(csv.DictReader(open(run + "/aishots_client.csv")))
    impacts = list(csv.DictReader(open(run + "/impacts_client.csv")))
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    head = np.array([float(pose[-1]["x"]), float(pose[-1]["y"]) + 1.6, float(pose[-1]["z"])])
    owner = {}
    for x in impacts:
        t = int(x["game_ms"])
        before = [j for j, c in enumerate(client) if int(c["game_ms"]) <= t <= int(c["game_ms"]) + 2000]
        if before:
            owner.setdefault(max(before, key=lambda j: int(client[j]["game_ms"])), x)
    dues = [r[0] for r in reports]
    filt = [sosfiltfilt(SOS, ch).astype(np.float32) for ch in (pcm[:, 0], pcm[:, 1])]
    out = []
    for i, r in enumerate(reports):
        x = owner.get(i)
        due, d = r[0], r[3]
        a, b = int((due - 0.06) / DT), int((due + 0.06) / DT)
        if x is None or a < 0 or b >= len(env):
            continue
        peak = a + int(np.argmax(env[a:b]))
        onset = peak
        while onset > peak - int(0.06 / DT) and env[onset - 1] > env[peak] - 6:
            onset -= 1
        fire = onset * DT - d / 330.0
        if fire < 1 or any(j != i and abs(dues[j] - (fire + 0.0)) < 0.35 for j in range(len(dues))):
            continue
        m = np.array([float(client[i][k]) for k in ("mx", "my", "mz")])
        q = np.array([float(x[k]) for k in ("x", "y", "z")])
        seg = q - m
        t = np.clip(np.dot(head - m, seg) / np.dot(seg, seg), 0, 1)
        closest = m + t * seg
        arrival = float(np.linalg.norm(closest - m)) / V
        center = int((fire + arrival) * RATE)
        s0, s1 = center - int(0.06 * RATE), center + int(0.22 * RATE)
        c1 = int((fire - 0.02) * RATE)
        c0 = c1 - (s1 - s0)
        k0, k1 = int(fire * RATE), s1
        lo, hi = min(c0, k0), max(s1, k1)
        chans = [f[lo:hi] for f in filt]
        out.append((chans, dict(search=(s0 - lo, s1 - lo), control=(c0 - lo, c1 - lo), crack=(k0 - lo, k1 - lo)),
                    dict(run=os.path.basename(run.rstrip("/\\"))[-15:], i=i, passed=float(np.linalg.norm(closest - head)), d=d, landed=float(np.linalg.norm(q - head)), arrival_ms=arrival * 1000)))
    print(f"{os.path.basename(run.rstrip('/'))}: clock {shift * 1000:+.0f} ms; {len(out)} shots qualify", flush=True)
    return out


if __name__ == "__main__":
    samples, runs = sys.argv[1], sys.argv[2:]
    lab = r"D:\users\tj\Projects\AssistZ\AssistZ\Lab"
    all_jobs = [j for run in runs for j in jobs(run, lab)]
    print(f"{len(all_jobs)} shots, {len(PITCHES)} pitches, 4 workers", flush=True)
    t0 = time.time()
    rows = []
    with mp.Pool(4, initializer=init, initargs=(samples,)) as pool:
        for n, r in enumerate(pool.imap_unordered(work, all_jobs), 1):
            rows.append(r)
            if n % 20 == 0:
                print(f"  {n}/{len(all_jobs)} in {time.time() - t0:.0f} s", flush=True)
    json.dump(rows, open(os.path.join(samples, "flyby_match2.json"), "w"))
    print(f"\n{len(rows)} shots. Best normalized correlation, median (share above 0.3):")
    print(f"  {'pass m':<8} {'n':>3}  {'flyby at arrival':>18}  {'flyby before fire':>18}  {'crack':>18}")
    for lo, hi in ((0, 3), (3, 5), (5, 8), (8, 12), (12, 18), (18, 25), (25, 35), (35, 1e9)):
        g = [r for r in rows if lo <= r["passed"] < hi]
        if g:
            cells = [f"{np.median([r[k] for r in g]):.3f} ({np.mean([r[k] > 0.3 for r in g]) * 100:3.0f}%)" for k in ("fly", "ctrl", "crack")]
            print(f"  {str(lo) + '-' + (str(hi) if hi < 1e9 else 'inf'):<8} {len(g):>3}  {cells[0]:>18}  {cells[1]:>18}  {cells[2]:>18}")
    close = [r["fly"] for r in rows if r["passed"] < 5]
    far = [r["fly"] for r in rows if r["passed"] >= 15]
    print(f"\nflyby correlation, under 5 m vs 15 m or more: {np.median(close):.3f} vs {np.median(far):.3f}, p = {mannwhitneyu(close, far, alternative='greater').pvalue:.2e}, AUC {np.mean([(a > b) + 0.5 * (a == b) for a in close for b in far]):.3f}")
    print(f"control on the same shots, flyby before the fire: {np.median([r['ctrl'] for r in rows]):.3f}; 99th percentile {np.percentile([r['ctrl'] for r in rows], 99):.3f}, max {max(r['ctrl'] for r in rows):.3f}")
    print(f"Spearman, pass distance vs flyby correlation {spearmanr([r['passed'] for r in rows], [r['fly'] for r in rows]).statistic:.3f}")
    print("\nClose passes, strongest first (sample, pitch, ms from the arrival):")
    for r in sorted([r for r in rows if r["passed"] < 5], key=lambda x: -x["fly"])[:30]:
        print(f"  {r['run']} shot {r['i']:2d}: passed {r['passed']:4.1f} m, landed {r['landed']:6.1f} m, flyby {r['fly']:.3f} {r['fly_sample'][7:14]} x{r['fly_pitch']:.3f} at {r['fly_ms']:+5.0f} ms; control {r['ctrl']:.3f}; crack {r['crack']:.3f} {r['crack_sample'][11:19]} x{r['crack_pitch']:.3f}")
    print("\nFar passes (>= 15 m), strongest flyby matches:")
    for r in sorted([r for r in rows if r["passed"] >= 15], key=lambda x: -x["fly"])[:8]:
        print(f"  {r['run']} shot {r['i']:2d}: passed {r['passed']:4.1f} m, flyby {r['fly']:.3f} {r['fly_sample'][7:14]} x{r['fly_pitch']:.3f} at {r['fly_ms']:+5.0f} ms; control {r['ctrl']:.3f}; crack {r['crack']:.3f}")

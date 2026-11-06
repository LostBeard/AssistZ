"""Two questions. (1) Does a thunder sample carry a second clap about 10 s in, and do the whole samples match a main burst plus its follow-up? (2) What is the 143 s of 215 events in 085810's last phase? Read-only."""
import csv, datetime, glob, json, multiprocessing as mp, os, sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, find_peaks, resample_poly, sosfiltfilt
from world_events import SOS, RATE

U = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
samples = sys.argv[1]
PAIRS = [("20261104_084227", 147.6), ("20261104_084227", 213.1), ("20261104_084227", 445.4), ("20261104_084227", 495.1), ("20261104_085810", 147.3), ("20261104_085810", 299.0), ("20261104_085810", 341.4)]
SINGLE = [("20261104_084227", 339.0), ("20261104_085810", 499.0), ("20261104_085810", 563.4)]
CONTROL = [("20261104_084227", 30.0), ("20261104_085810", 30.0), ("20261104_084227", 720.0)]


def load_templates():
    out = []
    for p in sorted(glob.glob(os.path.join(samples, "thunder", "*.ogg"))):
        x, r = sf.read(p, always_2d=True)
        y = resample_poly(x.mean(axis=1), RATE, r)
        for pitch in (0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06):
            n = int(round(len(y) / pitch))
            z = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
            out.append((os.path.basename(p)[7:-4], pitch, z / np.linalg.norm(z)))
    return out


def job(arg):
    name, t, tmpl = arg
    a = np.load(U + name + "/audio.npy", mmap_mode="r")
    seg = np.array(a[int(max(0, t - 8) * 48000): int((t + 30) * 48000)], dtype=np.float64)
    top = (0.0, 0.0, "", 1.0)
    for c in range(seg.shape[1]):
        s = resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32)
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for base, pitch, y in tmpl:
            if len(y) >= len(s):
                continue
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / e
            ncc[e < 0.0005] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > top[0]:
                top = (float(ncc[k]), max(0, t - 8) + k / RATE, base, pitch)
    return name, t, top


if __name__ == "__main__":
    print("(1) Envelope peaks of each thunder sample (1 s RMS, at least 25% of the loudest, 2 s apart): second, relative level")
    for p in sorted(glob.glob(os.path.join(samples, "thunder", "*.ogg"))):
        x, r = sf.read(p, always_2d=True)
        m = x.mean(axis=1)
        w = int(r * 0.5)
        env = np.sqrt(np.convolve(m ** 2, np.ones(w) / w, "same"))[::w // 5]
        rate = r / (w // 5)
        pk, _ = find_peaks(env, height=env.max() * 0.25, distance=int(2 * rate))
        print(f"  {os.path.basename(p)[7:-4]:<14} {len(m) / r:5.1f} s  " + ", ".join(f"{i / rate:.1f}s {20 * np.log10(env[i] / env.max()):.0f}dB" for i in pk))
    tmpl = load_templates()
    with mp.Pool(4) as pool:
        got = pool.map(job, [(n, t, tmpl) for n, t in PAIRS + SINGLE + CONTROL])
    print("\n(1) Whole-sample match, 7 pitches, window -8..+30 s (band as before)")
    for (n, t, top), kind in zip(got, ["burst + follow-up"] * len(PAIRS) + ["burst alone"] * len(SINGLE) + ["control"] * len(CONTROL)):
        print(f"  {kind:<18} {n[-6:]} {t:6.1f} s: {top[0]:.3f} {top[2]:<14} pitch {top[3]:.2f}, sample starts {top[1] - t:+5.1f} s from the burst")

    print("\n(2) 085810 from 640 s on")
    run = U + "20261104_085810"
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s0 = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    clock = lambda g: int(g) / 1000 + off - s0
    env = list(csv.DictReader(open(run + "/env.csv")))
    print("  env columns:", list(env[0].keys()))
    for e in env[::20]:
        if clock(e["game_ms"]) > 620:
            print("   ", round(clock(e["game_ms"])), {k: e[k] for k in e if k != "game_ms"})
    ps = [(clock(p["game_ms"]), float(p.get("x", 0)), float(p.get("z", 0))) for p in pose]
    print("  pose columns:", list(pose[0].keys())[:14])
    print("  pose at 640, 660, 700, 790 s:", [next(q for q in ps if q[0] >= tt) for tt in (640, 660, 700, 790)])
    ev = sorted((json.loads(x) for x in open(run + "/replay_vote_s0.jsonl", encoding="utf-8")), key=lambda e: e["sample"])
    late = [e for e in ev if e["sample"] / 48000 >= 655]
    ts = np.array([e["sample"] / 48000 for e in late])
    print(f"  {len(late)} events 655 s on; gaps s: median {np.median(np.diff(ts)):.2f}, p10 {np.percentile(np.diff(ts), 10):.2f}, p90 {np.percentile(np.diff(ts), 90):.2f}; level dB median {np.median([e['level'] for e in late]):.1f}; az median {np.median([e['az'] for e in late]):.1f}, sd {np.std([e['az'] for e in late]):.1f}")
    print("  first 25:", [(round(e["sample"] / 48000, 2), e["label"][:6], round(e["level"])) for e in late[:25]])
    a = np.load(run + "/audio.npy", mmap_mode="r")
    for t0 in (640, 650, 654, 656, 660, 700, 780):
        seg = np.array(a[int(t0 * 48000): int((t0 + 2) * 48000)], dtype=np.float64)
        mono = seg.mean(axis=1)
        sp = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
        f = np.fft.rfftfreq(len(mono), 1 / 48000)
        bands = [(0, 200), (200, 1000), (1000, 4000), (4000, 12000), (12000, 24000)]
        print(f"  {t0:4d} s level {10 * np.log10((mono ** 2).mean() + 1e-12):6.1f} dB, L-R corr {np.corrcoef(seg[:, 0], seg[:, 1])[0, 1]:.2f}, band % " + " ".join(f"{100 * sp[(f >= lo) & (f < hi)].sum() / sp.sum():.0f}" for lo, hi in bands))
    print("  Same check on 084227 (same phase, same settings, 3 events):")
    a2 = np.load(U + "20261104_084227/audio.npy", mmap_mode="r")
    for t0 in (650, 660, 700):
        mono = np.array(a2[int(t0 * 48000): int((t0 + 2) * 48000)], dtype=np.float64).mean(axis=1)
        sp = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) ** 2
        f = np.fft.rfftfreq(len(mono), 1 / 48000)
        print(f"  {t0:4d} s level {10 * np.log10((mono ** 2).mean() + 1e-12):6.1f} dB, band % " + " ".join(f"{100 * sp[(f >= lo) & (f < hi)].sum() / sp.sum():.0f}" for lo, hi in [(0, 200), (200, 1000), (1000, 4000), (4000, 12000), (12000, 24000)]))

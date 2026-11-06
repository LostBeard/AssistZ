"""What is a forced storm on Nasdara? Weather log, burst spacing, and the game's thunder and sand storm samples against the bursts. Read-only; samples stay local."""
import csv, datetime, glob, io, json, os, re, sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab\flyby")
from read_dayz_config import pbo_entries, lzss
from world_events import SOS, RATE

samples = sys.argv[1]
U = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_"
G = r"L:\SteamLibrary\steamapps\common\DayZ"
out = os.path.join(samples, "nasdara_weather")
os.makedirs(out, exist_ok=True)
f, e = pbo_entries(os.path.join(G, "nasdara", "addons", "sounds_nasdara.pbo"))
names = []
for n, (offset, size, mime, orig) in e.items():
    if not n.lower().endswith(".ogg") or not re.search(r"(?i)weather|sand_storm|thunder|helicrash", n):
        continue
    f.seek(offset)
    data = f.read(size)
    if mime == 0x43707273 and orig:
        data = lzss(data, orig)
    if data[:4] == b"OggS":
        p = os.path.join(out, os.path.basename(n.replace("\\", "/")))
        open(p, "wb").write(data)
        names.append(n.replace("\\", "/"))
print(len(names), "Nasdara weather sounds:", sorted({os.path.basename(x).rsplit("_", 1)[0] for x in names})[:30])


def load(paths, keep=3):
    t = []
    for p in paths:
        x, r = sf.read(p, always_2d=True)
        y = resample_poly(x.mean(axis=1), RATE, r)
        if len(y) < RATE // 4:
            continue
        k = min(len(y), keep * RATE)
        s = int(np.argmax(np.convolve(y ** 2, np.ones(k), "valid")))
        y = y[s: s + k]
        for pitch in (0.94, 0.97, 1.0, 1.03, 1.06):
            n = int(round(len(y) / pitch))
            z = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
            if np.linalg.norm(z) > 0:
                t.append((os.path.basename(p)[:30], z / np.linalg.norm(z)))
    return t


groups = {"Chernarus thunder": load(glob.glob(os.path.join(samples, "thunder", "*.ogg"))), "Nasdara weather": load(glob.glob(os.path.join(out, "*.ogg")))}


def best(a, t0, t1, tmpl):
    seg = np.array(a[int(max(0, t0) * 48000): int(t1 * 48000)], dtype=np.float64)
    top = (0.0, "")
    for c in range(seg.shape[1]):
        s = resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32)
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for name, y in tmpl:
            if len(y) >= len(s):
                continue
            en = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / en
            ncc[en < 0.0005] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > top[0]:
                top = (float(ncc[k]), name)
    return top


for name in ("20261104_170612", "20261104_171851"):
    run = U + name
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    s0 = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    world = [(int(e["game_ms"]) / 1000 + off - s0, e["event"]) for e in csv.DictReader(open(run + "/world.csv"))]
    t0 = next(t for t, n in world if n == "storm")
    t1 = next(t for t, n in world if n == "clear_after")
    env = [(int(e["game_ms"]) / 1000 + off - s0, e) for e in csv.DictReader(open(run + "/env.csv"))]
    a = np.load(run + "/audio.npy", mmap_mode="r")
    during = [e for t, e in env if t0 + 20 <= t < t1]
    print(f"\n{name}: storm {t0:.0f}-{t1:.0f} s; overcast {during[-1]['overcast']}, rain {max(float(e['rain']) for e in during)}, wind {during[-1]['wind']}")
    ev = sorted((json.loads(x) for x in open(f"{run}/replay_nas_s0.jsonl", encoding="utf-8")), key=lambda e: e["sample"])
    runs = []
    for e in ev:
        t = e["sample"] / 48000
        if not t0 <= t < t1:
            continue
        if runs and t - runs[-1][-1]["sample"] / 48000 < 0.5:
            runs[-1].append(e)
        else:
            runs.append([e])
    bursts = [r for r in runs if len(r) >= 3]
    st_times = [r[0]["sample"] / 48000 for r in bursts]
    print(f"  {len(bursts)} bursts; gaps {[round(float(x), 1) for x in np.diff(st_times)]} (setting 15); singles {len(runs) - len(bursts)}")
    print("  burst: start, n, loudest dB, labels; best Chernarus thunder; best Nasdara weather")
    for r in bursts[:5]:
        t = r[0]["sample"] / 48000
        c = {}
        for e in r:
            c[e["label"]] = c.get(e["label"], 0) + 1
        th = best(a, t - 2, t + 6, groups["Chernarus thunder"])
        na = best(a, t - 2, t + 6, groups["Nasdara weather"])
        print(f"   {t:6.1f} {len(r):2d} {max(e['level'] for e in r):6.1f} {c}  {th[0]:.3f} {th[1][:20]}  {na[0]:.3f} {na[1][:22]}")
    for lo, hi, note in ((t0 - 40, t0 - 20, "control, before the storm"),):
        print(f"  {note}: thunder {best(a, lo, hi, groups['Chernarus thunder'])[0]:.3f}, Nasdara weather {best(a, lo, hi, groups['Nasdara weather'])[0]:.3f}")
    seg = np.array(a[int(t0 * 48000) + 480000: int(t0 * 48000) + 960000], dtype=np.float64).mean(axis=1)
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    fr = np.fft.rfftfreq(len(seg), 1 / 48000)
    print("  10 s of storm, band % 0-200, 200-1k, 1-4k, 4-12k, 12k+:", [round(100 * float(sp[(fr >= x) & (fr < y)].sum() / sp.sum()), 1) for x, y in ((0, 200), (200, 1000), (1000, 4000), (4000, 12000), (12000, 24001))])
ch = np.load(U + "20261104_110232/audio.npy", mmap_mode="r")
seg = np.array(ch[100 * 48000: 110 * 48000], dtype=np.float64).mean(axis=1)
sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
fr = np.fft.rfftfreq(len(seg), 1 / 48000)
print("\nChernarus storm 110232 for comparison, band %:", [round(100 * float(sp[(fr >= x) & (fr < y)].sum() / sp.sum()), 1) for x, y in ((0, 200), (200, 1000), (1000, 4000), (4000, 12000), (12000, 24001))])

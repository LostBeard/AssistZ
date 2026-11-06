"""TJ by ear: the 121745 clip is a crow, plus a second bird. Do the game's bird samples match? Read-only; samples stay local, never committed. Band 500-6000 Hz, where the sound is."""
import io, os, re, sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt
from read_dayz_config import pbo_entries, lzss

ADDONS = r"L:\SteamLibrary\steamapps\common\DayZ\Addons"
RATE = 16000
SOS = butter(4, [500, 6000], "bandpass", fs=RATE, output="sos")
RUN = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261104_121745"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
tmpl, names = [], {}
for pbo in sorted(os.listdir(ADDONS)):
    if not re.match(r"(?i)sounds_", pbo) or not pbo.lower().endswith(".pbo"):
        continue
    f, entries = pbo_entries(os.path.join(ADDONS, pbo))
    for n, (offset, size, mime, orig) in entries.items():
        if not re.search(r"(?i)crow|raven|rook|magpie|jackdaw|\bcaw", n) or not n.lower().endswith(".ogg"):
            continue
        f.seek(offset)
        data = f.read(size)
        if mime == 0x43707273 and orig:
            data = lzss(data, orig)
        if data[:4] != b"OggS":
            continue
        base = pbo[:-4] + "__" + os.path.basename(n.replace("\\", "/"))
        open(os.path.join(out, base), "wb").write(data)
        x, r = sf.read(io.BytesIO(data), always_2d=True)
        y = resample_poly(x.mean(axis=1), RATE, r)
        names[base] = len(y) / RATE
        if len(y) < 0.1 * RATE or len(y) > 12 * RATE:
            continue
        y = sosfiltfilt(SOS, y)
        for pitch in (0.9, 0.95, 1.0, 1.05, 1.1):
            k = int(round(len(y) / pitch))
            z = np.interp(np.linspace(0, len(y) - 1, k), np.arange(len(y)), y)
            if np.linalg.norm(z) > 0:
                tmpl.append((base, pitch, (z / np.linalg.norm(z)).astype(np.float32)))
print(len(names), "bird samples:", list(names)[:14])
a = np.load(RUN + "/audio.npy", mmap_mode="r")


def best(t0, t1):
    seg = np.array(a[int(t0 * 48000): int(t1 * 48000)], dtype=np.float64)
    top = (0.0, 0.0, "", 1.0)
    for c in range(seg.shape[1]):
        s = sosfiltfilt(SOS, resample_poly(seg[:, c], RATE, 48000)).astype(np.float32)
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for base, pitch, y in tmpl:
            if len(y) >= len(s):
                continue
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / e
            ncc[e < 1e-5] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > top[0]:
                top = (float(ncc[k]), t0 + k / RATE, base, pitch)
    return top


for t0, t1, note in ((340, 344, "first caws"), (352, 356, "second caws"), (340.5, 343.5, "first, tight"), (86, 90, "same run, another group"), (292, 295, "another group")):
    m = best(t0, t1)
    print(f"  {note:<24} {m[0]:.3f} at {m[1]:.1f} s, {m[2][:44]}, pitch {m[3]}")
for name, t0, t1, note in (("20261104_121745", 346, 350, "control: between caws, same place"), ("20261104_133615", 200, 204, "control: another site"), ("20261104_132217", 100, 104, "control: another run")):
    if name != "20261104_121745":
        a = np.load("D:/users/tj/Projects/AssistZ/Lab/runs/sniper_" + name + "/audio.npy", mmap_mode="r")
    m = best(t0, t1)
    print(f"  {note:<36} {m[0]:.3f}, {m[2][:44]}")

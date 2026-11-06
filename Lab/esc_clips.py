"""ESC-50, a public set of everyday sounds from outside the game, as clips the way bank_clips.py cuts them, standing in for sounds a mod might add. Local only (CC BY-NC), for testing, never training. Usage: python esc_clips.py <zip> <out dir>"""
import csv, io, os, sys, zipfile
import numpy as np
import soundfile as sf
import soxr
zpath, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
RATE, CLIP, PRE = 48000, 48000, 4800
z = zipfile.ZipFile(zpath)
meta = next(n for n in z.namelist() if n.endswith("meta/esc50.csv"))
rows = list(csv.DictReader(io.TextIOWrapper(z.open(meta), encoding="utf-8")))
wavs = {os.path.basename(n): n for n in z.namelist() if n.endswith(".wav")}
fh = open(os.path.join(out, "clips_000.f32"), "wb")
man = csv.writer(open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8"))
man.writerow(["id", "file", "category"])
n = 0
for r in rows:
    data, sr = sf.read(io.BytesIO(z.read(wavs[r["filename"]])), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != RATE:
        mono = soxr.resample(mono, sr, RATE).astype(np.float32)
    if np.abs(mono).max() < 1e-4:
        continue
    peak = int(np.argmax(np.abs(mono)))
    s = peak - PRE
    c = np.zeros(CLIP, dtype=np.float32)
    a, b = max(0, s), min(len(mono), s + CLIP)
    c[a - s:b - s] = mono[a:b]
    fh.write(c.tobytes())
    man.writerow([n, r["filename"], r["category"]])
    n += 1
fh.close()
print(n, "clips", len(set(r["category"] for r in rows)), "categories")

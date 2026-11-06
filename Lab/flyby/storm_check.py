"""Are the storm's events thunder, and are the 'gas strike' events after it real? Searches the game's own samples locally, read-only; nothing is committed."""
import glob, json, os, re, sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly, sosfiltfilt
from read_dayz_config import pbo_entries, lzss, load, walk
from world_events import SOS, RATE, build

samples = sys.argv[1]
run = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261104_081350"
ADDONS = r"L:\SteamLibrary\steamapps\common\DayZ\Addons"
print("Sound sets naming thunder:")
for name, rap in load(ADDONS + r"\sounds_hpp.pbo"):
    out = []
    walk(rap, 16, "", re.compile(r"(?i)thunder|lightning"), 2, out)
    for full, node in out[:12]:
        print("  ", full[-52:], "| range", node["values"].get("range"), "|", str(node["values"].get("samples", node["values"].get("soundShaders", "")))[:90])
found = {}
for pbo in ("sounds_environment.pbo", "sounds_weapons.pbo", "sounds_bliss.pbo"):
    f, entries = pbo_entries(os.path.join(ADDONS, pbo))
    hits = {n: e for n, e in entries.items() if re.search(r"(?i)thunder|lightning", n) and n.lower().endswith((".ogg", ".wss"))}
    print(f"{pbo}: {len(hits)} thunder files; e.g. {list(hits)[:3]}")
    out = os.path.join(samples, "thunder")
    os.makedirs(out, exist_ok=True)
    for n, (offset, size, mime, orig) in hits.items():
        f.seek(offset)
        data = f.read(size)
        if mime == 0x43707273 and orig:
            data = lzss(data, orig)
        if data[:4] == b"OggS":
            p = os.path.join(out, os.path.basename(n.replace("\\", "/")))
            open(p, "wb").write(data)
            found[p] = 1
print(len(found), "thunder samples kept")
a = np.load(run + "/audio.npy").astype(np.float64)


def low(t0, t1):
    seg = a[int(t0 * 48000): int(t1 * 48000)]
    return [resample_poly(sosfiltfilt(SOS, seg[:, c]), RATE, 48000).astype(np.float32) for c in range(seg.shape[1])]


def templates(paths, keep=4):
    out = []
    for p in paths:
        x, r = sf.read(p, always_2d=True)
        x = resample_poly(x.mean(axis=1), RATE, r)
        k = min(len(x), keep * RATE)
        s = int(np.argmax(np.convolve(x ** 2, np.ones(k), "valid")))
        x = x[s: s + k]
        for pitch in np.exp(np.arange(np.log(0.85), np.log(1.18), 0.01)):
            n = int(round(len(x) / pitch))
            y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
            nrm = np.linalg.norm(y)
            if nrm > 0:
                out.append((os.path.basename(p), float(pitch), y / nrm))
    return out


def best(sig, tmpl):
    top = (0.0, 0.0, "", 0.0)
    for s in sig:
        cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
        for name, pitch, y in tmpl:
            if len(y) >= len(s):
                continue
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            ncc = np.abs(fftconvolve(s, y[::-1], mode="valid")) / e
            ncc[e < 0.0005] = 0
            k = int(np.argmax(ncc))
            if ncc[k] > top[0]:
                top = (float(ncc[k]), k / RATE, name, pitch)
    return top


ev = [json.loads(x) for x in open(run + "/replay_vote_s0.jsonl", encoding="utf-8")]
close = templates(glob.glob(os.path.join(samples, "world", "Artillery_Close*.ogg")))
for t0, t1, note in ((455, 515, "the gas strike words after the storm"), (200, 260, "control: rain, nothing logged")):
    m = best(low(t0, t1), close)
    print(f"Artillery_Close {note}: {m[0]:.3f} at {t0 + m[1]:.1f} s, {m[2]}")
print("gas strike events:", [round(e["sample"] / 48000, 1) for e in ev if e["label"] == "gas strike"])
if found:
    th = templates(list(found), keep=3)
    storm = [e for e in ev if 315 <= e["sample"] / 48000 < 465]
    print("Thunder samples around the storm's event clusters (sleigh crash words marked):")
    clusters = []
    for e in sorted(storm, key=lambda x: x["sample"]):
        t = e["sample"] / 48000
        if clusters and t - clusters[-1][-1][0] < 6:
            clusters[-1].append((t, e["label"]))
        else:
            clusters.append([(t, e["label"])])
    for c in clusters[:14]:
        t0, t1 = c[0][0], c[-1][0]
        m = best(low(max(0, t0 - 3), t1 + 8), th)
        print(f"  {t0:6.1f}-{t1:6.1f} s, {len(c):2d} events {sorted({w for _, w in c})[:3]}: thunder {m[0]:.3f} {m[2][:24]}")
    m = best(low(200, 260), th)
    print(f"  control, rain phase with no events 200-260 s: thunder {m[0]:.3f}")

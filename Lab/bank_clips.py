"""
The game's own sound bank as 1 s clips, the way the app cuts them: mono, 48 kHz, peak 4800 samples in. Reads the local install's Addons, writes to the lab disk. Samples and clips stay local, never committed or shipped; only fitted weights ship. A long file gives up to 3 clips: its peak, and two more spaced through it. Needs numpy, soundfile, soxr.
Usage: python bank_clips.py <out dir>
"""
import csv, glob, io, multiprocessing, os, re, sys
import numpy as np
import soundfile as sf
import soxr
sys.path.insert(0, "D:/users/tj/Projects/AssistZ/AssistZ/Lab/flyby")
from read_dayz_config import lzss, pbo_entries
ADDONS = r"L:\SteamLibrary\steamapps\common\DayZ\Addons"
RATE, CLIP, PRE, CHUNK = 48000, 48000, 4800, 2000


def clips_of(mono):
    """Windows of a sample: the peak's, and for a long one two more away from it."""
    n = len(mono)
    peak = int(np.argmax(np.abs(mono)))
    starts = [peak - PRE]
    if n > 2 * CLIP:
        for frac in (0.25, 0.75):
            s = int(n * frac) - CLIP // 2
            if all(abs(s - t) > CLIP // 2 for t in starts):
                starts.append(s)
    out = []
    for s in starts:
        c = np.zeros(CLIP, dtype=np.float32)
        a, b = max(0, s), min(n, s + CLIP)
        if b > a:
            c[a - s:b - s] = mono[a:b]
        out.append(c)
    return out


def job(task):
    pbo, name, offset, size, mime, orig = task
    try:
        with open(pbo, "rb") as f:
            f.seek(offset)
            raw = f.read(size)
        if mime != 0 and orig and orig != size:
            raw = lzss(raw, orig)
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if len(mono) < 100:
            return None
        if sr != RATE:
            mono = soxr.resample(mono, sr, RATE).astype(np.float32)
        return os.path.basename(pbo), name, sr, len(mono) / RATE, clips_of(mono)
    except Exception as ex:
        return os.path.basename(pbo), name, 0, 0.0, repr(ex)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    tasks = []
    for pbo in sorted(glob.glob(ADDONS + r"\sounds_*.pbo")):
        f, entries = pbo_entries(pbo)
        f.close()
        for name, (offset, size, mime, orig) in entries.items():
            if name.lower().endswith(".ogg"):
                tasks.append((pbo, name, offset, size, mime, orig))
    print(len(tasks), "samples", flush=True)
    man = open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8")
    w = csv.writer(man)
    w.writerow(["id", "chunk", "index", "bank", "path", "window", "rate", "seconds", "peak", "rms"])
    chunk, index, cid, failed = 0, 0, 0, []
    fh = open(os.path.join(out, f"clips_{chunk:03d}.f32"), "wb")
    with multiprocessing.Pool(6) as pool:
        for k, r in enumerate(pool.imap(job, tasks, chunksize=32)):
            if r is None:
                continue
            bank, name, sr, secs, clips = r
            if isinstance(clips, str):
                failed.append((name, clips))
                continue
            for wi, c in enumerate(clips):
                if index == CHUNK:
                    fh.close()
                    chunk += 1
                    index = 0
                    fh = open(os.path.join(out, f"clips_{chunk:03d}.f32"), "wb")
                fh.write(c.tobytes())
                w.writerow([cid, chunk, index, bank, name.replace("\\", "/"), wi, sr, round(secs, 3), round(float(np.abs(c).max()), 5), round(float(np.sqrt((c * c).mean())), 5)])
                index += 1
                cid += 1
            if k % 4000 == 0:
                print(f"{k}/{len(tasks)} samples, {cid} clips", flush=True)
    fh.close()
    man.close()
    print(f"done: {cid} clips in {chunk + 1} chunks; failed {len(failed)} {failed[:3]}", flush=True)

"""
Which world event sounds are in the recordings? The game's artillery and crash samples are searched in every recording by normalized cross-correlation, low-passed and at 4 kHz (long, low sounds). Read-only for the game
files; samples stay in the scratchpad. A control: the same search with the samples time-reversed, which has the same spectrum and length and cannot truly match.
Usage: python world_events.py <samples dir>
"""
import glob, json, multiprocessing as mp, os, re, sys, time
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfiltfilt
from read_dayz_config import pbo_entries, lzss

ADDONS = r"L:\SteamLibrary\steamapps\common\DayZ\Addons"
RUNS = r"D:\users\tj\Projects\AssistZ\Lab\runs"
RATE = 4000
SOS = butter(6, 1500, btype="lowpass", fs=48000, output="sos")
PITCHES = np.round(np.exp(np.arange(np.log(0.85), np.log(1.18), 0.01)), 4)
GROUPS = {"artillery_distant": r"Artillery_Distant_\d", "artillery_barrage": r"Artillery_Distant_Barrage", "artillery_close": r"Artillery_Close", "artillery_fall": r"Artillery_Fall", "helicrash": r"HeliCrash_Distant", "sledgecrash": r"SledgeCrash"}
_T = {}


def extract(out):
    os.makedirs(out, exist_ok=True)
    f, entries = pbo_entries(ADDONS + r"\sounds_environment.pbo")
    n = 0
    for name, (offset, size, mime, orig) in entries.items():
        if re.search(r"(?i)(^|[\\/])(artillery|helicrash)[\\/]", name):
            f.seek(offset)
            data = f.read(size)
            if mime == 0x43707273 and orig:
                data = lzss(data, orig)
            with open(os.path.join(out, os.path.basename(name.replace("\\", "/"))), "wb") as o:
                o.write(data)
            n += 1
    print("extracted", n, "samples to", out)


def build(out):
    tmpl = []
    for path in sorted(glob.glob(os.path.join(out, "*.ogg"))):
        base = os.path.basename(path)
        group = next((g for g, pat in GROUPS.items() if re.match(pat, base, re.I)), None)
        if not group:
            continue
        x, rate = sf.read(path, always_2d=True)
        x = x.mean(axis=1)
        x = resample_poly(x, RATE, rate)
        # The loudest 4 s (or all of it): enough to be specific, short enough to search.
        keep = min(len(x), 4 * RATE)
        energy = np.convolve(x ** 2, np.ones(keep), "valid")
        a = int(np.argmax(energy))
        x = x[a: a + keep]
        for pitch in PITCHES:
            n = int(round(len(x) / pitch))
            y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
            nrm = np.linalg.norm(y)
            if nrm > 0:
                tmpl.append((group, base, float(pitch), (y / nrm).astype(np.float32), (y[::-1] / nrm).astype(np.float32)))
    return tmpl


def init(out):
    _T["t"] = build(out)


def scan(run):
    a = np.load(run + "/audio.npy").astype(np.float64)
    meta = json.load(open(run + "/audio.json"))
    sig = []
    for c in range(a.shape[1]):
        sig.append(resample_poly(sosfiltfilt(SOS, a[:, c]), RATE, 48000).astype(np.float32))
    res = {}
    for group, base, pitch, y, rev in _T["t"]:
        for s in sig:
            if len(y) >= len(s):
                continue
            cum = np.concatenate(([0.0], np.cumsum(s.astype(np.float64) ** 2)))
            e = np.sqrt(cum[len(y):] - cum[:-len(y)]) + 1e-6
            for key, tpl in ((group, y), (group + "|rev", rev)):
                ncc = np.abs(fftconvolve(s, tpl[::-1], mode="valid")) / e
                # Ignore near silence: correlation is meaningless where there is no energy.
                ncc[e < 0.02] = 0
                k = int(np.argmax(ncc))
                if ncc[k] > res.get(key, (0,))[0]:
                    res[key] = (float(ncc[k]), k / RATE, base, pitch)
    return os.path.basename(run)[-15:], res


if __name__ == "__main__":
    out = os.path.join(sys.argv[1], "world")
    extract(out)
    t = build(out)
    assert t, "no templates: nothing to search"
    print(len(t), "templates:", {g: sum(x[0] == g for x in t) // len(PITCHES) for g in GROUPS})
    runs = [r for r in sorted(glob.glob(RUNS + r"\sniper_2026110[34]_*")) if os.path.exists(r + "/audio.npy") and os.path.exists(r + "/audio.json")]
    t0 = time.time()
    results = []
    with mp.Pool(4, initializer=init, initargs=(out,)) as pool:
        for n, r in enumerate(pool.imap_unordered(scan, runs), 1):
            results.append(r)
            print(f"  {n}/{len(runs)} at {time.time() - t0:.0f} s", flush=True)
    json.dump(results, open(os.path.join(sys.argv[1], "world_events.json"), "w"))
    results.sort()
    rev = [v[0] for _, res in results for k, v in res.items() if k.endswith("|rev")]
    limit = np.percentile(rev, 99)
    print(f"\nreversed-sample control: median {np.median(rev):.3f}, 99th percentile {limit:.3f}, max {max(rev):.3f}")
    print("Best match per sound group where it beats the control's maximum (recording: group correlation at seconds, sample, pitch):")
    for name, res in results:
        for g in GROUPS:
            if g in res and res[g][0] > max(rev):
                print(f"  {name}: {g:<18} {res[g][0]:.3f} at {res[g][1]:6.1f} s  {res[g][2]}  x{res[g][3]:.3f}")

"""
Read-only. A) Which config in the game names the flyby and crack sets, and on which classes. B) Duration and band energy of the game's own samples, in the bands analyze_flyby.py used.
"""
import glob, os, re, sys
import numpy as np
import soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_dayz_config import load

ADDONS = r"L:\SteamLibrary\steamapps\common\DayZ\Addons"
SAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_samples")
BANDS = ((300, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 20000))
WANT = re.compile(r"(?i)flyby|bulletfly|soniccrack|soundsetfly")

print("A. Configs naming these sets")
for pbo in sorted(glob.glob(ADDONS + r"\*.pbo")):
    if os.path.getsize(pbo) > 400_000_000:
        continue
    try:
        configs = load(pbo)
    except Exception:
        continue
    for cfg, rap in configs:
        if not re.search(rb"(?i)soundSetBulletFly|soundSetSonicCrack|bulletFlyBy|SonicCrack", rap.d):
            continue
        if os.path.basename(pbo) == "sounds_hpp.pbo":
            continue
        hits = []

        def visit(offset, path):
            node = rap.body(offset)
            for k, v in node["values"].items():
                if WANT.search(k) or WANT.search(str(v)):
                    hits.append((path, node["_parent"], k, v))
            for name, off in node["classes"].items():
                visit(off, path + "/" + name)
        try:
            visit(16, "")
        except Exception as e:
            print("   parse stopped:", e)
        print(f"  {os.path.basename(pbo)} {cfg}: {len(hits)} entries")
        for h in hits[:10]:
            print("     ", h[0][-70:], ":", h[1], " ", h[2], "=", str(h[3])[:90])
        fly = [h for h in hits if "fly" in (h[2] + str(h[3])).lower()]
        print(f"      flyby entries: {len(fly)}; classes with their own sound entry: {len({h[0] for h in hits})}")

print("\nB. The game's samples: dB by band relative to the sample's strongest band; effective duration (time between first and last 5 ms frame within 20 dB of the peak)")
print(f"  {'group':<22} {'n':>2} {'rate':>6} {'length ms':>9} {'effective ms':>12} " + " ".join(f"{lo//1000 if lo >= 1000 else lo}-{hi//1000}k".rjust(9) for lo, hi in BANDS) + "  top-band peak ms")
groups = {"bullet_flyby": [], "sonicCrack_Close": [], "sonicCrack_Dist": [], "sonicCrack (old)": []}
for path in sorted(glob.glob(SAMPLES + r"\*.ogg")):
    n = os.path.basename(path)
    key = "bullet_flyby" if n.startswith("bullet_flyby") else "sonicCrack_Close" if re.match(r"sonicCrack_Close_\d", n) else "sonicCrack_Dist" if n.startswith("sonicCrack_Dist") else "sonicCrack (old)" if re.match(r"sonicCrack\d", n) else None
    if not key:
        continue
    x, rate = sf.read(path, always_2d=True)
    groups[key].append((x.mean(axis=1), rate))
for key, items in groups.items():
    rows = []
    for x, rate in items:
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / rate)
        e = np.array([spec[(f >= lo) & (f < hi)].sum() for lo, hi in BANDS])
        db = 10 * np.log10(e / e.max() + 1e-12)
        hop = int(0.005 * rate)
        frames = np.array([np.sqrt((x[i:i + hop] ** 2).mean()) for i in range(0, max(1, len(x) - hop), hop)])
        env = 20 * np.log10(frames + 1e-9)
        on = np.flatnonzero(env > env.max() - 20)
        # When the top band peaks, measured from the sample's first audible frame.
        hp = np.fft.rfft(x)
        hp[f < 12000] = 0
        top = np.fft.irfft(hp, len(x))
        tf = np.array([np.sqrt((top[i:i + hop] ** 2).mean()) for i in range(0, max(1, len(x) - hop), hop)])
        rows.append((rate, 1000 * len(x) / rate, 5 * (on[-1] - on[0]) if len(on) else 0, db, 5 * (int(np.argmax(tf)) - int(on[0]))))
    if rows:
        d = np.mean([r[3] for r in rows], axis=0)
        print(f"  {key:<22} {len(rows):>2} {rows[0][0]:>6} {np.median([r[1] for r in rows]):>9.0f} {np.median([r[2] for r in rows]):>12.0f} " + " ".join(f"{v:9.1f}" for v in d) + f"  {np.median([r[4] for r in rows]):.0f}")

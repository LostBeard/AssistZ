"""
When does the high-band precursor arrive, relative to the shot? Two readings of DayZ's SuperSonicCrack sound sets
(sounds_hpp: range 500 / 100 m, InverseSquare1Curve, soundObjectsLimit 1; every Bullet_Base ammo uses them):
  A. played where/when the bullet passes the listener: arrives ~ d / v_bullet after the fire (v = 785 m/s 7.62x54);
  B. played at the shooter at the fire, attenuated from there: arrives ~ 0 after the fire, ~6 dB weaker per doubling of
     the shooter distance.
The fire time is taken from the report itself (report onset - d / 330, DayZ's measured sound delay), so the run's clock
alignment drops out. The report onset = the first 5 ms hop within 60 ms before the report's broadband peak where the
level is within 6 dB of the peak. For each clean shot (no other logged report within 0.45 s before), prints the
high-band (> 3 kHz) level over its background in 10 ms steps after the estimated fire, and where the precursor peak sits.
Usage: python analyze_crack_timing.py <run dir> [...]
"""
import csv, datetime, json, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from analyze_crack import band_env

def run(path):
    pcm = np.load(path + "/audio.npy").astype(np.float64)
    meta = json.load(open(path + "/audio.json"))
    rate = meta["rate"]
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(path + "/pose.csv")))
    g = np.array([int(r["game_ms"]) / 1000 for r in pose])
    u = np.array([sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) for r in pose], dtype=float)
    offset = np.median(u + 0.5 - g)
    hop = 240
    dt = hop / rate
    full = band_env(pcm, rate, hop)
    high = band_env(pcm, rate, hop, lo=3000)
    shots = list(csv.DictReader(open(path + "/shots_ai.csv")))
    dues = [int(x["game_ms"]) / 1000 + offset - start_s + float(x["distance"]) / 330.0 for x in shots]
    # The pose log's UTC column is whole seconds, so the game->recording clock is good to a few hundred ms only (it read
    # +95 ms on runs 4-6). Calibrate it on the data: the single shift that puts the most broadband energy at every shot's due
    # report time. The search spans +-5 s: at +-0.5 s run 7 read a false peak near zero, its real one is near -2.4 s (see
    # lab_clock.py). Then look for each report within +-60 ms of it - a wide window
    # picked loud impacts near the listener (lab mode 7's decoy stands 5-60 m away) as "reports".
    shifts = np.arange(-5.0, 5.0, dt)
    score = [np.mean([full[int((t + sh) / dt)] for t in dues if 0 <= int((t + sh) / dt) < len(full)]) for sh in shifts]
    shift = float(shifts[int(np.argmax(score))])
    dues = [t + shift for t in dues]
    print(f"   {path}: clock shift {shift * 1000:+.0f} ms")
    out = []
    for i, r in enumerate(shots):
        d = float(r["distance"])
        due = dues[i]
        a, b = int((due - 0.06) / dt), int((due + 0.06) / dt)
        if a < 0 or b >= len(full): continue
        peak = a + int(np.argmax(full[a:b]))
        onset = peak
        while onset > peak - int(0.06 / dt) and full[onset - 1] > full[peak] - 6: onset -= 1
        fire = onset * dt - d / 330.0
        if any(j != i and fire - 0.1 <= dues[j] <= onset * dt for j in range(len(dues))): continue   # another report in the way
        f0 = int(fire / dt)
        base = np.median(high[f0 - int(0.25 / dt): f0 - int(0.05 / dt)])
        lo, hi = f0 - int(0.03 / dt), onset - int(0.02 / dt)
        if hi <= lo: continue
        k = lo + int(np.argmax(high[lo:hi]))
        out.append((r["shooter"].replace("eAI_Survivor", ""), d, (k - f0) * dt * 1000, high[k] - base, d / 785.0 * 1000, path.rstrip("/").split("/")[-1], i))
    return out

rows = []
for p in sys.argv[1:]: rows += run(p)
print("shooter        dist  precursor after fire ms  over background dB   bullet arrival (A) ms")
for s, d, t, db, a, *_ in sorted(rows, key=lambda x: x[1]):
    print(f"{s:<12} {d:6.0f} {t:22.0f} {db:19.1f} {a:22.0f}")
for lo, hi in ((0, 60), (60, 120)):
    sel = [x for x in rows if lo <= x[1] < hi]
    if not sel: continue
    print(f"== shooter {lo}-{hi} m: {len(sel)} shots, precursor after fire median {np.median([x[2] for x in sel]):.0f} ms"
          f" (A predicts {np.median([x[4] for x in sel]):.0f}, B ~0), over background median {np.median([x[3] for x in sel]):.1f} dB")

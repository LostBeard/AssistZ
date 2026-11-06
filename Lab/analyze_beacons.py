"""
Analyses a loopback recording (record_loopback.py) against the beacon log (assistz_beacon_*.csv):
  - finds each detonation's onset in the recording,
  - arrival delay vs distance (does DayZ delay sound by d / 343 m/s?), using game_ms for sub-second timing,
  - broadband level and per-octave-band spectra per channel,
  - front/back mirror pairs at the same folded azimuth: back-minus-front spectrum (a front/back cue if non-zero),
  - level and high-frequency loss vs distance.
Usage: python analyze_beacons.py <recording.npy> <beacon.csv>
"""
import sys, json, csv, datetime
import numpy as np

rec_path, beacon_path = sys.argv[1], sys.argv[2]
pcm = np.load(rec_path).astype(np.float64)
meta = json.load(open(rec_path.replace(".npy", ".json")))
rate = meta["rate"]
rec_start = datetime.datetime.fromisoformat(meta["start_utc"])
beacons = list(csv.DictReader(open(beacon_path)))

def utc_seconds(hms, day_ref):
    h, m, s = map(int, hms.split(":"))
    return day_ref.replace(hour=h, minute=m, second=s, microsecond=0)

# Onset envelope: 5 ms RMS frames of the mono sum, onset = biggest rise in a search window.
hop = int(0.005 * rate)
mono = pcm.mean(axis=1)
nfr = len(mono) // hop
env = np.sqrt((mono[: nfr * hop].reshape(nfr, hop) ** 2).mean(axis=1) + 1e-20)
env_db = 20 * np.log10(env)

def find_onset(t_rec, before=0.6, after=2.6):
    a, b = max(0, int((t_rec - before) / 0.005)), min(nfr - 1, int((t_rec + after) / 0.005))
    if b <= a + 4:
        return None
    rise = env_db[a + 2 : b] - np.minimum(env_db[a : b - 2], env_db[a + 1 : b - 1])
    k = int(np.argmax(rise)) + a + 2
    return k * 0.005, float(rise.max()), float(env_db[k : k + 40].max())

bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
def band_db(seg):
    n = 1 << int(np.ceil(np.log2(len(seg))))
    win = np.hanning(len(seg))[:, None]
    spec = np.abs(np.fft.rfft(seg * win, n=n, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / rate)
    out = []
    for f in bands:
        sel = (freqs >= f / np.sqrt(2)) & (freqs < f * np.sqrt(2))
        out.append(10 * np.log10(spec[sel].sum(axis=0) + 1e-20))
    return np.array(out)  # [bands, channels]

rows = []
for b in beacons:
    trig = utc_seconds(b["utc"], rec_start)
    t_rec = (trig - rec_start).total_seconds()
    if t_rec < 0.5 or t_rec > len(mono) / rate - 2:
        continue
    found = find_onset(t_rec + 0.5)  # utc column is floored to the second
    if not found:
        continue
    t_on, rise, peak = found
    seg = pcm[int(t_on * rate) : int((t_on + 0.3) * rate)]
    bd = band_db(seg)
    rms = np.sqrt((seg ** 2).mean(axis=0))
    rows.append(dict(game_ms=int(b["game_ms"]), bearing=float(b["bearing_rel"]), dist=float(b["distance"]),
                     aim=float(b["aim_lr"]), t_on=t_on, rise=rise, peak=peak,
                     lvl=20 * np.log10(rms + 1e-20), bands=bd))

print(f"recording {len(mono) / rate:.1f} s from {meta['start_utc']}; {len(beacons)} beacons logged, {len(rows)} inside the recording")
if not rows:
    sys.exit(0)

# Arrival delay: t_on - game_ms/1000 = clock offset + propagation delay. Fit offset + d/c.
d = np.array([r["dist"] for r in rows]); lag = np.array([r["t_on"] - r["game_ms"] / 1000 for r in rows])
A = np.vstack([np.ones_like(d), d]).T
(off, slope), *_ = np.linalg.lstsq(A, lag, rcond=None)
print(f"\narrival delay vs distance: slope {slope * 1000:.2f} ms/m  (speed of sound would be {1000 / 343:.2f} ms/m)"
      + (f" -> {1 / slope:.0f} m/s" if slope > 1e-5 else " -> no propagation delay"))
for r in sorted(rows, key=lambda r: r["dist"]):
    print(f"  {r['dist']:5.0f} m  bearing {r['bearing']:5.0f}  delay-offset {(r['t_on'] - r['game_ms'] / 1000 - off) * 1000:7.1f} ms")

print("\nper shot: bearing, distance, L/R level dBFS, ILD (R-L), rise")
for r in rows:
    print(f"  {r['bearing']:5.0f} {r['dist']:5.0f} m  L {r['lvl'][0]:6.1f}  R {r['lvl'][1]:6.1f}  ILD {r['lvl'][1] - r['lvl'][0]:+6.1f}  rise {r['rise']:5.1f}")

def shot(bearing, dist):
    c = [r for r in rows if r["bearing"] == bearing and r["dist"] == dist]
    return c[0] if c else None

print("\nfront/back mirror pairs (50 m): back minus front, per octave band, mean of both channels (dB)")
print("  pair      " + " ".join(f"{f:>6}" for f in bands))
for front, back in ((45, 135), (315, 225), (0, 180)):
    a, b = shot(front, 50), shot(back, 50)
    if a and b:
        diff = (b["bands"] - a["bands"]).mean(axis=1)
        print(f"  {front:3.0f}/{back:3.0f}  " + " ".join(f"{x:6.1f}" for x in diff) + f"   broadband {np.mean(b['lvl'] - a['lvl']):+.1f}")
p90, p270 = shot(90, 50), shot(270, 50)
if p90 and p270:
    print(f"  90/270 (left/right control) ILD {p90['lvl'][1] - p90['lvl'][0]:+.1f} / {p270['lvl'][1] - p270['lvl'][0]:+.1f}")

print("\ndistance ladders: broadband level (mean ch) and HF tilt (8k+16k minus 250+500, dB)")
for bearing in (90, 0):
    pts = sorted([r for r in rows if r["bearing"] == bearing and r["dist"] in (35, 50, 70, 140, 280)], key=lambda r: r["dist"])
    for r in pts:
        tilt = r["bands"][7:9].mean() - r["bands"][2:4].mean()
        print(f"  {bearing:3.0f}  {r['dist']:5.0f} m  level {np.mean(r['lvl']):6.1f}  tilt {tilt:6.1f}")
    if len(pts) >= 3:
        x = np.log2([r["dist"] for r in pts]); y = [np.mean(r["lvl"]) for r in pts]
        k = np.polyfit(x, y, 1)[0]
        print(f"  -> {k:+.1f} dB per doubling of distance (inverse-square = -6.0)")

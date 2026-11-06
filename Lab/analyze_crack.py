"""
Is there a "being shot at" sound before the report? For each logged shot of a lab mode-6 run (shots_ai.csv from
aishots_to_shots.py: client fire time on the server clock, muzzle-to-head distance, barrel-line miss), against the
loopback recording:
  1. the report: the biggest broadband level peak within +-0.25 s of fire + distance / 330 m/s (DayZ's measured delay);
  2. a precursor: the largest high-band (> 3 kHz) level peak in [report - 0.35 s, report - 0.03 s], its lead over the
     report and its level against the report's high band.
A supersonic bullet passing close arrives before its report by about d * (1/330 - 1/v), v ~ 830 m/s (SVD) or ~ 900 (5.56):
if the precursor is the crack, its lead grows with the distance and it is missing for wide misses. Nothing about the
sound is modelled; the prediction is printed only to compare against.
Usage: python analyze_crack.py <run dir> [<run dir> ...]
"""
import csv, datetime, json, sys
import numpy as np

def band_env(pcm, rate, hop, lo=None):
    """Level envelope (dB) per hop; lo = high-pass cutoff in Hz via FFT per frame."""
    mono = pcm.mean(axis=1)
    n = len(mono) // hop
    frames = mono[: n * hop].reshape(n, hop)
    if lo is None:
        return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    spec = np.abs(np.fft.rfft(frames * np.hanning(hop), axis=1)) ** 2
    freqs = np.fft.rfftfreq(hop, 1 / rate)
    return 10 * np.log10(spec[:, freqs >= lo].sum(axis=1) / hop + 1e-12)

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
    hop = 240   # 5 ms
    dt = hop / rate
    full = band_env(pcm, rate, hop)
    high = band_env(pcm, rate, hop, lo=3000)
    rows = []
    shots = list(csv.DictReader(open(path + "/shots_ai.csv")))
    # Every shot's due report time: a precursor window that holds ANOTHER shot's report is not evidence (an AUG fires
    # every 0.27 s, so "a peak 275 ms before the report" is the previous report).
    dues = [int(x["game_ms"]) / 1000 + offset - start_s + float(x["distance"]) / 330.0 for x in shots]
    # Where each bullet landed (impacts_client.csv, the client's FirearmEffects log, client clock): the first impact of the
    # same ammo within 2 s after the shot's client fire time. shots_ai.csv holds server-clock times; aishots_client.csv
    # the client ones in the same order.
    import os
    impacts = list(csv.DictReader(open(path + "/impacts_client.csv"))) if os.path.exists(path + "/impacts_client.csv") else []
    client = list(csv.DictReader(open(path + "/aishots_client.csv"))) if impacts else []
    # One-to-one: each impact belongs to the LATEST shot fired before it (within 2 s); a bullet that left no impact (flew
    # out of the streamed world) gets none, instead of borrowing the next shot's.
    owner = {}
    for x in impacts:
        t = int(x["game_ms"])
        before = [j for j, c in enumerate(client) if int(c["game_ms"]) <= t <= int(c["game_ms"]) + 2000]
        if before:
            j = max(before, key=lambda j: int(client[j]["game_ms"]))
            owner.setdefault(j, x)
    def impact_of(i):
        x = owner.get(i)
        return float(x["head_dist"]) if x else None
    def true_miss(i):
        """The bullet's real closest approach to the head: muzzle (client log) -> impact segment, head from pose.csv at
        the shot (pose is ~1 s apart and the listener stands still in mode 6)."""
        x = owner.get(i)
        if not x: return None
        c = client[i]
        m = np.array([float(c[k]) for k in ("mx", "my", "mz")])
        q = np.array([float(x[k]) for k in ("x", "y", "z")])
        p = pose[-1]
        h = np.array([float(p["x"]), float(p["y"]) + 1.6, float(p["z"])])
        d = q - m
        t = np.clip(np.dot(h - m, d) / np.dot(d, d), 0, 1)
        return float(np.linalg.norm(m + t * d - h))
    for i, r in enumerate(shots):
        fire = int(r["game_ms"]) / 1000 + offset - start_s
        d, miss = float(r["distance"]), float(r["miss"])
        due = fire + d / 330.0
        a, b = int((due - 0.25) / dt), int((due + 0.25) / dt)
        if a < 0 or b >= len(full): continue
        rep = a + int(np.argmax(full[a:b]))
        pa, pb = rep - int(0.35 / dt), rep - int(0.03 / dt)
        seg = high[pa:pb]
        k = pa + int(np.argmax(seg))
        base = np.median(high[rep - int(0.6 / dt): rep - int(0.4 / dt)])
        clean = not any(j != i and rep * dt - 0.4 <= dues[j] <= rep * dt - 0.02 for j in range(len(dues)))
        rows.append(dict(impact=impact_of(i), true_miss=true_miss(i), clean=clean, shooter=r["shooter"].replace("eAI_Survivor", ""), weapon=r["weapon"], d=d, miss=miss,
                         rep_err=(rep * dt - due) * 1000, lead=(rep - k) * dt * 1000,
                         pre_vs_rep=high[k] - high[rep], pre_vs_base=high[k] - base))
    return rows

all_rows = []
for path in sys.argv[1:]:
    rows = run(path)
    all_rows += rows
    print(f"== {path}")
    print("  shooter      weapon     dist  miss  report vs due ms  precursor lead ms  pre-report dB  pre-background dB  predicted lead ms")
    for x in rows:
        v = 830.0 if x["weapon"] == "SVD" else 900.0
        pred = x["d"] * (1 / 330.0 - 1 / v) * 1000
        impact_note = '' if x['impact'] is None else "impact {:.1f} m from head, bullet passed {:.1f} m".format(x['impact'], x['true_miss'])
        print(f"  {x['shooter']:<12} {x['weapon']:<9} {x['d']:5.0f} {x['miss']:5.1f} {x['rep_err']:+12.0f} {x['lead']:16.0f} {x['pre_vs_rep']:14.1f} {x['pre_vs_base']:17.1f} {pred:16.0f}  {'' if x['clean'] else 'CONFOUNDED (another report in the window)'}  {impact_note}")
print("== clean shots only: precursor stands out (> 10 dB over the background before it), by distance and miss:")
for dlo, dhi in ((0, 60), (60, 120), (120, 1000)):
  for lo, hi in ((0, 6), (6, 12), (12, 30), (30, 1000)):
    sel = [x for x in all_rows if x["clean"] and lo <= x["miss"] < hi and dlo <= x["d"] < dhi]
    if not sel: continue
    stand = [x for x in sel if x["pre_vs_base"] > 10]
    leads = [x["lead"] for x in stand]
    print(f"  dist {dlo:>3}-{dhi:<4} m, miss {lo:>3}-{hi:<4} m: {len(stand)}/{len(sel)}  lead median {np.median(leads) if leads else float('nan'):.0f} ms")

with_impact = [x for x in all_rows if x["clean"] and x["impact"] is not None]
if with_impact:
    print("== clean shots with a logged impact: precursor (> 10 dB over background) by where the bullet LANDED:")
    for lo, hi in ((0, 10), (10, 30), (30, 100), (100, 100000)):
        sel = [x for x in with_impact if lo <= x["impact"] < hi]
        if not sel: continue
        stand = [x for x in sel if x["pre_vs_base"] > 10]
        print(f"  impact {lo:>4}-{hi:<6} m from head: {len(stand)}/{len(sel)}  (barrel-line miss {min(x['miss'] for x in sel):.1f}-{max(x['miss'] for x in sel):.1f} m, dist {min(x['d'] for x in sel):.0f}-{max(x['d'] for x in sel):.0f} m)")

if with_impact:
    print("== clean shots with a logged impact: precursor by the bullet's REAL closest approach (muzzle -> impact):")
    for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 30), (30, 100000)):
        sel = [x for x in with_impact if lo <= x["true_miss"] < hi]
        if not sel: continue
        stand = [x for x in sel if x["pre_vs_base"] > 10]
        print(f"  passed {lo:>3}-{hi:<6} m: {len(stand)}/{len(sel)}  (shooter at {min(x['d'] for x in sel):.0f}-{max(x['d'] for x in sel):.0f} m)")

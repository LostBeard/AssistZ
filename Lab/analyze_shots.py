"""
Crack-to-report analysis for a lab mode-4 run (server-fired SVD near-misses from a range ladder), against a loopback
recording (record_loopback.py: .npy + .json with the UTC start). For each shot: the onsets that follow it, the
first (supersonic crack) and the report (muzzle blast, arriving d / ~330 m/s after the shot), their gap vs the
prediction d * (1/c - 1/v_bullet), and the report's arrival delay vs range.
Usage: python analyze_shots.py <recording.npy> <shots.csv> <pose.csv>
"""
import sys, json, csv, datetime
import numpy as np

rec, shots_path, pose_path = sys.argv[1], sys.argv[2], sys.argv[3]
pcm = np.load(rec).astype(np.float64)
meta = json.load(open(rec.replace(".npy", ".json")))
rate = meta["rate"]
start = datetime.datetime.fromisoformat(meta["start_utc"])
start_s = start.hour * 3600 + start.minute * 60 + start.second + start.microsecond / 1e6

pose = list(csv.DictReader(open(pose_path)))
g = np.array([int(r["game_ms"]) / 1000 for r in pose])
u = np.array([sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) for r in pose], dtype=float)
offset = np.median(u + 0.5 - g)   # game seconds -> UTC seconds of day (utc floors to the second)

mono = np.abs(pcm).max(axis=1)
hop = int(0.002 * rate)            # 2 ms envelope
n = len(mono) // hop
env = mono[: n * hop].reshape(n, hop).max(axis=1)
env_db = 20 * np.log10(env + 1e-9)

C, V = 330.0, 830.0
print(f"recording {len(mono) / rate:.1f} s from {meta['start_utc']}")
print(" shot  dist  miss  fired  onsets(ms after shot, rise dB)                      gap ms  predicted ms")
gaps = []
for r in csv.DictReader(open(shots_path)):
    t = int(r["game_ms"]) / 1000 + offset - start_s
    d = float(r["distance"])
    # Mode 4 fires when the RPC arrives (log time = fire time): the report is due ~d/C after it, so search up to that
    # plus 0.6 s. Mode 5 ("sent"): the shooter's client aims first (0.25-4 s), so search 5 s more. In both, the gap is
    # between the first two onsets (crack, then report) - both from the game, no timing from here.
    extra = 5.0 if r.get("fired") == "sent" else 0.6
    a, b = int(t / 0.002), int((t + d / C + extra) / 0.002)
    if a < 5 or b >= n:
        continue
    # onsets: frames whose level jumps > 12 dB above the max of the previous 20 ms, at least 15 ms apart
    ons = []
    for k in range(a, b):
        prev = env_db[max(0, k - 10):k].max()
        rise = env_db[k] - prev
        if rise > 12 and env_db[k] > -50 and (not ons or k - ons[-1][0] > 7):
            ons.append((k, rise))
    desc = " ".join(f"{(k - a) * 2:5.0f}({rise:4.1f})" for k, rise in ons[:6])
    gap = (ons[1][0] - ons[0][0]) * 2 if len(ons) >= 2 else float("nan")
    pred = d * (1 / C - 1 / V) * 1000
    if len(ons) >= 2:
        gaps.append((d, gap, pred))
    print(f" {r['shot']:>4} {d:5.0f} {float(r.get('miss', 0)):5.0f}  {r['fired']:>5}  {desc:52} {gap:6.0f}  {pred:6.0f}")
if gaps:
    d = np.array([x[0] for x in gaps]); gap = np.array([x[1] for x in gaps])
    k = np.polyfit(d, gap, 1)
    print(f"gap vs distance: {k[0]:.2f} ms/m + {k[1]:.0f} ms (prediction {(1 / C - 1 / V) * 1000:.2f} ms/m)")

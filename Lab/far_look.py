"""First look: the shipped distance probe on the server-fired SVD ladders (50, 100, 200, 400 m), recordings it never trained on. The probe is computed from the embedding for every event at a shot, named gunfire or not, to see whether anything in the sound rises with distance."""
import collections, csv, datetime, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import envelope_db, HOP, SEARCH_SECONDS
ROOT = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261103_"
from distance_model import load
est = load("D:/users/tj/Projects/AssistZ/AssistZ/AssistZ/wwwroot/models/distance-probe.json")
rows = []
for n in sys.argv[1:]:
    run = ROOT + n
    pcm = np.load(run + "/audio.npy").astype(np.float64)
    meta = json.load(open(run + "/audio.json"))
    rate = meta["rate"]
    st = datetime.datetime.fromisoformat(meta["start_utc"])
    start_s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    g = np.array([int(r["game_ms"]) / 1000 for r in pose])
    u = np.array([sum(int(x) * k for x, k in zip(r["utc"].split(":"), (3600, 60, 1))) for r in pose], dtype=float)
    offset = np.median(u + 0.5 - g)
    shots = [s for s in csv.DictReader(open(run + "/shots.csv")) if s["fired"] == "true"]
    dues = [int(x["game_ms"]) / 1000 + offset - start_s + float(x["distance"]) / 330.0 for x in shots]
    dt = HOP / rate
    full = envelope_db(pcm)
    shifts = np.arange(-SEARCH_SECONDS, SEARCH_SECONDS, dt)
    score = [np.mean([full[int((t + sh) / dt)] for t in dues if 0 <= int((t + sh) / dt) < len(full)]) for sh in shifts]
    shift = float(shifts[int(np.argmax(score))])
    ev = [json.loads(x) for x in open(run + "/replay_far_s0.jsonl", encoding="utf-8")]
    hit = 0
    for t, s in zip(dues, shots):
        rt = t + shift
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - rt <= 0.15]
        e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - rt)) if near else None
        hit += e is not None
        rows.append(dict(run=n, d=float(s["distance"]), miss=float(s["miss"]), ev=e))
    print(f"{n}: {len(shots)} fired, clock {shift:+.3f} s, {hit} with an event, {len(ev)} events")
print()
print(" range  shots  event  gunfire   probe p25/med/p75 (all events)   probe med (named gunfire)   level")
for d in sorted({r["d"] for r in rows}):
    rr = [r for r in rows if r["d"] == d]
    e = [r["ev"] for r in rr if r["ev"]]
    gun = [x for x in e if x["raw"] in ("gunshot", "automatic fire")]
    p = np.array([est(x["embed"]) for x in e if "embed" in x])
    pg = np.array([x["dist"] for x in gun if "dist" in x])
    print(f"{d:6.0f} {len(rr):6d} {len(e):6d} {len(gun):8d}   " + (f"{np.percentile(p, 25):5.0f} {np.median(p):5.0f} {np.percentile(p, 75):5.0f}" if len(p) else "    -") + "                 " + (f"{np.median(pg):5.0f}" if len(pg) else "    -") + "                 " + (f"{np.median([x['level'] for x in e]):6.1f}" if e else "    -"))
print("\nwhat events at a shot were called, by range:")
for d in sorted({r["d"] for r in rows}):
    c = collections.Counter(r["ev"]["label"] for r in rows if r["d"] == d and r["ev"])
    print(f"  {d:.0f}: {dict(c.most_common(6))}")

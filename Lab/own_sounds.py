"""
What the character's own sounds do to the radar (lab mode 32). Each app event by the phase drive_own.py was in; truth that it moved and fired from the game's logs. Direction fields say whether own sounds stand apart: own sound should sit dead centre. Usage: own_sounds.py <tag> <run> [other players' footsteps run tag]
"""
import collections, csv, datetime, json, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\AssistZ")
TAG, run = sys.argv[1], sys.argv[2].replace("\\", "/").rstrip("/")
m = json.load(open(run + "/audio.json"))
st = datetime.datetime.fromisoformat(m["start_utc"]).timestamp()
drive = json.load(open(run + "/own_drive.json"))
if drive.get("stopped"):
    print("DRIVER STOPPED:", drive["stopped"])
phases = [(p["name"], datetime.datetime.fromisoformat(p["start_utc"]).timestamp() - st, datetime.datetime.fromisoformat(p["end_utc"]).timestamp() - st) for p in drive["phases"]]
pose = list(csv.DictReader(open(run + "/pose.csv")))
utc0 = datetime.datetime.fromisoformat(m["start_utc"])
start_s = utc0.hour * 3600 + utc0.minute * 60 + utc0.second + utc0.microsecond / 1e6
off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
pt = np.array([int(p["game_ms"]) / 1000 + off - start_s for p in pose])
px, pz = np.array([float(p["x"]) for p in pose]), np.array([float(p["z"]) for p in pose])
ev = [json.loads(x) for x in open(f"{run}/replay{TAG}_s0.jsonl", encoding="utf-8")]
print(f"{len(ev)} events; {len(phases)} phases")
try:
    shots = [r for r in csv.DictReader(open(run + "/aishots_client.csv"))] if os.path.exists(run + "/aishots_client.csv") else []
    print(f"game log: {len(shots)} shots fired on the client")
except Exception as e:
    print("no shot log", e)
print(f"{'phase':<13}{'s':>5}{'m/s':>6}{'events':>7}{'/min':>6}  |az| p50 p90  level   words")
GUN = ("gunshot", "automatic fire")
all_rows = []
for name, a, b in phases:
    g = [e for e in ev if a <= e["sample"] / 48000.0 < b]
    k = (pt >= a) & (pt < b)
    speed = 0.0
    if k.sum() > 2:
        speed = float(np.median(np.hypot(np.diff(px[k]), np.diff(pz[k])) / np.maximum(np.diff(pt[k]), 0.01)))
    az = np.abs([e["az"] for e in g]) if g else np.array([0.0])
    print(f"{name:<13}{b - a:>5.0f}{speed:>6.1f}{len(g):>7d}{60 * len(g) / max(b - a, 1):>6.1f}  {np.median(az):>6.1f}{np.percentile(az, 90):>5.1f}{np.median([e['level'] for e in g]) if g else 0:>7.1f}   {dict(collections.Counter(e['label'] for e in g).most_common(5))}")
    all_rows += [(name, e) for e in g]
print("\nWhat a player would see, all phases but the still ones:")
moving = [e for n, e in all_rows if not n.startswith("still")]
print("  " + str(dict(collections.Counter(e["label"] for e in moving).most_common(12))))
c = np.array([abs(e["az"]) for e in moving])
print(f"  centred, |az| under 3: {(c < 3).sum()}/{len(c)}; under 10: {(c < 10).sum()}/{len(c)}")
still = [e for n, e in all_rows if n.startswith("still")]
print(f"  still phases, nothing of ours: {len(still)} events: {dict(collections.Counter(e['label'] for e in still).most_common(6))}")

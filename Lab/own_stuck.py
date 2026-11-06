"""How much of the driver's movement was the character actually moving, or stuck on something? By run and phase, from the game's pose log, and how many sounds came in each. Usage: own_stuck.py <tag:run> [...]"""
import csv, datetime, json, sys
import numpy as np
R = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261105_"
tot = {}
for arg in sys.argv[1:]:
    tag, name = arg.split(":")
    run = R + name
    m = json.load(open(run + "/audio.json"))
    st = datetime.datetime.fromisoformat(m["start_utc"])
    stamp = st.timestamp()
    s = st.hour * 3600 + st.minute * 60 + st.second + st.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    t = np.array([int(p["game_ms"]) / 1000 + off - s for p in pose])
    x, z = np.array([float(p["x"]) for p in pose]), np.array([float(p["z"]) for p in pose])
    # Speed over 1 s windows, so the pose log's steps do not read as stops.
    drive = json.load(open(run + "/own_drive.json"))
    ev = [json.loads(l) for l in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
    et = np.array([e["sample"] / 48000.0 for e in ev])
    line = []
    for p in drive["phases"]:
        n = p["name"]
        if n not in ("walk", "jog", "sprint", "crouch_walk", "crawl"):
            continue
        a = datetime.datetime.fromisoformat(p["start_utc"]).timestamp() - stamp
        b = datetime.datetime.fromisoformat(p["end_utc"]).timestamp() - stamp
        speeds = []
        for c in np.arange(a, b - 1, 1.0):
            k0, k1 = np.searchsorted(t, c), np.searchsorted(t, c + 1)
            if k1 < len(t) and k0 < k1:
                speeds.append(np.hypot(x[k1] - x[k0], z[k1] - z[k0]) / max(t[k1] - t[k0], 0.1))
        speeds = np.array(speeds)
        count = int(((et >= a) & (et < b)).sum())
        stuck = float((speeds < 0.5).mean()) if len(speeds) else 0.0
        line.append(f"{n[:6]}: {np.percentile(speeds, 90):.1f} m/s top, stuck {100 * stuck:.0f}%, {count} sounds")
        d = tot.setdefault(n, [[], [], []])
        d[0].append(float(np.percentile(speeds, 90))); d[1].append(stuck); d[2].append(count)
    print(name, " | ".join(line))
print("\nall runs:")
for n, d in tot.items():
    print(f"  {n:<12} top speed {np.median(d[0]):.1f} m/s, stuck {100 * np.mean(d[1]):.0f}% of the time, sounds per run {d[2]}")

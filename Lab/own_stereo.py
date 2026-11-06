"""
Do the player's own sounds stand apart in stereo? Side energy against mid energy just before each event's end, from the recording: an exactly centred sound has no side at all. Groups: own footsteps and own shots (mode 32); other shooters truly ahead, from their logged positions (mode 28); other shooters anywhere; agroed infected (mode 30). Usage: own_stereo.py
"""
import collections, csv, datetime, json, math, os, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
R = "D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261105_"


def ratio(pcm, sample, rate=48000):
    a = max(0, sample - int(0.12 * rate))
    x = pcm[a:sample].astype(np.float64)
    if len(x) < 100:
        return None
    mid = (x[:, 0] + x[:, 1]) / 2
    side = (x[:, 0] - x[:, 1]) / 2
    return 10 * np.log10((side ** 2).mean() + 1e-14) - 10 * np.log10((mid ** 2).mean() + 1e-14)


groups = collections.defaultdict(list)
# Own sounds.
run = R + "134028"
m = json.load(open(run + "/audio.json"))
st = datetime.datetime.fromisoformat(m["start_utc"]).timestamp()
drive = json.load(open(run + "/own_drive.json"))
ph = [(p["name"], datetime.datetime.fromisoformat(p["start_utc"]).timestamp() - st, datetime.datetime.fromisoformat(p["end_utc"]).timestamp() - st) for p in drive["phases"]]
pcm = np.load(run + "/audio.npy", mmap_mode="r")
for line in open(run + "/replay_own_s0.jsonl", encoding="utf-8"):
    e = json.loads(line)
    t = e["sample"] / 48000.0
    name = next((n for n, a, b in ph if a <= t < b), None)
    r = ratio(pcm, e["sample"])
    if name and r is not None:
        groups["own shots" if name == "fire" else ("own steps etc" if not name.startswith("still") else "own still")].append((r, abs(e["az"]), e["level"], e["label"]))
# Other shooters: mode 28, near and clear.
for key in ("074844", "080020", "051052", "052232", "092756"):
    run = R + key
    tag = next((t for t in ("_f1", "_ship", "_ctl") if os.path.exists(f"{run}/replay{t}_s0.jsonl")), None)
    if not tag:
        continue
    reports, _ = calibrated_reports(run)
    shots = list(csv.DictReader(open(run + "/shots_ai.csv")))
    server = list(csv.DictReader(open(run + "/aishots_server.csv")))
    sg = np.array([int(s["game_ms"]) for s in server])
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    pg = np.array([int(p["game_ms"]) for p in pose])
    ev = [json.loads(x) for x in open(f"{run}/replay{tag}_s0.jsonl", encoding="utf-8")]
    pcm = np.load(run + "/audio.npy", mmap_mode="r")
    for r_, s in zip(reports, shots):
        g = int(s["game_ms"])
        k = int(np.argmin(np.abs(sg - g)))
        if abs(sg[k] - g) > 1500:
            continue
        p = pose[int(np.argmin(np.abs(pg - g)))]
        rel = (math.degrees(math.atan2(float(server[k]["mx"]) - float(p["x"]), float(server[k]["mz"]) - float(p["z"]))) - float(p["yaw"]) + 180) % 360 - 180
        near = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r_[0] <= 0.15]
        if not near:
            continue
        e = min(near, key=lambda x: abs(x["sample"] / 48000.0 - r_[0]))
        r = ratio(pcm, e["sample"])
        if r is None:
            continue
        row = (r, abs(e["az"]), e["level"], e["label"])
        groups["others, anywhere"].append(row)
        if abs(rel) <= 8 or abs(rel) >= 172:
            groups["others, true bearing dead ahead or behind"].append(row)
# Agroed infected, mode 30 and 31: fight state at under 3 m.
for key in ("112656", "114203", "122339", "124557"):
    run = R + key
    m = json.load(open(run + "/audio.json"))
    st_ = datetime.datetime.fromisoformat(m["start_utc"])
    s_ = st_.hour * 3600 + st_.minute * 60 + st_.second + st_.microsecond / 1e6
    pose = list(csv.DictReader(open(run + "/pose.csv")))
    off = np.median([sum(int(v) * k for v, k in zip(x["utc"].split(":"), (3600, 60, 1))) + 0.5 - int(x["game_ms"]) / 1000 for x in pose])
    env = list(csv.DictReader(open(run + "/env.csv")))
    et = np.array([int(e["game_ms"]) / 1000 + off - s_ for e in env])
    nr = np.array([float(e["infected_nearest"]) for e in env])
    stt = np.array([int(e["infected_state"]) for e in env])
    pcm = np.load(run + "/audio.npy", mmap_mode="r")
    for line in open(f"{run}/replay_st_s0.jsonl", encoding="utf-8"):
        e = json.loads(line)
        i = int(np.argmin(np.abs(et - e["sample"] / 48000.0)))
        if stt[i] == 4 and 0 <= nr[i] < 3:
            r = ratio(pcm, e["sample"])
            if r is not None:
                groups["agroed infected on you"].append((r, abs(e["az"]), e["level"], e["label"]))
print("side against mid, dB (lower = more exactly centred)")
print(f"{'group':<44}{'n':>6}{'p5':>8}{'p50':>8}{'p95':>8}   |az|<3   level p50")
for name in ("own steps etc", "own shots", "own still", "others, true bearing dead ahead or behind", "others, anywhere", "agroed infected on you"):
    g = groups[name]
    if not g:
        continue
    r = np.array([x[0] for x in g])
    az = np.array([x[1] for x in g])
    print(f"{name:<44}{len(g):>6}{np.percentile(r, 5):>8.1f}{np.median(r):>8.1f}{np.percentile(r, 95):>8.1f}{100 * (az < 3).mean():>8.0f}%{np.median([x[2] for x in g]):>9.1f}")
own = np.array([x[0] for x in groups["own steps etc"] + groups["own shots"]])
print("\nA cut on side against mid, plus |az| under 10: share each group would call own")
for cut in (-60, -50, -40, -35, -30, -25, -20, -15):
    line = f"  cut {cut:>4} dB: "
    for name in ("own steps etc", "own shots", "others, true bearing dead ahead or behind", "others, anywhere", "agroed infected on you"):
        g = groups[name]
        if g:
            n = sum(x[0] < cut and x[1] < 10 for x in g)
            line += f"{name.split(',')[0][:14]}{'*' if 'ahead' in name else ''} {100 * n / len(g):>3.0f}%  "
    print(line)
print("\nOwn events that look most like the world (side against mid over -30):")
for name in ("own steps etc", "own shots"):
    print(" ", name, sorted(round(float(x[0]), 1) for x in groups[name] if x[0] > -30)[:14])

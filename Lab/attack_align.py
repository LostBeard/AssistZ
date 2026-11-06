"""The app's infected attack labels against the game's own log of how near the infected were, in recording time. Clock: calibrated report time vs game_ms of the same shot. Also the score with shots too far to hear left out."""
import csv, json, sys
import numpy as np
sys.path.insert(0, r"D:\users\tj\Projects\AssistZ\AssistZ\Lab")
from lab_clock import calibrated_reports
for name in sys.argv[1:]:
    run = f"D:/users/tj/Projects/AssistZ/Lab/runs/sniper_20261105_{name}"
    rows = list(csv.DictReader(open(f"{run}/shots_ai.csv", encoding="utf-8")))
    reports, _ = calibrated_reports(run)
    # calibrated_reports keeps the csv's order and length here; checked by the distance matching below
    assert len(rows) == len(reports)
    assert all(abs(float(a["distance"]) - b[3]) < 0.11 for a, b in zip(rows, reports))
    off = np.median([float(a["game_ms"]) / 1000.0 - b[0] for a, b in zip(rows, reports)])
    env = list(csv.DictReader(open(f"{run}/env.csv", encoding="utf-8")))
    t = np.array([float(e["game_ms"]) for e in env]) / 1000.0 - off
    near = np.array([float(e["infected_nearest"]) for e in env])
    near[near < 0] = 1e9
    near_at = lambda x: near[np.argmin(np.abs(t - x))]
    ev = [json.loads(x) for x in open(f"{run}/replay_ship_s0.jsonl", encoding="utf-8")]
    att = [e["sample"] / 48000.0 for e in ev if e["label"] == "infected attack"]
    other = [e["sample"] / 48000.0 for e in ev if e["label"] != "infected attack"]
    print(f"{name}: game clock offset {off:.1f} s")
    print("  nearest infected at each attack label:", " ".join(f"{near_at(x):.0f}" for x in att[:40]), "..." if len(att) > 40 else "")
    for lim in (5, 15, 60):
        a = sum(near_at(x) <= lim for x in att)
        o = sum(near_at(x) <= lim for x in other)
        print(f"  infected within {lim} m: attack labels {a}/{len(att)}; other events {o}/{len(other)}; time {100 * (near <= lim).mean():.0f}%")
    close = t[near <= 5]
    if len(close):
        secs = np.arange(int(close.min()), int(close.max()) + 1)
        print(f"  infected within 5 m from {close.min():.0f} to {close.max():.0f} s; attack labels start {min(att):.0f}, end {max(att):.0f}")
    d = np.array([r[3] for r in reports])
    print(f"  shot distances: over 150 m {(d > 150).sum()} of {len(d)}; max {d.max():.0f}; spread of those: " + " ".join(f"{v:.0f}" for v in sorted(d[d > 150])[:8]))
    # Score again without shots too far to hear (not the registered number, only to see what they cost)
    tr = np.array([r[0] for r in reports])
    pair = []
    for r in reports:
        nearby = [e for e in ev if -0.05 <= e["sample"] / 48000.0 - r[0] <= 0.15]
        if nearby:
            e = min(nearby, key=lambda x: abs(x["sample"] / 48000.0 - r[0]))
            if "dist" in e:
                pair.append((r[3], e["dist"]))
    pair = np.array(pair)
    print(f"  scored pairs with truth over 150 m: {(pair[:, 0] > 150).sum()} of {len(pair)}")
    k = pair[:, 0] <= 150
    err = np.abs(pair[k, 1] - pair[k, 0])
    print(f"  without them: median error {np.median(err):.1f} m, within 25% {100 * (err <= 0.25 * pair[k, 0]).mean():.1f}%, correlation {np.corrcoef(np.log2(pair[k, 1]), np.log2(pair[k, 0]))[0, 1]:.2f}, n {k.sum()}")

"""Hypothesis for the false boxes: scenery stays put to the pixel, an infected walks. From the log, how often is a calm infected still for 5, 10, 20 s? From the boxes, would 'a track that has moved at least M px within N seconds' keep real ones and drop false ones? For a still camera only. Usage: see_still.py"""
import collections, csv, json, math, os
import numpy as np
PX = 8.8 * 1600 / 960
BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
W = "L:/AssistZLabWork/frames/see"
COND = {4: "dusk", 5: "dusk", 6: "dusk", 7: "rain", 8: "rain", 9: "rain", 10: "noon", 11: "noon", 12: "noon", 13: "night"}
# 1. How still do they stand? Metres moved in each 5 s window of each slot.
moved = []
for n in COND:
    run = open(W + str(n) + "/run.txt").read().strip().replace("\\", "/")
    env = [e for e in csv.DictReader(open(run + "/env.csv")) if float(e["placed_x"]) != 0]
    for win in (5, 10, 20):
        for a, b in zip(env, env[win:]):
            if int(b["game_ms"]) - int(a["game_ms"]) > (win + 2) * 1000:
                continue
            moved.append((win, math.hypot(float(a["placed_x"]) - float(b["placed_x"]), float(a["placed_z"]) - float(b["placed_z"]))))
m = np.array(moved)
for win in (5, 10, 20):
    k = m[:, 0] == win
    print(f"infected moved in {win:>2} s: under 0.2 m {100 * (m[k, 1] < 0.2).mean():.0f}% of windows, under 1 m {100 * (m[k, 1] < 1).mean():.0f}%, median {np.median(m[k, 1]):.1f} m")
# 2. Boxes as tracks; a track is true if most of its boxes sit at the infected.
Track = lambda: dict(boxes=[])
res = collections.defaultdict(lambda: collections.Counter())
for n, cond in COND.items():
    d = W + str(n)
    man = {r["frame"]: r for r in csv.DictReader(open(d + "/manifest.csv", encoding="utf-8"))}
    by = collections.defaultdict(list)
    for o in json.load(open(d + "_yolo.json")):
        by[o["file"]].append(o)
    frames = sorted(by, key=lambda f: float(man[f]["game"]))
    live = []
    done = []
    for f in frames:
        g = float(man[f]["game"])
        got = [b for o in by[f] if o["kind"] in BARS for b in o["found"] if b["label"] == "person" and b["conf"] >= BARS[o["kind"]]]
        rel = man[f]["placed_rel"]
        near = float(man[f]["placed_near"]) if man[f]["placed_near"] != "" else None
        x = 800 + 840.34 * math.tan(math.radians(float(rel))) if rel != "" and abs(float(rel)) < 89 else None
        for b in got:
            cx, cy = b["x"] + b["w"] / 2, b["y"] + b["h"]
            true = x is not None and abs(cx - x) <= max(25.0, 1260.0 / max(near, 1.0)) + b["w"] / 2
            for t in live:
                if g - t["last"] <= 5 and abs(t["cx"] - cx) <= 60 and abs(t["cy"] - cy) <= 60:
                    t.update(cx=cx, cy=cy, last=g)
                    t["pts"].append((g, cx, cy, true))
                    break
            else:
                live.append(dict(cx=cx, cy=cy, last=g, pts=[(g, cx, cy, true)]))
    for t in live:
        p = t["pts"]
        if len(p) < 2:
            continue
        true = sum(q[3] for q in p) > len(p) / 2
        span = p[-1][0] - p[0][0]
        # When, if ever, has the track moved at least M px from its first place?
        res[cond]["true tracks" if true else "false tracks"] += 1
        for M in (5, 10, 20):
            t0 = next((q[0] for q in p if math.hypot(q[1] - p[0][1], q[2] - p[0][2]) >= M), None)
            kept = t0 is not None
            res[cond][f"{'true' if true else 'false'} kept at {M}px"] += kept
            if true and kept:
                res[cond][f"true delay at {M}px"] += t0 - p[0][0]
for c, v in res.items():
    print(c, "; ".join(f"{k} {v[k]:.0f}" for k in sorted(v)))

"""What a player would get from YOLOv8: per infected (slot), was it spotted, how soon, how steadily; does box height give distance; does the column give the bearing. Whole frame and native tiles. Usage: yolo_slots.py <frames dir> <results json> [...]"""
import collections, csv, json, os, sys
import numpy as np
PX = 8.8 * 1600 / 960
args = sys.argv[1:]
slots = []
rest_boxes = []
pairs_h = []
col_err = []
for frames_dir, res in zip(args[0::2], args[1::2]):
    man = {r["frame"]: r for r in csv.DictReader(open(os.path.join(frames_dir, "manifest.csv"), encoding="utf-8"))}
    run = open(os.path.join(frames_dir, "run.txt")).read().strip().replace("\\", "/")
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    spans = [(int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000, a["event"], float(a["distance"])) for a, b in zip(world, world[1:])]
    results = json.load(open(res))
    by = collections.defaultdict(list)
    for o in results:
        by[o["file"]].append(o)
    frames = sorted(by, key=lambda f: float(man[f]["game"]))
    for lo, hi, ev, dist in spans:
        fs = [f for f in frames if lo + 1 <= float(man[f]["game"]) < hi]
        if not fs:
            continue
        if ev == "rest":
            for f in fs:
                for o in by[f]:
                    for d in o["found"]:
                        if d["label"] == "person" and d["conf"] >= 0.5:
                            rest_boxes.append((os.path.basename(frames_dir), f, o["kind"], d))
            continue
        for bar in (0.5, 0.35):
            hits, first = 0, None
            looked = 0
            for f in fs:
                m = man[f]
                if not m["rel"] or abs(float(m["rel"])) >= 50 or float(m["state"]) != 0 and bar > 2:
                    continue
                looked += 1
                x = 800 + float(m["rel"]) * PX
                best = None
                for o in by[f]:
                    for d in o["found"]:
                        if d["label"] == "person" and d["conf"] >= bar and d["x"] - 25 <= x <= d["x"] + d["w"] + 25:
                            if best is None or d["conf"] > best["conf"]:
                                best = d
                if best:
                    hits += 1
                    if first is None:
                        first = float(m["game"]) - lo
                    if bar == 0.5:
                        pairs_h.append((float(m["near"]), best["h"], best["conf"], int(m["state"])))
                        col_err.append(((best["x"] + best["w"] / 2) - x) / PX)
            slots.append(dict(run=os.path.basename(frames_dir), dist=dist, bar=bar, hits=hits, looked=looked, first=first, near=np.median([float(man[f]["near"]) for f in fs])))
print("Per infected (one slot, 40 s), spotted at least once, person box at its column; share of frames; seconds to first sight:")
for bar in (0.5, 0.35):
    print(f" bar {bar}")
    for d in sorted({s["dist"] for s in slots}):
        g = [s for s in slots if s["dist"] == d and s["bar"] == bar and s["looked"] > 0]
        if g:
            print(f"   placed at {d:>4.0f} m: spotted {sum(s['hits'] > 0 for s in g)}/{len(g)}; frames {sum(s['hits'] for s in g)}/{sum(s['looked'] for s in g)}; first sight {[round(s['first']) if s['first'] is not None else '-' for s in g]}")
print(f"\nFalse boxes at bar 0.5 in rest frames: {len(rest_boxes)}; {collections.Counter(b[0] for b in rest_boxes)}")
if pairs_h:
    a = np.array(pairs_h)
    good = a[(a[:, 0] > 0)]
    print(f"\nBox height against distance, {len(good)} boxes: correlation of log height and log distance {np.corrcoef(np.log(good[:, 1]), np.log(good[:, 0]))[0, 1]:.2f}")
    for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 80)):
        k = (good[:, 0] >= lo) & (good[:, 0] < hi)
        if k.sum():
            print(f"   {lo}-{hi} m: height p25 {np.percentile(good[k, 1], 25):.0f} p50 {np.median(good[k, 1]):.0f} p75 {np.percentile(good[k, 1], 75):.0f} px, n {k.sum()}, states {dict(collections.Counter(good[k, 3].astype(int).tolist()))}")
    # A single constant: distance = K / height.
    K = np.median(good[:, 0] * good[:, 1])
    est = K / good[:, 1]
    ratio = est / good[:, 0]
    print(f"   distance = {K:.0f} / height: within 25% {100 * (np.abs(est - good[:, 0]) <= 0.25 * good[:, 0]).mean():.0f}%, within a factor of 2 {100 * ((ratio < 2) & (ratio > 0.5)).mean():.0f}%, median error {np.median(np.abs(est - good[:, 0])):.1f} m (fitted on these same boxes)")
    e = np.abs(col_err)
    print(f"Bearing from the box, degrees off the log: median {np.median(e):.1f}, p90 {np.percentile(e, 90):.1f}")

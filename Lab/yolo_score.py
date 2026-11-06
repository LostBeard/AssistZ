"""
Does YOLOv8 see the infected? For each frame with one spawned and in view: a person box (or any box) at its column, by distance and confidence bar; false alarms in rest frames. Natural infected may be about, so false alarms are an upper bound. Usage: yolo_score.py <frames dir> <results json> [...]
"""
import collections, csv, json, os, sys
import numpy as np
PX = 8.8 * 1600 / 960
args = sys.argv[1:]
pairs = list(zip(args[0::2], args[1::2]))
rows = []
for frames_dir, res in pairs:
    man = {r["frame"]: r for r in csv.DictReader(open(os.path.join(frames_dir, "manifest.csv"), encoding="utf-8"))}
    run = open(os.path.join(frames_dir, "run.txt")).read().strip().replace("\\", "/")
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    spans = [(int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000, a["event"], float(a["distance"])) for a, b in zip(world, world[1:])]
    results = json.load(open(res))
    by = collections.defaultdict(list)
    for o in results:
        by[o["file"]].append(o)
    for f, os_ in by.items():
        m = man[f]
        g = float(m["game"])
        span = next((s for s in spans if s[0] + 3 <= g < s[1]), None)
        found = {k["kind"]: k["found"] for k in os_}
        rows.append(dict(run=os.path.basename(frames_dir), frame=f, kind=span[2] if span else None, slot=span[3] if span else 0, near=float(m["near"]), rel=float(m["rel"]) if m["rel"] else None, found=found))
print(f"{len(rows)} frames: infected out {sum(r['kind'] == 'infected' for r in rows)}, rest {sum(r['kind'] == 'rest' for r in rows)}")
print("everything it ever says (whole frame):", dict(collections.Counter(d["label"] for r in rows for d in r["found"].get("full", [])).most_common(10)))
print("everything it ever says (tiles):", dict(collections.Counter(d["label"] for r in rows for k, v in r["found"].items() if k != "full" for d in v).most_common(10)))


def hit(r, kinds, bar, labels):
    x = 800 + r["rel"] * PX
    for k in kinds:
        for d in r["found"].get(k, []):
            if d["conf"] >= bar and (labels is None or d["label"] in labels) and d["x"] - 25 <= x <= d["x"] + d["w"] + 25:
                return True
    return False


def any_box(r, kinds, bar, labels):
    return any(d["conf"] >= bar and (labels is None or d["label"] in labels) for k in kinds for d in r["found"].get(k, []))


TILES = ("tile0", "tile1", "tile2")
print("\nFrames with a person box at the infected's column, by distance:")
print(f"{'set':<22}{'bar':>5} " + "".join(f"{b:>10}" for b in ("0-15", "15-30", "30-50", "50-70", "70-100", "100-150")) + "   rest frames with any person")
for name, kinds in (("whole frame", ("full",)), ("3 native tiles", TILES), ("both", ("full",) + TILES)):
    if not any(k in rows[0]["found"] for k in kinds):
        continue
    for bar in (0.1, 0.25, 0.5):
        line = f"{name:<22}{bar:>5} "
        for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 70), (70, 100), (100, 150)):
            g = [r for r in rows if r["kind"] == "infected" and r["rel"] is not None and abs(r["rel"]) < 50 and lo <= r["near"] < hi]
            line += f"{sum(hit(r, kinds, bar, {'person'}) for r in g):>5}/{len(g):<4}"
        rest = [r for r in rows if r["kind"] == "rest"]
        line += f"   {sum(any_box(r, kinds, bar, {'person'}) for r in rest)}/{len(rest)}"
        print(line)
print("\nPer run, tiles, bar 0.25:")
for run in sorted({r["run"] for r in rows}):
    line = f"  {run:<8}"
    for lo, hi in ((0, 15), (15, 30), (30, 50), (50, 70), (70, 100), (100, 150)):
        g = [r for r in rows if r["run"] == run and r["kind"] == "infected" and r["rel"] is not None and abs(r["rel"]) < 50 and lo <= r["near"] < hi]
        line += f"{sum(hit(r, TILES, 0.25, {'person'}) for r in g):>5}/{len(g):<4}"
    print(line)

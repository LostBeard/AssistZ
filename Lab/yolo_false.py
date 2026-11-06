"""YOLOv8 person boxes not at the placed infected's column, in every frame, grouped by where they sit in the picture: a box that stays put is scenery (or something standing still). Usage: yolo_false.py <frames dir> <results json> [...]"""
import collections, csv, json, os, sys
PX = 8.8 * 1600 / 960
args = sys.argv[1:]
for frames_dir, res in zip(args[0::2], args[1::2]):
    man = {r["frame"]: r for r in csv.DictReader(open(os.path.join(frames_dir, "manifest.csv"), encoding="utf-8"))}
    run = open(os.path.join(frames_dir, "run.txt")).read().strip().replace("\\", "/")
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    spans = [(int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000, a["event"], float(a["distance"])) for a, b in zip(world, world[1:])]
    print(os.path.basename(frames_dir), "manifest columns", list(next(iter(man.values())).keys()))
    results = json.load(open(res))
    frames = {o["file"] for o in results}
    for bar in (0.5, 0.35):
        groups = collections.defaultdict(list)
        total = 0
        false_frames = set()
        for o in results:
            m = man[o["file"]]
            g = float(m["game"])
            span = next((s for s in spans if s[0] + 1 <= g < s[1]), None)
            for d in o["found"]:
                if d["label"] != "person" or d["conf"] < bar:
                    continue
                at = span and span[2] == "infected" and m["rel"] and d["x"] - 25 <= 800 + float(m["rel"]) * PX <= d["x"] + d["w"] + 25
                if at:
                    continue
                total += 1
                false_frames.add(o["file"])
                groups[(round((d["x"] + d["w"] / 2) / 60) * 60, round((d["y"] + d["h"] / 2) / 40) * 40, round(d["h"] / 30) * 30)].append((round(g), o["kind"], round(d["conf"], 2), span[2] if span else None, span[3] if span else None, m["rel"], m["near"]))
        print(f" bar {bar}: {total} boxes not at the column, in {len(false_frames)}/{len(frames)} frames, {len(groups)} places")
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:8]:
            print(f"   centre x {k[0]} y {k[1]} height {k[2]}: {len(v)}; game {v[0][0]}..{v[-1][0]} s; first {v[0][1:]}")

"""
Scores person boxes against the placed infected's logged place (docs/PLAN.md, limits for a calm infected on the radar). A sighting: a person box on two of three consecutive frames within 60 px. The frame rate of the results is what the app would have to reach, so say what it was.
Usage: see_score.py <bar> <kinds: plan|plan2|plan3|plan4|tiles|full|both|zoom|zoomtiles> <frames dir> <results json> [...]
SEE_CUT=see27:665 ends that run at that game second. SEE_TOL=25 keeps the first, flat 25 px allowance. The default follows the truth's error: the log updates once a second and the infected walks, so its column is off by up to 1.5 m at its distance (840 px focal length at this view), and never under 25 px. plan2 is plan without boxes touching a tile's top. plan3 is setup 3 in docs/PLAN.md: plan2, and a sighting needs a track that has moved 10 px and had one box at 0.6.
"""
import collections, csv, json, math, os, sys
import numpy as np
# Focal length in px at 1600 wide, from the measured 8.8 px per degree at 960 wide at the centre. Flat projection: 87 degrees across.
F = 8.8 * 1600 / 960 * 180 / math.pi
VIEW = math.degrees(math.atan(800 / F))


def column(rel):
    return 800 + F * math.tan(math.radians(rel)) if abs(rel) < 89 else (-1e9 if rel < 0 else 1e9)


K = 1268
BAR = float(sys.argv[1])
# The setup docs/PLAN.md fixed before the fresh runs. The bar argument is ignored for it.
PLAN_BARS = {"full": 0.25, "tile0": 0.35, "tile1": 0.35, "tile2": 0.35}
KINDS = {"plan": tuple(PLAN_BARS), "plan2": tuple(PLAN_BARS), "plan3": tuple(PLAN_BARS), "plan4": tuple(PLAN_BARS), "tiles": ("tile0", "tile1", "tile2"), "full": ("full",), "both": ("full", "tile0", "tile1", "tile2"), "zoom": tuple(f"zoom{i}" for i in range(12)), "zoomtiles": tuple(f"zoom{i}" for i in range(12)) + ("tile0", "tile1", "tile2")}[sys.argv[2]]
args = sys.argv[3:]
BANDS = ((0, 10), (10, 20), (20, 30), (30, 50), (50, 70), (70, 100), (100, 150))


def boxes(os_):
    plan = sys.argv[2] in ("plan", "plan2", "plan3", "plan4")
    return [d for o in os_ if o["kind"] in KINDS for d in o["found"] if d["label"] == "person" and d["conf"] >= (PLAN_BARS[o["kind"]] if plan else BAR) and not (sys.argv[2] in ("plan2", "plan3", "plan4") and o["kind"] != "full" and abs(d["y"] - 120) < 2)]


def allowance(near):
    fixed = os.environ.get("SEE_TOL")
    return float(fixed) if fixed else max(25.0, 1260.0 / max(near, 1.0))


def moving(frames):
    """plan3. Links boxes into tracks, up to 5 s apart and within 60 px; a box counts once its track has moved SEE_MOVE px from its first place and has had one box at SEE_CONF."""
    need = float(os.environ.get("SEE_MOVE", "10"))
    strong = float(os.environ.get("SEE_CONF", "0.6"))
    tracks, out = [], []
    for g, bs in frames:
        for b in bs:
            cx, by = b["x"] + b["w"] / 2, b["y"] + b["h"]
            t = next((t for t in tracks if g - t["last"] <= 5 and abs(t["cx"] - cx) <= 60 and abs(t["by"] - by) <= 60), None)
            if t is None:
                t = dict(x0=cx, y0=by, moved=False, best=0.0)
                tracks.append(t)
            t.update(cx=cx, by=by, last=g)
            if not t["moved"] and math.hypot(cx - t["x0"], by - t["y0"]) >= need:
                t["moved"] = True
            t["best"] = max(t["best"], b["conf"])
            if t["moved"] and t["best"] >= strong:
                out.append((g, b))
    return out


def sightings(frames):
    """frames: list of (game, boxes). Yields (game, box) when a box has a neighbour on one of the previous two frames."""
    if sys.argv[2] in ("plan3", "plan4"):
        return moving(frames)
    out = []
    for i, (g, bs) in enumerate(frames):
        for b in bs:
            cx = b["x"] + b["w"] / 2
            if any(abs(p["x"] + p["w"] / 2 - cx) <= 60 for _, ps in frames[max(0, i - 2):i] for p in ps):
                out.append((g, b))
    return out


slot_rows, band = [], collections.defaultdict(lambda: [0, 0])
rest_seconds, rest_events, rest_boxes = 0.0, 0, []
slot_seconds, slot_false, slot_false_boxes = 0.0, 0, []
hit_y = []
near_err, dist_pairs = [], []
for frames_dir, res in zip(args[0::2], args[1::2]):
    man = {r["frame"]: r for r in csv.DictReader(open(os.path.join(frames_dir, "manifest.csv"), encoding="utf-8"))}
    run = open(os.path.join(frames_dir, "run.txt")).read().strip().replace("\\", "/")
    world = [w for w in csv.DictReader(open(run + "/world.csv")) if w["event"] in ("infected", "rest")]
    by = collections.defaultdict(list)
    results = json.load(open(res))
    for o in results:
        by[o["file"]].append(o)
    frames = sorted(by, key=lambda f: float(man[f]["game"]))
    # plan4: vision says nothing while the inventory is open.
    if sys.argv[2] == "plan4":
        shut = set(json.load(open(os.path.join(frames_dir, "inventory.json"))))
        print(f"   inventory: {sum(f in shut for f in frames)} of {len(frames)} frames left out")
        frames = [f for f in frames if f not in shut]
    # SEE_CUT=see27:665 stops a run there, when something outside the test changed it. Say so wherever it is used.
    for spec in filter(None, os.environ.get("SEE_CUT", "").split(",")):
        name, cut = spec.split(":")
        if name == os.path.basename(frames_dir):
            frames = [f for f in frames if float(man[f]["game"]) < float(cut)]
            print(f"   CUT at game {cut} s")
    step = float(np.median(np.diff([float(man[f]["game"]) for f in frames])))
    print(f"{os.path.basename(frames_dir)}: {len(frames)} frames, {step:.2f} s apart; infected at most {max(int(man[f]['count']) for f in frames)} at once")
    # A still-camera run must not have turned: a stray mouse did once (2026-11-06). The truth reads yaw per frame, the movement rule assumes none.
    pose = list(csv.DictReader(open(run + "/pose.csv")))[20:]
    # Not pitch or roll: the body settles on the ground by up to 9 degrees with nobody touching it. Aim drifts by up to 2.5 at rest.
    for col, limit in (("yaw", 0.5), ("aim_lr", 5.0), ("aim_ud", 5.0)):
        if col in pose[0]:
            v = [float(p[col]) for p in pose]
            d = [((x - v[0] + 180) % 360) - 180 for x in v]
            if max(d) - min(d) > limit:
                at = next(int(p["game_ms"]) / 1000 for p, x in zip(pose, d) if abs(x) > limit / 2)
                print(f"   VIEW MOVED: {col} range {max(d) - min(d):.1f} from game {at:.0f} s")
    for a, b in zip(world, world[1:]):
        lo, hi = int(a["game_ms"]) / 1000, int(b["game_ms"]) / 1000
        fs = [f for f in frames if lo + 1 <= float(man[f]["game"]) < hi]
        if not fs:
            continue
        seq = [(float(man[f]["game"]), boxes(by[f])) for f in fs]
        if a["event"] == "rest":
            rest_seconds += hi - lo - 1
            s = sightings(seq)
            # One event: sightings closer together than 5 s are the same one.
            last = None
            for g, box in s:
                if last is None or g - last > 5:
                    rest_events += 1
                    rest_boxes.append((os.path.basename(frames_dir), round(g), round(box["conf"], 2), round(box["x"]), round(box["y"]), round(box["h"])))
                last = g
            continue
        at = []
        for f, (g, bs) in zip(fs, seq):
            m = man[f]
            if m["placed_rel"] == "":
                at.append((g, [], None, None))
                continue
            rel, near = float(m["placed_rel"]), float(m["placed_near"])
            x = column(rel)
            tol = allowance(near)
            hit = [d for d in bs if d["x"] - tol <= x <= d["x"] + d["w"] + tol]
            # On runs that log it, in view also means at least one of three rays from the eye reaches it.
            los = m.get("placed_los", "")
            view = abs(rel) < VIEW and (los in ("", None) or int(los) >= 1)
            for lo_b, hi_b in BANDS:
                if lo_b <= near < hi_b and view:
                    band[(lo_b, hi_b)][1] += 1
                    band[(lo_b, hi_b)][0] += bool(hit)
            at.append((g, hit, near, view))
            if hit and view:
                best = max(hit, key=lambda d: d["conf"])
                near_err.append(math.degrees(math.atan((best["x"] + best["w"] / 2 - 800) / F)) - rel)
                dist_pairs.append((near, best["h"]))
                hit_y.append((best["y"], best["y"] + best["h"]))
        # With an infected out, any sighting away from it is false too (natural ones are removed on these runs).
        away = []
        for f, (g, bs) in zip(fs, seq):
            m = man[f]
            if m["placed_rel"] == "":
                away.append((g, bs))
                continue
            x = column(float(m["placed_rel"]))
            tol = allowance(float(m["placed_near"]))
            away.append((g, [d for d in bs if not d["x"] - tol <= x <= d["x"] + d["w"] + tol]))
        slot_seconds += hi - lo - 1
        last = None
        for g, box in sightings(away):
            if last is None or g - last > 5:
                slot_false += 1
                slot_false_boxes.append((os.path.basename(frames_dir), round(g), round(box["conf"], 2), round(box["x"]), round(box["y"]), round(box["h"])))
            last = g
        # Time in view at 10-30 m, and the first sighting there.
        inview = [(g, h) for g, h, n, v in at if v and n is not None and 10 <= n < 30]
        start = inview[0][0] if inview else None
        s = sightings([(g, h) for g, h in inview])
        slot_rows.append(dict(run=os.path.basename(frames_dir), placed=float(a["distance"]), seconds=step * len(inview), first=(s[0][0] - start) if s else None, states=collections.Counter(man[f]["placed_state"] for f in fs).most_common(2)))
print(f"\nBar {BAR}, {sys.argv[2]}. Frames with a person box at the infected's column while in view, by logged distance:")
for b in BANDS:
    h, n = band[b]
    print(f"  {b[0]:>3}-{b[1]:<3} m: {h}/{n}" + (f" ({100 * h / n:.0f}%)" if n else ""))
ok = [s for s in slot_rows if s["seconds"] >= 5]
got = [s for s in ok if s["first"] is not None and s["first"] <= 5]
print(f"\nLIMIT sees it: slots in view at 10-30 m for 5 s or more: {len(ok)}; sighted within 5 s: {len(got)} ({100 * len(got) / max(1, len(ok)):.0f}%, limit 90, at least 12 slots)")
print("  those missed:", [(s["run"], s["placed"], s["seconds"], s["first"], s["states"]) for s in ok if s not in got])
print(f"LIMIT false: {rest_events} sightings in {rest_seconds / 60:.1f} min of rest frames ({rest_events / max(rest_seconds / 300, 1e-9):.2f} per 5 min, limit 1); {rest_boxes[:10]}")
print(f"   and during slots, away from the infected (no limit written, reported): {slot_false} in {slot_seconds / 60:.1f} min ({slot_false / max(slot_seconds / 300, 1e-9):.2f} per 5 min); {slot_false_boxes[:6]}")
if hit_y:
    y = np.array(hit_y)
    print(f"   boxes at the infected, top y p5 {np.percentile(y[:, 0], 5):.0f} p50 {np.median(y[:, 0]):.0f}; bottom y p50 {np.median(y[:, 1]):.0f} p95 {np.percentile(y[:, 1], 95):.0f}")
if near_err:
    e = np.abs(near_err)
    print(f"LIMIT bearing: median {np.median(e):.2f} deg, p90 {np.percentile(e, 90):.2f} (limit median 3), {len(e)} boxes")
    a = np.array(dist_pairs)
    est = K / a[:, 1]
    print(f"LIMIT distance: K {K} fixed: within 25% {100 * (np.abs(est - a[:, 0]) <= 0.25 * a[:, 0]).mean():.0f}% (limit 75), median error {np.median(np.abs(est - a[:, 0])):.1f} m, {len(a)} boxes; K that would fit here {np.median(a[:, 0] * a[:, 1]):.0f}")

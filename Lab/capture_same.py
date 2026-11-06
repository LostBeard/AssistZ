"""Does graph capture change what YOLOv8 finds? Same 80 frames, capture off and on: every person box at bar 0.25 must match, by confidence and place."""
import collections, json
W = "L:/AssistZLabWork/frames/"
off = json.load(open(W + "capture_640_tiles__capture_off.json"))
on = json.load(open(W + "capture_640_tiles__capture_on.json"))
key = lambda o: (o["file"], o["kind"])
a = {key(o): o["found"] for o in off}
b = {key(o): o["found"] for o in on}
print("inferences", len(a), len(b), "same keys", a.keys() == b.keys())
same = diff = 0
worst_conf = worst_px = 0.0
extra = 0
for k in a:
    x = sorted((d for d in a[k] if d["conf"] >= 0.1), key=lambda d: (d["label"], d["x"], d["y"]))
    y = sorted((d for d in b[k] if d["conf"] >= 0.1), key=lambda d: (d["label"], d["x"], d["y"]))
    if len(x) != len(y) or [d["label"] for d in x] != [d["label"] for d in y]:
        extra += 1
        continue
    for p, q in zip(x, y):
        worst_conf = max(worst_conf, abs(p["conf"] - q["conf"]))
        worst_px = max(worst_px, *(abs(p[f] - q[f]) for f in "xywh"))
    same += 1
print(f"inferences with the same boxes at bar 0.1: {same}; different box lists: {extra}; worst confidence change {worst_conf:.4f}; worst box change {worst_px:.1f} px (logged to whole px)")

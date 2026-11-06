"""The inventory as a rule of numbers: three flat grey strips at fixed fractions of the picture, either side of each header's text. Fit the strip colours on see17 and see27, then count on every run how many strips a frame matches. Lab/see_inventory.py writes the per-frame result for the scorer."""
import csv, json, os, sys
import numpy as np
from PIL import Image
SZ = (400, 225)
# (x0, x1, y0, y1) at 400 x 225: each header's ends, away from its centred text.
STRIPS = {"vicinity_l": (26, 56, 17, 24), "vicinity_r": (98, 128, 17, 24), "equipment_l": (272, 302, 17, 24), "equipment_r": (344, 374, 17, 24), "hands_l": (149, 179, 152, 159), "hands_r": (218, 248, 152, 159)}
def feats(path):
    a = np.asarray(Image.open(path).convert("RGB").resize(SZ), dtype=np.float32)
    out = {}
    for k, (x0, x1, y0, y1) in STRIPS.items():
        r = a[y0:y1, x0:x1]
        out[k] = (r.mean(axis=(0, 1)), float(r.std()))
    return out
if __name__ == "__main__":
    S = "C:/Users/TJ/AppData/Local/Temp/claude/631e58d4667bd5fc/ae7b0fb7-365f-4d31-b173-80673f47f899/scratchpad/"
    open_frames = json.load(open(S + "inv_open.json"))
    per = {k: [] for k in STRIPS}
    stds = {k: [] for k in STRIPS}
    for p in open_frames:
        f = feats(p)
        for k in STRIPS:
            per[k].append(f[k][0]); stds[k].append(f[k][1])
    for k in STRIPS:
        a = np.array(per[k])
        print(f"{k}: colour mean {a.mean(axis=0).round(1)} range {a.min(axis=0).round(1)} to {a.max(axis=0).round(1)}; std median {np.median(stds[k]):.1f} max {np.max(stds[k]):.1f}")

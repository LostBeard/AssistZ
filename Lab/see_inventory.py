"""
Is this frame the inventory screen? Six flat, neutral grey strips at fixed fractions of the picture, either side of the VICINITY, EQUIPMENT and HANDS headers' text. Numbers only, so no game art ships. Writes inventory.json beside each run's frames, and reports how many frames match. Default interface scale at 16:9 only, as far as tested.
Usage: see_inventory.py <run number> [...]
"""
import json, multiprocessing, os, sys
import numpy as np
from PIL import Image
SZ = (400, 225)
STRIPS = ((26, 56, 17, 24), (98, 128, 17, 24), (272, 302, 17, 24), (344, 374, 17, 24), (149, 179, 152, 159), (218, 248, 152, 159))
LOW, HIGH, TINT, ROUGH = 110.0, 142.0, 25.0, 9.0


def matches(path):
    a = np.asarray(Image.open(path).convert("RGB").resize(SZ), dtype=np.float32)
    count = 0
    for x0, x1, y0, y1 in STRIPS:
        r = a[y0:y1, x0:x1]
        m = r.mean(axis=(0, 1))
        count += bool(LOW <= m.min() and m.max() <= HIGH and m.max() - m.min() <= TINT and r.std() <= ROUGH)
    return count


def job(p):
    return os.path.basename(p), matches(p)


if __name__ == "__main__":
    for n in sys.argv[1:]:
        d = f"L:/AssistZLabWork/frames/see{n}"
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".jpg"))
        with multiprocessing.Pool(6) as pool:
            res = pool.map(job, files, chunksize=40)
        hist = np.bincount([c for _, c in res], minlength=7)
        hits = [f for f, c in res if c == 6]
        json.dump(hits, open(d + "/inventory.json", "w"))
        print(f"see{n}: {len(files)} frames; strips matched 0..6: {hist.tolist()}; inventory {len(hits)}", flush=True)

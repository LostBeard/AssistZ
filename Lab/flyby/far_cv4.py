"""
Does far gunfire in training hide heli crash or gas? Each world run held out, with and without the far rows: its events by class. Far gunfire was being called heli crash, so that class is the one at risk.
Usage (from Lab/flyby): python far_cv4.py <cache.pkl>
"""
import pickle, sys
import numpy as np
from world_classes import CLASSES, WORLD, gunstorm_rows
from far_cv import FAR, fit

base, DX, Dy = pickle.load(open(sys.argv[1], "rb"))
far = [r for m in FAR for r in gunstorm_rows(m, "_far")]
print(f"far rows: gunfire {sum(r['y'] == 'gunfire' for r in far)}, other {sum(r['y'] == 'other' for r in far)}, thunder {sum(r['y'] == 'thunder' for r in far)}")
tot = {}
for n in WORLD:
    rest = [r for r in base if not (r["run"] == n and n in WORLD)]
    held = [r for r in base if r["run"] == n]
    for tag, rows in (("without far", rest), ("with far", rest + far)):
        A, B = fit(rows)
        got = [CLASSES[int(np.argmax(A @ np.asarray(r["x"], dtype=np.float64) + B))] for r in held]
        for c in ("heli_crash", "gas_strike", "other"):
            g = [w for r, w in zip(held, got) if r["y"] == c]
            t = tot.setdefault((tag, c), [0, 0, 0])
            t[0] += sum(w == c for w in g); t[1] += len(g); t[2] += sum(w == "gunfire" for w in g)
for (tag, c), v in tot.items():
    print(f"  {tag:<12} {c:<11} right {v[0]}/{v[1]}, called gunfire {v[2]}")

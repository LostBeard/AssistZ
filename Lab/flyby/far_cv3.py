"""
Screen for the candidate namer on phase_gates7.py's holdouts, from stored embeddings (tag _r7): candidate labels as _off, the same rows with no far runs as _off0, so raw word is compared with raw word. No burst vote: a screen, not the gate.
Usage (from Lab/flyby): python far_cv3.py <cache.pkl>
"""
import json, pickle, sys
import numpy as np
from world_classes import CLASSES, U, gunstorm_rows
from far_cv import FAR, fit

base, DX, Dy = pickle.load(open(sys.argv[1], "rb"))
far_rows = {n: gunstorm_rows(n, "_far") for n in FAR}
fits = {"_off": fit(base + [r for m in FAR for r in far_rows[m]]), "_off0": fit(base)}
keys = ["221544", "223129", "224711", "144529", "145757", "152249", "151013", "121745", "135013", "085810", "110232", "133615", "160723", "162444", "215004", "215827", "220700"]
for tag, (A, B) in fits.items():
    for k in keys:
        run = U + "20261104_" + k
        with open(f"{run}/replay{tag}_s0.jsonl", "w", encoding="utf-8") as out:
            for line in open(f"{run}/replay_r7_s0.jsonl", encoding="utf-8"):
                e = json.loads(line)
                w_ = CLASSES[int(np.argmax(A @ np.asarray(e["embed"], dtype=np.float64) + B))]
                e["label"] = "gunshot" if w_ == "gunfire" else (w_.replace("_", " ") if w_ != "other" else (e["clap"] if e["clap"] not in ("gunshot", "machine gun fire") else "rain"))
                out.write(json.dumps({k2: v for k2, v in e.items() if k2 not in ("scores", "embed")}) + "\n")
print("written")

"""Is CLAP the same between two replays of one recording? Same events, labels, scores and embeddings. Usage: clap_same.py <run> <tag a> <tag b>"""
import json, sys
import numpy as np
run, a, b = sys.argv[1].replace("\\", "/"), sys.argv[2], sys.argv[3]
A = [json.loads(x) for x in open(f"{run}/replay{a}_s0.jsonl", encoding="utf-8")]
B = [json.loads(x) for x in open(f"{run}/replay{b}_s0.jsonl", encoding="utf-8")]
print(f"{len(A)} and {len(B)} events")
ka, kb = {e["sample"]: e for e in A}, {e["sample"]: e for e in B}
both = sorted(set(ka) & set(kb))
print(f"same positions {len(both)}; only in one: {len(set(ka) ^ set(kb))}")
label = sum(ka[s]["label"] != kb[s]["label"] for s in both)
clip = sum(ka[s]["clip"] != kb[s]["clip"] for s in both)
scores = max(float(np.max(np.abs(np.array(ka[s]["scores"]) - np.array(kb[s]["scores"])))) for s in both)
embed = max(float(np.max(np.abs(np.array(ka[s]["embed"]) - np.array(kb[s]["embed"])))) for s in both if "embed" in ka[s] and "embed" in kb[s])
print(f"labels differ {label}, clips differ {clip}; largest score difference {scores:.6f}, largest embedding difference {embed:.6f}")
print("BIT-IDENTICAL" if label == 0 and clip == 0 and scores == 0 and embed == 0 else "NOT IDENTICAL")

"""Same recording replayed on two builds: are clips, scores and embeddings bit-identical? Usage: python compare_bits.py <run> <tagA> <tagB>"""
import json, sys
import numpy as np

run, A, B = sys.argv[1:4]
a = [json.loads(x) for x in open(f"{run}/replay{A}_s0.jsonl", encoding="utf-8")]
b = [json.loads(x) for x in open(f"{run}/replay{B}_s0.jsonl", encoding="utf-8")]
print(len(a), len(b), "events")
assert len(a) == len(b)
clip = label = scores = embed = 0
smax = emax = 0.0
for x, y in zip(a, b):
    clip += x["clip"] == y["clip"] and x["sample"] == y["sample"]
    label += x["label"] == y["label"] and x["clap"] == y["clap"]
    sa, sb = np.array(x["scores"], dtype=np.float64) if not isinstance(x["scores"], dict) else np.array(list(x["scores"].values()), dtype=np.float64), np.array(y["scores"], dtype=np.float64) if not isinstance(y["scores"], dict) else np.array(list(y["scores"].values()), dtype=np.float64)
    ea, eb = np.array(x["embed"], dtype=np.float64), np.array(y["embed"], dtype=np.float64)
    scores += bool(np.array_equal(sa, sb))
    embed += bool(np.array_equal(ea, eb))
    smax = max(smax, float(np.abs(sa - sb).max()))
    emax = max(emax, float(np.abs(ea - eb).max()))
n = len(a)
print(f"same clip {clip}/{n}, same label {label}/{n}, scores bit-identical {scores}/{n} (max diff {smax:.3g}), embeddings bit-identical {embed}/{n} (max diff {emax:.3g})")

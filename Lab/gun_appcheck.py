"""The app's gun call against the JSON on the same embedding. Usage: gun_appcheck.py <replay jsonl> <probe json>"""
import collections, json, sys
import numpy as np
ev = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8")]
p = json.load(open(sys.argv[2], encoding="ascii"))
mean, std, W, B = np.array(p["Mean"]), np.array(p["Std"]), np.array(p["Weights"]), np.array(p["Bias"])
KNOWN = {"AKM": ("AKM", "assault"), "AugShort": ("AUG", "assault"), "M4A1": ("M4A1", "assault"), "SVD": ("SVD", "sniper"), "Winchester70": ("Winchester", "sniper"), "CZ75": ("CZ75", "pistol"), "FNX45": ("FNX45", "pistol"), "Mosin9130": (None, "sniper"), "FAL": (None, "assault"), "MP5K": (None, "smg")}
same = diff = 0
said = collections.Counter()
for e in ev:
    if e["raw"] not in ("gunshot", "automatic fire") or "embed" not in e:
        continue
    s = W @ ((np.array(e["embed"]) - mean) / std) + B
    q = np.exp(s - s.max()); q /= q.sum()
    top = p["Classes"][int(np.argmax(q))]
    name, family = None, None
    if q.max() >= 0.8 and KNOWN[top][0]:
        name, family = KNOWN[top]
    else:
        fam = collections.Counter()
        for c, v in zip(p["Classes"], q):
            fam[KNOWN[c][1]] += v
        f, v = fam.most_common(1)[0]
        family = f if v >= 0.8 else None
    a = (e.get("gun"), e.get("family"))
    if a == (name, family):
        same += 1
    else:
        diff += 1
    said[a] += 1
print(f"app against the JSON: same {same}, different {diff}; said {said.most_common(8)}")
